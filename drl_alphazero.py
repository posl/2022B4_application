from functools import singledispatchmethod
from math import sqrt
from collections import deque

import numpy as np

from inada_framework import Model, cuda, optimizers, no_grad
import inada_framework.layers as dzl
import inada_framework.functions as dzf
from drl_train_utilities import SelfMatch, simple_plan, corners_plan
from board import Board



# PV-MCTS での過去の探索状況を近似するニューラルネットワーク
class PolicyValueNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.p = dzl.Affine(action_size)
        self.v = dzl.Affine(1)

    def forward(self, x):
        policy = dzf.softmax(self.p(x))
        value = dzf.tanh(self.v(x))
        return policy, value




# モンテカルロ木探索の要領で、深層強化学習を駆使し、行動選択をするエージェント (コンピュータとしても使う)
class AlphaZeroAgent:
    def __init__(self, action_size, simulations_num = 800, c_puct = 1.0, alpha = 0.35, epsilon = 0.25):
        self.network = PolicyValueNet(action_size)

        # ハイパーパラメータ
        assert simulations_num > 0
        self.simulations_num = simulations_num
        self.c_puct = c_puct
        self.alpha = alpha
        self.epsilon = epsilon

    # 引数はパラメータが保存されたファイルの名前か、同じモデルのインスタンス
    def reset(self, arg):
        self.load(arg)

        # それぞれ、過去の探索割合を近似したある種の方策、過去の勝率も参考にした累計行動価値、各状態・行動の探索回数
        self.P = {}
        self.T = {}
        self.N = {}

        self.seen_state = set()
        self.placable = {}
        self.rng = np.random.default_rng()


    # クラスメソッドのオーバーロードによって、コンピュータとしての使用時と、学習時で処理を分ける
    @singledispatchmethod
    def load(self, arg):
        message = f"\"{arg.__class__}\" is not supported."
        raise TypeError(message)

    @load.register(str)
    def __(self, file_name):
        self.network.load_weights(SelfMatch.get_path(file_name).format("parameters"))

    @load.register(PolicyValueNet)
    def __(self, model):
        self.network.copy_weights(model)


    def __call__(self, board):
        return self.get_action(board, False)

    def get_action(self, board, train_flag = True):
        root_state = board.state
        state, policy = self.__search(board, root_state)

        indices = np.where(policy == policy.max())[0]
        placable = self.placable[root_state]
        action = placable[indices[0] if len(indices) == 1 else self.rng.choice(indices)]

        if train_flag:
            # 方策の形状はニューラルネットワークのものと合わせる
            mcts_policy = np.zeros(board.action_size, dtype = np.float32)
            mcts_policy[np.array(placable)] = policy
            return action, state, mcts_policy

        # コンピュータとして使う場合の処理を軽くしている
        return action


    # シミュレーションを行なって得た方策を返すメソッド
    def __search(self, board, root_state):
        if root_state in self.seen_state:
            state_ndarray = board.get_state_ndarray()
        else:
            state_ndarray = self.__expand(board, root_state, True)

        # 探索の初期状態ではランダムな手が選ばれやすくなるように、ノイズをかける
        if self.alpha is not None:
            P = self.P[root_state]
            P = (1. - self.epsilon) * P + self.epsilon * self.rng.dirichlet(np.full(len(P), self.alpha))

        # PUCT アルゴリズムで指定回数だけシミュレーションを行う
        for __ in range(self.simulations_num):
            board.set_state(root_state)
            self.__evaluate(board, root_state)

        # 評価値が高いほど探索回数が多くなるように収束するので、それを方策として使う
        N = self.N[root_state]
        mcts_policy = N / N.sum()
        return state_ndarray, mcts_policy


    # 子盤面を展開し、その方策・累計行動価値・探索回数の初期化を行うメソッド
    def __expand(self, board, state, root_flag = False):
        self.seen_state.add(state)
        state_ndarray = board.get_state_ndarray()
        policy, value = self.network(state_ndarray[None, :])

        placable = board.list_placable()
        self.placable[state] = placable

        self.P[state] = policy[0, np.array(placable)]
        self.T[state] = np.zeros(len(placable), dtype = np.float32)
        self.N[state] = np.zeros(len(placable), dtype = np.float32)

        # ルート盤面の展開時は評価地の代わりに、学習時用にニューラルネットワークへの入力としての状態データを返す
        if root_flag:
            return state_ndarray

        # それ以外は、ニューラルネットワークの出力である過去の勝率を返す
        return value[0, 0]


    # 初到達の盤面の展開をしてゲーム木を大きくしながら、ゲームの報酬や過去のデータをもとに評価値を返すメソッド
    def __evaluate(self, board, state, continue_flag = 1):
        if not continue_flag:
            # ゲーム終了時は最後に手を打ったプレイヤーの報酬を返す
            return board.reward

        if state in self.seen_state:
            # 右辺の第１項が過去の結果を勘案しつつ探索を促進する項で、第２項が勝率を見て活用を促進する項
            N = self.N[state]
            pucts = (self.c_puct * sqrt(N.sum()) * self.P[state]) / (N + 1) + self.T[state] / (N + 1e-15)

            # np.argmax を使うと選択が前にある要素に偏るため、np.where で取り出したインデックスからランダムに選ぶ
            indices = np.where(pucts == pucts.max())[0]
            action_index = indices[0] if len(indices) == 1 else self.rng.choice(indices)

            action = self.placable[state][action_index]
            board.put_stone(action)
            value = self.__evaluate(board, board.state, board.can_continue())

            # 結果を反映させる
            self.T[state][action_index] += value
            self.N[state][action_index] += 1
        else:
            # 展開していなかった盤面の場合は、ニューラルネットワークの出力である過去の勝率を評価値として返す
            value = self.__expand(board, state)

        # １つ前に手を打ったプレイヤーにとっての評価値を返す (手番が変わっていた場合は、視点が逆になる)
        if continue_flag == 1:
            return -value
        return value




# 過去の探索の記録を残し、それらを順に取り出すことのできるイテラブル
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size = 64):
        self.buffer = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.indices = None
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.buffer)

    # １エピソード分のデータがリストとして、まとめて格納される
    def add(self, datas: list):
        self.buffer.extend(datas)

    def __iter__(self):
        self.indices = self.rng.permutation(len(self.buffer))
        return self

    # ランダムに並び替えた経験データをバッチサイズ分ずつ順に取り出す
    def __next__(self):
        indices = self.indices
        if indices:
            indices, self.indices = np.split(indices, [self.batch_size])
        else:
            self.indices = None
            raise StopIteration()

        buffer = self.buffer
        selected = [buffer[i] for i in indices]

        states = np.stack([x[0] for x in selected])
        mcts_policys = np.stack([x[1] for x in selected])
        rewards = np.array([x[2] for x in selected], dtype = np.float32)

        return states, mcts_policys, rewards




# 自己対戦によって、パラメータを学習するための闘技場
class AlphaZeroArena:
    def __init__(self, action_size):
        self.network = PolicyValueNet(action_size)

    def save(self, file_name):
        self.network.save_weights(SelfMatch.get_path(file_name).format("parameters"))

    def fit(self):
        # self_plays -> update
        pass

    def self_plays(self):
        pass

    def __self_play(self):
        pass

    def update(self):
        pass