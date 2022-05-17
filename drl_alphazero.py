from functools import singledispatchmethod
import random
from math import sqrt
from dataclasses import dataclass
from collections import deque

import numpy as np
import ray

from inada_framework import Model, optimizers, no_grad
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
    def __init__(self, action_size, sampling_limits = 20, c_puct = 1.0):
        self.network = PolicyValueNet(action_size)

        # ハイパーパラメータ
        self.sampling_limits = sampling_limits
        self.c_puct = c_puct

    # 引数はパラメータが保存されたファイルの名前か、同じモデルのインスタンス
    def reset(self, arg, simulations_num):
        self.load(arg)
        self.simulations_num = simulations_num

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

    @load.register(dict)
    def __(self, weights):
        self.network.set_weights(weights)


    def __call__(self, board):
        return self.get_action(board)

    def get_action(self, board, count = None):
        root_state = board.state
        state, policy = self.__search(board, root_state)
        placable = self.placable[root_state]

        if count is not None and count < self.sampling_limits:
            action = random.choices(placable, policy)[0]
        else:
            indices = np.where(policy == policy.max())[0]
            action_index = indices[0] if len(indices) == 1 else random.choice(indices)
            action = placable[action_index]

        # コンピュータとして使う場合、以降の処理は不要
        if count is None:
            return action

        # 方策の形状をニューラルネットワークのものと合わせる
        mcts_policy = np.zeros(board.action_size, dtype = np.float32)
        mcts_policy[np.array(placable)] = policy
        return action, state, mcts_policy


    # シミュレーションを行なって得た方策を返すメソッド
    def __search(self, board, root_state):
        if root_state in self.seen_state:
            state_ndarray = board.get_state_ndarray()
        else:
            state_ndarray = self.__expand(board, root_state, True)

        # 探索の初期状態ではランダムな手が選ばれやすくなるように、ノイズをかける
        P = self.P[root_state]
        P = 0.75 * P + 0.25 * self.rng.dirichlet(alpha = np.full(len(P), 0.35))

        # PUCT アルゴリズムで指定回数だけシミュレーションを行う
        for __ in range(self.simulations_num):
            self.__evaluate(board, root_state)
            board.set_state(root_state)

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
            action_index = indices[0] if len(indices) == 1 else random.choice(indices)

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




# 実態はクラスだが、リストやタプルに比べて、メモリ容量は小さく、アクセスも速い
@dataclass(repr = False, eq = False)
class ReplayData:
    state: np.ndarray
    mcts_policy: np.ndarray
    reward: int


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

        states = np.stack([x.state for x in selected])
        mcts_policys = np.stack([x.mcts_policy for x in selected])
        rewards = np.array([x.reward for x in selected], dtype = np.float32)

        return states, mcts_policys, rewards




@ray.remote(num_cpus = 1, num_gpus = 0)
def alphazero_play(simulations_num, weights: dict):
    # 環境
    board = Board()

    # エージェント (先攻・後攻で分けない)
    agent = AlphaZeroAgent(board.action_size)
    agent.reset(weights, simulations_num)

    # 実際に１ゲームプレイして、対局データを収集する
    memory = []
    for count in range(board.action_size):
        action, state, mcts_policy = agent.get_action(board, count)
        memory.append(ReplayData[state, mcts_policy, board.turn])

        board.put_stone(action)
        if not board.can_continue():
            break

    # 最後に手を打ったプレイヤーとその報酬の情報で、各対局データの報酬を確定する
    final_player = board.turn
    reward = board.reward
    for data in memory:
        data.reward = reward if data.reward == final_player else -reward

    return memory




# 自己対戦によって、パラメータを学習するための闘技場
class AlphaZeroArena:
    def __init__(self, action_size, buffer_size, batch_size, simulations_num = 800):
        self.network = PolicyValueNet(action_size)
        self.optimizer = optimizers.Momentum(lr = 0.0005).setup(self.network)
        self.optimizer.add_hook(ff)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

    def save(self, file_name):
        self.network.save_weights(SelfMatch.get_path(file_name).format("parameters"))

    def fit(self, episodes, update_interval, simulations_num = 800):
        assert not episodes % update_interval

        # 並列実行を行うための初期設定
        ray.init()

        for __ in range(episodes // update_interval):
            weights = ray.put(self.network.get_weights())
            remains = [alphazero_play.remote(simulations_num, weights) for __ in range(update_interval)]
            self.self_match(remains)

    def self_match(self, remains):
        while remains:
            finished, remains = ray.wait(remains, num_returns = 1)
            self.replay_buffer.add(ray.get(finished[0]))

    def update(self, update_epochs = 5):
        for __ in range(update_epochs):
            for states, mcts_policys, rewards in self.replay_buffer:
                policy, value = self.network(states)

                self.optimizer.update()