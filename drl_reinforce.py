import random

import numpy as np

from inada_framework import Model, cuda, optimizers, no_grad
import inada_framework.layers as dzl
import inada_framework.functions as dzf
from drl_utilities import SelfMatch, eval_computer
from board import Board



# 確率形式に変換する前の最適方策を出力するニューラルネットワーク
class PolicyNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = dzl.Affine(512)
        self.l2 = dzl.Affine(action_size)

    def forward(self, x):
        x = dzf.relu(self.l1(x))
        return self.l2(x)


# モンテカルロ法でパラメータを修正する方策ベースのエージェント
class ReinforceAgent:
    def __init__(self, action_size, gamma = 0.9, lr = 0.0005, to_gpu = False):
        self.memory = []
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.use_gpu = to_gpu and cuda.gpu_enable

    # エージェントを動かす前に呼ぶ必要がある
    def reset(self):
        self.pi = PolicyNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr).setup(self.pi)
        if self.use_gpu:
            self.pi.to_gpu()

    def save(self, file_path, is_yet = None):
        self.pi.save_weights(file_path + ".npz")

    def load_to_restart(self, file_path):
        self.pi.load_weights(file_path + ".npz")
        if self.use_gpu:
            self.pi.to_gpu()


    # エージェントを関数形式で使うとその方策に従った行動が得られる
    def __call__(self, board):
        action, __ = self.get_action(board)
        return action

    def get_action(self, board):
        xp = cuda.cp if self.use_gpu else np
        state = board.get_state_ndarray(xp)
        policy = self.pi(state[None, :])

        # 学習時は方策を合法手のみに絞って、確率形式に変換し、それと学習の進行状況に応じて行動を選択する
        placable = board.list_placable()
        probs = dzf.softmax(policy[:, np.array(placable)])

        if len(placable) == 1:
            action_index = 0
        else:
            action_index = random.choices(range(len(placable)), probs.data[0])[0]

        # 行動が選ばれる確率も一緒に出力する
        return placable[action_index], probs[0, action_index]


    def add(self, data):
        self.memory.append(data)

    # ニューラルネットワークで近似したある方策に従った時の収益の期待値の勾配を求め、パラメータを更新する
    def update(self):
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = self.gamma * G + reward

            # 負けの場合は、報酬に１回分余分に gamma が掛かっている
            if prob is not None:
                loss += -G * dzf.log(prob)

        # メモリの解放を先に行う
        self.memory.clear()

        self.pi.clear_grads()
        loss.backward()
        self.optimizer.update()




# 実際にコンピュータとして使われるクラス
class ReinforceComputer:
    def __init__(self, action_size, to_gpu = False):
        self.each_pi = []
        self.action_size = action_size
        self.use_gpu = to_gpu and cuda.gpu_enable

    def reset(self, file_name, gamma, turn):
        file_path = Reinforce.get_path(file_name).format("parameters")
        file_path += "-{}_{}".format(str(gamma * 100)[:2], turn)

        # 各エージェントの方策を表すインスタンス変数をリセットし、新たに登録する
        each_pi = self.each_pi
        each_pi.clear()
        use_gpu = self.use_gpu

        # 同じ条件で学習した３人のエージェントの中から、２人だけ、ランダムに選ぶ
        for i in random.sample(range(3), 2):
            pi = PolicyNet(self.action_size)
            each_pi.append(pi)

            pi.load_weights(file_path + f"{i}.npz")
            if use_gpu:
                pi.to_gpu()

    def __call__(self, board):
        placable = board.list_placable()
        if len(placable) == 1:
            return placable[0]

        xp = cuda.cp if self.use_gpu else np
        state = board.get_state_ndarray(xp)[None, :]

        # 学習済みのパラメータを使うだけなので、動的に計算グラフを構築する必要はない
        with no_grad():
            pi0, pi1 = self.each_pi
            policy = pi0(state).data + pi1(state).data

        # 各エージェントが提案するスコア値の和をとり、それを元にした重み付きランダムサンプリングで行動を選択する
        policy = policy[0, np.array(placable)]
        return random.choices(placable, xp.exp(policy))[0]




class Reinforce(SelfMatch):
    def fit_episode(self, progress = None):
        board = self.board
        board.reset()

        while True:
            agent = self.agents[board.turn]
            action, prob = agent.get_action(board)
            board.put_stone(action)

            # 報酬はゲーム終了まで出ない
            flag = board.can_continue()
            if flag:
                agent.add((0, prob))

            # エージェントの学習はエピソードが終わるごとに行う
            else:
                reward = board.reward
                agent.add((reward, prob))
                agent.update()
                break

        agent = self.agents[board.turn ^ 1]
        agent.add((-reward, None))
        agent.update()


# ガンマの値が学習結果に大きく寄与することがわかったため、それを変更しながら学習して、結果を比較する
def fit_reinforce_agent(to_gpu, gammas, file_name, episodes = 100000, restart = False):
    # ハイパーパラメータ設定
    gamma = None
    lr = 0.000025

    # 環境
    board = Board()

    # エージェント
    agent_args = board.action_size, gamma, lr, to_gpu
    first_agent = ReinforceAgent(*agent_args)
    second_agent = ReinforceAgent(*agent_args)

    # 自己対戦
    self_match = Reinforce(board, first_agent, second_agent)

    for gamma in gammas:
        print(f"\n\"gamma = {gamma}\" is started.\n")
        first_agent.gamma = gamma
        second_agent.gamma = gamma
        self_match.fit(3, episodes, restart, file_name + "-{}".format(str(gamma * 100)[:2]))
        restart = False
        print(f"\ndone!")




if __name__ == "__main__":
    to_gpu = False
    gammas = [0.98]
    file_name = "reinforce"

    # 学習の進行具合によって変更する必要がある変数
    restart = False

    # 学習用コード
    fit_reinforce_agent(to_gpu, gammas, file_name, restart = restart)

    # 評価用コード
    gammas = 0.90, 0.92, 0.95, 0.98
    eval_computer(ReinforceComputer, to_gpu, gammas, file_name, graph_index = 4)