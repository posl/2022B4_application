import random

import numpy as np

from inada_framework import Model, cuda, optimizers, no_grad
from drl_utilities import SimpleCNN, SelfMatch
import inada_framework.layers as dzl
from inada_framework.functions import relu, softmax, log
from board import Board



# =============================================================================
# 方策を近似するニューラルネットワーク
# =============================================================================

class PolicyNet(Model):
    def __init__(self, action_size):
        super().__init__()

        self.cnn = SimpleCNN(use_batch_norm = False)
        self.l1 = dzl.Affine(512)
        self.l2 = dzl.Affine(action_size)

    def forward(self, x):
        x = self.cnn(x)
        x = relu(self.l1(x))
        return self.l2(x)




# =============================================================================
# モンテカルロ法・方策ベースでパラメータを学習するエージェント
# =============================================================================

class ReinforceAgent:
    def __init__(self, action_size, gamma = 0.88, lr = 0.000025, to_gpu = False):
        self.memory1 = []
        self.memory0 = []

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


    def __call__(self, board):
        action, __ = self.get_action(board)
        return action

    def get_action(self, board, placable = None):
        if placable is None:
            placable = board.list_placable()

        xp = cuda.cp if self.use_gpu else np
        state = board.get_img(xp)
        policy = self.pi(state[None, :])
        probs = softmax(policy[:, np.array(placable)])

        if len(placable) == 1:
            action_index = 0
        else:
            action_index = random.choices(range(len(placable)), probs.data[0])[0]

        # 行動が選ばれる確率も一緒に出力する
        return placable[action_index], probs[0, action_index]


    # 先攻・後攻で行動が選ばれる確率を保存するリストを分ける
    def add(self, prob, turn):
        memory = getattr(self, f"memory{turn}")
        memory.append(prob)

    # ニューラルネットワークで近似したある方策に従った時の収益の期待値の勾配を求め、パラメータを更新する
    def update(self, reward, final_turn):
        loss, gamma = 0, self.gamma

        for turn in range(2):
            memory = getattr(self, f"memory{turn}")
            G = reward if turn == final_turn else -reward

            for prob in reversed(memory):
                loss += -G * log(prob)
                G *= gamma

            # メモリの解放を先に行う
            memory.clear()

        self.pi.clear_grads()
        loss.backward()
        self.optimizer.update()




# =============================================================================
# 自己対戦による学習
# =============================================================================

class Reinforce(SelfMatch):
    def fit_episode(self, progress = None):
        board = self.board
        agent = self.agent

        board.reset()
        placable = board.list_placable()

        while placable:
            action, prob = agent.get_action(board, placable)
            board.put_stone(action)

            # 報酬はゲーム終了まで出ない
            agent.add(prob, board.turn)
            placable = board.can_continue_placable()

        # エージェントの学習はエピソードごとに行う
        agent.update(board.reward, board.turn)


def fit_reinforce_agent(episodes = 200000, restart = False):
    # ハイパーパラメータ設定
    gamma = 0.92
    lr = 0.000025
    to_gpu = True

    # 環境
    board = Board()

    # エージェント
    agent = ReinforceAgent(board.action_size, gamma, lr, to_gpu)

    # 自己対戦
    self_match = Reinforce(board, agent)
    self_match.fit(3, episodes, restart, "reinforce")




# =============================================================================
# 実際に使われることを想定したコンピュータ
# =============================================================================

class ReinforceComputer:
    def __init__(self, action_size, file_name = "reinforce", to_gpu = False):
        file_path = Reinforce.get_path(file_name).format("parameters")
        each_pi = []
        use_gpu = to_gpu and cuda.gpu_enable

        # 同じ条件で学習した３人のエージェントの中から、２人だけ、ランダムに選ぶ
        for i in random.sample(range(3), 2):
            pi = PolicyNet(action_size)
            each_pi.append(pi)

            pi.load_weights(file_path + f"-{i}.npz")
            if use_gpu:
                pi.to_gpu()

        self.each_pi = each_pi
        self.use_gpu = use_gpu

    def __call__(self, board):
        placable = board.list_placable()
        if len(placable) == 1:
            return placable[0]

        xp = cuda.cp if self.use_gpu else np
        state = board.get_img(xp)[None, :]

        # 学習済みのパラメータを使うだけなので、動的に計算グラフを構築する必要はない
        with no_grad():
            pi0, pi1 = self.each_pi
            policy = pi0(state).data + pi1(state).data

        # 各エージェントが提案するスコア値の和をとり、それを元にした重み付きランダムサンプリングで行動を選択する
        policy = policy[0, np.array(placable)]
        policy -= policy.max()
        return random.choices(placable, xp.exp(policy))[0]




if __name__ == "__main__":
    # 学習用コード
    fit_reinforce_agent(restart = True)