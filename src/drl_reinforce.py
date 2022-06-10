from random import choices, sample

import numpy as np

from inada_framework import Model, cuda, optimizers, no_train
from drl_utilities import SelfMatch, eval_computer
import inada_framework.layers as dzl
from inada_framework.functions import relu, flatten, softmax, log
from board import Board



# =============================================================================
# 方策を近似するニューラルネットワーク
# =============================================================================

class PolicyNet(Model):
    def __init__(self, action_size):
        super().__init__()

        self.conv1 = dzl.Conv2d(64, 3, 1, 1)
        self.bn1 = dzl.BatchNorm()
        self.conv2 = dzl.Conv2d(64, 3, 1, 1)
        self.bn2 = dzl.BatchNorm()
        self.conv3 = dzl.Conv2d1x1(64)
        self.bn3 = dzl.BatchNorm()

        self.fc1 = dzl.Affine(512)
        self.fc2 = dzl.Affine(512)
        self.fc3 = dzl.Affine(action_size)

    def forward(self, x):
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)) + self.bn3(self.conv3(x)))
        x = flatten(x)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return self.fc3(x)




# =============================================================================
# モンテカルロ法・方策ベースでパラメータを学習するエージェント
# =============================================================================

class ReinforceAgent:
    def __init__(self, action_size, gamma = 0.90, lr = 0.00002, to_gpu = False):
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
        with no_train():
            action, __ = self.get_action(board)
        return action

    def get_action(self, board, placable = None):
        if placable is None:
            placable = board.list_placable()

        xp = cuda.cp if self.use_gpu else np
        state = board.get_img(xp)[None, :]
        mask = np.array(placable)

        policy = self.pi(state)[:, mask]
        probs = softmax(policy)

        if len(placable) == 1:
            action_index = 0
        else:
            action_index = choices(range(len(placable)), probs.data[0])[0]

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


def fit_reinforce_agent(episodes = 10000, restart = False):
    # ハイパーパラメータ設定
    gamma = 0.90
    lr = 0.00001
    to_gpu = False

    # 環境
    board = Board()

    # エージェント
    agent = ReinforceAgent(board.action_size, gamma, lr, to_gpu)

    # 自己対戦
    self_match = Reinforce(board, agent)
    self_match.fit(5, episodes, restart, "reinforce")




# =============================================================================
# 実際に使われることを想定したコンピュータ
# =============================================================================

class ReinforceComputer:
    def __init__(self, action_size, file_name = "reinforce", to_gpu = False):
        self.action_size = action_size
        self.file_path = Reinforce.get_path(file_name).format("parameters")
        self.use_gpu = to_gpu and cuda.gpu_enable
        self.reset()

    def reset(self):
        file_path = self.file_path

        each_pi = []
        for i in sample(range(5), 3):
            pi = PolicyNet(self.action_size)
            pi.load_weights(f"{file_path}-{i}.npz")
            if self.use_gpu:
                pi.to_gpu()

            each_pi.append(pi)
        self.each_pi = each_pi

    def __call__(self, board):
        placable = board.list_placable()
        if len(placable) == 1:
            return placable[0]

        for action in placable:
            with board.log_runtime():
                board.put_stone(action)
                flag = board.can_continue()

                if not flag:
                    result = board.black_num - board.white_num
                    is_win = (result > 0) if board.turn else (result < 0)

                    # 必ず勝つ手が存在するならば、そこに置く
                    if is_win:
                        return action

        xp = cuda.cp if self.use_gpu else np
        state = board.get_img(xp)[None, :]
        mask = np.array(placable)

        with no_train():
            probs = 0
            for pi in self.each_pi:
                policy = pi(state)[:, mask]
                probs += softmax(policy).data[0]

        # 複数人のエージェントが意見を出し合って、確率的サンプリングで行動を決める
        action = choices(placable, probs)[0]
        return action




if __name__ == "__main__":
    # 学習用コード
    # fit_reinforce_agent(restart = True)

    # 評価用コード
    eval_computer(ReinforceComputer, "REINFORCE")