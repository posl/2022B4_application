import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from inada_framework import Model, cuda, optimizers, no_grad
import inada_framework.layers as dzl
import inada_framework.functions as dzf
from drl_utilities import SelfMatch
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

    def get_action(self, board, placable = None):
        if placable is None:
            placable = board.list_placable()

        xp = cuda.cp if self.use_gpu else np
        state = board.get_state_ndarray(xp)
        policy = self.pi(state[None, :])
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




class Reinforce(SelfMatch):
    def fit_episode(self, progress = None):
        board = self.board
        agents = self.agents

        board.reset()
        placable = board.list_placable()

        while True:
            turn = board.turn
            agent = agents[turn]

            action, prob = agent.get_action(board, placable)
            board.put_stone(action)

            # 報酬はゲーム終了まで出ない
            placable = board.can_continue_placable()
            if placable:
                agent.add((0, prob))

            # エージェントの学習はエピソードが終わるごとに行う
            else:
                reward = board.reward
                agent.add((reward, prob))
                agent.update()
                break

        agent = agents[turn ^ 1]
        agent.add((-reward, None))
        agent.update()


# ガンマの値が学習結果に大きく寄与することがわかったため、それを変更しながら学習して、結果を比較する
def fit_reinforce_agent(gammas, episodes = 150000, restart = False):
    # ハイパーパラメータ設定
    gamma = None
    lr = 0.000025
    to_gpu = False

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
        self_match.fit(5, episodes, restart, "reinforce-{}".format(str(gamma * 100)[:2]))
        restart = False
        print(f"\ndone!")




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

        # 同じ条件で学習した３人のエージェントの中から、２人だけ、ランダムに選ぶ
        for i in random.sample(range(3), 2):
            pi = PolicyNet(self.action_size)
            each_pi.append(pi)

            pi.load_weights(file_path + f"{i}.npz")
            if self.use_gpu:
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


# コンピュータの性能を割引率の値ごとに評価し、グラフ形式で保存する関数
def eval_reinforce_computer(gammas):
    # 環境
    board = Board()

    # コンピュータ
    computer = ReinforceComputer(board.action_size)

    # 対戦場
    self_match = SelfMatch(board, computer, computer)

    # 描画用配列
    length = len(gammas)
    win_rates = np.empty((2, length))

    # 先攻か後攻か・難易度ごとに、コンピュータの評価を合計 20 回行い、その平均をグラフに描画する勝率とする
    for i, gamma in enumerate(gammas):
        print(f"\n\033[92mgamma = {gamma}\033[0m")

        for turn in (1, 0):
            target = f"turn {turn}"
            win_rate = 0

            for __ in tqdm(range(20), desc = target, leave = False):
                computer.reset("reinforce", gamma, turn)
                win_rate += self_match.eval(turn)

            win_rate /= 20
            win_rates[turn, i] = win_rate
            print(f"{target}: {win_rate:.5g} %")

    # グラフの目盛り位置を設定するための変数
    width = 1 / 3
    left = np.arange(length)
    center = left + width

    # 左が先攻、右が後攻の勝率となるような棒グラフを画像保存する
    plt.bar(left, win_rates[1], width = width, align = "edge", label = "first")
    plt.bar(center, win_rates[0], width = width, align = "edge", label = "second")
    plt.xticks(ticks = center, labels = gammas)
    plt.legend()

    plt.ylim(-5, 105)
    plt.xlabel("Gamma")
    plt.ylabel("Winning Percentage")
    plt.savefig(self_match.get_path("reinforce_bar").format("graphs"))
    plt.clf()




if __name__ == "__main__":
    # 学習用コード
    gammas = 0.93, 0.98
    fit_reinforce_agent(gammas, restart = True)

    # 評価用コード
    gammas = 0.88, 0.93, 0.98
    eval_reinforce_computer(gammas)