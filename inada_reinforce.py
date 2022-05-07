from inada_framework import cuda, Model, optimizers
import inada_framework.layers as dzl
import inada_framework.functions as dzf
import numpy as np
xp = cuda.cp if cuda.gpu_enable else np
from board import Board
from inada_selfmatch import REINFORCE, simple_plan
from inada_dqn import SumTree


# 確率形式に変換する前の最適方策を出力するニューラルネットワーク
class PolicyNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = dzl.Affine(512)
        self.l2 = dzl.Affine(512)
        self.l3 = dzl.Affine(action_size)

    def forward(self, x):
        x = dzf.relu(self.l1(x))
        x = dzf.relu(self.l2(x))
        return self.l3(x)


# モンテカルロ法でパラメータを修正する方策ベースのエージェント
class ReinforceAgent:
    def __init__(self, action_size, gamma = 0.99, lr = 0.0002):
        self.memory = []

        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr

    # エージェントを動かす前に呼ぶ必要がある
    def reset(self):
        self.pi = PolicyNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr).setup(self.pi)

        self.current_stage = 0
        self.rng = xp.random.default_rng()

    def save_weights(self, file_name):
        self.pi.save_weights(file_name)


    # エージェントを関数形式で使うとその方策に従った行動が得られる
    def __call__(self, board):
        action, _ = self.get_action(board)
        return action

    def get_action(self, board):
        state = board.state2ndarray(board.state, xp)
        policy = self.pi(state[None, :])
        placable = board.list_placable()

        # 学習時は方策を合法手のみに絞って、確率形式に変換し、それと学習の進行状況に応じて行動を選択する
        probs = dzf.softmax(policy[:, placable])

        if len(placable) == 1:
            action_index = 0
        else:
            action_index = self.rng.choice(len(placable), p = probs.data[0])

        # 行動が選ばれる確率も一緒に出力する
        return placable[action_index], probs[0, action_index]


    def add(self, data):
        self.memory.append(data)

    # ニューラルネットワークで近似したある方策に従った時の収益の期待値の勾配を求め、パラメータを更新する
    def update(self, progress):
        # 学習の進行度合いに応じて、lr を調整する
        stage = progress // 0.25
        if self.current_stage < stage:
            self.current_stage = stage
            self.optimizer.lr = self.lr / (stage + 1.0)

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G *= self.gamma
            G += reward

            # 負けの場合は、報酬に１回分余分に gamma が掛かっている (負け方じゃなく、勝ち方が知りたいわけだからいいか)
            if prob.data:
                loss += -G * dzf.log(prob)

        # メモリの解放を先に行う
        self.memory.clear()

        self.pi.clear_grads()
        loss.backward()
        self.optimizer.update()


def fit_reinforce_agent(runs, episodes, version = None):
    # ハイパーパラメータ設定
    gamma = 0.99
    lr = 0.0002

    file_name = "reinforce" if version is None else ("reinforce" + version)

    # 環境とエージェント
    board = Board()
    first_agent = ReinforceAgent(board.action_size, gamma, lr)
    second_agent = ReinforceAgent(board.action_size, gamma, lr)

    # 自己対戦
    self_match = REINFORCE(board, first_agent, second_agent)
    self_match.fit(runs, episodes, file_name)




# 実際にコンピュータとして使われるクラス (get_action() は親クラスのものを使う)
class ReinforceComputer(ReinforceAgent):
    def __init__(self, action_size):
        self.each_pi = []
        self.action_size = action_size

    def reset(self, file_name, turn, player_num):
        self.rng = xp.random.default_rng()

        # ファイル名は先攻と後攻で異なる
        file_name += f"{turn}_"

        # 何人のエージェントを行動選択に使うかによって、難易度を変えることができる
        assert 1 <= player_num, player_num <= 8
        self.player_probs = SumTree(player_num)
        self.player_probs.reset()

        # 各エージェントの方策を表すインスタンス変数をリセットし、新たに登録する
        each_pi = self.each_pi
        each_pi.clear()
        for i in self.rng.choice(8, player_num, replace = False):
            pi = PolicyNet(self.action_size)
            pi.load_weights(file_name + f"{i}.npz")
            each_pi.append(pi)

    def __call__(self, board):
        player_probs = self.player_probs
        actions = []

        for pi in self.each_pi:
            self.pi = pi
            action, prob = self.get_action(board)

            # 各エージェントが行動を選ぶ確率を重みとする
            player_probs[len(actions)] = float(prob.data)
            actions.append(action)

        # 複数人のエージェントの意見から重み付きランダムサンプリングして、選ばれたものをコンピュータの行動とする
        return actions[player_probs.sample()]


def eval_reinforce_computer(player_num, enemy_plan, version = None):
    file_name = "reinforce" if version is None else ("reinforce" + version)

    # 環境とエージェント
    board = Board()
    first_agent = ReinforceComputer(board.action_size)
    second_agent = ReinforceComputer(board.action_size)

    # エージェントの初期化
    first_agent.reset(file_name, 1, player_num)
    second_agent.reset(file_name, 0, player_num)
    self_match = REINFORCE(board, first_agent, second_agent)

    # 評価
    print("enemy:", enemy_plan.__name__)
    print("player_num:", player_num)
    print("first: {} %".format(self_match.eval(1, enemy_plan, verbose = True) / 10))
    print("second: {} %\n".format(self_match.eval(0, enemy_plan, verbose = True) / 10))




if __name__ == "__main__":
    # fit_reinforce_agent(runs = 100, episodes = 10000, version = "_v2_")

    import random
    def random_computer(board : Board):
        return random.choice(board.list_placable())

    eval_reinforce_computer(player_num = 8, enemy_plan = simple_plan, version = None)