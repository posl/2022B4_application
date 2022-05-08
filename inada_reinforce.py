from inada_framework import Model, optimizers, cuda, no_grad
import inada_framework.layers as dzl
import inada_framework.functions as dzf
import numpy as np
xp = cuda.cp if cuda.gpu_enable else np
from board import Board
from inada_selfmatch import MultiAgentComputer, REINFORCE, simple_plan, random_plan


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
    def __init__(self, action_size, gamma = 0.99, lr = 0.0002, bias = 0.1):
        self.memory = []

        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.bias = bias

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

    def get_action(self, board, progress = 1.0):
        placable = board.list_placable()
        state = board.state2ndarray(board.state, xp)[None, :]
        policy = self.pi(state)

        # 学習時は方策を合法手のみに絞って、確率形式に変換し、それと学習の進行状況に応じて行動を選択する
        policy *= (1. + self.bias * progress)
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
    def update(self):
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


# 実際にコンピュータとして使われるクラス
class ReinforceComputer(MultiAgentComputer):
    network_class = PolicyNet

    def __call__(self, board):
        placable = board.list_placable()
        state = board.state2ndarray(board.state, xp)[None, :]

        # 学習済みのパラメータを使うだけなので、動的に計算グラフを構築する必要はない
        with no_grad():
            for pi in self.each_net:
                try:
                    policy += pi(state)
                except NameError:
                    policy = pi(state)

            # 各エージェントが提案するスコア値の和をとり、それを元に確率付きランダムサンプリングで行動を選択する
            probs = dzf.softmax(policy[:, placable])
            probs = probs.data[0]

        if len(placable) == 1:
            action_index = 0
        else:
            action_index = self.rng.choice(len(placable), p = probs)

        return placable[action_index]




def fit_reinforce_agent(runs, episodes, version = None):
    file_name = "reinforce" if version is None else ("reinforce" + version)

    # ハイパーパラメータ設定
    gamma = 0.98
    lr = 0.0001
    bias = 0.1

    # 環境
    board = Board()

    # エージェント
    agent_args = board.action_size, gamma, lr, bias
    first_agent = ReinforceAgent(*agent_args)
    second_agent = ReinforceAgent(*agent_args)

    # 自己対戦
    self_match = REINFORCE(board, first_agent, second_agent)
    self_match.fit(runs, episodes, file_name)


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
    print(f"player_num: {player_num}")
    print("first: {} %".format(self_match.eval(1, enemy_plan, verbose = True) / 10))
    print("second: {} %\n".format(self_match.eval(0, enemy_plan, verbose = True) / 10))




if __name__ == "__main__":
    # 学習用コード
    fit_reinforce_agent(runs = 20, episodes = 10000, version = None)

    # 評価用コード
    # eval_reinforce_computer(player_num = 8, enemy_plan = simple_plan, version = None)