import numpy as np

from inada_framework import Model, optimizers, cuda, no_grad
import inada_framework.layers as dzl
import inada_framework.functions as dzf
xp = cuda.cp if cuda.gpu_enable else np
from drl_selfmatch import SelfMatch, simple_plan, corners_plan
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
    def __init__(self, action_size, gamma = 0.99, lr = 0.0001):
        self.memory = []

        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr

    # エージェントを動かす前に呼ぶ必要がある
    def reset(self):
        pi = PolicyNet(self.action_size)
        self.pi = pi
        self.optimizer = optimizers.Adam(self.lr).setup(pi)
        self.rng = xp.random.default_rng()

        if cuda.gpu_enable:
            pi.to_gpu()

    def save(self, file_name, is_yet = None):
        self.pi.save_weights(file_name + ".npz")

    def load_to_restart(self, file_name):
        self.pi.load_weights(file_name + ".npz")
        if cuda.gpu_enable:
            self.pi.to_gpu()


    # エージェントを関数形式で使うとその方策に従った行動が得られる
    def __call__(self, board):
        action, __ = self.get_action(board)
        return action

    def get_action(self, board):
        placable = board.list_placable()
        state = board.get_state_ndarray(xp)
        policy = self.pi(state[None, :])

        # 学習時は方策を合法手のみに絞って、確率形式に変換し、それと学習の進行状況に応じて行動を選択する
        probs = dzf.softmax(policy[:, np.array(placable)])

        if len(placable) == 1:
            action_index = 0
        else:
            action_index = int(self.rng.choice(len(placable), p = probs.data[0]))

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

            # 負けの場合は、報酬に１回分余分に gamma が掛かっている
            if prob is not None:
                loss += -G * dzf.log(prob)

        # メモリの解放を先に行う
        self.memory.clear()

        self.pi.clear_grads()
        loss.backward()
        self.optimizer.update()




class Reinforce(SelfMatch):
    def fit_one_episode(self, progress = None):
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

    def save(self, turn, file_name, index):
        agent = self.agents[turn]
        agent.save(f"{file_name}{turn}_{index}")


def fit_reinforce_agent(episodes, trained_num = 0, restart = 0, version = None):
    file_name = "reinforce" if version is None else ("reinforce" + version)

    # ハイパーパラメータ設定
    gamma = 0.97
    lr = 0.00005

    # 環境
    board = Board()

    # エージェント
    agent_args = board.action_size, gamma, lr
    first_agent = ReinforceAgent(*agent_args)
    second_agent = ReinforceAgent(*agent_args)

    # 自己対戦
    self_match = Reinforce(board, first_agent, second_agent)
    self_match.fit(8, episodes, file_name, trained_num, restart)




# 実際にコンピュータとして使われるクラス
class ReinforceComputer:
    def __init__(self, action_size):
        self.each_pi = []
        self.action_size = action_size

    def reset(self, file_name, turn, agent_num):
        file_name = Reinforce.get_path(file_name).format("parameters")
        file_name += f"{turn}_"
        self.rng = xp.random.default_rng()

        # 何人のエージェントを行動選択に使うかによって、難易度を変えることができる (上限は８人)
        assert isinstance(agent_num, int)
        assert 1 <= agent_num <= 8

        # 各エージェントの方策を表すインスタンス変数をリセットし、新たに登録する
        each_pi = self.each_pi
        each_pi.clear()

        for i in self.rng.choice(8, agent_num, replace = False):
            pi = PolicyNet(self.action_size)
            each_pi.append(pi)

            pi.load_weights(file_name + f"{i}.npz")
            if cuda.gpu_enable:
                self.pi.to_gpu()

    def __call__(self, board):
        placable = board.list_placable()
        if len(placable) == 1:
            return placable[0]

        # 学習済みのパラメータを使うだけなので、動的に計算グラフを構築する必要はない
        state = board.get_state_ndarray(xp)[None, :]
        with no_grad():
            for pi in self.each_pi:
                try:
                    policy += pi(state)
                except NameError:
                    policy = pi(state)

            # 各エージェントが提案するスコア値の和をとり、それを元に確率付きランダムサンプリングで行動を選択する
            probs = dzf.softmax(policy[:, np.array(placable)])
            probs = probs.data[0]

        return placable[int(self.rng.choice(len(placable), p = probs))]


def eval_reinforce_computer(agent_num, enemy_plan, version = None):
    file_name = "reinforce" if version is None else ("reinforce" + version)

    # 環境とエージェント
    board = Board()
    first_agent = ReinforceComputer(board.action_size)
    second_agent = ReinforceComputer(board.action_size)

    # エージェントの初期化
    first_agent.reset(file_name, 1, agent_num)
    second_agent.reset(file_name, 0, agent_num)
    self_match = Reinforce(board, first_agent, second_agent)

    # 評価
    print("enemy:", enemy_plan.__name__)
    print(f"agent_num: {agent_num}")
    print("first: {} %".format(self_match.eval(1, enemy_plan, verbose = True) / 10))
    print("second: {} %\n".format(self_match.eval(0, enemy_plan, verbose = True) / 10))




if __name__ == "__main__":
    # 学習用コード
    fit_reinforce_agent(episodes = 100000, trained_num = 0, restart = 0, version = None)

    # 評価用コード
    # eval_reinforce_computer(agent_num = 8, enemy_plan = simple_plan, version = None)
    # eval_reinforce_computer(agent_num = 8, enemy_plan = corners_plan, version = None)