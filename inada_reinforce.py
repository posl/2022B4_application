from inada_framework import cuda, Model, optimizers
import inada_framework.layers as dzl
import inada_framework.functions as dzf
import numpy as np
xp = cuda.cp if cuda.gpu_enable else np
from inada_dqn import SumTree
from board import Board
from inada_selfmatch import REINFORCE


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
    def __init__(self, action_size, gamma = 0.98, lr = 0.0002):
        self.memory = []

        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr

    # エージェントを動かす前に呼ぶ必要がある
    def reset(self):
        self.pi = PolicyNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr).setup(self.pi)

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

        # 方策を合法手のみに絞って、確率形式に変換し、それに従って行動を選択する
        placable = board.list_placable()
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




# 実際にコンピュータとして使われるクラス (get_action() は親クラスのものを使う)
class ReinforceComputer(ReinforceAgent):
    def __init__(self, action_size):
        self.pi = PolicyNet(action_size)

    def reset(self, player_num, turn, file_name):
        # 何人のエージェントを行動選択に使うかによって、難易度を変えることができる
        assert 1 <= player_num, player_num <= 8
        self.player_probs = SumTree(player_num)

        # ファイル名は先攻と後攻で異なる
        self.file_name = file_name + f"{turn}_"

        # 行動選択に使うエージェントの重みファイルの索引番号をこの時点で決定する
        self.rng = xp.random.default_rng()
        self.file_indexs = self.rng.choice(8, player_num, replace = False)

    def __call__(self, board):
        player_probs = self.player_probs
        player_probs.reset()
        actions = []

        for index in self.file_indexs:
            self.pi.load_weights(self.file_name + str(index))
            action, prob = self.get_action(board)

            # 各エージェントが行動を選ぶ確率を重みとする
            player_probs[len(actions)] = float(prob.data)
            actions.append(action)

        # 複数人のエージェントの意見から重み付きランダムサンプリングして、選ばれたものをコンピュータの行動とする
        return actions[player_probs.sample()]




if __name__ == "__main__":
    board = Board()
    first_agent = ReinforceAgent(board.action_size, gamma = 0.98, lr = 0.0002)
    second_agent = ReinforceAgent(board.action_size, gamma = 0.98, lr = 0.0002)

    self_match = REINFORCE(board, first_agent, second_agent)
    self_match.fit(runs = 100, episodes = 10000, file_name = "reinforce")