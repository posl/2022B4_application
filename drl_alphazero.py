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


class AlphaZeroAgent:
    def __init__(self, action_size, c_puct = 1.0):
        self.network = PolicyValueNet(action_size)
        self.c_puct = c_puct

    # 引数はパラメータが保存されたファイルの名前か、同じモデルのインスタンス
    def reset(self, arg):
        self.load(arg)

        # それぞれ、過去の探索割合を近似したある種の方策、平均の評価値、探索回数 (計算の都合上、探索回数は常に１多い)
        self.P = {}
        self.V = {}
        self.N = {}

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
        pass

    def get_action(self, board):
        placable = self.board.list_placable()
        state = board.state
        N = self.N[state]

        # 右辺の第１項が過去の探索割合を勘案しながら探索回数の少ない手を選ぶための項で、第２項が勝率を見て活用を行うための項
        pucts = (self.c_puct * sqrt(N.sum() - len(placable)) * self.P[state]) / N + self.V[state]

        # np.argmax を使うと選択が前にある要素に偏るため、np.where で取り出したインデックスからランダムに選ぶ
        action_indexs = np.where(pucts == pucts.max())[0]

        if len(action_indexs) == 1:
            action_index = action_indexs[0]
        else:
            action_index = self.rng.choice(action_indexs)
        return placable[action_index]


    def search(self):
        pass

    def __expand(self):
        pass

    def __evaluate(self):
        pass




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

        xp = cuda.cp if cuda.gpu_enable else np
        states = xp.stack([state2ndarray_func(x[0], xp) for x in selected])

        mcts_policy = xp.stack([x[1] for x in selected], dtype = np.float32)
        rewards = xp.array([x[2] for x in selected], dtype = np.float32)

        return states, mcts_policy, rewards




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