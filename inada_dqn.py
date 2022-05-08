from functools import lru_cache
from inada_framework import Layer, Parameter, cuda, Model, Function, optimizers, no_grad
import numpy as np
xp = cuda.cp if cuda.gpu_enable else np
import inada_framework.functions as dzf
from board import Board
import pickle
import zlib
from inada_selfmatch import MultiAgentComputer, DQN, simple_plan, random_plan


# 強化学習の探索に必要なランダム性をネットワークに持たせるためのレイヤ (Noisy Network)
class NoisyAffine(Layer):
    def __init__(self, out_size, activation = None):
        super().__init__()
        self.out_size = out_size
        self.activation = activation

        # 重みは学習が開始したときに動的に生成する
        self.init_flag = True
        self.W_mu = Parameter(None, name = "W_mu")
        self.W_sigma = Parameter(None, name = "W_sigma")
        self.b_mu = Parameter(None, name = "b_mu")
        self.b_sigma = Parameter(None, name = "b_sigma")

        self.rng = xp.random.default_rng()

    # 通常の Affine レイヤのパラメータが正規分布に従う乱数であるかのような実装
    def forward(self, x):
        in_size = x.shape[1]
        out_size = self.out_size
        if self.init_flag:
            self.init_flag = False
            self.init_params(in_size, out_size)

        # Factorized Gaussian Noise (正規分布からのサンプリング数を減らす工夫) を使っている
        epsilon_in = self.noise_f(self.rng.normal(0.0, 1.0, size = (in_size, 1)).astype(np.float32))
        epsilon_out = self.noise_f(self.rng.normal(0.0, 1.0, size = (1, out_size)).astype(np.float32))
        W_epsilon = xp.matmul(epsilon_in, epsilon_out)
        b_epsilon = epsilon_out

        # 本当はパラメータが従う正規分布の平均・分散を学習したいが、それだと逆伝播ができないので、再パラメータ化を用いる
        W = self.W_mu + self.W_sigma * W_epsilon
        b = self.b_mu + self.b_sigma * b_epsilon
        x = dzf.affine(x, W, b)

        # 活性化関数を挟む場合は、その関数を経由する
        activation = self.activation
        if activation is not None:
            x = activation(x)

        return x

    # 重みの初期化方法はオリジナルの Rainbow のものを採用する
    def init_params(self, in_size, out_size):
        stdv = 1. / np.sqrt(in_size)
        self.W_mu.data = self.rng.uniform(-stdv, stdv, size = (in_size, out_size)).astype(np.float32)
        self.b_mu.data = self.rng.uniform(-stdv, stdv, size = (1, out_size)).astype(np.float32)

        initial_sigma = 0.5 * stdv
        self.W_sigma.data = xp.full((in_size, out_size), initial_sigma, dtype = np.float32)
        self.b_sigma.data = xp.full((1, out_size), initial_sigma, dtype = np.float32)

    @staticmethod
    def noise_f(x):
        return xp.sign(x) * xp.sqrt(xp.abs(x))


# 行動価値関数を状態価値関数とアドバンテージ関数に分けて Q 関数を近似するニューラルネットワーク (Dueling DQN)
class DuelingNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.v1 = NoisyAffine(512, activation = dzf.relu)
        self.v2 = NoisyAffine(1)

        self.a1 = NoisyAffine(512, activation = dzf.relu)
        self.a2 = NoisyAffine(action_size)

    def forward(self, x):
        value = self.v2(self.v1(x))
        advantage = self.a2(self.a1(x))

        advantage -= advantage.mean(axis = 1, keepdims = True)
        return value + advantage


# 重み付きの平均二乗誤差 (Function インスタンス)
class MeanSquaredWeightedError(Function):
    def __init__(self, t, W = None):
        self.t = t
        self.W = W

    def forward(self, x):
        diff = x - self.t
        diff **= 2.0
        if self.W is not None:
            diff *= self.W
        return 0.5 * (diff.sum() / len(diff))

    def backward(self, gy):
        x = self.inputs[0].data
        diff = x - self.t
        if self.W is not None:
            diff *= self.W
        return gy * (diff / len(diff))




# ランダムサンプリングを高速化するためのセグメントツリー
class SumTree:
    def __init__(self, capacity):
        self.capacity = self.__get_capacity(capacity)

    # 引数以上の数で最小の２のべき乗を取得する (容量は必ず２のべき乗であるものとする)
    @staticmethod
    def __get_capacity(capacity: int):
        if capacity > 0:
            # 指定された値が２のべき乗でない場合は目的の出力を作成する
            if capacity & (capacity - 1):
                return 1 << capacity.bit_length()
        elif capacity:
            message = f"argument must be positive integer, but \"{capacity}\" were given."
            raise AssertionError(message)

        return capacity

    # 使用する前に呼ぶ必要がある
    def reset(self):
        self.tree = np.zeros(2 * self.capacity)
        self.rng = np.random.default_rng()


    # 外部には木構造であることを見せない
    def __str__(self):
        return str(self.tree[self.capacity:])

    def __getitem__(self, indices):
        return self.tree[self.capacity:][indices]

    # 優先度のセットは１つずつ行うものとする
    def __setitem__(self, index, value):
        assert value >= 0

        # 木構造を保持する ndarray の後半半分が実際の優先度データを格納する部分
        index += self.capacity
        tree = self.tree
        tree[index] = value

        # 親ノードに２つの子ノードの和が格納されている状態を保つように更新する (インデックス１が最上位の親ノード)
        parent = index // 2
        while parent:
            left_child = 2 * parent
            right_child = left_child + 1
            tree[parent] = tree[left_child] + tree[right_child]
            parent //= 2


    # 優先度付きランダムサンプリングを行う (重複なしではない)
    def sample(self, batch_size = 1):
        indices = [self.__sample() for _ in range(batch_size)]
        return indices if len(indices) > 1 else indices[0]

    def __sample(self):
        z = self.rng.uniform(0, self.sum())
        current_index = 1

        # 実際の優先度データが格納されているインデックスに行き着くまでループを続ける
        tree = self.tree
        while current_index < self.capacity:
            left_child = 2 * current_index
            right_child = left_child + 1

            # 乱数 z が左子ノードより大きい場合は、z を左部分木にある全要素の和の分減じてから、右子ノードに進む
            left_value = tree[left_child]
            if z > left_value:
                current_index = right_child
                z -= left_value
            else:
                current_index = left_child

        # 見かけ上のインデックスに変換してから返す
        return current_index - self.capacity

    # 全優先度データを確率形式で返す
    @property
    def probs(self):
        return self.tree[self.capacity:] / self.sum()

    def sum(self):
        return self.tree[1]


# 学習に使う経験データ間の相関を弱め、また経験データを繰り返し使うためのバッファ (経験再生)
class ReplayBuffer:
    def __init__(self, buffer_size, prioritized = True, compress = False, gamma = 0.99):
        self.buffer = []
        self.buffer_size = buffer_size

        self.prioritized = prioritized
        self.compress = compress

        # Multi-step Q Learning を実現するためのバッファ
        self.tmp_buffer = []
        self.gamma = gamma

        # 優先度付き経験再生のハイパーパラメータは、オリジナルの Rainbow のものを採用する
        if prioritized:
            self.priorities = SumTree(buffer_size)
            self.epsilon = 0.01
            self.alpha = 0.6
            beta = 0.4
            self.beta = lambda progress: beta + (1 - beta) * progress

    # 使用する前に呼ぶ必要がある
    def reset(self):
        self.buffer.clear()
        self.tmp_buffer.clear()
        self.count = 0

        if self.prioritized:
            self.priorities.reset()
            self.max_priority = self.epsilon
        else:
            self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.buffer)


    def add(self, data):
        next_state, reward = data[2:]
        data = data[:2]
        self.tmp_buffer.append(data)

        # 報酬が出てから (エピソードが終了してから) そのエピソードで収集したデータを経験バッファに入れる
        if not reward:
            return

        nstep_gamma = 1
        count = self.count

        for state, action in reversed(self.tmp_buffer):
            nstep_reward = reward * nstep_gamma
            nstep_gamma *= self.gamma
            nstep_data = state, action, next_state, nstep_reward, nstep_gamma

            # 経験データ数がバッファサイズを超えたら、古いものから上書きしていく
            if count == self.buffer_size:
                count = 0

            if self.prioritized:
                # 少なくとも１回は学習に使われてほしいので、優先度の初期値は今までで最大のものとする
                self.priorities[count] = self.max_priority

            if self.compress:
                # pickle.dump : ファイルに書き込む, pickle.dumps : 戻り値として返す
                nstep_data = zlib.compress(pickle.dumps(nstep_data))

            try:
                self.buffer[count] = nstep_data
            except IndexError:
                self.buffer.append(nstep_data)

            count += 1

        self.tmp_buffer.clear()
        self.count = count

    def get_batch(self, batch_size, progress):
        if self.prioritized:
            # 重複なしではないが処理上問題はなく、buffer_size >> batch_size なので大丈夫
            indices = self.priorities.sample(batch_size)

            # 重みは等確率でサンプリングした時との確率の比の β 乗で、最大値が１になるように変換したものを使用する
            weights = (self.priorities.probs[indices] * len(self.buffer)) ** (-1 * self.beta(progress))
            weights /= weights.max()
        else:
            # ジェネレータによる重複なしランダムサンプリング
            indices = self.rng.choice(len(self.buffer), batch_size, replace = False)
            weights = None

        if self.compress:
            # pickle.load : ファイルから読み込む, pickle.loads : 引数を使う
            selected = [pickle.loads(zlib.decompress(self.buffer[i])) for i in indices]
        else:
            selected = [self.buffer[i] for i in indices]

        state2ndarray_func = Board.state2ndarray
        state = xp.stack([state2ndarray_func(x[0], xp) for x in selected])
        next_state = xp.stack([state2ndarray_func(x[2], xp) for x in selected])

        action = xp.array([x[1] for x in selected], dtype = np.int32)
        reward = xp.array([x[3] for x in selected], dtype = np.float32)
        gamma = xp.array([x[4] for x in selected], dtype = np.float32)

        return (state, action, next_state, reward, gamma), indices, weights

    def update_priorities(self, deltas, indices):
        if self.prioritized:
            # 優先度 = (|TD 誤差| + ε) ^ α
            new_priorities = (abs(deltas) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, new_priorities.max())

            priorities = self.priorities
            for i, index in enumerate(indices):
                priorities[index] = new_priorities[i]




# TD 法の Q 学習でパラメータを修正する価値ベースのエージェント
class DQNAgent:
    def __init__(self, action_size, batch_size, buffer_size, prioritized = True, compress = False,
                 gamma = 0.99, lr = 0.0005, exec_start = 50000, exec_interval = 4, sync_interval = 10000):
        self.action_size = action_size
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size, prioritized, compress, gamma)

        self.lr = lr / 4.0 if prioritized else lr

        # update メソッド内の条件分岐に使う属性
        self.exec_start = exec_start
        self.exec_interval = exec_interval
        self.sync_interval = sync_interval

    # エージェントを動かす前に呼ぶ必要がある
    def reset(self):
        self.qnet = DuelingNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr).setup(self.qnet)

        # TD ターゲットの安定性のために、それを生成するためのネットワークは別で用意する (ターゲットネットワーク)
        self.qnet_target = DuelingNet(self.action_size)

        self.total_steps = 0
        self.replay_buffer.reset()
        self.rng = np.random.default_rng()

    # ターゲットネットワークはパラメータを学習せず、定期的に学習対象のネットワークと同期させることで学習を進行させる
    def sync_qnet(self):
        self.qnet_target.copy_weights(self.qnet)

    def save_weights(self, file_name):
        self.qnet.save_weights(file_name)


    # エージェントを関数形式で使うとその方策に従った行動が得られる
    def __call__(self, board):
        action, _ = self.get_action(board)
        return action

    # ニューラルネットワーク内のランダム要素が探索の役割を果たす
    def get_action(self, board):
        placable = board.list_placable()
        state = board.state

        # オセロ盤の状態情報を変換して、ニューラルネットワークに流す
        with no_grad():
            qs = self.qnet(board.state2ndarray(state, xp)[None, :])
            qs = qs.data

        # 合法手の中から Q 関数が最大のものを選択し、オセロ盤の状態情報と一緒に出力する
        qs = qs[0, placable]
        return placable[qs.argmax()], state


    def update(self, data, progress):
        total_steps = self.total_steps
        self.total_steps = total_steps + 1
        replay_buffer = self.replay_buffer
        replay_buffer.add(data)

        if total_steps % self.exec_interval or total_steps < self.exec_start:
            return
        if not total_steps % self.sync_interval:
            self.sync_qnet()

        # バッチサイズ分の経験データと優先度更新用のインデックス、平均二乗誤差に使う重みを取得する
        experiences, indices, weights = replay_buffer.get_batch(self.batch_size, progress)
        state, action, next_state, reward, gamma = experiences

        # ニューラルネットワークの出力の形状は (batch_size, action_size)
        qnet = self.qnet
        qs = qnet(state)
        sequence = np.arange(self.batch_size)
        qs = qs[sequence, action]

        # 誤差を含む出力に max 演算子を使うことで過大評価を起こさないように、２つの Q 関数を使う (Double DQN)
        with no_grad():
            next_qs = self.qnet_target(next_state)
            next_qs = next_qs.data[sequence, qnet(next_state).data.argmax(axis = 1)]
            target = reward + (reward == 0) * gamma * next_qs

        # TD 誤差を使って、経験データの優先度を更新する
        replay_buffer.update_priorities(qs.data - target, indices)

        # Q 関数を TD ターゲットに近づけるようにパラメータを更新する
        loss = MeanSquaredWeightedError(target, weights)(qs)

        qnet.clear_grads()
        loss.backward()
        self.optimizer.update()


# 実際にコンピュータとして使われるクラス
class DQNComputer(MultiAgentComputer):
    network_class = DuelingNet

    def __call__(self, board):
        placable = board.list_placable()
        state = board.state2ndarray(board.state, xp)[None, :]

        # 学習済みのパラメータを使うだけなので、動的に計算グラフを構築する必要はない
        actions = []
        with no_grad():
            for qnet in self.each_net:
                qs = qnet(state)
                qs = qs.data[0, placable]
                actions.append(placable[qs.argmax()])

        # 各エージェントがそれぞれ選んだ行動価値が最大の行動の中からランダムに選択する
        return int(self.rng.choice(actions))




def fit_dqn_agent(runs, episodes, version = None):
    file_name = "dqn" if version is None else ("dqn" + version)

    # ハイパーパラメータ設定
    buffer_size = 1000000
    batch_size = 32
    gamma = 0.98
    lr = 0.0001
    prioritized = True
    compress = True

    # 環境
    board = Board()

    # エージェント
    agent_args = board.action_size, batch_size, buffer_size, prioritized, compress, gamma, lr
    first_agent = DQNAgent(*agent_args)
    second_agent = DQNAgent(*agent_args)

    # 自己対戦
    self_match = DQN(board, first_agent, second_agent)
    self_match.fit(runs, episodes, file_name)


def eval_dqn_computer(agent_num, enemy_plan, version = None):
    file_name = "dqn" if version is None else ("dqn" + version)

    # 環境とエージェント
    board = Board()
    first_agent = DQNComputer(board.action_size)
    second_agent = DQNComputer(board.action_size)

    # エージェントの初期化
    first_agent.reset(file_name, 1, agent_num)
    second_agent.reset(file_name, 0, agent_num)
    self_match = DQN(board, first_agent, second_agent)

    # 評価
    print("enemy:", enemy_plan.__name__)
    print(f"agent_num: {agent_num}")
    print("first: {} %".format(self_match.eval(1, enemy_plan, verbose = True) / 10))
    print("second: {} %\n".format(self_match.eval(0, enemy_plan, verbose = True) / 10))




if __name__ == "__main__":
    # 学習用コード
    fit_dqn_agent(runs = 1, episodes = 3000000, version = None)

    # 評価用コード
    # eval_dqn_computer(agent_num = 8, enemy_plan = simple_plan, version = None)