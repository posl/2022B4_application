from inada_framework import cuda, Model, Function, optimizers, no_grad
import inada_framework.layers as dzl
import inada_framework.functions as dzf
from board import Board
import numpy as np
from collections import deque
import pickle
import zlib
xp = cuda.cp if cuda.gpu_enable else np
import copy
from inada_selfmatch import DQN


# Q 関数を近似するニューラルネットワーク
class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = dzl.Affine(512)
        self.l2 = dzl.Affine(512)
        self.l3 = dzl.Affine(action_size)

    def forward(self, x):
        x = dzf.relu(self.l1(x))
        x = dzf.relu(self.l2(x))
        return self.l3(x)


# 行動価値関数を状態価値関数とアドバンテージ関数に分けて Q 関数を近似するニューラルネットワーク (Dueling DQN)
class DuelingNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = dzl.Affine(512)
        self.l2 = dzl.Affine(512)
        self.l_value = dzl.Affine(1)
        self.l_advantage = dzl.Affine(action_size)

    def forward(self, x):
        x = dzf.relu(self.l1(x))
        x = dzf.relu(self.l2(x))
        advantage = self.l_advantage(x)
        advantage -= advantage.mean(axis = 1, keepdims = True)
        return self.l_value(x) + advantage


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
    def __init__(self, buffer_size, step_num, gamma, prioritized = True, compress = False):
        self.buffer = []
        self.buffer_size = buffer_size

        # Multi-step Q Learning を実現するためのバッファ
        self.step_num = step_num
        self.gamma = gamma
        self.tmp_buffer = deque(maxlen = step_num)

        self.prioritized = prioritized
        self.compress = compress

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
        tmp_buffer = self.tmp_buffer
        tmp_buffer.append(data)
        if len(tmp_buffer) < self.step_num:
            return

        # n step 先までの報酬の重み付き和を考える  (data = state, action, next_state, reward)
        nstep_reward = 0
        for i, data in enumerate(tmp_buffer):
            next_state, reward = data[2:]

            # 報酬が出るのは、ゲーム終了時だけ
            if reward:
                nstep_reward = (self.gamma ** i) * reward
                break

        state, action = tmp_buffer[0][:2]
        nstep_data = state, action, next_state, nstep_reward

        # 経験データ数がバッファサイズを超えたら、古いものから上書きしていく
        if self.count == self.buffer_size:
            self.count = 0

        # 少なくとも１回は学習に使われてほしいので、優先度の初期値は今までで最大のものとする
        self.priorities[self.count] = self.max_priority

        if self.compress:
            # pickle.dump : ファイルに書き込む, pickle.dumps : 戻り値として返す
            nstep_data = zlib.compress(pickle.dumps(nstep_data))

        try:
            self.buffer[self.count] = nstep_data
        except IndexError:
            self.buffer.append(nstep_data)

        self.count += 1

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

        state = xp.stack([Board.state2ndarray(x[0], xp) for x in selected])
        next_state = xp.stack([Board.state2ndarray(x[2], xp) for x in selected])

        action = xp.array([x[1] for x in selected], dtype = np.int32)
        reward = xp.array([x[3] for x in selected], dtype = np.float32)

        return (state, action, next_state, reward), indices, weights

    def update_priorities(self, deltas, indices):
        if self.prioritized:
            # 優先度 = (|TD 誤差| + ε) ^ α
            priorities = (abs(deltas) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priorities.max())

            for i, index in enumerate(indices):
                self.priorities[index] = priorities[i]




# TD 法の Q 学習でパラメータを修正する価値ベースのエージェント
class DQNAgent:
    def __init__(self, qnet_class, replay_buffer, action_size, batch_size, step_num, gamma):
        self.qnet_class = qnet_class
        self.replay_buffer = replay_buffer

        self.action_size = action_size
        self.batch_size = batch_size

        # Q Learning 用のハイパーパラメータ
        self.step_num = step_num
        self.gamma = gamma ** step_num
        lr = 0.0005
        self.lr = lr / 4.0 if replay_buffer.prioritized else lr
        self.epsilon = lambda progress: max(1.0 - 2.0 * progress, 0.1)

    # エージェントを動かす前に呼ぶ必要がある
    def reset(self):
        self.qnet = self.qnet_class(self.action_size)
        self.optimizer = optimizers.Adam(self.lr).setup(self.qnet)

        # TD ターゲットの安定性のために、それを生成するためのネットワークは別で用意する (ターゲットネットワーク)
        self.qnet_target = self.qnet_class(self.action_size)

        self.replay_buffer.reset()
        self.rng = np.random.default_rng()

    # ターゲットネットワークはパラメータを学習せず、定期的に学習対象のネットワークと同期させることで学習を進行させる
    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def save_weights(self, file_name):
        self.qnet.save_weights(file_name)


    # エージェントを関数形式で使うとその方策に従った行動が得られる
    def __call__(self, board):
        action, _ = self.get_action(board)
        return action

    # ε の確率で探索、1 - ε の確率で活用を行う (ε-greedy 法)
    def get_action(self, board, progress = None):
        if progress is not None and np.random.rand() < self.epsilon(progress):
            return int(self.rng.choice(self.action_size))

        # オセロ盤の状態情報を変換して、ニューラルネットワークに流す
        state = board.state
        qs = self.qnet(board.state2ndarray(state, xp)[None, :]).data

        # 合法手の中から Q 関数が最大のものを選択する
        placable = board.list_placable()
        qs = qs[0, placable]

        # オセロ盤の状態情報も一緒に出力する
        return placable[qs.argmax()], state


    def update(self, data, progress):
        replay_buffer = self.replay_buffer
        batch_size = self.batch_size

        replay_buffer.add(data)
        if len(replay_buffer) < batch_size:
            return

        # バッチサイズ分の経験データと優先度更新用のインデックス、平均二乗誤差に使う重みを取得する
        (state, action, next_state, reward), indices, W = replay_buffer.get_batch(batch_size, progress)

        # ニューラルネットワークの出力の形状は (batch_size, action_size)
        qs = self.qnet(state)
        sequence = np.arange(batch_size)
        qs = qs[sequence, action]

        # 誤差を含む出力に max 演算子を使うことで過大評価を起こさないように、２つの Q 関数を使う (Double DQN)
        with no_grad():
            next_qs = self.qnet_target(next_state)
            next_qs = next_qs.data[sequence, self.qnet(next_state).data.argmax(axis = 1)]
            target = reward + (reward == 0) * self.gamma * next_qs

        # TD 誤差を使って、経験データの優先度を更新する
        replay_buffer.update_priorities(qs.data - target, indices)

        # Q 関数を TD ターゲットに近づけるようにパラメータを更新する
        loss = MeanSquaredWeightedError(target, W)(qs)

        self.qnet.clear_grads()
        loss.backward()
        self.optimizer.update()




# 実際にコンピュータとして使われるクラス (__call__(), get_action() は親クラスのものを使う)
class DQNComputer(DQNAgent):
    def __init__(self, qnet_class, action_size):
        self.qnet = qnet_class(action_size)

    # 何ステップ先の報酬まで見て学習したものを使うか選ぶことによって、難易度を変えることができる
    def reset(self, step_num, turn, file_name):
        self.qnet.load_weights(file_name + f"{turn}_{step_num}")




if __name__ == "__main__":
    # ハイパーパラメータ設定
    buffer_size = 1000000
    step_num = 6
    gamma = 0.99
    prioritized = True
    compress = False

    qnet_class = DuelingNet
    batch_size = 32


    # 経験再生バッファ
    replay_buffer1 = ReplayBuffer(buffer_size, step_num, gamma, prioritized, compress)
    replay_buffer2 = ReplayBuffer(buffer_size, step_num, gamma, prioritized, compress)

    # 環境とエージェント
    board = Board()
    first_agent = DQNAgent(qnet_class, replay_buffer1, board.action_size, batch_size, step_num, gamma)
    second_agent = DQNAgent(qnet_class, replay_buffer2, board.action_size, batch_size, step_num, gamma)


    # 自己対戦
    self_match = DQN(board, first_agent, second_agent)
    self_match.fit(runs = 100, episodes = 3000, file_name = "dqn")