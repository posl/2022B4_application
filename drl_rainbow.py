from math import sqrt
from functools import cache
from collections import deque
from bz2 import BZ2File
import pickle
import zlib

import numpy as np

from inada_framework import Layer, Parameter, Model, Function, optimizers, no_train
from inada_framework.cuda import get_array_module, as_cupy, as_numpy, gpu_enable
from inada_framework.functions import affine, relu, flatten, broadcast_to, sum_to
from drl_utilities import ResNet50, preprocess_to_gpu, SelfMatch
import inada_framework.layers as dzl
from inada_framework.utilities import reshape_for_broadcast
from board import Board

from speedup import get_qmax_action, update_sumtree, weighted_sampling



# =============================================================================
# 行動価値関数を近似するニューラルネットワーク
# =============================================================================

# 強化学習の探索に必要なランダム性をネットワークに持たせるためのレイヤ (Noisy Network)
class NoisyAffine(Layer):
    def __init__(self, out_size, activation = None):
        super().__init__()
        self.out_size = out_size
        self.activation = activation

        # 重みは学習が開始したときに動的に生成する
        self.W_mu = Parameter(None, name = "W_mu")
        self.W_sigma = Parameter(None, name = "W_sigma")
        self.b_mu = Parameter(None, name = "b_mu")
        self.b_sigma = Parameter(None, name = "b_sigma")

    # 通常の Affine レイヤのパラメータが正規分布に従う乱数であるかのような実装
    def forward(self, x):
        xp = get_array_module(x)
        in_size = x.shape[1]
        out_size = self.out_size

        # 乱数ジェネレータの取得
        try:
            rng = getattr(self, "rng")
        except AttributeError:
            rng = xp.random.default_rng()
            self.rng = rng

        # 重みの初期化
        if self.W_mu.data is None:
            self.init_params(xp, in_size, out_size)

        # Factorized Gaussian Noise (正規分布からのサンプリング数を減らす工夫) を使用
        epsilon_in = self.noise_f(rng.standard_normal(size = (in_size, 1), dtype = np.float32), xp)
        epsilon_out = self.noise_f(rng.standard_normal(size = (1, out_size), dtype = np.float32), xp)
        W_epsilon = epsilon_in.dot(epsilon_out)
        b_epsilon = epsilon_out

        # 本当はパラメータが従う正規分布の平均・分散を学習したいが、それだと逆伝播ができないので、再パラメータ化を用いる
        W = self.W_mu + self.W_sigma * W_epsilon
        b = self.b_mu + self.b_sigma * b_epsilon
        x = affine(x, W, b)

        # 活性化関数を挟む場合は、その関数を経由する
        activation = self.activation
        if activation is not None:
            x = activation(x)

        return x

    # 重みの初期化方法はオリジナルの Rainbow のものを採用する
    def init_params(self, xp, in_size, out_size):
        stdv = 1. / sqrt(in_size)
        self.W_mu.data = xp.random.uniform(-stdv, stdv, size = (in_size, out_size)).astype(np.float32)
        self.b_mu.data = xp.random.uniform(-stdv, stdv, size = (1, out_size)).astype(np.float32)

        initial_sigma = 0.5 * stdv
        self.W_sigma.data = xp.full((in_size, out_size), initial_sigma, dtype = np.float32)
        self.b_sigma.data = xp.full((1, out_size), initial_sigma, dtype = np.float32)

    @staticmethod
    def noise_f(x, xp):
        return xp.sign(x) * xp.sqrt(xp.abs(x))




# 分位点数を指定して、行動価値分布に対応する累積分布関数の逆関数を近似する (QR-DQN)
class RainbowNet(Model):
    def __init__(self, action_size, quantiles_num, use_gpu = False):
        super().__init__()

        self.sizes = action_size, quantiles_num
        self.use_gpu = use_gpu

        # 全結合層への入力形式を学習する、畳み込み層
        self.cnn = ResNet50()

        # 行動価値関数をアドバンテージ分布と状態価値分布に分けて学習する (Dueling DQN)
        self.conv_a = dzl.Conv2d1x1(16)
        self.bn_a = dzl.BatchNorm()
        self.a1 = NoisyAffine(512, relu)
        self.a2 = NoisyAffine(action_size * quantiles_num)

        self.conv_v = dzl.Conv2d1x1(8)
        self.bn_v = dzl.BatchNorm()
        self.v1 = NoisyAffine(512, relu)
        self.v2 = NoisyAffine(quantiles_num)

    def forward(self, x):
        batch_size = len(x)
        action_size, quantiles_num = self.sizes

        if self.use_gpu:
            x = as_cupy(x)
        x = self.cnn(x)

        # 学習の円滑化のためにアドバンテージ分布は中心化する
        advantages = relu(self.bn_a(self.conv_a(x)))
        advantages = self.a2(self.a1(flatten(x)))
        advantages = advantages.reshape((batch_size, action_size, quantiles_num))
        advantages -= advantages.mean(axis = 1, keepdims = True)

        values = relu(self.bn_v(self.conv_v(x)))
        values = self.v2(self.v1(flatten(x)))
        values = values.reshape((batch_size, 1, quantiles_num))
        return values + advantages

    # 合法手の中から行動価値関数が最大の行動を選択する
    def get_actions(self, states, placables):
        quantile_values = self(states)
        quantile_values = quantile_values.data

        # 分位点が等間隔であることを想定しているので、確率変数の期待値は分位点の単純平均に一致する
        qs = quantile_values.mean(axis = 2)
        qs = as_numpy(qs)

        # エピソード中の行動選択時での引数の渡し方は、バッチ軸を追加した states, １次元リストの placables とする
        if len(qs) == 1:
            return get_qmax_action(qs[0], placables)

        # インデックスとして使うので、高速化のために np.ndarray に変換する
        actions = [get_qmax_action(q, placable) for q, placable in zip(qs, placables)]
        return np.array(actions)




# =============================================================================
# 重み付きの分位点 Huber 誤差 (損失関数)
# =============================================================================

class QuantileHuberLoss(Function):
    def __init__(self, t, quantiles_num, use_gpu = False):
        self.t = t
        self.quantiles = quantiles(quantiles_num, use_gpu)
        self.N = quantiles_num

    def forward(self, x):
        xp = get_array_module(x)

        #  TD 誤差の形状は、(batch_size, quantiles_num, quantiles_num)
        td_error = self.t - x

        # δ = 1.0 の Huber 誤差
        abs_error = abs(td_error)
        loss = xp.where(abs_error < 1., 0.5 * td_error * td_error, abs_error - 0.5)

        # Huber 誤差に分位点重みを掛ける
        loss *= abs(self.quantiles - (td_error < 0))

        # 次元２方向に平均を取り、次元１方向に和を取る操作を簡略化している
        return loss.sum(axis = (1, 2)) / self.N

    def backward(self, gy):
        x = self.inputs[0].data
        td_error = self.t - x
        abs_error = abs(td_error)

        td_indicator = (td_error < 0)
        abs_indicator = (abs_error < 1.)

        # 和と平均の逆伝播
        gy /= self.N
        td_error_shape = td_error.shape
        gy = reshape_for_broadcast(gy, td_error_shape, axis = (1, 2), keepdims = False)
        gy = broadcast_to(gy, td_error_shape)

        # Huber 誤差の逆伝播
        gy *= abs(self.quantiles - td_indicator)
        mask = (~abs_indicator).astype(gy.dtype)
        mask[td_indicator] = -1.
        mask[abs_indicator] = td_error[abs_indicator]
        gy *= mask

        return -sum_to(gy, x.shape)


@cache
def quantiles(quantiles_num, use_gpu = False):
    step = 1 / quantiles_num
    start = step / 2.
    a = np.array([start + i * step for i in range(quantiles_num)], dtype = np.float32)

    if use_gpu:
        a = as_cupy(a)
    return a




# =============================================================================
# ランダムサンプリングを高速化するためのセグメントツリー
# =============================================================================

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

    def save(self, file_path):
        np.save(file_path + "_tree.npy", self.tree)

    def load(self, file_path):
        self.tree = np.load(file_path + "_tree.npy")


    # セグメントツリーの更新は１つずつのみとする (技巧的な方法と比べてもこれが一番速かった)
    def __setitem__(self, index, value):
        update_sumtree(self.tree, index, value)

    # 優先度付きランダムサンプリングを行う (重複なしではない)
    def sample(self, batch_size):
        return weighted_sampling(self.tree, self.rng.uniform(0, self.sum, batch_size))

    # 全優先度データを確率形式で返す
    @property
    def probs(self):
        return self.tree[self.capacity:] / self.sum

    @property
    def sum(self):
        return self.tree[1]




# =============================================================================
# （優先度付き) 経験再生バッファ
# =============================================================================

# 学習に使う経験データ間の相関を弱め、また経験データを繰り返し使うためのバッファ (経験再生)
class ReplayBuffer:
    def __init__(self, buffer_size, prioritized = True, compress = False, step_num = 3, gamma = 0.9):
        self.buffer = []
        self.buffer_size = buffer_size

        self.prioritized = prioritized
        self.compress = compress

        # Multi-step Q Learning を実現するためのバッファ
        self.step_num = step_num
        self.tmp_buffer = deque(maxlen = step_num), deque(maxlen = step_num)
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
        self.tmp_buffer[1].clear()
        self.tmp_buffer[0].clear()
        self.count = 0

        if self.prioritized:
            self.priorities.reset()
            self.max_priority = self.epsilon
        else:
            self.rng = np.random.default_rng()

    # エージェントの情報である合計ステップ数も一緒に保存する
    def save(self, file_path, total_steps):
        save_data = [self.buffer, self.tmp_buffer, self.count, total_steps]

        if self.prioritized:
            self.priorities.save(file_path)
            save_data.append(self.max_priority)

        fout = BZ2File(file_path + "_buffer.bz2", "wb", compresslevel = 9)
        try:
            save_data = pickle.dumps(save_data)
            fout.write(save_data)
        finally:
            fout.close()

    def load(self, file_path):
        fin = BZ2File(file_path + "_buffer.bz2", "rb")
        try:
            load_data = fin.read()
            load_data = pickle.loads(load_data)
        finally:
            fin.close()

        if self.prioritized:
            self.priorities.load(file_path)
            self.max_priority = load_data.pop()

        self.buffer, self.tmp_buffer, self.count, total_steps = load_data
        return total_steps


    def __len__(self):
        return len(self.buffer)

    def add(self, data, turn):
        tmp_buffer = self.tmp_buffer[turn]
        tmp_buffer.append(data)
        if len(tmp_buffer) < self.step_num:
            return

        nstep_reward = 0
        nstep_gamma = 1.
        gamma = self.gamma

        # data = (state, action, reward, next_state, next_placable)
        for data in tmp_buffer:
            reward, next_state, next_placable = data[2:]
            reward_gamma = nstep_gamma
            nstep_gamma *= gamma

            # 報酬が出るのはエピソード終了時のみなので、報酬を加算していく必要はない
            if reward:
                nstep_reward = reward_gamma * reward
                break

        state, action = tmp_buffer[0][:2]
        nstep_data = state, action, nstep_reward, nstep_gamma, next_state, next_placable

        # 経験データ数がバッファサイズを超えたら、古いものから上書きしていく
        count = self.count
        if count == self.buffer_size:
            count = 0

        if self.compress:
            nstep_data = zlib.compress(pickle.dumps(nstep_data))

        try:
            self.buffer[count] = nstep_data
        except IndexError:
            self.buffer.append(nstep_data)

        # 少なくとも１回は学習に使われてほしいので、優先度の初期値は今までで最大のものとする
        if self.prioritized:
            self.priorities[count] = self.max_priority

        self.count = count + 1


    def get_batch(self, batch_size, progress):
        priorities = self.priorities
        buffer = self.buffer

        if self.prioritized:
            indices = priorities.sample(batch_size)

            # 重みは等確率でサンプリングした時との確率の比の β 乗で、最大値が１になるように変換したものを使用する
            weights = (priorities.probs[np.array(indices)] * len(buffer)) ** (-1 * self.beta(progress))
            weights /= weights.max()
        else:
            # ジェネレータによる重複なしランダムサンプリング
            indices = self.rng.choice(len(buffer), batch_size, replace = False)
            weights = None

        if self.compress:
            selected = [pickle.loads(zlib.decompress(buffer[i])) for i in indices]
        else:
            selected = [buffer[i] for i in indices]

        # 状態データはニューラルネットワークに入力してから、必要に応じて GPU 対応を行うので、np.ndarray でよい
        states = np.stack([x[0] for x in selected])
        next_states = np.stack([x[4] for x in selected])

        # actions はインデックスとして使うだけなので、np.ndarray でよい
        actions = np.array([x[1] for x in selected])

        # パラメータの学習で使用する前に、必要に応じて GPU メモリへの非同期転送を行うので、np.ndarray でよい
        rewards = np.array([x[2] for x in selected], dtype = np.float32)
        gammas = np.array([x[3] for x in selected], dtype = np.float32)

        # 抽出した経験データごとに長さが違う可能性があるため、通常のリスト
        next_placables = [x[5] for x in selected]

        return (states, actions, rewards, gammas, next_states, next_placables), indices, weights


    def update_priorities(self, deltas, indices):
        if self.prioritized:
            # 優先度 = (|TD 誤差| + ε) ^ α
            new_priorities = (abs(deltas) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, new_priorities.max())

            priorities = self.priorities
            for index, value in zip(indices, new_priorities):
                priorities[index] = value




# =============================================================================
# TD 法・価値ベースでパラメータを学習するエージェント
# =============================================================================

class RainbowAgent:
    def __init__(self, replay_buffer, batch_size, action_size, quantiles_num, lr, to_gpu = False):
        assert isinstance(replay_buffer, ReplayBuffer)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.action_size = action_size
        self.quantiles_num = quantiles_num
        self.lr = lr / 4. if replay_buffer.prioritized else lr
        self.use_gpu = to_gpu and gpu_enable

    # エージェントを動かす前に呼ぶ必要がある
    def reset(self):
        qnet_args = self.action_size, self.quantiles_num, self.use_gpu
        self.qnet = RainbowNet(*qnet_args)
        self.optimizer = optimizers.Adam(self.lr).setup(self.qnet)

        # TD ターゲットの安定性のために、それを生成するためのネットワークは別で用意する (ターゲットネットワーク)
        self.qnet_target = RainbowNet(*qnet_args)

        self.replay_buffer.reset()
        self.total_steps = 0

        if self.use_gpu:
            self.qnet.to_gpu()
            self.qnet_target.to_gpu()

    # キーボード例外で学習を中断した場合は、再開に必要な情報も保存する
    def save(self, file_path, is_yet = False):
        if is_yet:
            self.qnet.save_weights(file_path + "_online.npz")
            self.qnet_target.save_weights(file_path + "_target.npz")
            self.replay_buffer.save(file_path, self.total_steps)
        else:
            self.qnet.save_weights(file_path + ".npz")

    # 同じ条件での途中再開に必要な情報を読み込む
    def load_to_restart(self, file_path):
        self.qnet.load_weights(file_path + "_online.npz")
        self.qnet_target.load_weights(file_path + "_target.npz")
        self.total_steps = self.replay_buffer.load(file_path)

        if self.use_gpu:
            self.qnet.to_gpu()
            self.qnet_target.to_gpu()

    # ターゲットネットワークはパラメータを学習せず、定期的に学習対象のネットワークと同期させることで学習を進行させる
    def sync_qnet(self):
        self.qnet_target.copy_weights(self.qnet)


    # エージェントを関数形式で使うとその方策に従った行動が得られる
    def __call__(self, board):
        action, __ = self.get_action(board)
        return action

    # ニューラルネットワーク内のランダム要素が探索の役割を果たす
    def get_action(self, board, placable = None):
        if placable is None:
            placable = board.list_placable()

        with no_train():
            state = board.get_img()
            action = self.qnet.get_actions(state[None, :], placable)
            return action, state


    def update(self, data, turn, progress):
        total_steps = self.total_steps
        self.total_steps = total_steps + 1
        replay_buffer = self.replay_buffer
        replay_buffer.add(data, turn)

        # 各種インターバルや開始タイミングは、おおよそオリジナルのものに従う
        if total_steps % 4 or total_steps < 65536:
            return
        if not total_steps % 32768:
            self.sync_qnet()

        # バッチサイズ分の経験データと優先度更新用のインデックス、重点サンプリングの重みを取得する
        batch_size = self.batch_size
        experiences, indices, weights = replay_buffer.get_batch(batch_size, progress)
        states, actions, rewards, gammas, next_states, next_placables = experiences

        # GPU を使用する場合は、非同期メモリ転送を開始する
        use_gpu = self.use_gpu
        if use_gpu:
            rewards, source, rewards_stream = preprocess_to_gpu(rewards)
            rewards.set(source, stream = rewards_stream)

            gammas, source, gammas_stream = preprocess_to_gpu(gammas)
            gammas.set(source, stream = gammas_stream)

            weights, source, weights_stream = preprocess_to_gpu(weights)
            weights.set(source, stream = weights_stream)

        # 誤差を含む出力に max 演算子を使うことで過大評価を起こさないように、行動はオンライン分布から取る (Double DQN)
        with no_train():
            sequence = np.arange(batch_size)
            next_actions = self.qnet.get_actions(next_states, next_placables)

            next_quantile_values = self.qnet_target(next_states)
            next_quantile_values = next_quantile_values.data[sequence, next_actions]

            if use_gpu:
                rewards_stream.synchronize()
                gammas_stream.synchronize()

            # ターゲットネットワークの出力にベルマンオペレータを適用する
            rewards = rewards[:, None]
            target_quantile_values = rewards + (rewards == 0) * gammas[:, None] * next_quantile_values

            # TD ターゲットの形状は、(batch_size, 1, quantiles_num)
            target_quantile_values = target_quantile_values[:, None]

        # 分位点数データの形状は、(batch_size, quantiles_num, 1)
        quantile_values = self.qnet(states)
        quantile_values = quantile_values[sequence, actions]
        quantile_values = quantile_values.reshape((batch_size, self.quantiles_num, 1))

        # 出力の TD 誤差の形状は、(batch_size, )
        td_loss = QuantileHuberLoss(target_quantile_values, self.quantiles_num, use_gpu)(quantile_values)

        # TD 誤差を使って、経験データの優先度を更新する
        replay_buffer.update_priorities(td_loss.data, indices)

        # 優先度付き経験再生を使っている場合は、重点サンプリングの重みを掛ける
        if weights is not None:
            if use_gpu:
                weights_stream.synchronize()
            td_loss *= weights

        loss = td_loss.mean()

        self.qnet.clear_grads()
        loss.backward()
        self.optimizer.update()




# =============================================================================
# 自己対戦による学習
# =============================================================================

class Rainbow(SelfMatch):
    def fit_episode(self, progress):
        board = self.board
        agent = self.agent
        transition_infos = deque(), deque()

        board.reset()
        placable = board.list_placable()

        while placable:
            action, state = agent.get_action(board, placable)
            board.put_stone(action)

            # 遷移情報を一時バッファに格納する
            turn = board.turn
            buffer = transition_infos[turn]
            buffer.append((placable, state, action))

            # 遷移情報２つセットで１回の update メソッドを呼ぶ
            if len(buffer) == 2:
                __, state, action = buffer.popleft()
                next_placable, next_state, __ = buffer[0]

                # 報酬はゲーム終了まで出ない
                agent.update((state, action, 0, next_state, next_placable), turn, progress)

            placable = board.can_continue_placable()

        reward = board.reward
        next_state = board.get_img()

        # 遷移情報のバッファを先攻・後攻とも空にしたらエピソード終了
        __, state, action = buffer.popleft()
        agent.update((state, action, reward, next_state, placable), turn, progress)

        turn ^= 1
        buffer = transition_infos[turn]
        __, state, action = buffer.popleft()
        agent.update((state, action, -reward, next_state, placable), turn, progress)


def fit_rainbow_agent(episodes = 150000, restart = False):
    # ハイパーパラメータ設定
    buffer_size = 100000
    prioritized = True
    compress = False
    step_num = 3
    gamma = 0.95
    batch_size = 32
    quantiles_num = 50
    lr = 0.0005
    to_gpu = True

    # 環境
    board = Board()

    # 経験再生バッファ
    replay_buffer = ReplayBuffer(buffer_size, prioritized, compress, step_num, gamma)

    # エージェント
    agent = RainbowAgent(replay_buffer, batch_size, board.action_size, quantiles_num, lr, to_gpu)

    # 自己対戦
    self_match = Rainbow(board, agent)
    self_match.fit(1, episodes, restart, file_name = "rainbow")




# =============================================================================
# 実際に使われることを想定したコンピュータ
# =============================================================================

class RainbowComputer(RainbowAgent):
    def __init__(self, action_size, quantiles_num = 50, file_name = "rainbow-0", to_gpu = False):
        use_gpu = to_gpu and gpu_enable
        qnet = RainbowNet(action_size, quantiles_num, use_gpu)
        self.qnet = qnet

        file_path = Rainbow.get_path(file_name).format("parameters")
        qnet.load_weights(file_path + ".npz")
        if use_gpu:
            qnet.to_gpu()




if __name__ == "__main__":
    # 学習用コード
    fit_rainbow_agent(restart = False)