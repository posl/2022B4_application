from math import log, sqrt, ceil
from random import Random, sample, choices
from glob import glob
from collections import deque, defaultdict
from bz2 import BZ2File
import pickle
from time import time

import numpy as np
import ray
try:
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from matplotlib import collections, patches
except ImportError:
    pass

from inada_framework import Model, Function, no_train
from drl_utilities import ResNet50, SelfMatch, preprocess_to_gpu
import inada_framework.layers as dzl
from inada_framework.functions import relu, flatten, tanh
from inada_framework.cuda import get_array_module, gpu_enable, as_cupy
from board import Board
from inada_framework.optimizers import Adam, WeightDecay
from inada_framework.utilities import make_dir_exist

from mc_primitive import PrimitiveMonteCarlo
from mc_tree_search import MonteCarloTreeSearch
from gt_alpha_beta import AlphaBeta
from drl_reinforce import ReinforceComputer
from drl_rainbow import RainbowComputer



# =============================================================================
# PV-MCTS での過去の探索状況を近似するニューラルネットワーク
# =============================================================================

class PolicyValueNet(Model):
    def __init__(self, action_size):
        super().__init__()

        # 全結合層への入力形式を学習する、畳み込み層
        self.cnn = ResNet50()

        # policy head
        self.conv_p = dzl.Conv2d1x1(2)
        self.bn_p = dzl.BatchNorm()
        self.fc_p = dzl.Affine(action_size)

        # value head
        self.conv_v = dzl.Conv2d1x1(1)
        self.bn_v = dzl.BatchNorm()
        self.fc_v = dzl.Affine(1)

    def forward(self, x):
        x = self.cnn(x)
        score = self.fc_p(flatten(relu(self.bn_p(self.conv_p(x)))))
        value = self.fc_v(flatten(relu(self.bn_v(self.conv_v(x)))))
        return score, tanh(value)


def softmax(score, xp = np):
    # オーバーフロー対策に最大要素との差を取る
    score -= score.max(axis = 1, keepdims = True)

    score = xp.exp(score)
    policy = score / score.sum(axis = 1, keepdims = True)
    return policy




# =============================================================================
# SoftmaxWith CrossEntropyError  +  MeanSquaredError  (損失関数)
# =============================================================================

class AlphaZeroLoss(Function):
    def __init__(self, mcts_policys, rewards):
        self.mcts_policys = mcts_policys
        self.rewards = rewards

    def forward(self, score, value):
        xp = get_array_module(score)
        policy = softmax(score, xp)

        # ソフトマックス関数の出力を使うと逆伝播が簡単になるので、インスタンス変数として保存しておく
        self.policy = policy

        # 方策の損失関数は多クラス交差エントロピー誤差
        policy = xp.clip(policy, a_min = 1e-15, a_max = None)
        loss = self.mcts_policys * xp.log(policy)
        loss = -loss.sum(axis = 1, keepdims = True)

        # 状態価値関数の損失関数は平均二乗誤差
        loss += 0.5 * ((value - self.rewards) ** 2.)

        return xp.average(loss)

    def backward(self, gy):
        value = self.inputs[1].data

        # 誤差関数の逆伝播
        gy /= len(value)
        gscore = gy * (self.policy - self.mcts_policys)
        gvalue = gy * (value - self.rewards)

        # 不要となった配列をメモリから消去する
        self.mcts_policys = None
        self.rewards = None
        self.policy = None

        return gscore, gvalue




# =============================================================================
# PV-MCTS に基づいて行動選択をするエージェント
# =============================================================================

class AlphaZeroAgent:
    def __init__(self, action_size, arg = None, simulations = 800, seed = None):
        self.network = PolicyValueNet(action_size)
        self.reset(arg)

        # ハイパーパラメータ
        self.simulations = simulations
        self.sampling_limits = 15
        self.c_puct = lambda T: (log(1 + (1 + T) / 19652) + 1.25) * sqrt(T)

        # 並列実行によって、同じ乱数の種を使わないようにするために、明示的にインスタンスを生成する
        random = Random(seed)
        self.__choices = random.choices
        self.__choice = random.choice

        rng = np.random.default_rng(seed)
        self.__dirichlet = rng.dirichlet


    # arg は、パラメータが保存されたファイルのインデックスか、同じモデルのインスタンス
    def reset(self, arg):
        self.load(arg)

        # それぞれ、過去の探索割合を近似したある種の方策、過去の勝率も参考にした累計行動価値、各状態・行動の探索回数
        self.P = {}
        self.W = {}
        self.N = {}

        # 再計算を無くすための辞書
        self.next_states_dict = {}
        self.placable_dict = {}

    def load(self, weights):
        self.network.set_weights(weights)


    def __call__(self, board):
        with no_train():
            return self.get_action(board)

    def get_action(self, board, count = None):
        # 必要に応じて、ルート盤面の合法手リストの登録も行うので、先に呼ぶ必要がある
        placable = self.__get_placable(board)

        selfplay_flag = (count is not None)
        board_img, policy = self.__search(board, board.state, selfplay_flag)

        if selfplay_flag and (count < self.sampling_limits):
            action = self.__choices(placable, policy)[0]
        else:
            indices = np.where(policy == policy.max())[0]
            index = indices[0] if len(indices) == 1 else self.__choice(indices)
            action = placable[index]

        if selfplay_flag:
            # 方策の形状をニューラルネットワークの出力と合わせる
            mcts_policy = np.zeros(board.action_size, dtype = np.float32)
            mcts_policy[np.array(placable)] = policy
            return action, board_img, mcts_policy

        # 学習時以外、方策の計算などの処理は不要
        return action


    # シミュレーションを行なって得た方策を返すメソッド
    def __search(self, board, root_state, selfplay_flag):
        if root_state in self.P:
            board_img = board.get_img()
        else:
            board_img = self.__expand(board, root_state, root_flag = True)

        # 自己対戦時の探索初期状態では、ランダムな手が選ばれやすくなるように、ノイズをかける
        if selfplay_flag:
            P = self.P[root_state]
            self.P[root_state] = 0.75 * P + 0.25 * self.__dirichlet(alpha = np.full(len(P), 0.35))

        # PUCT アルゴリズムで指定回数だけシミュレーションを行う
        for __ in range(self.simulations):
            self.__evaluate(board, root_state)
            board.set_state(root_state)

        # 評価値が高いほど探索回数が多くなるように収束するので、それを方策として使う
        N = self.N[root_state]
        policy = N / N.sum()
        return board_img, policy


    # 子盤面を展開し、その方策・累計行動価値・探索回数の初期化を行うメソッド
    def __expand(self, board, state, root_flag = False):
        board_img = board.get_img()
        score, value = self.network(board_img[None, :])
        policy = softmax(score.data)

        placable = self.placable_dict[state]
        self.P[state] = policy[0, np.array(placable)]
        self.W[state] = np.zeros(len(placable), dtype = np.float32)
        self.N[state] = np.zeros(len(placable), dtype = np.float32)

        # ルート盤面の展開時は評価値の代わりに、学習時用にニューラルネットワークへの入力としての画像データを返す
        if root_flag:
            return board_img

        # それ以外は、ニューラルネットワークの出力である過去の勝率を返す
        return value.data[0, 0]


    # 初到達の盤面の展開をしてゲーム木を大きくしながら、ゲームの報酬や過去のデータをもとに評価値を返すメソッド
    def __evaluate(self, board, state, continue_flag = 1):
        if not continue_flag:
            # ゲーム終了時は最後に手を打ったプレイヤーの報酬を返す
            return board.reward

        elif state in self.P:
            W = self.W[state]
            N = self.N[state]

            # 右辺の第１項が過去の結果を勘案しつつ探索を促進する項で、第２項が勝率を見て活用を促進する項
            pucts = self.c_puct(N.sum()) * self.P[state] / (1 + N) + W / (N + 1e-15)

            # np.argmax を使うと選択が前にある要素に偏るため、np.where で取り出したインデックスからランダムに選ぶ
            indices = np.where(pucts == pucts.max())[0]
            index = indices[0] if len(indices) == 1 else self.__choice(indices)

            action = self.placable_dict[state][index]
            self.board_put_stone(board, action)

            # 手番交代によって次の状態が変わる可能性があることに注意
            next_continue_flag = self.board_can_continue(board)
            value = self.__evaluate(board, board.state, next_continue_flag)

            # 結果を反映させる
            W[index] += value
            N[index] += 1.

        else:
            # 展開していなかった盤面の場合は、ニューラルネットワークの出力である過去の勝率を評価値として返す
            value = self.__expand(board, state)

        # １つ前に手を打ったプレイヤーにとっての評価値を返す (手番が変わっていた場合は、視点が逆になる)
        if continue_flag == 1:
            return -value
        return value


    # 高速化のため、独自の put_stone メソッドを使う
    def board_put_stone(self, board, action):
        next_states_dict = self.next_states_dict
        key = board.state
        key += (action, )

        if key in next_states_dict:
            board.set_state(next_states_dict[key])
        else:
            board.put_stone(action)
            next_states_dict[key] = board.state


    # 合法手のリストの取得の仕方が異なるため、独自の can_continue メソッドを使う
    def board_can_continue(self, board, pass_flag = False):
        board.turn ^= 1
        if self.__get_placable(board):
            return 1 + pass_flag
        elif pass_flag:
            return 0
        else:
            return self.board_can_continue(board, True)

    def __get_placable(self, board):
        placable_dict = self.placable_dict
        state = board.state

        if state in placable_dict:
            return placable_dict[state]
        else:
            placable = board.list_placable()
            placable_dict[state] = placable
            return placable




# =============================================================================
# 実際に使われることを想定したコンピュータ
# =============================================================================

class AlphaZeroComputer(AlphaZeroAgent):
    def load(self, index):
        self.network.load_weights(self.get_trained_path(index))

    @staticmethod
    def get_trained_path(index):
        file_path = SelfMatch.get_path("alphazero").format("parameters")

        if isinstance(index, int) and (0 <= index <= 9):
            file_path += f"-{index}.npz"
        else:
            # インデックスの指定がなかった場合は、最も成熟したパラメータを使う
            file_path += "-[0-9].npz"
            file_paths = glob(file_path)
            file_paths.sort()
            file_path = file_paths.pop()

        return file_path




# =============================================================================
# 経験再生バッファ
# =============================================================================

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.indices = None
        self.rng = np.random.default_rng()

    @property
    def max_iter(self):
        return ceil(len(self.buffer) / self.batch_size)

    def save(self, file_path, step):
        fout = BZ2File(file_path, "wb", compresslevel = 9)
        try:
            save_data = pickle.dumps((self.buffer, step))
            fout.write(save_data)
        finally:
            fout.close()

    def load(self, file_path):
        fin = BZ2File(file_path, "rb")
        try:
            load_data = fin.read()
            self.buffer, step = pickle.loads(load_data)
        finally:
            fin.close()
        return step


    # １エピソード分のデータがリストとして、まとめて格納される
    def add(self, datas):
        self.buffer.extend(datas)

    def __iter__(self):
        self.indices = self.rng.permutation(len(self.buffer))
        return self

    # ランダムに並び替えた経験データをバッチサイズ分ずつ順に取り出す
    def __next__(self):
        indices = self.indices
        if indices.size:
            indices, self.indices = np.split(indices, [self.batch_size])
        else:
            self.indices = None
            raise StopIteration()

        buffer = self.buffer
        selected = [buffer[i] for i in indices]

        states = np.stack([x[0] for x in selected])
        mcts_policys = np.stack([x[1] for x in selected])
        rewards = np.array([x[2] for x in selected], dtype = np.float32)

        # ゲームの報酬はニューラルネットワークの教師データとして使うので、形状をそれと合わせる
        rewards = rewards[:, None]

        return states, mcts_policys, rewards




# =============================================================================
# 並列実行により、自己対戦・エージェントの評価を高速化するための関数
# =============================================================================

@ray.remote(num_cpus = 1, num_gpus = 0)
def play(weights, simulations, seed):
    # 環境
    board = Board()
    board.reset()

    # エージェント (先攻・後攻で分けない)
    agent = AlphaZeroAgent(board.action_size, weights, simulations, seed)

    # 実際に１ゲームプレイして、対局データを収集する
    memory = []
    for count in range(board.action_size):
        action, state, mcts_policy = agent.get_action(board, count)
        memory.append([state, mcts_policy, board.turn])

        agent.board_put_stone(board, action)
        if not agent.board_can_continue(board):
            break

    final_player = board.turn
    reward = board.reward

    # 最後に手を打ったプレイヤーとその報酬の情報で、各対局データの報酬を確定する
    for data in memory:
        player = data.pop()
        data.append(reward if player == final_player else -reward)

    return memory


@ray.remote(num_cpus = 1, num_gpus = 0)
def test(weights, simulations, seed, turn, enemy):
    # 環境
    board = Board()
    board.reset()

    # エージェント (先攻・後攻で分けない)
    agent = AlphaZeroAgent(board.action_size, weights, simulations, seed)

    # 方策の設定・１ゲーム勝負
    plans = (agent, enemy) if turn else (enemy, agent)
    board.set_plan(*plans)
    board.game()

    # 勝利か否かを取得
    result = board.black_num - board.white_num
    return (result > 0) if turn else (result < 0)


@ray.remote(num_cpus = 1, num_gpus = 0)
def verify(weights1, weights0, seed1, seed0, couple):
    # 環境
    board = Board()
    board.reset()

    # エージェント
    agent1 = AlphaZeroAgent(board.action_size, arg = weights1, seed = seed1)
    agent0 = AlphaZeroAgent(board.action_size, arg = weights0, seed = seed0)

    # 方策の設定・１ゲーム勝負
    board.set_plan(agent1, agent0)
    board.game()

    # 先攻が勝利したか否かをを取得
    result = board.black_num - board.white_num
    if result:
        result = result > 0, result < 0
    else:
        result = 0, 0
    return couple, result




# =============================================================================
# 自己対戦・探索状況の反映・エージェントの評価を繰り返すための闘技場
# =============================================================================

class AlphaZero:
    def __init__(self, buffer_size = 76800, batch_size = 128, lr = 0.001, decay = 0.0001, to_gpu = True):
        # 学習対象のニューラルネットワーク
        self.network = PolicyValueNet(Board.action_size)

        # 経験再生バッファ
        self.buffer = ReplayBuffer(buffer_size, batch_size)

        # 最適化手法
        self.optimizer = Adam(lr).setup(self.network)
        self.optimizer.add_hook(WeightDecay(decay))

        # 学習部分は GPU による高速化の余地がある
        self.use_gpu = to_gpu and gpu_enable


    def fit(self, updates = 300, episodes = 128, epochs = 5, simulations = 800, restart = False):
        network = self.network
        buffer = self.buffer
        optimizer = self.optimizer
        use_gpu = self.use_gpu

        # 各ファイルパス
        file_path = SelfMatch.get_path("alphazero")
        is_yet_path = file_path.format("is_yet")
        params_path = file_path.format("parameters")
        graphs_path = file_path.format("graphs")
        del file_path

        # ディレクトリが存在しなければ作る
        make_dir_exist(is_yet_path)
        make_dir_exist(graphs_path)


        if restart:
            # 学習の途中再開に必要なデータをファイルから読み込む
            network.load_weights(f"{is_yet_path}_weights.npz")
            restart = buffer.load(f"{is_yet_path}_buffer.bz2")
            history = np.load(f"{is_yet_path}_history.npy")

            # 学習率を進捗に応じて変更する
            n = 4. ** (restart // 100)
            optimizer.lr /= n

            # 次のステップから学習を再開する
            restart += 1

        else:
            # 学習開始前にダミーの入力をニューラルネットワークに流して、初期の重みを確定する
            with no_train():
                network(np.zeros((1, 2, Board.height, Board.width), dtype = np.float32))

            restart = 1
            history = np.zeros((2, 100), dtype = np.int32)


        # 並列実行を行うための初期設定
        ray.init()

        # ray の共有メモリへの重みパラメータのコピーを明示的に行うことで、以降の処理を高速化する
        weights = ray.put(network.get_weights())

        # GPU を使用するかどうかの表示
        answer = "Yes, it do." if self.use_gpu else "No, it don't."
        print(f"Q: Will this script use GPU?\nA: {answer}\n")
        del answer


        # 評価結果の表示
        print("\033[92m=== Winning Percentage ===\033[0m")
        print("progress || first | second")
        print("===========================")

        # 変数定義
        population = range(episodes * 1024)

        assert not updates % 100
        eval_interval = updates // 100

        start_time = time()


        try:
            for step in range(restart, updates + 1):
                with tqdm(desc = f"step {step}", total = episodes, leave = False) as pbar:
                    with no_train():
                        # まだ完遂していないタスクがリストの中に残るようになる
                        seeds = sample(population, episodes)
                        remains = [play.remote(weights, simulations, seed) for seed in seeds]

                        # タスクが１つ終了するたびに、経験データをバッファに格納するような同期処理
                        while remains:
                            finished, remains = ray.wait(remains, num_returns = 1)
                            buffer.add(ray.get(finished[0]))
                            pbar.update(1)

                if step >= 5:
                    # episodes だけゲームをこなすごとに、epochs で指定したエポック数だけ、パラメータを学習する
                    with tqdm(total = epochs * buffer.max_iter, leave = False) as pbar:
                        if use_gpu:
                            # GPU を使う場合は、モデルの重みをそれ対応にする
                            network.to_gpu()

                        for epoch in range(1, epochs + 1):
                            pbar.set_description(f"fit {step}, epoch {epoch}")

                            for states, mcts_policys, rewards in buffer:
                                if use_gpu:
                                    # GPU を使う場合は、そのメモリへデータを非同期転送する
                                    mcts_policys, source, stream_m = preprocess_to_gpu(mcts_policys)
                                    mcts_policys.set(source, stream = stream_m)

                                    rewards, source, stream_r = preprocess_to_gpu(rewards)
                                    rewards.set(source, stream = stream_r)

                                    # すぐに使うデータに非同期転送は使わない
                                    states = as_cupy(states)

                                score, value = network(states)

                                if use_gpu:
                                    # データを使う前に非同期の転送処理と実行スクリプトを同期させる
                                    stream_m.synchronize()
                                    stream_r.synchronize()

                                loss = AlphaZeroLoss(mcts_policys, rewards)(score, value)

                                # 勾配が加算されていかないように、先にリセットしてから逆伝播を行う
                                network.clear_grads()
                                loss.backward()
                                optimizer.update()

                                pbar.update(1)

                        if use_gpu:
                            # GPU を使った場合は、モデルの重みを CPU 対応に戻す
                            network.to_cpu()

                    if not step % 100:
                        optimizer.lr /= 4.


                    # パラメータを更新したので、新しく ray の共有メモリに重みをコピーする
                    weights = ray.put(network.get_weights())

                    eval_q, eval_r = divmod(step, eval_interval)
                    if not eval_r:
                        save_q, save_r = divmod(eval_q, 10)

                        # パラメータの保存 (合計 10 回)
                        if not save_r:
                            network.save_weights(params_path + "-{}.npz".format(save_q - 1))

                        # エージェントの評価 (合計 100 回)
                        win_rates = self.eval(weights)
                        history[:, eval_q - 1] = win_rates
                        print("{:>6} % || {:>3} % | {:>3} %".format(eval_q, *win_rates), end = "   ")

                        # 累計経過時間の表示
                        print("({:.5g} min elapsed)".format((time() - start_time) / 60.))


                # 途中再開に必要な暫定の情報を上書きする
                print("  now saving: \"Don't suspend right now, please.\"", end = "\r")
                network.save_weights(f"{is_yet_path}_weights.npz")
                buffer.save(f"{is_yet_path}_buffer.bz2", step)
                np.save(f"{is_yet_path}_history.npy", history)

            print("\033[K")

        finally:
            # 学習の進捗を x 軸、その時の勝率を y 軸とするグラフを描画し、画像保存する
            x = np.arange(1, 101)
            plt.plot(x, history[0], label = "first")
            plt.plot(x, history[1], label = "second")
            plt.legend()

            plt.ylim(-5, 105)
            plt.xlabel("Progress Rate")
            plt.ylabel("Winning Percentage")
            plt.title("AlphaZero's Changes")

            plt.savefig(graphs_path)
            plt.clf()


    @staticmethod
    def eval(weights, simulations = 100, enemy = AlphaBeta()):
        N = 20
        population = range(N * 1024)

        with tqdm(desc = "now evaluating", total = N * 2, leave = False) as pbar:
            enemy = ray.put(enemy)
            win_rates = []

            for turn in (1, 0):
                seeds = sample(population, N)
                remains = [test.remote(weights, simulations, seed, turn, enemy) for seed in seeds]

                # タスクが１つ終了するたびに、勝利数を加算していくような同期処理
                win_count = 0
                while remains:
                    finished, remains = ray.wait(remains, num_returns = 1)
                    win_count += ray.get(finished[0])
                    pbar.update(1)

                win_rates.append(int(win_count * (100 / N)))
        return win_rates




# =============================================================================
# エージェントの最終評価を行う関数
# =============================================================================

def eval_alphazero_computer(index = None):
    # ファイルのパス
    params_path = AlphaZeroComputer.get_trained_path(index)
    graphs_path = params_path.replace("parameters", "graphs")[:-6]
    print(f"\nuse {params_path}")

    # 並列実行を行うための初期化
    ray.shutdown()
    ray.init()

    # パラメータを共有メモリにコピー
    network = PolicyValueNet(Board.action_size)
    network.load_weights(params_path)
    weights = ray.put(network.get_weights())
    del params_path, network


    # 対戦する相手の設定  (name, alias, instance)
    enemys = []
    enemys.append(("Rainbow",  "rain",  RainbowComputer(Board.action_size)))
    enemys.append(("REINFORCE",  "rein",  ReinforceComputer(Board.action_size)))
    enemys.append(("Alpha Beta",  "ab",  AlphaBeta()))
    enemys.append(("MCTS",  "mcts",  MonteCarloTreeSearch()))
    enemys.append(("MC Primitive",  "mcp",  PrimitiveMonteCarlo()))


    # 棒グラフの設定
    bar_fig, bar_ax = plt.subplots(tight_layout = True)

    bar_num = 3
    sims_array = 50 * (4 ** np.arange(bar_num))
    bar_record = np.empty((2, bar_num), dtype = np.int32)

    bar_width = 1 / 3
    bar_left = np.arange(bar_num)
    bar_center = bar_left + bar_width

    # 円グラフの設定
    pie_fig, pie_axes = plt.subplots(2, 3, tight_layout = True)

    pie_colors = ["orange", "greenyellow", "aqua"]
    edge_infos = {"linewidth" : 2, "edgecolor" : "white"}
    pie_configs = {"startangle" : 90, "counterclock" : False, "wedgeprops" : edge_infos}


    # 対戦相手を変化させながら、コンピュータを評価する
    start = time()
    for i in range(6):
        pie_ax = pie_axes[divmod(i, 3)]
        try:
            name, alias, enemy = enemys.pop()
        except IndexError:
            pie_fig.delaxes(pie_ax)
            continue

        # 対戦部分
        print(f"\n\033[92m>> vs. {name}\033[0m")
        print("sims || first | second")
        print("=======================")

        for i, simulations in enumerate(sims_array):
            win_rates = AlphaZero.eval(weights, simulations, enemy)
            bar_record[:, i] = win_rates
            print("{:>4} || {:>3} % | {:>3} %".format(simulations, *win_rates))

        finish = time()
        print("done!  (took {:5g} minutes)".format((finish - start) / 60.))
        start = finish


        # シミュレーション回数を変動させた時の勝率を示す棒グラフを描画し、画像保存する
        bar_ax.bar(bar_left, bar_record[0], width = bar_width, align = "edge", label = "first")
        bar_ax.bar(bar_center, bar_record[1], width = bar_width, align = "edge", label = "second")
        bar_ax.legend()

        # 一目で判断できるようにするために、棒の中に勝率を表示する
        for i in range(2):
            for j, x in enumerate(bar_left + (bar_width / 2)):
                win_rate = bar_record[i][j]
                text = f"{win_rate} %"
                bar_ax.text(x + i * bar_width, -5, text, ha = "center")

        bar_ax.set_xticks(ticks = bar_center, labels = sims_array)
        bar_ax.set_ylim(-10, 105)
        bar_ax.set_xlabel("The Number of Simulations")
        bar_ax.set_ylabel("Winning Percentage")
        bar_ax.set_title(f"AlphaZero vs. {name}", fontsize = 14, color = "red")

        bar_fig.savefig(f"{graphs_path}_{alias}")
        bar_ax.cla()


        # 最大のシミュレーション回数での勝率を円グラフを描画する
        pie_record = np.zeros(3, dtype = np.int32)
        pie_record[:-1] = win_rates
        pie_record[-1] = 200 - pie_record.sum()

        pie_ax.pie(pie_record, colors = pie_colors, autopct = "%.1f%%", **pie_configs)
        pie_ax.set_title(f"vs. {name}    ", fontsize = 12)

    # 凡例・タイトルを付け加えて、画像保存する
    labels = ["AlphaZero Win  (Black)", "AlphaZero Win  (White)", "Draw or Lose"]
    pie_fig.legend(labels, loc = (0.65, 0.2), fontsize = 10)
    pie_fig.suptitle("AlphaZero's Winning Percentage", fontsize = 14, color = "red")
    pie_fig.savefig(f"{graphs_path}_vs")


    # 図のクリア
    bar_fig.clf()
    pie_fig.clf()




def comp_alphazero_computer():
    # ファイルのパス
    graphs_path = SelfMatch.get_path("alphazero").format("graphs")
    params_path = graphs_path.replace("graphs", "parameters")
    params_paths = glob(f"{params_path}-[0-9].npz")
    params_paths.sort()

    # 並列実行を行うための初期化
    ray.shutdown()
    ray.init()

    # パラメータを共有メモリにコピー
    network = PolicyValueNet(Board.action_size)
    weights_list = []

    for params_path in params_paths:
        network.load_weights(params_path)
        weights_list.append(ray.put(network.get_weights()))

    N = len(weights_list)
    del params_path, params_paths, network


    # 先攻・後攻入れ替えて M 回ずつ、自己対戦を行う
    print("\n\033[92m>>Evaluatation by self match\033[0m")
    start = time()

    M = 25
    n_gen = range(0, M * 1024, 1024)

    remains = []
    for i in range(N):
        for j in range(i + 1, N):
            couple = i, j

            for k in (1, -1):
                l, m = couple[::k]
                WL, WM = weights_list[l], weights_list[m]

                seeds = [choices(range(i, i + 1024), k = 2) for i in n_gen]
                remains.extend([verify.remote(WL, WM, SL, SM, (l, m)) for SL, SM in seeds])

    # 並列実行を行う
    with tqdm(desc = "now evaluating", total = N * (N - 1) * M, leave = False) as pbar:
        results = np.zeros((N, N, 2), dtype = np.int32)

        while remains:
            finished, remains = ray.wait(remains, num_returns = 1)
            couple, result = ray.get(finished[0])
            results[couple] += result
            pbar.update(1)

    print("done!  (took {:5g} minutes)".format((time() - start) / 60.))


    # 図の生成
    fig, ax = plt.subplots()
    ax.clear()

    # タイトルを設定
    fig.suptitle("AlphaZero's Winning Percentages In Self Match", fontsize = 14, color = "red")
    ax.set_title("(Between Agents With Different Progress Rates)", fontsize = 12)

    # ラベルを描画しないように設定
    ax.tick_params(labelbottom = False, labelleft = False, labelright = False, labeltop = False)

    # 目盛り線・グリッド線を描画
    HW = N + 1
    ax.set_xticks(range(HW + 1))
    ax.set_yticks(range(HW + 1))
    ax.grid(True, color = "darkgray")

    # 各軸方向の範囲を設定
    ax.set_xlim(0, HW)
    ax.set_ylim(0, HW)

    # 総当たり表を見やすくするための直線を描画
    lines = [(0, HW), (HW, 0)], [(1, 0), (1, HW)], [(0, N), (HW, N)]
    ax.add_collection(collections.LineCollection(lines, color = "gray"))

    # 先攻・後攻を表現するための楕円を描画
    xy = np.array([0, N])
    ax.add_patch(patches.Ellipse(xy = xy + 0.3, width = 0.2, height = 0.25, fc = "black", ec = "black"))
    ax.add_patch(patches.Ellipse(xy = xy + 0.7, width = 0.2, height = 0.25, fc = "white", ec = "black"))

    # コンピュータそれぞれの学習の進捗率を示す列を追加
    for i in range(1, HW):
        progress = "{} %".format(10 * i)
        ax.text(0.5, (N - i) + 0.35, progress, ha = "center", size = "small")
        ax.text(i + 0.5, N + 0.35, progress, ha = "center", size = "small")


    # 自己対戦の結果を表に反映させる
    lines_dict = defaultdict(lambda: [])
    for i in range(N):
        y = N - (i + 1)

        for j in range(N):
            if i == j:
                continue

            x = j + 1
            black_wins, white_wins = results[i, j]
            result = f"{black_wins} - {white_wins}"

            ax.text(x + 0.5, y + 0.4, result, ha = "center", size = "small")

            diff = black_wins - white_wins
            if diff:
                x_offset = 0.05 + 0.075 * (7 - len(result))
                lines_dict[diff].append([(x + x_offset, y + 0.3), (x + (1. - x_offset), y + 0.3)])

    # 結果を分かりやすくするための下線を描画
    for diff, lines in lines_dict.items():
        if lines:
            S = max(0.9 - abs(diff) / M, 0)
            RGBA = (1., S, S, 0.8) if diff > 0 else (S, S, 1., 0.8)
            ax.add_collection(collections.LineCollection(lines, color = RGBA))

    # 図の保存
    fig.savefig(f"{graphs_path}_self")
    fig.clf()




if __name__ == "__main__":
    # 学習用コード
    # arena = AlphaZero()
    # arena.fit(restart = False)

    # 評価用コード
    eval_alphazero_computer(index = None)
    comp_alphazero_computer()