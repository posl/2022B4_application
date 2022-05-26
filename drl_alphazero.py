from math import log, sqrt, ceil
from random import choices, choice
from collections import deque
from bz2 import BZ2File
import pickle
from time import time

import numpy as np
import ray
from tqdm import tqdm
import matplotlib.pyplot as plt

from inada_framework import Model, Function, no_train
from drl_utilities import ResNet50, SelfMatch, preprocess_to_gpu, corners_plan
import inada_framework.layers as dzl
from inada_framework.functions import relu, flatten, tanh
from inada_framework.cuda import get_array_module, gpu_enable, as_cupy
from board import Board
from inada_framework.optimizers import Adam, WeightDecay

from mc_tree_search import MonteCarloTreeSearch
from mc_primitive import PrimitiveMonteCarlo
from gt_alpha_beta import AlphaBeta
from drl_reinforce import ReinforceComputer
from drl_rainbow import RainbowComputer



# PV-MCTS での過去の探索状況を近似するニューラルネットワーク
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


# ニューラルネットワークの出力を確率形式に変換するための関数
def softmax(score, xp = np):
    # オーバーフロー対策に最大要素との差を取る
    score -= score.max(axis = 1, keepdims = True)

    score = xp.exp(score)
    policy = score / score.sum(axis = 1, keepdims = True)
    return policy




# ニューラルネットワークの出力が過去の探索結果を近似するように学習するための損失関数
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




# モンテカルロ木探索の要領で、深層強化学習を駆使し、行動選択をするエージェント (コンピュータとしても使う)
class AlphaZeroAgent:
    def __init__(self, action_size, sampling_limits = 15, c_base = 19652, c_init = 1.25):
        self.network = PolicyValueNet(action_size)

        # ハイパーパラメータ
        self.sampling_limits = sampling_limits
        self.c_puct = lambda T: (log(1 + (1 + T) / c_base) + c_init) * sqrt(T)

    # 引数はパラメータが保存されたファイルの名前か、同じモデルのインスタンス
    def reset(self, arg, simulations = 800):
        self.load(arg)
        self.simulations = simulations

        # それぞれ、過去の探索割合を近似したある種の方策、過去の勝率も参考にした累計行動価値、各状態・行動の探索回数
        self.P = {}
        self.W = {}
        self.N = {}

        self.placable_dict = {}
        self.rng = np.random.default_rng()

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

        if selfplay_flag and count < self.sampling_limits:
            action = choices(placable, policy)[0]
        else:
            indices = np.where(policy == policy.max())[0]
            action_index = indices[0] if len(indices) == 1 else choice(indices)
            action = placable[action_index]

        if selfplay_flag:
            # 方策の形状をニューラルネットワークのものと合わせる
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
            self.P[root_state] = 0.75 * P + 0.25 * self.rng.dirichlet(alpha = np.full(len(P), 0.35))

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
            # 右辺の第１項が過去の結果を勘案しつつ探索を促進する項で、第２項が勝率を見て活用を促進する項
            N = self.N[state]
            pucts = self.c_puct(N.sum()) * self.P[state] / (1 + N) + self.W[state] / (N + 1e-15)

            # np.argmax を使うと選択が前にある要素に偏るため、np.where で取り出したインデックスからランダムに選ぶ
            indices = np.where(pucts == pucts.max())[0]
            action_index = indices[0] if len(indices) == 1 else choice(indices)

            action = self.placable_dict[state][action_index]
            board.put_stone(action)

            # 手番交代によって次の状態が変わる可能性があることに注意
            next_continue_flag = self.board_can_continue(board)
            value = self.__evaluate(board, board.state, next_continue_flag)

            # 結果を反映させる
            self.W[state][action_index] += value
            self.N[state][action_index] += 1.

        else:
            # 展開していなかった盤面の場合は、ニューラルネットワークの出力である過去の勝率を評価値として返す
            value = self.__expand(board, state)

        # １つ前に手を打ったプレイヤーにとっての評価値を返す (手番が変わっていた場合は、視点が逆になる)
        if continue_flag == 1:
            return -value
        return value


    # 合法手のリストの取得の仕方が異なるため、独自の can_continue メソッドを使う
    def board_can_continue(self, board, pass_flag = False):
        board.turn ^= 1
        if self.__get_placable(board):
            return 1 + pass_flag
        elif pass_flag:
            return 0
        else:
            return self.board_can_continue(board, True)


    # 木の構築と探索がボトルネックなので、それを解消するように board.list_placable() の再計算を無くすためのメソッド
    def __get_placable(self, board):
        placable_dict = self.placable_dict
        state = board.state

        if state in placable_dict:
            return placable_dict[state]
        else:
            placable = board.list_placable()
            placable_dict[state] = placable
            return placable




# 実際にコンピュータとして使われるクラス (上のエージェントをそのまま使っても良い)
class AlphaZeroComputer(AlphaZeroAgent):
    def load(self, file_name):
        self.network.load_weights(SelfMatch.get_path(file_name + ".npz").format("parameters"))




# 過去の探索の記録を残し、それらを順に取り出すことのできるイテラブル
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




@ray.remote(num_cpus = 1, num_gpus = 0)
def alphazero_play(weights, simulations):
    # 環境
    board = Board()
    board.reset()

    # エージェント (先攻・後攻で分けない)
    agent = AlphaZeroAgent(board.action_size)
    agent.reset(weights, simulations)

    # 実際に１ゲームプレイして、対局データを収集する
    memory = []
    for count in range(board.action_size):
        action, state, mcts_policy = agent.get_action(board, count)
        memory.append([state, mcts_policy, board.turn])

        board.put_stone(action)
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
def alphazero_test(weights, simulations, turn, enemy):
    # 環境
    board = Board()
    board.reset()

    # エージェント (先攻・後攻で分けない)
    agent = AlphaZeroAgent(board.action_size)
    agent.reset(weights, simulations)

    # 方策の設定・１ゲーム勝負
    plans = (agent, enemy) if turn else (enemy, agent)
    board.set_plan(*plans)
    board.game()

    # 勝利か否かを取得
    result = board.black_num - board.white_num
    return (result > 0) if turn else (result < 0)




class AlphaZero:
    def __init__(self, buffer_size = 60000, batch_size = 128, lr = 0.0005, decay = 0.001, to_gpu = True):
        # 学習対象のニューラルネットワーク
        self.network = PolicyValueNet(Board.action_size)

        # 経験再生バッファ
        self.buffer = ReplayBuffer(buffer_size, batch_size)

        # 最適化手法
        self.optimizer = Adam(lr).setup(self.network)
        self.optimizer.add_hook(WeightDecay(decay))

        # 学習部分は GPU による高速化の余地がある
        self.use_gpu = to_gpu and gpu_enable


    def fit(self, updates = 500, episodes = 100, epochs = 5, simulations = 300, restart = False):
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

        if restart:
            # 学習の途中再開に必要なデータをファイルから読み込む
            network.load_weights(f"{is_yet_path}_weights.npz")
            restart = buffer.load(f"{is_yet_path}_buffer.bz2")
            history = np.load(f"{is_yet_path}_history.npy")

            if restart > 300:
                optimizer.lr /= 10.

        else:
            # 学習開始前にダミーの入力をニューラルネットワークに流して、初期の重みを確定する
            with no_train():
                network(np.zeros((1, 2, Board.height, Board.width), dtype = np.float32))

            restart = 1
            history = np.zeros((2, 100), dtype = np.int32)

        # 画面表示
        print("\033[92m=== Winning Percentage ===\033[0m")
        print("progress || first | second")
        print("===========================")

        # 変数定義
        assert not updates % 100
        eval_interval = updates // 100
        start_time = time()


        try:
            # 並列実行を行うための初期設定
            ray.init()

            # ray の共有メモリへの重みパラメータのコピーを明示的に行うことで、以降の処理を高速化する
            weights = ray.put(network.get_weights())

            for step in range(restart, updates + 1):
                with tqdm(desc = f"step {step}", total = episodes, leave = False) as pbar:
                    with no_train():
                        # まだ完遂していないタスクがリストの中に残るようになる
                        remains = [alphazero_play.remote(weights, simulations) for __ in range(episodes)]

                        # タスクが１つ終了するたびに、経験データをバッファに格納するような同期処理
                        while remains:
                            finished, remains = ray.wait(remains, num_returns = 1)
                            buffer.add(ray.get(finished[0]))
                            pbar.update(1)

                if step < 5:
                    continue

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

                if step == 300:
                    optimizer.lr /= 10.

                # パラメータを更新したので、新しく ray の共有メモリに重みをコピーする
                weights = ray.put(network.get_weights())

                eval_q, eval_r = divmod(step, eval_interval)
                if not eval_r:
                    save_q, save_r = divmod(eval_q, 10)

                    # パラメータの保存 (合計 10 回)
                    if not save_r:
                        network.save_weights(params_path + "-{}.npz".format(save_q - 1))

                    # エージェントの評価 (合計 100 回)
                    win_rates = self.eval(weights, simulations)
                    history[:, eval_q - 1] = win_rates
                    print("{:>6} % || {:>3} % | {:>3} %".format(eval_q, *win_rates), end = "   ")

                    # 累計経過時間の表示
                    print("({:.5g} min elapsed)".format((time() - start_time) / 60.))


                # 途中再開に必要な暫定の情報を上書きする
                print("  now saving: \"Don't suspend right now, please.\"", end = "\r")
                network.save_weights(f"{is_yet_path}_weights.npz")
                buffer.save(f"{is_yet_path}_buffer.bz2", step)
                np.save(f"{is_yet_path}_history.npy", history)

        finally:
            # 学習の進捗を x 軸、その時の勝率を y 軸とするグラフを描画し、画像保存する
            x = np.arange(1, 101)
            plt.plot(x, history[0], label = "first")
            plt.plot(x, history[1], label = "second")
            plt.legend()

            plt.ylim(-5, 105)
            plt.xlabel("Progress Rate")
            plt.ylabel("Winning Percentage")
            plt.savefig(graphs_path)
            plt.clf()


    @staticmethod
    def eval(weights, simulations, enemy = PrimitiveMonteCarlo(2048)):
        with tqdm(desc = "now evaluating", total = 40, leave = False) as pbar:
            win_rates = []

            for turn in (1, 0):
                remains = [alphazero_test.remote(weights, simulations, turn, enemy) for __ in range(20)]

                # タスクが１つ終了するたびに、勝利数を加算していくような同期処理
                win_count = 0
                while remains:
                    finished, remains = ray.wait(remains, num_returns = 1)
                    win_count += ray.get(finished[0])
                    pbar.update(1)

                win_rates.append(win_count * 5)
        return win_rates




def eval_alphazero_computer(file_name):
    file_path = SelfMatch.get_path(file_name)
    weights = __eval_preprocess(file_path, "sim.")

    # 描画用配列
    length = 6
    simulations_array = 25 * (2 ** np.arange(length))
    win_rates = np.empty((2, length))

    # 行動選択時のシミュレーション回数を推移させながら、コンピュータを評価する
    for i, simulations in enumerate(simulations_array):
        win_rates[:, i] = AlphaZero.eval(weights, simulations)
        print("{:>4} || {:>3} % | {:>3} %".format(simulations, *win_rates[:, i]))


    # グラフの目盛り位置を設定するための変数
    width = 1 / 3
    left = np.arange(length)
    center = left + width

    # 左が先攻、右が後攻の勝率となるような棒グラフを画像保存する
    plt.bar(left, win_rates[0], width = width, align = "edge", label = "first")
    plt.bar(center, win_rates[1], width = width, align = "edge", label = "second")
    plt.xticks(ticks = center, labels = simulations_array)
    plt.legend()

    plt.ylim(-5, 105)
    plt.xlabel("The Number of Simulations")
    plt.ylabel("Winning Percentage")
    plt.title("AlphaZero")
    plt.savefig(file_path.format("graphs") + "_bar")
    plt.clf()




def vs_alphazero_computer(file_name, simulations = 800):
    file_path = SelfMatch.get_path(file_name)
    weights = __eval_preprocess(file_path, "no.")

    # 対戦する相手の設定
    enemys = []
    enemys.append(("Rainbow", RainbowComputer(Board.action_size)))
    enemys.append(("REINFORCE", ReinforceComputer(Board.action_size)))
    enemys.append(("Alpha Beta", AlphaBeta()))
    enemys.append(("MCTS", MonteCarloTreeSearch()))
    enemys.append(("Primitive MC", PrimitiveMonteCarlo()))


    # 図の生成
    fig, axes = plt.subplots(2, 3, tight_layout = True)

    # 各種設定
    pie_colors = ["orange", "greenyellow", "aqua"]
    edge_infos = {"linewidth" : 2, "edgecolor" : "white"}
    pi_configs = {"startangle" : 90, "counterclock" : False, "wedgeprops" : edge_infos}

    # 行動選択時のシミュレーション回数を推移させながら、指定された方策でコンピュータを評価し、円グラフに描画する
    for i in range(6):
        ax = axes[divmod(i, 3)]
        try:
            name, enemy = enemys.pop()
        except IndexError:
            fig.delaxes(ax)
            continue

        win_rates = AlphaZero.eval(weights, simulations, enemy)
        print("{:>2}  || {:>3} % | {:>3} %".format(i, *win_rates))

        record = np.zeros(3)
        record[:-1] = win_rates
        record[-1] = 200 - record.sum()

        ax.pie(record, colors = pie_colors, autopct = "%.1f%%", **pi_configs)
        ax.set_title(f"vs. {name}    ", fontsize = 12)

    # 凡例・タイトルを付け加えて、画像保存する
    labels = ["AlphaZero Win  (Black)", "AlphaZero Win  (White)", "Draw or Lose"]
    fig.legend(labels, loc = (0.65, 0.2), fontsize = 10)
    fig.suptitle(f"AlphaZero's Winning Percentage (Simulations = {simulations})", fontsize = 14)
    fig.savefig(file_path.format("graphs") + f"-{simulations}_vs")
    fig.clf()




def __eval_preprocess(file_path, key_str):
    network = PolicyValueNet(Board.action_size)
    network.load_weights(file_path.format("parameters") + ".npz")

    # 並列実行を行うための前処理
    ray.init()
    weights = ray.put(network.get_weights())
    del network

    # 画面表示
    print("\033[92m=== Winning Percentage ===\033[0m")
    output = key_str + " || first | second"
    print(output)
    print("=" * (len(output) + 1))

    return weights




if __name__ == "__main__":
    # 学習用コード
    arena = AlphaZero()
    arena.fit(restart = False)

    # 評価用コード
    # eval_alphazero_computer(file_name = "alphazero-9")
    # vs_alphazero_computer(file_name = "alphazero-9", simulations = 800)