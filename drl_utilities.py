import random
from os.path import join, dirname
from time import time

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from inada_framework import Layer, cuda
import inada_framework.layers as dzl
from inada_framework.functions import relu, flatten
from inada_framework.utilities import make_dir_exist



# =============================================================================
# オセロボードの画像処理用 CNN
# =============================================================================

class SimpleCNN(Layer):
    def __init__(self, use_batch_norm = True):
        super().__init__()

        conv_layer = BatchNormConv2d if use_batch_norm else dzl.Conv2d
        self.conv1 = conv_layer(32, 3, 1, 1)
        self.conv2 = conv_layer(32, 3, 1, 1)
        self.conv3 = conv_layer(64, 3, 1, 0)
        self.conv4 = conv_layer(64, 3, 1, 0)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        return flatten(x)


class BatchNormConv2d(Layer):
    def __init__(self, out_channels, filter_size, stride, padding):
        super().__init__()

        self.conv = dzl.Conv2d(out_channels, filter_size, stride, padding)
        self.bn = dzl.BatchNorm()

    def forward(self, x):
        return self.bn(self.conv(x))




# 前のレイヤが学習しきれなかった残余を次の層に渡すという工程を繰り返すネットワーク (Residual Networks)
class ResNet50(Layer):
    def __init__(self):
        super().__init__()

        self.conv1 = dzl.Conv2d(8, 3, 1, 1, nobias = True)
        self.bn1 = dzl.BatchNorm()

        # 画像データの高さ・幅はそのままで、チャネル数を倍々にしていく
        self.res2 = BuildingBlock(3, 8, 32)
        self.res3 = BuildingBlock(4, 16, 64)
        self.res4 = BuildingBlock(6, 32, 128)
        self.res5 = BuildingBlock(3, 64, 256)

    def forward(self, x):
        x = relu(self.bn1(self.conv1(x)))
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        return x


# 通常の building block と同等の計算量で、層をさらに深くできる残余ブロック (bottleneck building block)
class BuildingBlock(Layer):
    def __init__(self, n_layers, mid_channels, out_channels):
        super().__init__()

        a = BottleneckA(mid_channels, out_channels)
        self.a = a
        self.__forward = [a]

        for i in range(1, n_layers):
            b = BottleneckB(mid_channels, out_channels)
            setattr(self, f"b{i}", b)
            self.__forward.append(b)

    def forward(self, x):
        for layer in self.__forward:
            x = layer(x)
        return x




# building block の最初に組み込む残余ブロック
class BottleneckA(Layer):
    def __init__(self, mid_channels, out_channels):
        super().__init__()

        # 入力チャネル数と出力チャネル数が異なる
        self.res = BottleneckC(mid_channels, out_channels)
        self.conv = dzl.Conv2d1x1(out_channels)
        self.bn = dzl.BatchNorm()

    # Skip Connection により勾配消失問題を解決
    def forward(self, x):
        return relu(self.res(x) + self.bn(self.conv(x)))


# building block のループ部分に組み込む残余ブロック
class BottleneckB(Layer):
    def __init__(self, mid_channels, out_channels):
        super().__init__()

        self.res = BottleneckC(mid_channels, out_channels)

    # Skip Connection により勾配消失問題を解決
    def forward(self, x):
        return relu(self.res(x) + x)


# 入力チャネル数と出力チャネル数が等しい残余ブロック
class BottleneckC(Layer):
    def __init__(self, mid_channels, out_channels):
        super().__init__()

        # 1×1 の畳み込みで次元削減を行った後、3×3 の畳み込みを行い、1×1 の畳み込みで次元を元に戻している
        self.conv1 = dzl.Conv2d1x1(mid_channels)
        self.bn1 = dzl.BatchNorm()
        self.conv2 = dzl.Conv2d(mid_channels, 3, 1, 1, nobias = True)
        self.bn2 = dzl.BatchNorm()
        self.conv3 = dzl.Conv2d1x1(out_channels)
        self.bn3 = dzl.BatchNorm()

    def forward(self, x):
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        return self.bn3(self.conv3(x))




# =============================================================================
# GPU 対応
# =============================================================================

def preprocess_to_gpu(array):
        # 非同期で GPU メモリへのデータ転送を行うために、スワップアウトされないメモリ領域にソース配列をコピーする
        xp = cuda.cp
        p_mem = xp.cuda.alloc_pinned_memory(array.nbytes)
        source = np.frombuffer(p_mem, array.dtype, array.size).reshape(array.shape)
        source[...] = array

        # ストリームを用意し、GPU メモリ上の領域を確保する
        stream = xp.cuda.Stream(non_blocking = True)
        destination = xp.ndarray(source.shape, source.dtype)
        return destination, source, stream




# =============================================================================
# 評価用の単純な方策
# =============================================================================

def simple_plan(board, placable = None):
    if placable is None:
        placable = board.list_placable()

    # 30 % の確率でランダムな合法手を打ち、70 % の確率で取れる石の数が最大の合法手を打つ
    if random.random() < 0.3:
        if len(placable) == 1:
            return placable[0]
        return random.choice(placable)

    current_stone_num = board.get_stone_num()
    flip_nums = np.array([board.get_next_stone_num(n) - current_stone_num for n in placable])

    # np.argmax を使うと選択が前にある要素に偏るため、np.where で取り出した最大手であるインデックスからランダムに選ぶ
    indices = np.where(flip_nums == flip_nums.max())[0]
    action_index = indices[0] if len(indices) == 1 else random.choice(indices)
    return placable[action_index]


def corners_plan(board):
    placable = board.list_placable()

    # 四隅に自身の石を置くことができる場合は必ずそこに置く
    width = board.width
    action_size = board.action_size
    corners = set(placable) & {0, width - 1, action_size - width, action_size - 1}
    if corners:
        return corners.pop()

    # そうでなければ、30 % の確率でランダムな合法手を打ち、70 % の確率で取れる石の数が最大の合法手を打つ
    return simple_plan(board, placable)




# =============================================================================
# 自己対戦による学習の基底クラス
# =============================================================================

class SelfMatch:
    def __init__(self, board, agent):
        self.board = board
        self.agent = agent

    @staticmethod
    def get_path(file_name):
        return join(dirname(__file__), "data", "{}", file_name)


    # 前回の状態を引き継いで、学習を途中再開することができる
    def fit(self, runs, episodes, restart = False, file_name = "params"):
        file_path = SelfMatch.get_path(file_name)
        is_yet_path = file_path.format("is_yet")
        params_path = file_path.format("parameters")
        graphs_path = file_path.format("graphs")
        del file_path

        # ディレクトリが存在しなければ作る
        make_dir_exist(is_yet_path)
        make_dir_exist(graphs_path)

        # エージェントの初期化
        self.agent.reset()

        if restart:
            # 学習を途中再開する場合は、描画用配列と開始番号も引き継ぐ
            history = np.load(f"{is_yet_path}_history.npy")
            run, restart = history[:, -1].astype(int)

            # チェックポイント時点の次のエピソードから学習を再開する
            restart += 1

            # 前回の run が終わった直後か否かで学習を途中再開するかどうかが決まる
            if restart <= episodes:
                self.agent.load_to_restart(is_yet_path)

        else:
            # 勝率の推移を描画するための配列 (最後の列は学習再開に使う変数を記録するための領域)
            history = np.zeros((2, 101), dtype = np.int32)
            run, restart = 1, 1


        # GPU を使用するかどうかの表示
        answer = "Yes, it do." if self.agent.use_gpu else "No, it don't."
        print(f"Q: Will this script use GPU?\nA: {answer}\n")
        del answer

        # 評価結果の表示
        print("\033[92m=== Winning Percentage ===\033[0m")
        print("run || first | second")
        print("======================")

        # 変数定義
        assert not episodes % 100
        eval_interval = episodes // 100
        start_time = time()

        try:
            for run in range(run, runs + 1):
                with tqdm(range(restart, episodes + 1), desc = f"run {run}", leave = False) as pbar:
                    for episode in pbar:
                        self.fit_episode(progress = episode / episodes)

                        eval_q, eval_r = divmod(episode, eval_interval)
                        if not eval_r:
                            pbar.set_description("now evaluating")
                            pbar.set_postfix(dict(caution = "\"Don't suspend right now, please.\""))

                            # エージェントの評価 (合計 100 回)
                            win_rates = self.eval()
                            history[:, eval_q - 1] += win_rates

                            save_q, save_r = divmod(eval_q, 10)
                            if not save_r:
                                pbar.set_description(f"now saving")

                                # パラメータの最終保存 (合計 1 回)  or  学習再開に必要な情報の保存 (合計 9 回)
                                if save_q == 10:
                                    self.save(params_path, run - 1)
                                else:
                                    self.save(is_yet_path)

                                # メタ情報の保存 (合計 10 回)
                                history[:, -1] = run, episode
                                np.save(f"{is_yet_path}_history.npy", history)

                            pbar.set_description(f"run {run}")
                            pbar.set_postfix(dict(rates = "({}%, {}%)".format(*win_rates)))

                try:
                    win_rates
                except NameError:
                    pass
                else:
                    print("{:>3} || {:>3} % | {:>3} %".format(run, *win_rates), end = "   ")
                    print("({:.5g} min elapsed)".format((time() - start_time) / 60.))

                restart = 1


        finally:
            # 学習の進捗を x 軸、その時の勝率の平均を y 軸とするグラフを描画し、画像保存する
            x = np.arange(1, 101)
            y = history[:, :-1].astype(np.float64)

            if run > 1:
                try:
                    index = eval_q
                except NameError:
                    index = 100

                y[:, :index] /= run
                y[:, index:] /= run - 1

            plt.plot(x, y[0], label = "first")
            plt.plot(x, y[1], label = "second")
            plt.legend()
            plt.ylim(-5, 105)
            plt.xlabel("Progress Rate")
            plt.ylabel("Mean Winning Percentage")
            plt.savefig(graphs_path)
            plt.clf()


    # このメソッドは、このクラスを継承した子クラスが実装する
    def fit_episode(self, progress):
        raise NotImplementedError()


    # エージェントを指定した敵と 100 回戦わせた時の勝利数を取得する
    def eval(self, enemy = corners_plan):
        board = self.board
        agent = self.agent
        win_rates = []

        for turn in (1, 0):
            plans = (agent, enemy) if turn else (enemy, agent)
            board.set_plan(*plans)

            win_count = 0
            for __ in range(100):
                board.reset()
                board.game()

                result = board.black_num - board.white_num
                win_count += (result > 0) if turn else (result < 0)

            win_rates.append(win_count)
        return win_rates


    def save(self, file_path, index = None):
        if index is None:
            self.agent.save(file_path, is_yet = True)
        else:
            # パラメータの最終保存を行う場合は、その後エージェントの初期化も行う
            agent = self.agent
            agent.save(file_path + f"-{index}")
            agent.reset()