import random
from os.path import join, dirname
from time import time
from math import ceil

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from inada_framework import Layer, cuda
import inada_framework.layers as dzl
from inada_framework.functions import relu



# Residual Networks (残余ネットワーク : 前のレイヤが学習しきれなかった残余を次の層に渡すという工程を繰り返す)
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


# bottleneck building block : 通常の building block と同等の計算量で、層をさらに深くできる残余ブロック
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




# エージェントの評価に使う単純な方策
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




# 自己対戦で学習・評価を行うためのクラスの基底クラス
class SelfMatch:
    def __init__(self, board, first_agent, second_agent):
        self.board = board
        self.agents = second_agent, first_agent

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

        # エージェントの評価は、学習中にちょうど 100 回だけ行う
        assert not episodes % 100
        eval_interval = episodes // 100
        win_rates = np.zeros(2, dtype = np.int32)
        self.max_win_rates = np.full((2, 3), -1, dtype = np.int32)

        if restart:
            # 学習を途中再開する場合は、描画用配列と開始インデックスも引き継ぐ
            historys = np.load(f"{is_yet_path}_history.npy")
            run_start, start = historys[:, -1].astype(int)

            # 前回保存した学習途中のデータを読み込むために、エージェントの初期化も行う
            for turn in {1, 0}:
                agent = self.agents[turn]
                agent.reset()
                agent.load_to_restart(f"{is_yet_path}_{turn}")

            # ここで定義した変数は学習中ずっと残ることになるので、不要なものは削除する
            del restart, turn, agent

        else:
            # 勝率の推移を描画するための配列 (最後の列は学習再開に使う変数を記録するための領域)
            historys = np.zeros((2, 101), dtype = np.float32)
            run_start, start = 1, 1

            # エージェントの初期化
            self.agents[1].reset()
            self.agents[0].reset()


        # 累計経過時間の表示
        print("\033[92m=== Total Elapsed Time ===\033[0m")
        start_time = time()

        try:
            for run in range(run_start, runs + 1):
                index = ceil(start / eval_interval) - 1

                with tqdm(range(start, episodes + 1), desc = f"run {run}", leave = False) as pbar:
                    for episode in pbar:
                        self.fit_episode(progress = episode / episodes)

                        # 定期的に現在の方策を評価し、現在の勝率をプログレスバーの後ろに出力して、描画用配列に追加する
                        if not episode % eval_interval:
                            win_rates[...] = self.eval(0), self.eval(1)
                            pbar.set_postfix(dict(rates = "({}%, {}%)".format(*win_rates[::-1])))

                            # 学習の中断によって描画用配列の整合性を損なうことがないように、ここの反映はまとめて行う
                            historys[:, index] += (win_rates - historys[:, index]) / run
                            index += 1

                # パラメータの保存と累計経過時間の表示
                self.save(params_path, win_rates)
                print("run {}: {:.5g} min".format(run, (time() - start_time) / 60))
                start = 1

        finally:
            # 配列に学習を途中再開するために必要な情報も入れる
            historys[:, -1] = run, episode

            np.save(f"{is_yet_path}_history.npy", historys)
            self.save(is_yet_path)

            # 学習の進捗を x 軸、その時の勝率の平均を y 軸とするグラフを描画し、画像保存する
            x = np.arange(100)
            y = historys[:, :-1]
            plt.plot(x, y[1], label = "first")
            plt.plot(x, y[0], label = "second")
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
    def eval(self, turn, enemy_plan = corners_plan):
        board = self.board
        agent = self.agents[turn]
        plans = (agent, enemy_plan) if turn else (enemy_plan, agent)
        board.set_plan(*plans)

        win_count = 0
        for __ in range(100):
            board.reset()
            board.game()

            result = board.black_num - board.white_num
            win_count += (result > 0) if turn else (result < 0)

        return win_count


    # win_rates が渡されるのは、学習済みのパラメータを保存するときのみ
    def save(self, file_path: str, win_rates = None):
        agents = self.agents
        file_path += "_{}"

        if win_rates is None:
            is_yet = True
            agents[1].save(file_path.format(1), is_yet)
            agents[0].save(file_path.format(0), is_yet)

        else:
            max_win_rates = self.max_win_rates

            for turn in {1, 0}:
                win_rate = win_rates[turn]
                agent = agents[turn]

                # 同じ条件で学習したエージェントのうち、評価相手への勝率が高いものを最大３人分保存する
                index = max_win_rates[turn].argmin()
                if max_win_rates[turn, index] < win_rate:
                    max_win_rates[turn, index] = win_rate
                    agent.save(file_path.format(turn) + f"{index}")

                # エージェントの初期化も同時に行う
                agent.reset()