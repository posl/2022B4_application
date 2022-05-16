import random
from os.path import join, dirname
from time import time
from math import ceil

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from board import Board



# エージェントの評価に使う単純な方策
def simple_plan(board, placable = None):
    if placable is None:
        placable = board.list_placable()

    # 30 % の確率でランダムな合法手を打ち、70 % の確率で取れる石の数が最大の合法手を打つ
    if np.random.rand() < 0.3:
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
        file_name = self.get_path(file_name)

        # エージェントの評価は、学習中にちょうど 100 回だけ行う
        assert not episodes % 100
        eval_interval = episodes // 100
        win_rates = np.zeros(2, dtype = np.int32)

        if restart:
            # 学習を途中再開する場合は、描画用配列と開始インデックスも引き継ぐ
            load_file = file_name.format("is_yet")
            historys = np.load(f"{load_file}history.npy")
            run_start, start = historys[:, -1].astype(int)

            # 前回保存した学習途中のデータを読み込むために、エージェントの初期化も行う
            for turn in {1, 0}:
                agent = self.agents[turn]
                agent.reset()
                agent.load_to_restart(f"{load_file}{turn}")

            # ここで定義した変数は学習中ずっと残ることになるので、不要なものは削除する
            del restart, load_file, turn, agent

        else:
            # 勝率の推移を描画するための配列 (最後の列は学習再開に使う変数を記録するための領域)
            historys = np.zeros((2, 101), dtype = np.float32)
            run_start, start = 1, 1

            # エージェントの初期化
            self.agents[1].reset()
            self.agents[0].reset()


        # 累計経過時間の表示
        print("\n\033[92m=== Total Elapsed Time ===\033[0m")
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
                self.save(file_name.format("parameters"), run - 1)
                print("{:.5g} min".format((time() - start_time) / 60))
                start = 1

        except KeyboardInterrupt:
            # 配列に学習を途中再開するために必要な情報も入れる
            historys[:, -1] = run, episode

            save_file = file_name.format("is_yet")
            np.save(f"{save_file}history.npy", historys)
            self.save(save_file)

            raise KeyboardInterrupt()

        finally:
            # 学習の進捗を x 軸、その時の勝率の平均を y 軸とするグラフを描画し、画像保存する
            x = np.arange(100)
            y = historys[:, :-1]
            plt.plot(x, y[1], label = "first")
            plt.plot(x, y[0], label = "second")
            plt.ylim(-5, 105)

            plt.xlabel("Progress Rate")
            plt.ylabel("Mean Winning Percentage")
            plt.legend()
            plt.savefig(file_name.format("graphs"))


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
            is_win = (result > 0) if turn else (result < 0)
            if is_win:
                win_count += 1

        return win_count


    def save(self, file_path: str, index = None):
        file_path += "{}"

        if index is None:
            is_yet = True
        else:
            is_yet = False
            file_path += f"{index}"

        for turn in {1, 0}:
            agent = self.agents[turn]
            agent.save(file_path.format(turn), is_yet)
            agent.reset()




# コンピュータの性能を割引率の値ごとに評価し、グラフ形式で保存する関数
def eval_computer(computer_class, to_gpu, gammas, file_name):
    # 環境
    board = Board()

    # コンピュータ
    computer_args = board.action_size, to_gpu
    first_computer = computer_class(*computer_args)
    second_computer = computer_class(*computer_args)

    # その他の設定
    self_match = SelfMatch(board, first_computer, second_computer)
    length = len(gammas)
    rates = np.empty((2, length))

    # 先攻か後攻か・難易度ごとに、コンピュータの評価を合計 20 回行い、その平均をグラフに描画する勝率とする
    for turn in (1, 0):
        for i, gamma in enumerate(gammas):
            current_target = f"turn {turn}, gamma {gamma}"

            win_rate = 0
            for __ in tqdm(range(20), desc = current_target, leave = False):
                first_computer.reset(file_name, gamma, 1)
                second_computer.reset(file_name, gamma, 0)
                win_rate += self_match.eval(turn)

            win_rate /= 20
            rates[turn, i] = win_rate
            print(f"{current_target}: {win_rate:.5g} %")

    # グラフの目盛り位置を設定するための変数
    width = 2.0 / length
    left = np.arange(length)
    right = left + width

    # 左が先攻、右が後攻の勝率となるような棒グラフを画像保存する
    plt.bar(left, rates[1], width = width, align = "edge", label = "first")
    plt.bar(right, rates[0], width = width, align = "edge", label = "second")
    plt.xticks(ticks = right, labels = gammas)
    plt.legend()

    plt.ylim(-5, 105)
    plt.xlabel("Gamma")
    plt.ylabel("Winning Percentages")
    plt.title(f"{computer_class.__name__}")
    plt.savefig(self_match.get_path(file_name + "_bar").format("graphs"))