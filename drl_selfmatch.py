import random
from os.path import join, dirname
from time import time
from math import ceil

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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

    # np.argmax を使うと選択が偏るため、np.where で取り出したインデックスからランダムに選ぶ
    action_indexs = np.where(flip_nums == flip_nums.max())[0]

    if len(action_indexs) == 1:
        action_index = action_indexs[0]
    else:
        action_index = random.choice(action_indexs)
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
    def fit(self, runs, episodes, file_name, trained_num: int = 0, restart: int = 0):
        assert runs > trained_num
        assert not episodes % 100
        file_name = self.get_path(file_name)

        # 勝率の推移を描画するための配列の用意 (エージェントの評価は、学習中の 100 回 + 最終評価の 1 回)
        if restart:
            load_file = file_name.format("is_yet")
            eval_historys = np.load(load_file + "_history.npy")
        else:
            eval_historys = np.zeros((2, 101), dtype = np.float32)

        # エージェントの初期化
        for turn in {1, 0}:
            agent = self.agents[turn]
            agent.reset()
            if restart:
                agent.load_to_restart(f"{load_file}{turn}")

        # ここで定義した変数は学習中ずっと残ることになるので、不要なものは削除する
        if restart:
            del load_file, turn, agent
        else:
            del turn, agent

        print("\033[92m=== Final Winning Percentage (Total Elapsed Time) ===\033[0m")
        print(" run || first | second")
        win_rates = np.zeros(2, dtype = np.int32)
        start_time = time()

        try:
            for run in range(trained_num + 1, runs + 1):
                self.fit_one_run(run, restart, episodes, eval_historys, win_rates)
                restart = 0

                # パラメータの保存、最終評価、エージェントの初期化 (パラメータの保存を優先する)
                for turn in {1, 0}:
                    self.save(turn, file_name.format("parameters"), index = run - 1)
                    win_rates[turn] = self.eval(turn)
                    self.agents[turn].reset()

                # 評価結果と累計経過時間の表示
                print("{:>4} || {:>3} % | {:>3} %".format(run, win_rates[1], win_rates[0]), end = "  ")
                print("({:.5g} min)".format((time() - start_time) / 60))

                # 評価結果の描画用配列への反映
                eval_historys[:, 100] += (win_rates - eval_historys[:, 100]) / run

        except KeyboardInterrupt:
            is_yet = True
        else:
            is_yet = False
        finally:
            self.plot(eval_historys, file_name, is_yet)
            print()

    def fit_one_run(self, run, restart, episodes, eval_historys, win_rates):
        with tqdm(range(restart, episodes), desc = f"run {run}", leave = False) as pbar:
            eval_interval = episodes // 100
            index = ceil(restart / eval_interval)

            for episode in pbar:
                self.fit_one_episode(progress = (episode + 1) / episodes)

                # 定期的に現在の方策を評価し、現在の勝率をプログレスバーの後ろに出力して、描画用配列に追加する
                if not episode % eval_interval:
                    win_rates[...] = self.eval(0), self.eval(1)
                    pbar.set_postfix(dict(rates = "({}%, {}%)".format(win_rates[1], win_rates[0])))

                    # 学習の中断によって描画用配列の整合性を損なうことがなるべくないように、この反映はまとめて行う
                    eval_historys[:, index] += (win_rates - eval_historys[:, index]) / run
                    index += 1

    # このメソッドを継承した子クラスが実装する
    def fit_one_episode(self, progress):
        raise NotImplementedError()


    # エージェントを指定した敵と定数回戦わせた時の勝利数を取得する
    def eval(self, turn, enemy_plan = corners_plan, verbose = False):
        board = self.board
        agent = self.agents[turn]
        plans = (agent, enemy_plan) if turn else (enemy_plan, agent)
        board.set_plan(*plans)

        # 外部からこのメソッドを呼び出すときに冗長要素を加えることができる
        if verbose:
            n_gen = tqdm(range(1000), desc = f"turn {turn}", leave = False)
        else:
            n_gen = range(100)

        win_count = 0
        for __ in n_gen:
            board.reset()
            board.game()

            result = board.black_num - board.white_num
            is_win = (result > 0) if turn else (result < 0)
            if is_win:
                win_count += 1

        return win_count

    # このメソッドを継承した子クラスが実装する
    def save(self, turn, file_name, index):
        raise NotImplementedError()

    # キーボード割り込みによって途中終了した場合は、パラメータの保存も同時に行う
    def plot(self, eval_historys, file_name, is_yet = False):
        if is_yet:
            save_file = file_name.format("is_yet")
            np.save(save_file + "_history.npy", eval_historys)

        for turn in (1, 0):
            if is_yet:
                self.agents[turn].save(f"{save_file}{turn}", is_yet)

            eval_history = eval_historys[turn] / 100.
            plt.plot(np.arange(len(eval_history)), eval_history, label = "first" if turn else "second")

        plt.xlabel("Thousands of Episodes")
        plt.ylabel("Mean Winning Percentage")
        plt.legend()
        plt.ylim(-0.1, 1.1)
        plt.savefig(file_name.format("graphs"))