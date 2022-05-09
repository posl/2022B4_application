import numpy as np
import random
from os.path import join
from time import time
from math import ceil
from tqdm import tqdm
import matplotlib.pyplot as plt
from inada_framework.layers import parameters_dir
from collections import deque
from inada_framework import Variable


# エージェントの評価に使う単純な方策
def simple_plan(board, placable = None):
    if placable is None:
        placable = board.list_placable()

    rng = np.random.default_rng()

    # 30 % の確率でランダムな合法手を打ち、70 % の確率で取れる石の数が最大の合法手を打つ
    if np.random.rand() < 0.3:
        length = len(placable)
        if length == 1:
            return placable[0]
        return placable[rng.choice(length)]

    current_stone_num = board.get_stone_num()
    flip_nums = np.array([board.get_next_stone_num(n) - current_stone_num for n in placable])

    # np.argmax を使うと選択が偏るため、np.where で取り出したインデックスからランダムに選ぶ
    action_indexs = np.where(flip_nums == max(flip_nums))[0]

    if len(action_indexs) == 1:
        action_index = action_indexs[0]
    else:
        action_index = rng.choice(action_indexs)
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


def random_plan(board):
    return random.choice(board.list_placable())




# 自己対戦で学習・評価を行うためのクラスの基底クラス
class SelfMatch:
    def __init__(self, board, first_agent, second_agent):
        self.board = board
        self.agents = second_agent, first_agent

    def fit(self, runs, episodes, eval_interval, file_name):
        # 前回の続きから学習をスタートする場合は、パラメータを読み込む
        for turn in (1, 0):
            agent = self.agents[turn]
            try:
                agent.reset()
                agent.load_weights(join("is_yet", file_name + f"{turn}_yet.npz"))
            except FileNotFoundError:
                pass
        del turn, agent

        print("\033[92m=== Final Winning Percentage (Total Elapsed Time) ===\033[0m")
        print(" run || first | second")
        start = time()

        history_length = ceil(episodes / eval_interval) + 1
        eval_historys = np.zeros((2, history_length))
        del history_length

        try:
            for run in range(1, runs + 1):
                self.fit_one_run(run, episodes, eval_interval, eval_historys)

                # 最終評価の表示
                print(f"{run:>4}", end = " || ")

                for turn in (1, 0):
                    win_rate = self.eval(turn)
                    print(f"{win_rate:>3} %", end = " | " if turn else "  ")

                    # 描画用の時系列データに最終評価を反映させる
                    eval_historys[turn, -1] += (win_rate - eval_historys[turn, -1]) / run

                    # パラメータの保存、エージェントの初期化
                    self.save(turn, win_rate, file_name)
                    self.agents[turn].reset()

                # 累計経過時間の表示
                print("({:.5g} min)".format((time() - start) / 60))

        except KeyboardInterrupt:
            is_yet = True
        else:
            is_yet = False
        finally:
            self.plot(eval_historys, file_name, run, is_yet)
            print()

    def fit_one_run(self, run, episodes, eval_interval, eval_historys):
        with tqdm(range(episodes), desc = f"run {run}", leave = False) as pbar:
            index = 0

            for episode in pbar:
                self.fit_one_episode(progress = (episode + 1) / episodes)

                # 定期的に現在の方策を評価し、現在の勝率を配列に追加して、プログレスバーの後ろに出力する
                if not episode % eval_interval:
                    win_rates = []

                    for turn in (1, 0):
                        win_rate = self.eval(turn)
                        eval_historys[turn, index] += (win_rate - eval_historys[turn, index]) / run
                        win_rates.append(f"{win_rate}%")

                    pbar.set_postfix(dict(rates = win_rates))
                    index += 1

    # このメソッドを継承した子クラスが実装する
    def fit_one_episode(self, progress):
        raise NotImplementedError()


    # エージェントを指定した敵と定数回戦わせて、その時の勝利数を返す
    def eval(self, turn, enemy_plan = corners_plan, verbose = False):
        board = self.board

        # 方策の登録
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
    def save(self, turn, win_rate, file_name):
        raise NotImplementedError()

    # キーボード割り込みによって途中終了した場合は、パラメータの保存も同時に行う
    def plot(self, eval_historys, file_name, run, is_yet = False):
        for turn in (1, 0):
            if is_yet:
                self.agents[turn].save_weights(join("is_yet", file_name + f"{turn}_yet"))

            eval_history = eval_historys[turn] / 100.
            history_label = "first" if turn else "second"
            plt.plot(np.arange(len(eval_history)), eval_history, label = history_label)

        plt.xlabel("Thousands of Episodes")
        plt.ylabel("Mean Winning Percentage")
        plt.title(file_name + f" (runs = {run})")
        plt.legend()
        plt.ylim(-0.1, 1.1)
        plt.savefig(join(parameters_dir, "graphs", file_name))




class Rainbow(SelfMatch):
    def fit_one_episode(self, progress):
        board = self.board
        board.reset()
        transition_infos = deque(), deque()
        flag = 1

        while flag:
            turn = board.turn
            agent = self.agents[turn]

            placable = board.list_placable()
            state = board.state
            action = agent.get_action(board, placable)

            board.put_stone(action)
            flag = board.can_continue()

            # 遷移情報を一時バッファに格納する
            buffer = transition_infos[turn]
            buffer.append((placable, state, action))

            # 遷移情報２つセットで１回の update メソッドが呼べる
            if len(buffer) == 2:
                state, action = buffer.popleft()[1:]
                next_placable, next_state = buffer[0][:2]

                # 報酬はゲーム終了まで出ない
                agent.update((state, action, 0, next_state, next_placable), progress)

        reward = board.reward
        next_state = board.state
        next_placable = []

        # 遷移情報のバッファが先攻・後攻とも空になったらエピソード終了
        while True:
            state, action = buffer.popleft()[1:]
            agent.update((state, action, reward, next_state, next_placable), progress)

            if turn == board.turn:
                turn ^= 1
                buffer = transition_infos[turn]
                agent = self.agents[turn]
                reward = -reward
            else:
                break

    def save(self, turn, win_rate, file_name):
        try:
            name = f"max_win_rate{turn}"
            max_win_rate = getattr(self, name)
        except AttributeError:
            pass
        else:
            if max_win_rate > win_rate:
                return

        setattr(self, name, win_rate)
        agent = self.agents[turn]
        agent.save_weights(file_name + f"{turn}_{agent.quantiles_num}")




class Reinforce(SelfMatch):
    def fit_one_episode(self, progress = None):
        board = self.board
        board.reset()

        while True:
            agent = self.agents[board.turn]
            action, prob = agent.get_action(board)
            board.put_stone(action)

            # 報酬はゲーム終了まで出ない
            flag = board.can_continue()
            if flag:
                agent.add((0, prob))

            # エージェントの学習 (update メソッド) はエピソードが終わるたびに行う
            else:
                reward = board.reward
                agent.add((reward, prob))
                agent.update()
                break

        agent = self.agents[board.turn ^ 1]
        agent.add((-reward, Variable(np.array(0))))
        agent.update()

    # 評価用方策に対しての勝率の高い順で８人分のパラメータを保存する (先攻・後攻は別々のファイル)
    def save(self, turn, win_rate, file_name):
        try:
            name = f"max_win_rates{turn}"
            max_win_rates = getattr(self, name)
        except AttributeError:
            max_win_rates = []
            setattr(self, name, max_win_rates)

        length = len(max_win_rates)
        if length < 8:
            index = length
            max_win_rates.append(win_rate)
        else:
            index = np.argmin(max_win_rates)
            if max_win_rates[index] >= win_rate:
                return
            max_win_rates[index] = win_rate

        self.agents[turn].save_weights(file_name + f"{turn}_{index}")