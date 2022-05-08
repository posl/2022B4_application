import numpy as np
import random
from math import ceil
from time import time
import matplotlib.pyplot as plt
from inada_framework.layers import parameters_dir
import os
from tqdm import tqdm
from collections import deque
from inada_framework import Variable


# エージェントの評価に使う単純な方策
def simple_plan(board, placable = None):
    if placable is None:
        placable = board.list_placable()

    # 30 % の確率でランダムな合法手を打ち、70 % の確率で取れる石の数が最大の合法手を打つ
    rng = np.random.default_rng()
    if np.random.rand() < 0.3:
        if len(placable) == 1:
            return placable[0]
        return int(rng.choice(placable))

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


# 何人のエージェントを行動選択に使うかによって、難易度を変えることができるオセロ対戦用コンピュータの基底クラス
class MultiAgentComputer:
    network_class = None

    def __init__(self, action_size):
        self.each_net = []
        self.action_size = action_size

    def reset(self, file_name, turn, agent_num):
        self.rng = np.random.default_rng()
        file_name += f"{turn}_"

        # エージェント数の上限は８人
        assert 1 <= agent_num, agent_num <= 8

        # 各エージェントの方策を表すインスタンス変数をリセットし、新たに登録する
        each_net = self.each_net
        each_net.clear()
        for i in self.rng.choice(8, agent_num, replace = False):
            network = self.network_class(self.action_size)
            network.load_weights(file_name + f"{i}.npz")
            each_net.append(network)




# 自己対戦で学習・評価を行うためのクラスの基底クラス
class SelfMatch:
    def __init__(self, board, first_agent, second_agent):
        self.board = board
        self.agents = [second_agent, first_agent]

    def fit(self, runs, episodes, file_name):
        print("\033[92m=== Final Winning Percentage (Total Elapsed Time) ===\033[0m")
        print(" run || first | second")

        history_length = ceil(episodes / 1000) + 1
        eval_historys = [np.zeros(history_length), np.zeros(history_length)]
        start = time()

        try:
            for run in range(runs):
                self.agents[1].reset()
                self.agents[0].reset()
                self.fit_one_run(run, episodes, eval_historys)

                # 最終結果と経過時間のの表示、パラメータの保存
                print(f"{run:>4}", end = " || ")

                for turn in (1, 0):
                    win_rate = self.eval(turn)
                    eval_historys[turn][-1] += win_rate
                    self.save(turn, win_rate, file_name)

                    print(f"{win_rate:>3} %", end = " | " if turn else "  ")
                print("({:.5g} min)".format((time() - start) / 60))

        except KeyboardInterrupt:
            if run:
                for turn in (1, 0):
                    eval_history = eval_historys[turn] / 100.
                    history_label = "first" if turn else "second"
                    plt.plot(np.arange(history_length), eval_history, label = history_label)

                plt.xlabel("Thousands of Episodes")
                plt.ylabel("Mean Winning Percentage")
                plt.title(file_name + f" (runs = {run})")
                plt.legend()
                plt.ylim(0., 1.1)
                plt.savefig(os.path.join(parameters_dir, "graphs", file_name))
        finally:
            print()

    def fit_one_run(self, run, episodes, eval_historys):
        with tqdm(range(episodes), desc = f"run {run}", leave = False) as pbar:
            run += 1
            index = 0

            for episode in pbar:
                self.fit_one_episode(progress = (episode + 1) / episodes)

                # 定期的に現在の方策を評価し、現在の勝率を配列に追加して、プログレスバーの後ろに出力する
                if not episode % 1000:
                    win_rates = ()

                    for turn in (1, 0):
                        win_rate = self.eval(turn)
                        eval_history = eval_historys[turn]
                        eval_history[index] += (win_rate - eval_history[index]) / run
                        win_rates += (f"{win_rate}%", )

                    pbar.set_postfix(dict(rates = win_rates))
                    index += 1

    # このメソッドを継承した子クラスが実装する
    def fit_one_episode(self, progress):
        raise NotImplementedError()


    # エージェントを指定した敵と定数回戦わせて、その時の勝利数を返す
    def eval(self, turn, enemy_plan = corners_plan, verbose = False):
        board = self.board
        agent_plan = self.agents[turn]
        plans = (agent_plan, enemy_plan) if turn else (enemy_plan, agent_plan)
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

        agent = self.agents[turn]
        agent.save_weights(file_name + f"{turn}_{index}")




class DQN(SelfMatch):
    def fit_one_episode(self, progress):
        board = self.board
        board.reset()
        transition_infos = deque(), deque()
        flag = 1

        while flag:
            turn = board.turn
            agent = self.agents[turn]
            action, state = agent.get_action(board)
            board.put_stone(action)

            # 遷移情報を一時バッファに格納する
            buffer = transition_infos[turn]
            buffer.append((state, action))

            # 学習開始前にターゲットネットワークを現在の学習対象ネットワークと同期させる
            if len(buffer) == 2:
                state, action = buffer.popleft()
                next_state = buffer[0][0]

                # 報酬はゲーム終了まで出ない
                agent.update((state, action, next_state, 0), progress)

            flag = board.can_continue()

        # 遷移情報のバッファが先攻・後攻とも空になったらエピソード終了
        reward = board.reward
        while True:
            state, action = buffer.popleft()
            agent.update((state, action, board.state, reward), progress)

            if turn == board.turn:
                turn ^= 1
                buffer = transition_infos[turn]
                agent = self.agents[turn]
                reward = -reward
            else:
                break


class REINFORCE(SelfMatch):
    def fit_one_episode(self, progress):
        board = self.board
        board.reset()

        while True:
            agent = self.agents[board.turn]
            action, prob = agent.get_action(board, progress)
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