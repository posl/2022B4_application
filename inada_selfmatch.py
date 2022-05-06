import numpy as np
from tqdm import tqdm
from collections import deque
from inada_framework import Variable


# エージェントの評価に使う単純な方策
def simple_plan(board):
    placable = board.list_placable()

    # 四隅に自身の石を置くことができる場合は必ずそこに置く
    width = board.width
    action_size = board.action_size
    corners = set(placable) & {0, width - 1, action_size - width, action_size - 1}
    if corners:
        return corners.pop()

    # そうでなければ、30 % の確率でランダムな合法手を打ち、70 % の確率で取れる石の数が最大の合法手を打つ
    rng = np.random.default_rng()
    if np.random.rand() < 0.3:
        if len(placable) == 1:
            return placable[0]
        return rng.choice(placable)

    current_stone_num = board.get_stone_num()
    flip_nums = np.array([board.get_next_stone_num(n) - current_stone_num for n in placable])

    # np.argmax を使うと選択が偏るため、np.where で取り出したインデックスからランダムに選ぶ
    action_indexs = np.where(flip_nums == max(flip_nums))[0]

    if len(action_indexs) == 1:
        action_index = action_indexs[0]
    else:
        action_index = rng.choice(action_indexs)
    return placable[action_index]


# 自己対戦の基底クラス
class SelfMatch:
    def __init__(self, board, first_agent, second_agent):
        self.board = board
        self.agents = [second_agent, first_agent]

    def fit(self, runs, episodes, file_name):
        print("\033[92m=== Final Winning Percentage ===\033[0m")
        print(" run || first | second")

        for run in range(1, runs + 1):
            for turn in (1, 0):
                self.agents[turn].reset()

            with tqdm(range(0, episodes + 1), desc = f"run {run}", leave = False) as pbar:
                for episode in pbar:
                    self.fit_one_episode(progress = episode / episodes)

                    # 定期的に、プログレスバーの後ろに現在の勝率を出力する
                    if not episode % 100:
                        win_rate = (self.eval(0) + self.eval(1)) / 2
                        pbar.set_postfix(dict(rate = f"{win_rate}%"))

            # 最終結果の表示とパラメータの保存
            print(f"{run:>4}", end = " || ")
            for turn in (1, 0):
                win_rate = self.eval(turn)
                print(f"{win_rate:>3} %", end = " | " if turn else None)
                self.save(turn, win_rate, file_name)

    # このメソッドは継承した子クラスが実装する
    def fit_one_episode(self, progress):
        raise NotImplementedError()

    def eval(self, turn):
        board = self.board
        agent_plan = self.agents[turn]
        plans = (agent_plan, simple_plan) if turn else (simple_plan, agent_plan)
        board.set_plan(*plans)

        win_count = 0
        for _ in range(100):
            board.reset()
            board.game()
            reward = -board.reward if turn ^ board.turn else board.reward
            if reward > 0:
                win_count += 1

        return win_count

    # このメソッドは継承した子クラスが実装する
    def save(self, turn, win_rate, file_name):
        raise NotImplementedError()




class DQN(SelfMatch):
    def fit_one_episode(self, progress):
        board = self.board
        board.reset()
        transition_infos = deque(), deque()
        flag = 1

        while flag:
            agent = self.agents[board.turn]
            action, state = agent.get_action(board)
            board.put_stone(action)

            # 遷移情報を一時バッファに格納する
            buffer = transition_infos[board.turn]
            buffer.append((state, action))

            if len(buffer) == 2:
                state, action = buffer.popleft()
                next_state = buffer[0][0]

                # 報酬はゲーム終了まで出ない
                agent.update((state, action, next_state, 0), progress)

            flag = board.can_continue()

        # 遷移情報のバッファが先攻・後攻とも空になったらエピソード終了 (先攻・後攻とも１回ずつ update が呼ばれる)
        reward = board.reward
        while True:
            buffer = transition_infos[board.turn]
            try:
                state, action = buffer.popleft()
            except IndexError:
                break
            else:
                agent.update((state, action, board.state, reward), progress)
                reward = -reward

    # 先攻か後攻か、何ステップ先の報酬まで見て学習したものかによってファイル名の接尾語を変える
    def save(self, turn, win_rate, file_name):
        try:
            name = f"max_win_rate{turn}"
            max_win_rate = getattr(self, name)
        except AttributeError:
            max_win_rate = 0

        # 評価用方策に対しての勝率の最大値を更新したら、パラメータを上書き保存する
        if max_win_rate < win_rate:
            setattr(self, name, win_rate)
            agent = self.agents[turn]
            agent.save_weights(file_name + f"{turn}_{agent.step_num}")




class REINFORCE(SelfMatch):
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
            name = f"max_win_rate{turn}"
            max_win_rate = getattr(self, name)
        except AttributeError:
            setattr(self, name, [])
            max_win_rate = getattr(self, name)

        length = len(max_win_rate)
        if length < 8:
            index = length
            max_win_rate.append(win_rate)
        else:
            index = np.argmin(max_win_rate)
            if max_win_rate[index] >= win_rate:
                return
            max_win_rate[index] = win_rate

        agent = self.agents[turn]
        agent.save_weights(file_name + f"{turn}_{index}")