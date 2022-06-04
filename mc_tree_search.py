from math import sqrt, log
from random import choice, randrange

from board import Board
from speedup import nega_alpha, count_stand_bits


class MonteCarloTreeSearch:
    expanding_threshold = 5

    def __init__(self, max_tries = 65536):
        self.max_tries = max_tries

    # ゲーム毎に呼ぶ必要がある
    def reset(self):
        self.visits_dict = {}
        self.wins_dict = {}
        self.placable_dict = {}
        self.children_state_dict = {}

    def __call__(self, board : Board):
        return self.monte_carlo_tree_search(board)

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

    # 独自のゲーム関数
    def game_simulation(self, board : Board):
        flag = 1
        while flag:
            n = board.get_action()
            board.put_stone(n)
            flag = self.board_can_continue(board)

        return board.black_num - board.white_num

    # ランダムで手を打つplan
    def random_action(self, board):
        return choice(self.__get_placable(board))

    # 優先度（utcによる）の高い子ノードを選ぶ（複数ある場合ランダム）
    # 子供を持たない時はNoneが返る
    def selection(self, state):
        visits_dict = self.visits_dict
        if state in visits_dict:
            visit_list = visits_dict[state]
            sum_visits = sum(visit_list)
            ucts = [w / (v + 1e-15) + sqrt(2 * log(sum_visits) / (v + 1e-15)) for v, w in zip(visit_list, self.wins_dict[state])]
            max_uct = max(ucts)
            max_index = [i for i, u in enumerate(ucts) if u == max_uct]
            return choice(max_index)
        else:
            return None

    # 木を拡張し、拡張した葉ノードへのランダムなインデックスを返す
    def expansion(self, board : Board, state):
        placable = self.placable_dict[state]

        children_state = []
        for move in placable:
            board.set_state(state)
            board.put_stone(move)
            self.board_can_continue(board)
            children_state.append(board.state)

        self.children_state_dict[state] = children_state

        size = len(placable)
        self.visits_dict[state] = [0] * size
        self.wins_dict[state] = [0] * size

        return randrange(size)

    # 1プレイ分、再帰で実装
    def play(self, board, state):
        placable_dict = self.placable_dict

        # ゲーム終了時
        if not placable_dict[state]:
            return count_stand_bits(state[0]) - count_stand_bits(state[1])

        next_index = self.selection(state)

        # 葉ノードに到達したらプレイアウト
        if next_index is None:
            board.set_state(state)
            return self.game_simulation(board)

        else:
            children_state_dict = self.children_state_dict
            visits_dict = self.visits_dict
            wins_dict = self.wins_dict

            next_state = children_state_dict[state][next_index]

            # 選んだ子ノードの訪問回数が初めて閾値に到達した時、木を展開
            if visits_dict[state][next_index] == self.expanding_threshold and placable_dict[ children_state_dict[state][next_index] ]:
                expansion_index = self.expansion(board, next_state)
                expansion_state = children_state_dict[next_state][expansion_index]

                # uct式の性質上ここで一層深くはせずにここで同時に子ノードも評価
                if placable_dict[expansion_state]:
                    board.set_state(expansion_state)
                    game_res = self.game_simulation(board)

                else:
                    game_res = count_stand_bits(expansion_state[0]) - count_stand_bits(expansion_state[1])
                
                wins_dict[next_state][expansion_index] += (state[2] == (game_res > 0)) if game_res else 0
                visits_dict[next_state][expansion_index] += 1
                
            # 一層深く
            else:
                game_res = self.play(board, next_state)

        # 結果を記録    
        wins_dict[state][next_index] += (state[2] == (game_res > 0)) if game_res else 0
        visits_dict[state][next_index] += 1

        return game_res

    # 呼ばれる本体
    def monte_carlo_tree_search(self, board : Board):
        original_plans = board.plans

        board.set_plan(self.random_action, self.random_action)

        state = board.state
        visits_dict = self.visits_dict

        # 探索木で訪れてない場合に呼ぶ
        if state not in visits_dict:
            placable = self.__get_placable(board)
            expansion_index = self.expansion(board, state)
            expansion_state = self.children_state_dict[state][expansion_index]
            board.set_state(expansion_state)
            game_res = self.game_simulation(board)
            self.wins_dict[state][expansion_index] += (state[2] == (game_res > 0)) if game_res else 0
            visits_dict[state][expansion_index] += 1

        # max_tries回シミュレーションする
        for _ in range(self.max_tries):
            self.play(board, state)

        # 状態を回復する
        board.set_plan(*original_plans)
        board.set_state(state)

        # 着手を選ぶ
        visits = visits_dict[state]
        max_visit = max(visits)
        max_index = [i for i, v in enumerate(visits) if v == max_visit]
        return self.placable_dict[state][choice(max_index)]


class NAMonteCarloTreeSearch(MonteCarloTreeSearch):
    def __init__(self, max_tries = 65536, limit_time = 10):
        self.max_tries = max_tries
        self.limit_time = limit_time

    def __call__(self, board : Board):
        placable = board.list_placable()

        if count_stand_bits(board.stone_black | board.stone_white) > 44:
            move = nega_alpha(*board.players_board, self.limit_time)
            if move in placable:
                print("checkmate")
                return move

        return self.monte_carlo_tree_search(board)