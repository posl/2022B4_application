from __future__ import annotations #dataclassにおいて自信をメンバに置くために使用

from dataclasses import dataclass, field
from math import sqrt, log
from random import choice

from board import Board
from speedup import nega_alpha, count_stand_bits


@dataclass
class Node:
    parent : Node #Python 4.0 では標準になるらしい
    children : list = field(default_factory=list)
    untried_move : list = field(default_factory=list)
    state : tuple = ()
    move : int = -1
    wins : int = 0  # 記録されているターンプレイヤーの勝ち数でないことに注意
    visits : int = 0
    is_inner_node : int = 1


class MonteCarloTreeSearch:
    expanding_threshold = 10

    def __init__(self, max_tries = 65536):
        self.max_tries = max_tries

    def reset(self):
        self.is_first_call = True

    def __call__(self, board : Board):
        move = self.monte_carlo_tree_search(board)
        self.is_first_call = False
        return move

    # ランダムで手を打つplan
    def random_action(self, board : Board):
        return choice(board.list_placable())

    # 優先度（uctによる）の高い子ノードを選ぶ（複数ある場合ランダム）
    def selection(self, node : Node):
        ucts = [child.wins / child.visits + sqrt(2 * log(node.visits) / child.visits) for child in node.children]
        max_uct = max(ucts)
        max_index = [i for i, u in enumerate(ucts) if u == max_uct]
        return node.children[choice(max_index)]

    # 木を拡張する
    def expansion(self, board : Board, node : Node):
        next_move = choice(node.untried_move)
        node.untried_move.remove(next_move)

        board.put_stone(next_move)
        continue_flag = board.can_continue()
        child = Node(parent = node, untried_move = board.list_placable(), state = board.state, move = next_move, is_inner_node = continue_flag)
        node.children.append(child)
        
        return child
    
    # 結果を記録
    def backup(self, node : Node, game_result):
        while node.parent:
            parent_turn = node.parent.state[2]
            node.visits += 1
            if parent_turn == (game_result > 0):
                node.wins += 1
            node = node.parent
        node.visits += 1

    # rootを設定
    def set_root(self, board : Board):
        root = Node(parent = None, state = board.state, untried_move = board.list_placable())
        for _ in range(len(root.untried_move)):
            board.set_state(root.state)
            node = self.expansion(board, root)
            if node.is_inner_node:
                board.internal_game()
            self.backup(node, board.black_num - board.white_num)
        self.root = root

    # max_try分ロールアウト
    def play(self, board : Board):
        for _ in range(self.max_tries):
            node = self.root

            while not node.untried_move and node.children:
                node = self.selection(node)
            
            board.set_state(node.state)

            if node.is_inner_node:
                if node.visits > self.expanding_threshold:
                    node = self.expansion(board, node)
 
                if node.is_inner_node:
                    board.internal_game()

            game_result = board.black_num - board.white_num
            
            self.backup(node, game_result)

    # 呼ばれる本体
    def monte_carlo_tree_search(self, board : Board):
        original_plans = board.plans
        
        board.set_plan(self.random_action, self.random_action)

        if self.is_first_call:
            self.set_root(board)
        else:
            put_log = board.put_log
            print(put_log)
            root = self.root
            print(root.move)
            for m in put_log[put_log.index(root.move) + 1:]:
                print(m)
                for child in root.children:
                    if child.move == m:
                        root = child
                        break
            self.root = root

        # for child in self.root.children:
        #     print(child.move, child.wins, child.visits)
        #     for c in child.children:
        #         print("    ", c.move, c.wins, c.visits)

        self.play(board)

        root = self.root

        # for child in root.children:
        #     print(child.move, child.wins, child.visits)
        #     for c in child.children:
        #         print("    ", c.move, c.wins, c.visits)
        #         for d in c.children:
        #             print("         ", d.move, d.wins, d.visits)

        #wins = [child.wins for child in root.children]
        visits = [child.visits for child in root.children]
        max_visit = max(visits)
        max_index = [i for i, v in enumerate(visits) if v == max_visit]
        choiced_index = choice(max_index)
        move = root.children[choiced_index].move
        self.root = root.children[choiced_index]
        
        board.set_state(root.state)
        board.set_plan(*original_plans)
        
        # print([i.move for i in root.children])
        # print(wins)
        # print(visits)
        # print("put", move)

        return move




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


if __name__ == "__main__":
    def player(board : Board):
        while 1:
            try:
                n = int(input("enter n : "))
                if board.is_placable(n, *board.players_board):
                    return n
            except:
                print("error")
                continue

    import cProfile
    import pstats

    from drl_alphazero import AlphaZeroComputer


    pr = cProfile.Profile()
    pr.enable()

    mcts = MonteCarloTreeSearch()
    mcts.reset()
    alphazero = AlphaZeroComputer(64)
    alphazero.reset("alphazero-6")
    board = Board()
    # board.set_state((0b00000000_00001110_11000111_00101011_11101111_01101110_00111100_00111000, 
    #                 0b00011100_00010000_00111000_11010100_00010000_00010000_00000000_00000000, 
    # 1))

    # board.set_state((0b00000000_00000000_00000000_00011110_00001000_00011000_00001000_00001000, 
    #                  0b00010000_00011101_01111111_01100001_11110111_11100111_00110100_00100100,
    # 1))

    board.debug_game(alphazero, mcts)

    print("game set")
    print("black:", board.black_num)
    print("white:", board.white_num)

    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('tottime')
    stats.print_stats()

