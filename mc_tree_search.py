from math import sqrt, log
from random import choice

from board import Board
from board_speedup import nega_alpha, count_stand_bits



class Node:
    def __init__(self, board : Board, move, parent):
        self.parent = parent
        self.children = list()
        self.state = board.state
        self.turn = board.turn
        self.move = move
        self.wins = 0   # 記録されているターンプレイヤーの勝ち数でないことに注意
        self.visits = 0
        self.untried_move = board.list_placable()
        

    def uct(self, total_visits):
        return self.wins / self.visits + sqrt(log(total_visits) / self.visits)


class MonteCarloTreeSearch:
    def __init__(self, max_tries = 65536):
        self.max_tries = max_tries

    def __call__(self, board : Board):
        placable = board.list_placable()
        if len(placable) == 1:
            return placable[0]

        return self.monte_carlo_tree_search(board)

    # ランダムで手を打つplan
    def random_action(self, board : Board):
        return choice(board.list_placable())

    # 優先度（utcによる）の高い子孫を選ぶ（複数ある場合ランダム）
    def select_child(self, node : Node):
        ucts = [child.uct(self.root.visits) for child in node.children]
        max_index = [i for i, u in enumerate(ucts) if u == max(ucts)]
        return node.children[choice(max_index)]

    # 未試行の手をランダムに選ぶ
    def expand_child(self, board : Board, node : Node):
        move = choice(node.untried_move)
        node.untried_move.remove(move)

        board.put_stone(move)
        continuable_flag = board.can_continue()

        child = Node(board, move, node)
        node.children.append(child)

        return child, continuable_flag

    # 各ノードに勝敗と訪問を記録
    def back_propagate(self, board : Board, node : Node):
        # 黒における勝敗
        if board.black_num == board.white_num:
            judge = 0
        elif board.black_num > board.white_num:
            judge = 0.5
        else:
            judge = -0.5

        while node:
            node.visits += 1
            # 記録されているターンと勝ちは反転していることに注意
            node.wins += judge * (-1 if node.turn else 1) + 0.5

            node = node.parent

    # モンテカルロ木探索
    def monte_carlo_tree_search(self, board : Board):
        # 原状回復用
        original_state = board.state
        original_plans = board.plans

        # シミュレーション用の行動をセット
        board.set_plan(self.random_action, self.random_action)

        # 木を生成
        self.root = Node(board, None, None)  

        # 試行回数分繰り返す
        for _ in range(self.max_tries):
            node = self.root

            # 行われていないシミュレーションが存在するノードに移動する
            while not node.untried_move and node.children:
                node = self.select_child(node)

            # 盤面をセット
            board.set_state(node.state)
            
            # シミュレーションする手を決定し、木を拡張
            if node.untried_move:
                node, continuable_flag = self.expand_child(board, node)

            # シミュレーション
            if continuable_flag:
                continuable_flag = 0
                board.game()
            
            # プレイアウトの結果を伝播
            self.back_propagate(board, node)
        
        # 原状回復
        board.set_state(original_state)
        board.set_plan(*original_plans)


        # print([i.move for i in self.root.children])
        wins = [child.wins for child in self.root.children]
        # print(wins)
        visits = [child.visits for child in self.root.children]
        # print(visits)
        max_index = [i for i, v in enumerate(visits) if v == max(visits)]
        move = self.root.children[choice(max_index)].move
        # print("put", move)
        return move


class NAMonteCarloTreeSearch(MonteCarloTreeSearch):
    def __init__(self, max_tries = 65536, limit_time = 10):
        self.max_tries = max_tries
        self.limit_time = limit_time
    
    def __call__(self, board : Board):
        placable = board.list_placable()
        if len(placable) == 1:
            return placable[0]

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
                if board.is_placable(n):
                    return n
            except:
                print("error")
                continue

    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()

    mcts = MonteCarloTreeSearch()
    abmcts = ABMonteCarloTreeSearch()
    board = Board()
    board.debug_game(abmcts, mcts)

    print("game set")
    print("black:", board.black_num)
    print("white:", board.white_num)

    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('tottime')
    stats.print_stats()
