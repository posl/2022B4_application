from random import choice

from board import Board
from board_speedup import alpha_beta, count_stand_bits


class PrimitiveMonteCarlo:
    def __init__(self, max_try = 6400, limit_time = 10):
        self.max_try = max_try
        self.limit_time = limit_time

    def __call__(self, board : Board):
        placable = board.list_placable()
        if len(placable) == 1:
            return placable[0]
        return self.primitive_monte_carlo(board)

    # ランダムで手を打つplan
    def random_action(self, board : Board):
        return choice(board.list_placable())

    # nに石を置いた時の勝利回数を返す
    def play_out(self, board : Board, my_turn):
        # 途中復帰用
        state = board.state

        score = 0
        for _ in range(self.max_try):
            board.set_state(state)

            board.game()
            if board.black_num == board.white_num:
                score += 0.5
            elif ((board.black_num > board.white_num) == my_turn):
                score += 1

        return score

    # 次の手を原始モンテカルロ法で決定
    def primitive_monte_carlo(self, board : Board):
        # 原状回復用
        original_plans = board.plans
        original_turn = board.turn

        # 合法手の内、勝率が最もよかったものを置く
        placable = board.list_placable()
        scores = list()
        board.set_plan(self.random_action, self.random_action)
        for n in placable:
            with board.log_runtime():
                board.put_stone(n)

                if not board.can_continue():
                    # ゲームが終了して勝つならその手を打つ
                    if board.black_num == board.white_num:
                        scores.append(self.max_try // 2)
                        continue
                    if (board.black_num > board.white_num) == original_turn:
                        scores.append(self.max_try + 1)
                        break
                    else:
                        scores.append(-1)
                        continue
                else:
                    score = self.play_out(board, original_turn)
                    scores.append(score)
                    print(n, score)

        # 原状回復
        board.set_plan(*original_plans)

        # スコアが同点ならランダムで打つ
        max_index = [i for i, s in enumerate(scores) if s == max(scores)]
        move = placable[choice(max_index)]
        print("put : ", move)
        return move

class ABPrimitiveMonteCarlo(PrimitiveMonteCarlo):
    def __init__(self):
        super().__init__()
    
    def __call__(self, board : Board):
        placable = board.list_placable()
        if len(placable) == 1:
            return placable[0]

        if count_stand_bits(board.stone_black | board.stone_white) > 44:
            move = alpha_beta(*board.players_board, self.limit_time)
            print("zazazazazazaz", move)
            if move in board.list_placable():
                print("checkmate")
                return move

        return self.primitive_monte_carlo(board)


if __name__ == "__main__":
    def player(board : Board):
        while 1:
            try:
                n = int(input("enter n : "))
                if n in board.list_placable():
                    return n
            except:
                print("error")
                continue


    pMC = PrimitiveMonteCarlo()
    abpmc = ABPrimitiveMonteCarlo()
    board = Board()
    board.debug_game(abpmc, pMC)

    print("game set")
    print("black:", board.black_num)
    print("white:", board.white_num)