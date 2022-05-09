import numpy as np

from board import Board


class PrimitiveMonteCarlo:
    def __init__(self, max_try = 200):
        self.max_try = max_try
        self.rng = np.random.default_rng()

    # ランダムで手を打つplan
    def random_action(self, board : Board):
        return int(self.rng.choice(board.list_placable()))

    # nに石を置いた時の勝利回数を返す
    def pmc_method(self, board : Board, my_turn):
        # 途中復帰用
        state = board.state
        turn = board.turn

        score = 0
        for _ in range(self.max_try):
            board.set_state(state)
            board.turn = turn

            board.game()
            if (board.black_num > board.white_num) == my_turn:
                score += 1
        
        return score

    # 次の手を原始モンテカルロ法で決定
    def get_next_move_by_pmc(self, board : Board):
        # 原状回復用
        original_plans = board.plans
        original_turn = board.turn

        # 合法手の内、勝率が最もよかったものを置く
        candidate = board.list_placable()
        scores = list()
        board.set_plan(self.random_action, self.random_action)
        for n in candidate:
            with board.log_runtime(n):
                if not board.list_placable():
                    # ゲームが終了して勝つならその手を打つ
                    if (board.black_num > board.white_num) == original_turn:
                        scores.append(self.max_try)
                        break
                    else:
                        continue
                else:
                    score = self.pmc_method(board, original_turn)
                    scores.append(score)
                    print(n, score)
       
        # 原状回復
        board.set_plan(*original_plans)
        board.turn = original_turn

        # スコアが同点ならランダムで打つ
        max_index = [i for i, s in enumerate(scores) if s == max(scores)]
        print("put", candidate[self.rng.choice(max_index)])
        return candidate[self.rng.choice(max_index)]


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

    board = Board()
    pMC = PrimitiveMonteCarlo(200)
    board.play(pMC.random_action, pMC.get_next_move_by_pmc)

    print("game set")
    print("black:", board.black_num)
    print("white:", board.white_num)