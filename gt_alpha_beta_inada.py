from board import Board



class AlphaBeta:
    def __init__(self, depth = 6):
        self.__max_depth = depth

        # 学習済みのマスの評価点
        corner = 0.6799049534960244
        around_corner = 0.5004023729821491
        edge = 0.5544923898781453
        around_edge = -0.11575315250855303
        others = 0.06616793321648566

        # 置ける場所の数による評価点にかかる係数
        self.place = 0.5390611408271047

        # 各マスの評価点を表現する辞書
        block_points = {}
        self.block_points = block_points

        for i in range(Board.action_size):
            x, y = divmod(i, Board.width)
            x, y = min(x, Board.height - x - 1), min(y, Board.width - y - 1)

            if not (x or y):
                # ４隅の評価点
                block_point = corner
            elif x + y == 1:
                # ４隅と隣接する８マスの評価点
                block_point = around_corner
            elif not (x and y):
                # ボードの辺の評価点
                block_point = edge
            elif x == 1 or y == 1:
                # ボードの中辺の評価点
                block_point = around_edge
            else:
                # それ以外 (中央の 4x4 マス) の評価値
                block_point = others

            block_points[i] = block_point


    # 評価値を返す
    def __eval_board(self, board):
        value = self.__eval_exist_stones(board)
        value += self.place * self.__eval_valid_actions(board)
        return value

    # 既に置かれた石による評価
    def __eval_exist_stones(self, board: Board):
        stone_black, stone_white = board.stone_black, board.stone_white
        block_points = self.block_points
        black_value = 0

        for i in range(board.action_size):
            if (stone_black >> i) & 1:
                black_value += block_points[i]
            elif (stone_white >> i) & 1:
                black_value -= block_points[i]

        return black_value if board.turn else -black_value

    # 置けるマスの数による評価
    def __eval_valid_actions(self, board: Board, flag = 2):
        turn = board.turn
        board.turn = 1
        black_value = len(board.list_placable())
        board.turn = 0
        black_value -= len(board.list_placable())
        board.turn = turn
        return black_value if turn else -black_value


    def __call__(self, board):
        return self.get_next_move(board)

    # 評価が最大値となる場所を求める
    def get_next_move(self, board: Board, alpha = -999, beta = 999):
        self.turn = board.turn
        place_list = board.list_placable()
        place_max = place_list[0]

        for i in place_list:
            with board.log_runtime():
                board.put_stone(i)
                board.can_continue()
                tmp_value = self.__node(board, 1, alpha, beta)
            if tmp_value > alpha:
                alpha = tmp_value
                place_max = i

        return place_max

    # 評価値を求める
    def __node(self, board: Board, depth, alpha, beta, can_continue_flag = 1):
        if (depth == self.__max_depth) or (not can_continue_flag):
            return self.__eval_board(board)

        # 求める評価値が最大か最小か決定する
        is_max = not (self.turn ^ board.turn)
        value = -999 if is_max else 999

        for i in board.list_placable():
            with board.log_runtime():
                board.put_stone(i)
                flag = board.can_continue()
                tmp_value = self.__node(board, depth + 1, alpha, beta, flag)

            if is_max:
                if tmp_value >= beta:
                    return tmp_value
                if tmp_value > value:
                    value = tmp_value
                    if value > alpha:
                        alpha = value
            else:
                if tmp_value <= alpha:
                    return tmp_value
                if tmp_value < value:
                    value = tmp_value
                    if value < beta:
                        beta = value

        return value