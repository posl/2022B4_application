from board import Board



class AlphaBeta:
    def __init__(self, depth = 6):
        self.__max_depth = depth

        # 学習済みパラメータ
        self.corner = 0.6576514321742223
        self.around_corner = 0.4757884128809289
        self.edge = 0.5309212414274704
        self.around_edge = 0.13153629267654865
        self.others = -0.2540768710622537
        self.place = 0.42422219310905573

        # 各マスの評価点を表現する辞書
        block_points = {}
        self.block_points = block_points

        for i in range(Board.action_size):
            x, y = divmod(i, Board.width)
            x, y = min(x, Board.height - x - 1), min(y, Board.width - y - 1)

            if not (x or y):
                # ４隅の評価点
                block_point = self.corner
            elif x + y == 1:
                # ４隅と隣接する８マスの評価点
                block_point = self.around_corner
            elif not (x and y):
                # ボードの辺の評価点
                block_point = self.edge
            elif x == 1 or y == 1:
                # ボードの中辺の評価点
                block_point = self.around_edge
            else:
                # それ以外 (中央の 4x4 マス) の評価値
                block_point = self.others

            block_points[i] = block_point

    # 評価値を返す
    def __eval_board(self, board):
        value = self.__eval_exist_stones(board)
        value += self.place * self.__eval_valid_actions()
        return value

    # 既に置かれた石による評価
    def __eval_exist_stones(self, board: Board):
        block_points = self.block_points
        black_value = 0

        for i in range(board.action_size):
            block_point = block_points[i]
            if (board.stone_black >> i) & 1:
                black_value += block_point
            elif (board.stone_white >> i) & 1:
                black_value -= block_point

        return black_value if board.turn else -black_value

    # 置けるマスの数による評価
    def __eval_valid_actions(self, board: Board, flag = 2):
        # ゲームが終了した場合、置けるマスは０
        if not flag:
            return 0

        p_list = board.list_placable()
        p_length = len(p_list)

        # 手番が交代した場合は、相手の置けるマスの数にマイナスを掛けたものを評価点に使う
        if flag == 1:
            return -p_length

        p_value = 0
        for i in p_list:
            with board.log_runtime():
                board.put_stone(i)
                flag = board.can_continue()
                p_value += self.__eval_valid_actions(board, flag)

        # 置けるマスの数を評価点とし、実際に石を置いた後の、置けるマスの数による評価点の平均をそれに加算して、出力とする
        p_length += p_value / p_length
        return p_length


    def __call__(self, board):
        return self.get_next_move(board)

    # 次の手を示す
    def get_next_move(self, board):
        self.turn = board.turn
        return int(self.__first_max_node(board, -999, 999))

    # 評価が最大値となる場所を求める
    def __first_max_node(self, board: Board, alpha, beta):
        p_list = board.list_placable()
        place_max = p_list[0]

        for i in p_list:
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