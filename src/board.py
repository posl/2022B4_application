from functools import cache
from contextlib import contextmanager

import numpy as np

import pyx.speedup as sp
from pyx.speedup import get_board_img, get_stand_bits



# 下のジェネレータの引数となる (step, num) を８方向分生成するジェネレータ
class StepNumGenerator:
    def __init__(self, startpoint):
        up, left = Board.n2t(startpoint)
        right = Board.width - 1 - left
        down = Board.height - 1 - up

        self.nums = [up, right, down, left]

    # 引数には正負の符号を指定する
    def __gen(self, sign):
        nums = self.nums

        # 水平方向探索用の (step, num)
        yield sign, nums[sign]

        # 垂直方向探索用の (step, num)
        step_base = sign * Board.width
        num_base = nums[sign + 1]
        yield step_base, num_base

        # 左斜め方向探索用の (step, num)
        yield step_base - 1, min(num_base, nums[-1])

        # 右斜め方向探索用の (step, num)
        yield step_base + 1, min(num_base, nums[1])

    @property
    def generator(self):
        yield from self.__gen(1)
        yield from self.__gen(-1)


# 引数の組み合わせを極力減らした上でキャッシュを利用することによる高速化 (引数は、304 通り -> 40 kbyte 消費される)
@cache
def element_num_range(startpoint, step, num):
    startpoint += step
    start = startpoint + step
    stop = startpoint + step * num

    # 静的型付き言語で書かれているため処理が速い、 range を使うことによる高速化
    return startpoint, range(start, stop, step)


# オセロ盤の特定のマスから全方位探索を行うためのイテラブル (再利用は想定していないので、扱い的にはジェネレータと等価)
class OmniDirectionalSearcher:
    def __init__(self, startpoint):
        self.step_num = StepNumGenerator(startpoint).generator
        self.startpoint = startpoint

    # このメソッドで自身をジェネレータとして返す
    def __iter__(self):
        return self

    def __next__(self):
        # この next() の呼び出しで生じる StopIteration 例外をこのジェネレータが生じさせた例外として使う
        step, num = next(self.step_num)

        if num > 1:
            return element_num_range(self.startpoint, step, num)
        return next(self)




class Board:
    height = 8
    width = 8
    action_size = height * width

    # インスタンスを生成する前にクラス属性をチェックするようにする
    def __new__(cls):
        assert not cls.height & 1
        assert not cls.width & 1

        return super().__new__(cls)

    def __init__(self):
        # list_placable : 30 ~ 40 倍、reverse : 3 倍  (大体の平均)
        if self.height == self.width == 8:
            self.__count_bits = sp.count_stand_bits
            self.__reverse = sp.get_reverse_board
            self.__list_placable = sp.get_legal_list
        else:
            self.__count_bits = self.__count_bits_python
            self.__reverse = self.__reverse_python
            self.__list_placable = self.__list_placable_python

        # オセロ盤の状態を表現する整数、ターンを表す整数 (先攻(黒) : 1, 後攻(白) : 0)
        self.stone_black = 0
        self.stone_white = 0
        self.turn = 1

        # オセロ盤の状態のログを取って、前の状態に戻ることを可能にするためのスタック
        self.log_stack = []

        # プレイヤーの方策を設定するための属性
        self.player1_plan = None
        self.player2_plan = None

        # 画面表示用の属性
        self.click_attr = None
        self.reversed = 0
        self.log_list = []
        self.brate_list = []


    @property
    def state(self):
        return self.stone_black, self.stone_white, self.turn

    def set_state(self, state):
        self.stone_black, self.stone_white, self.turn = state


    # ボードの状態を一時保存するランタイムコンテキストを生成するマネージャ
    @contextmanager
    def log_runtime(self):
        self.add_state()
        yield
        self.undo_state()

    def add_state(self):
        self.log_stack.append(self.state)

    def undo_state(self):
        self.set_state(self.log_stack.pop())


    def add_playlog(self, brate):
        self.log_list.append(self.state)
        self.brate_list.append(brate)

    def clear_playlog(self):
        self.log_list.clear()
        self.brate_list.clear()


    # オセロ盤の状態を画像データとして取得する (形状は (2, height, width))
    def get_img(self, xp = np):
        img = get_board_img(*self.players_board, self.height, self.width)
        if xp != np:
            return xp.asarray(img)
        return img


    # オセロ盤を初期状態にセットする
    def reset(self):
        height, width = self.height, self.width
        bottom_left = self.t2n(((height >> 1), ((width >> 1) - 1)))
        upper_left = bottom_left - width
        self.stone_black = (0b10 << upper_left) + (0b01 << bottom_left)
        self.stone_white = (0b01 << upper_left) + (0b10 << bottom_left)
        self.turn = 1


    # オセロ盤の各位置を表す、行列のインデックス (tuple) を通し番号に変換する
    @staticmethod
    def t2n(t):
        return Board.width * t[0] + t[1]

    # オセロ盤の各位置を表す、通し番号を行列のインデックス (tuple) に変換する
    @staticmethod
    def n2t(n):
        return divmod(n, Board.width)


    @property
    def black_num(self):
        return self.__count_bits(self.stone_black)

    @property
    def white_num(self):
        return self.__count_bits(self.stone_white)

    @staticmethod
    def __count_bits_python(x):
        return bin(x).count("1")


    @property
    def black_positions(self):
        return get_stand_bits(self.stone_black)

    @property
    def white_positions(self):
        return get_stand_bits(self.stone_white)

    @property
    def reverse_positions(self):
        return get_stand_bits(self.reversed)


    # ゲーム終了後、勝敗に応じて報酬を与えるための属性 (最後の手番の人から見て、勝ち : 1, 負け : -1, 分け : 0)
    @property
    def reward(self):
        diff = self.black_num - self.white_num
        if diff:
            if (diff > 0) ^ self.turn:
                return -1.
            return 1.
        return 0

    def get_stone_num(self):
        if self.turn:
            return self.black_num
        return self.white_num

    # 通し番号 n に自身の石を打ったとき、自身の石の数が幾つになるかを取得する
    def get_next_stone_num(self, n):
        with self.log_runtime():
            self.put_stone(n)
            return self.get_stone_num()


    @property
    def plans(self):
        return self.player1_plan, self.player2_plan

    def set_plan(self, player1_plan, player2_plan):
        self.player1_plan = player1_plan
        self.player2_plan = player2_plan

    # 手番のプレイヤーが置くマスの通し番号を取得する
    def get_action(self):
        if self.turn:
            return self.player1_plan(self)
        return self.player2_plan(self)


    # 手番のプレイヤーとその相手のプレイヤーを判別し、それらのボード表現をタプルで取得する
    @property
    def players_board(self):
        if self.turn:
            return self.stone_black, self.stone_white
        return self.stone_white, self.stone_black

    # 各プレイヤーのボード表現を更新する
    def set_players_board(self, move_player_mask, opposition_player_mask):
        if self.turn:
            self.stone_black ^= move_player_mask
            self.stone_white ^= opposition_player_mask
        else:
            self.stone_white ^= move_player_mask
            self.stone_black ^= opposition_player_mask


    # 通し番号 n に自身の石を置き、挟まれた石を返す
    def put_stone(self, n):
        move_player, opposition_player = self.players_board
        mask = self.__reverse(n, move_player, opposition_player)
        self.set_players_board(mask | (1 << n), mask)
        return mask

    # n に置いた時に返るマスを返す
    @staticmethod
    def __reverse_python(startpoint, move_player, opposition_player):
        mask = 0
        for n, n_gen in OmniDirectionalSearcher(startpoint):
            if (opposition_player >> n) & 1:
                tmp_mask = 1 << n

                for n in n_gen:
                    if (opposition_player >> n) & 1:
                        tmp_mask |= 1 << n
                        continue
                    elif (move_player >> n) & 1:
                        mask |= tmp_mask
                    break
        return mask


    # プレイヤーが石を置ける箇所の通し番号をリストで取得する
    def list_placable(self):
        return self.__list_placable(*self.players_board)

    def __list_placable_python(self, move_player, opposition_player):
        stone_exist = move_player | opposition_player
        is_placable = self.is_placable
        p_list = []

        for n in range(self.action_size):
            # 合法手判定は石が存在しない箇所だけでよい
            if not ((stone_exist >> n) & 1) and is_placable(n, move_player, opposition_player):
                p_list.append(n)

        return p_list

    # 空きマスに自身の石を置けるかどうかの真偽値を取得する
    @staticmethod
    def is_placable(startpoint, move_player, opposition_player):
        for n, n_gen in OmniDirectionalSearcher(startpoint):
            # 相手の石が連続して、その終端の先に自分の石がある場合だけが合法手である
            if (opposition_player >> n) & 1:
                for n in n_gen:
                    if (opposition_player >> n) & 1:
                        continue
                    elif (move_player >> n) & 1:
                        return True
                    break
        return False


    # 終了 : 0, 手番を交代 : 1, 手番そのままで続行 : 2
    def can_continue(self, pass_flag = False):
        self.turn ^= 1
        if self.list_placable():
            return 1 + pass_flag
        elif pass_flag:
            return 0
        else:
            return self.can_continue(True)

    # フラグの代わりに合法手のリストを返すようにした can_continue メソッド (パスがあったかどうかは判定できない)
    def can_continue_placable(self, pass_flag = False):
        self.turn ^= 1
        placable = self.list_placable()
        if placable:
            return placable
        elif pass_flag:
            return []
        else:
            return self.can_continue_placable(True)


    # ゲーム本体
    def game(self, render_func = None):
        render_flag = render_func is not None

        flag = 1
        while flag:
            n = self.get_action()
            mask = self.put_stone(n)
            flag = self.can_continue()

            if render_flag:
                render_func(mask, flag, n)