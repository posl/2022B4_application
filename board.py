from functools import cache
from contextlib import contextmanager
from math import ceil
import random
import os

import numpy as np
import pygame

from board_speedup import get_legal_board
import display


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
        if self.height == self.width == 8:
            self.__list_placable = self.list_placable_cython
        else:
            self.__list_placable = self.list_placable_python

        # オセロ盤の状態を表現する整数、ターンを表す整数 (先攻(黒) : 1, 後攻(白) : 0)
        self.stone_black = 0
        self.stone_white = 0
        self.turn = 1

        # オセロ盤の状態のログを取って、前の状態に戻ることを可能にするためのスタック
        self.log_state = []
        self.log_plans = []

        # can_continue で計算した、合法手のリストを再利用するための属性
        self.p_list = None

        # 画面表示用の属性
        self.click_attr = None


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
        self.log_state.append(self.state)

    def undo_state(self):
        self.set_state(self.log_state.pop())


    # オセロ盤の状態情報である２つの整数を 8 bit 区切りで ndarray に格納して、それを出力する
    @property
    def state_ndarray(self, xp = np):
        n_gen = range(ceil(Board.action_size / 8))
        stone_black, stone_white = self.state

        state_list = [(stone_black >> (i << 3)) & 0xff for i in n_gen]
        state_list += [(stone_white >> (i << 3)) & 0xff for i in n_gen]

        # 正規化してから出力する
        ndarray = xp.array(state_list, dtype = np.float32)
        return ndarray / 255.


    # オセロ盤の各位置を表す、行列のインデックス (tuple) を通し番号に変換する
    @staticmethod
    def t2n(t):
        return Board.width * t[0] + t[1]

    # オセロ盤の各位置を表す、通し番号を行列のインデックス (tuple) に変換する
    @staticmethod
    def n2t(n):
        return divmod(n, Board.width)

    # オセロ盤を初期状態にセットする
    def reset(self):
        height, width = self.height, self.width
        bottom_left = self.t2n(((height >> 1), ((width >> 1) - 1)))
        upper_left = bottom_left - width
        self.stone_black = (0b10 << upper_left) + (0b01 << bottom_left)
        self.stone_white = (0b01 << upper_left) + (0b10 << bottom_left)
        self.turn = 1


    @property
    def black_num(self):
        return self.stone_black.bit_count()

    @property
    def white_num(self):
        return self.stone_white.bit_count()

    # ゲーム終了後、勝敗に応じて報酬を与えるための属性 (最後の手番の人から見て、勝ち : 1, 負け : -1, 分け : 0)
    @property
    def reward(self):
        diff = self.black_num - self.white_num
        if diff:
            if (diff > 0) ^ self.turn:
                return -1
            return 1
        return 0


    def get_stone_num(self):
        if self.turn:
            return self.black_num
        return self.white_num

    # 通し番号 n に自身の石を打ったとき、自身の石の数が幾つになるかを取得する
    def get_next_stone_num(self, n):
        with self.log_runtime(n):
            self.put_stone(n)
            return self.get_stone_num()


    # ボードを表現する整数を引数として、１が立っている箇所のリストを取得する
    def get_stand_bits(self, x):
        return [n for n in range(self.action_size) if (x >> n) & 1]

    @property
    def black_positions(self):
        return self.get_stand_bits(self.stone_black)

    @property
    def white_positions(self):
        return self.get_stand_bits(self.stone_white)


    # 空きマスに自身の石を置けるかどうかの真偽値を取得する
    def is_placable(self, startpoint, move_player, opposition_player):
        for n, n_gen in OmniDirectionalSearcher(startpoint):
            # 相手の石が連続して、その終端の先に自分の石がある場合だけが合法
            if (opposition_player >> n) & 1:
                for n in n_gen:
                    if (opposition_player >> n) & 1:
                        continue
                    elif (move_player >> n) & 1:
                        return True
                    break
        return False

    # エージェントが石を置ける箇所の番号をリストで取得する
    def list_placable_python(self, players):
        move_player, opposition_player = players
        stone_exist = move_player | opposition_player
        is_placable = self.is_placable
        p_list = []

        for n in range(self.action_size):
            # 合法手判定は石が存在しない箇所だけでよい
            if not ((stone_exist >> n) & 1) and is_placable(n, move_player, opposition_player):
                p_list.append(n)

        return p_list

    # C 言語で実装した高速なコードが利用できる条件を満たす時、それを使う
    def list_placable_cython(self, players):
        return self.get_stand_bits(get_legal_board(*players))


    # 手番のプレイヤーとその相手のプレイヤーを判別し、それらのボード表現をタプルで取得する
    @property
    def players(self):
        if self.turn:
            return self.stone_black, self.stone_white
        return self.stone_white, self.stone_black

    def list_placable(self, save_flag = False):
        p_list = self.p_list

        # can_continue で計算したものが存在する場合は、それを利用する
        if p_list is not None:
            self.p_list = None
            return p_list

        p_list = self.__list_placable(self.players)
        if save_flag:
            self.p_list = p_list
        return p_list


    # 盤面リセット、戦略、先攻後攻(奇数:player1先行)を設定
    def set_plan(self, player1_plan, player2_plan):
        self.player1_plan = player1_plan
        self.player2_plan = player2_plan

    @property
    def plans(self):
        return self.player1_plan, self.player2_plan

    # 置くマスを取得
    def get_action(self):
        if self.turn:
            return self.player1_plan(self)
        return self.player2_plan(self)


    # n に石を置き、返す
    def put_stone(self, n):
        
        self.__set_stone(n)
        self.__reverse(n)

    # n に石を置く
    def __set_stone(self, n):
        self.setbit_stone_exist(n)
        if self.turn:
            self.setbit_stone_black(1 << n)

    # n に置いた時に返るマスを返す
    def __reverse(self, startpoint):
        for n, n_gen in OmniDirectionalSearcher(startpoint):
            if self.getbit_stone_exist(n) and (self.getbit_stone_black(n) ^ self.turn):
                mask = 1 << n

                for n in n_gen:
                    if self.getbit_stone_exist(n):
                        if self.getbit_stone_black(n) ^ self.turn:
                            mask |= 1 << n
                            continue
                        self.setbit_stone_black(mask)
                    break


    # 終了 : 0, 手番を交代 : 1, 手番そのままで続行 : 2
    def can_continue(self, pass_flag = False):
        self.turn ^= 1
        if self.list_placable(True):
            return 1 + pass_flag

        if pass_flag:
            return 0
        else:
            return self.can_continue(True)


    # ゲーム本体
    def game(self, render_flag = False):
        flag = 1
        while flag:
            n = self.get_action()
            self.put_stone(n)
            flag = self.can_continue()

            if render_flag:
                # 引数は pass があったかどうかの真偽値
                self.render(flag)

    # エピソード中の画面表示メソッド
    def render(self, flag):
        self.tkapp.game_page.canvas_update()


    def play(self):
        # ページたち
        self.start_page = display.StartPage()
        self.option_page = display.OptionPage()
        self.game_page = display.GamePage()

        # サウンド
        sound_folder_path = os.path.normpath(os.path.join(os.path.abspath(__file__),  "sound"))
        self.bgm1 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou09.mp3"))
        self.bgm1.play(loops=-1)
        self.se1 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou47.wav"))
        self.se2 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou41.wav"))
        self.se3 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou48.wav"))
        self.se4 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou19.wav"))

        # tkapp の初期化
        while True:
            self.start_page.tkraise()
            self.main_loop()
            if self.click_attr:
                self.__play()
            else:
                break

    def __play(self):
        self.main_loop()
        player1_plan, player2_plan = self.click_attr
        self.set_plan(player1_plan, player2_plan)

        # 最初の盤面表示
        self.reset()
        self.print_state()

        self.game(self.print_state)


    # 一時的な盤面表示
    def print_board(self, x):
        for i in range(8):
            print(format(x & 0b1111_1111, "08b")[::-1])
            x >>= 8
    def print_state(self):
        self.print_board(self.stone_exist)
        print("-" * 20)
        self.print_board(self.stone_black)
        print("black:", self.black_num, "   white:", self.white_num)
        print(self.list_placable())
        print()




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

    def com_random(board : Board):
        return random.choice(board.list_placable())

    board = Board()

    def f():
        return com_random, com_random

    # それぞれのプレイヤーの戦略の関数をわたす
    # プレイヤー先行でゲーム開始
    #board.set_plan(player, com_random, 1)
    #board.set_plan(player, player, 1)
    board.play(player, com_random)

    print("game set")
    print("black:", board.black_num)
    print("white:", board.white_num)


    #詰み手順の確認
    #37, 43, 34, 29, 52, 45, 38, 44, 20