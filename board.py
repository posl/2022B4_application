import numpy as np
from math import ceil
from contextlib import contextmanager
import random


# 下のジェネレータの引数となる (step, num) を８方向分生成するジェネレータ
class StepNumGenerator:
    def __init__(self, startpoint):
        up, left = divmod(startpoint, Board.width)
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


# 開始の数とステップ数、要素数を指定する range イテラブル (再利用は想定していないので、扱い的にはジェネレータと等価)
class ElementNumRange:
    def __init__(self, startpoint):
        self.args = [startpoint, None, None]

    # クラスからのインスタンス生成の回数を減らして、処理速度改善
    def reset(self, step_num):
        self.args[1:] = step_num
        self.count = 0
        return self

    def __iter__(self):
        return self

    # getattr() の呼び出しが少なくなるようにして、処理速度改善
    def __next__(self):
        startpoint, step, num = self.args
        count = self.count

        if count >= num:
            raise StopIteration()

        count += 1
        self.count = count
        return startpoint + step * count


# オセロ盤の特定のマスから全方位探索を行うためのイテラブル (再利用は想定していないので、扱い的にはジェネレータと等価)
class OmniDirectionalSearcher:
    def __init__(self, startpoint):
        self.step_num = StepNumGenerator(startpoint).generator
        self.range = ElementNumRange(startpoint)

    # このメソッドで自身をジェネレータとして返す
    def __iter__(self):
        return self

    def __next__(self):
        # この next() の呼び出しで生じる StopIteration 例外をこのジェネレータが生じさせた例外として使う
        step_num = next(self.step_num)
        n_gen = self.range.reset(step_num)

        try:
            n = next(n_gen)
        except StopIteration:
            return next(self)

        return n, n_gen




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
        # オセロ盤の状態を表現する整数、ターンを表す整数 (先攻(黒) : 1, 後攻(白) : 0)
        self.stone_exist = 0
        self.stone_black = 0
        self.turn = 1

        # オセロ盤の状態のログを取って、前の状態に戻ることを可能にするためのスタック
        self.log_state = []
        self.log_plans = []

        # somewhere_placable() で保存した n を次の list_placable() に使うための変数
        self.tmp_n = None


    # オセロ盤の情報である 64 bit 整数を 8 bit 区切りで状態として取得する
    @property
    def state(self):
        return self.stone_exist, self.stone_black

    def set_state(self, state):
        self.stone_exist, self.stone_black = state

    # オセロ盤の状態情報である２つの 64 bit 整数を 8 bit 区切りで ndarray に格納して、それを出力する
    @staticmethod
    def state2ndarray(state, xp = np):
        size = ceil(Board.action_size / 8)
        box = xp.empty(size << 1, dtype = np.float32)
        stone_exist, stone_black = state

        for i in range(size):
            box[i] = stone_exist & 0xff
            box[i + size] = stone_black & 0xff
            stone_exist >>= 8
            stone_black >>= 8

        # 正規化してから出力する
        return box / 255.


    # オセロ盤の各位置を表す、行列のインデックス (tuple) を通し番号に変換する
    @staticmethod
    def t2n(t):
        return Board.width * t[0] + t[1]

    # オセロ盤を初期状態にセットする
    def reset(self):
        width = self.width
        shift = self.t2n(((self.height >> 1) - 1, (width >> 1) - 1))
        self.stone_exist = (0b11 << shift) + (0b11 << (shift + width))
        self.stone_black = (0b10 << shift) + (0b01 << (shift + width))
        self.turn = 1


    @property
    def black_num(self):
        return self.stone_black.bit_count()

    @property
    def white_num(self):
        return (self.stone_exist ^ self.stone_black).bit_count()

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
            return self.get_stone_num()


    # 指定された 64 bit 整数の下から n bit 目の値を取得する
    def __getbit(self, name, n):
        return (getattr(self, name) >> n) & 1

    def getbit_stone_exist(self, n):
        return self.__getbit("stone_exist", n)

    def getbit_stone_black(self, n):
        return self.__getbit("stone_black", n)


    # 石が存在するかどうかを示す変数、または存在する石が黒かどうかを示す変数を更新する
    def setbit_stone_exist(self, n):
        self.stone_exist |= 1 << n

    def setbit_stone_black(self, mask):
        self.stone_black ^= mask


    # 空きマスに自身の石を置けるかどうかの真偽値を取得する
    def is_placable(self, startpoint):
        for n, n_gen in OmniDirectionalSearcher(startpoint):
            if self.getbit_stone_exist(n) and (self.getbit_stone_black(n) ^ self.turn):
                for n in n_gen:
                    if self.getbit_stone_exist(n):
                        if self.getbit_stone_black(n) ^ self.turn:
                            continue
                        return True
                    break
        return False

    # 石を置ける箇所がどこかにあるかどうかの真偽値を取得する
    def somewhere_placable(self):
        for n in range(self.action_size):
            if not self.getbit_stone_exist(n) and self.is_placable(n):
                self.tmp_n = n
                return True
        return False

    # エージェントが石を置ける箇所の番号をリストで取得する
    def list_placable(self):
        p_list = []

        # このメソッドの呼び出しの直前の somewhere_placable() の結果を引き継ぐことができる
        n = self.tmp_n
        if n is None:
            start = 0
        else:
            p_list.append(n)
            start = n + 1
            self.tmp_n = None

        for n in range(start, self.action_size):
            if not self.getbit_stone_exist(n) and self.is_placable(n):
                p_list.append(n)
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


    # ターンを交代
    def turn_change(self):
        self.turn ^= 1

    # 終了 : 0, 手番を交代 : 1, 手番そのままで続行 : 2
    def can_continue(self, pass_flag = False):
        self.turn_change()
        if self.somewhere_placable():
            return 1 + pass_flag

        if pass_flag:
            return 0
        else:
            return self.can_continue(True)


    # ゲーム本体
    def game(self, render_func = None):
        flag = 1
        while flag:
            n = self.get_action()
            self.put_stone(n)
            flag = self.can_continue()

            if render_func is not None:
                # 引数は pass があったかどうかの真偽値
                render_func()

    def play(self, player1_plan, player2_plan):
        self.reset()

        # # start 表示 (コンピュータの設定選択)
        # player1_plan, player2_plan = f()

        self.set_plan(player1_plan, player2_plan)

        # 最初の盤面表示
        self.print_state()

        self.game(self.print_state)

        # finish 表示 (結果の表示)


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


    # 実際に手を打たずに、打った時の状況を検証するためのランタイムコンテキストを生成するマネージャ
    @contextmanager
    def log_runtime(self, n, info = "state"):
        add_log = getattr(self, "add_" + info)
        add_log()
        self.put_stone(n)
        flag = self.can_continue()
        yield

        if flag == 1:
            self.turn_change()
        undo_log = getattr(self, "undo_" + info)
        undo_log()

    def add_state(self):
        self.log_state.append(self.state)

    def undo_state(self):
        self.set_state(self.log_state.pop())


    def add_state_plans(self):
        self.add_state()
        self.log_plans.append(self.plans)

    def undo_state_plans(self):
        self.undo_state()
        self.set_plan(*self.log_plans.pop())




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