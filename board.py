import numpy as np
from math import ceil
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


# 開始の数とステップ数、要素数を指定する range ジェネレータ
class ElementNumRange:
    def __init__(self, startpoint):
        self.startpoint = startpoint

    def reset(self, step_num):
        self.step, self.num = step_num
        self.value = self.startpoint
        self.count = 0
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.num:
            raise StopIteration()

        self.count += 1
        self.value += self.step
        return self.value


# オセロ盤の特定のマスから全方位探索を行うためのジェネレータ
class OmniDirectionalSearcher:
    def __init__(self, startpoint):
        self.step_num = StepNumGenerator(startpoint).generator
        self.range = ElementNumRange(startpoint)

    def __iter__(self):
        return self

    def __next__(self):
        # この next() の呼び出しで生じる StopIteration 例外をこのジェネレータが生じさせた例外として使う
        n_gen = self.range.reset(next(self.step_num))

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
        assert cls.action_size.bit_length() < 256

        return super().__new__(cls)

    def __init__(self):
        self.stone_exist = 0
        self.stone_black = 0
        self.turn = 1


    # オセロ盤の情報である 64 bit 整数を 8 bit 区切りで状態として取得する
    @property
    def state(self):
        return self.stone_exist, self.stone_black

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


    # 指定された 64 bit 整数の下から n bit 目の値を取得する
    def __getbit(self, name, n):
        return (getattr(self, name) >> n) & 1

    def getbit_stone_exist(self, n):
        return self.__getbit("stone_exist", n)

    def getbit_stone_black(self, n):
        return self.__getbit("stone_black", n)


    # 引数以上の数で最小の２のべき乗を取得する (任意の bit 数の整数に対応可能)
    @staticmethod
    def get_powerof2(x: int):
        assert x > 0

        # 指定された値が２のべき乗でない場合は目的の出力を作成する
        if x & (x - 1):
            return 1 << x.bit_length()
        return x

    # 1 が立っているビットの数を取得する (255 bit の整数まで対応可能)
    def __bits_count(self, x):
        # 2 bit ごとにブロック分けして、それぞれのブロックにおいて１が立っているビット数を各ブロックの 2 bit で表現する
        x -= (x >> 1) & 0x5555555555555555

        # 4 bit ごとにブロック分けして、各ブロックに 上位 2 bit + 下位 2 bit を計算した値を入れる
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)

        # 8 bit ごとにブロック分けして、各ブロックに 上位 4 bit + 下位 4 bit を計算した値を入れる
        x += x >> 4
        x &= 0x0f0f0f0f0f0f0f0f

        # 以下、同様
        n = 8
        max_n = self.get_powerof2(x)
        while n < max_n:
            x += x >> n
            n <<= 1

        # 下位 8 bit のみを取り出して出力とする (出力は 0 ~ 255 になる)
        return x & 0xff

    @property
    def black_num(self):
        return self.__bits_count(self.stone_black)

    @property
    def white_num(self):
        return self.__bits_count(self.stone_exist ^ self.stone_black)


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
                return True
        return False

    # エージェントが石を置ける箇所の番号をリストで取得する
    def list_placable(self):
        p_list = []
        for n in range(self.action_size):
            if not self.getbit_stone_exist(n) and self.is_placable(n):
                p_list.append(n)
        return p_list


    # 置くマスを取得
    def get_num(self):
        if self.turn == self.player_turn:
            return self.player1_plan(self)
        else:
            return self.player2_plan(self)

    # n に駒を置く
    def __set(self, n):
        self.setbit_stone_exist(n)
        if self.turn == 1:
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

    # n に駒を置き、返す
    def put(self, n):
        self.__set(n)
        self.__reverse(n)

    # ターンを交代
    def turn_change(self):
        self.turn ^= 1

    # 盤面リセット、戦略、先攻後攻(奇数:player1先行)を設定
    def pre(self, player1_plan, player2_plan, first_play):
        self.reset()
        self.player1_plan = player1_plan
        self.player2_plan = player2_plan
        self.player_turn = first_play % 2

    # ゲームが正常に続行できるか判定
    def can_continue(self):
        if self.list_placable():
            return "true"
        else:
            self.turn ^= 1
            if self.list_placable():
                self.turn ^= 1
                return "pass"
            else:
                self.turn ^= 1
                return "gameset"

    # ゲーム終了時
    def gameset(self):
        print("owa\nblack:", self.black_num, "   white:", self.white_num) #表示

    # ゲームの本体
    def game(self):
        print("start") #表示
        while 1:
            #ゲームが正常に続行されるか
            flag = self.can_continue()
            if flag == "pass":
                print("pass") #表示
                self.turn_change()
                continue
            elif flag == "gameset":
                break

            self.print_state()

            # 置く場所を取得
            n = self.get_num()
            print("put : ", n)

            # 駒を置く
            self.put(n)

            # ターン更新
            self.turn_change()

        print("gameset") #表示
        self.gameset()


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


# ログ機能を持ち，以前の盤面に戻ることができる
class LogBoard(Board):
    def __init__(self):
        super().__init__()
        self.stone_exist_stack = [None] * Board.action_size
        self.stone_black_stack = [None] * Board.action_size
        self.stack_num = 0

    # ログを追加する
    def add_log(self):
        if self.stack_num == Board.action_size:
            return
        self.stone_exist_stack[self.stack_num] = self.stone_exist
        self.stone_black_stack[self.stack_num] = self.stone_black
        self.stack_num += 1

    # 最新のログの盤面に戻る
    def undo_log(self):
        if self.stack_num == 0:
            return
        self.stack_num -= 1
        self.stone_exist = self.stone_exist_stack[self.stack_num]
        self.stone_black = self.stone_black_stack[self.stack_num]

    # ゲーム木用
    # Boardから必要な情報をセットする
    def set_board(self, board : Board):
        self.stone_exist = board.stone_exist
        self.stone_black = board.stone_black
        self.turn = board.turn




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
    # それぞれのプレイヤーの戦略の関数をわたす
    # プレイヤー先行でゲーム開始
    #board.pre(player, com_random, 1)
    board.pre(player, player, 1)
    board.game()


    #詰み手順の確認
    #37, 43, 34, 29, 52, 45, 38, 44, 20