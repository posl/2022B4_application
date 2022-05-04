import numpy as np
import copy
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
        # 水平方向探索用の (step, num)
        yield sign, self.nums[sign]

        step_base = sign * Board.width
        num_base = self.nums[sign + 1]

        for bias in range(-1, 2):
            if bias:
                # 斜め方向探索用の (step, num)
                yield step_base + bias, min(num_base, self.nums[bias])
            else:
                # 垂直方向探索用の (step, num)
                yield step_base, num_base

    @property
    def generator(self):
        yield from self.__gen(1)
        yield from self.__gen(-1)


# オセロ盤の各位置から８方向それぞれに、存在する石を探索するときに使うジェネレータ
class ElementNumRange:
    def __init__(self, startpoint, step, num):
        self.value = startpoint
        self.step = step
        self.count = 0
        self.num = num

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.num:
            raise StopIteration()

        self.count += 1
        self.value += self.step
        return self.value




class Board:
    height = 8
    width = 8
    action_size = height * width

    # インスタンスを生成する前にクラス属性をチェックするようにする
    def __new__(cls):
        assert not cls.height & 1
        assert not cls.width & 1
        assert cls.action_size <= 64

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
        box = xp.empty(16, dtype = np.float32)
        stone_exist, stone_black = state

        for i in range(8):
            box[i] = stone_exist & 0xff
            box[i + 8] = stone_black & 0xff
            stone_exist >>= 8
            stone_black >>= 8

        # 正規化してから出力する
        return box / 255.


    # オセロ盤の各位置を表す、行列のインデックス (tuple) を通し番号に変換するためのメソッド
    @staticmethod
    def t2n(t):
        return Board.width * t[0] + t[1]

    # オセロ盤を初期状態にセットするためのメソッド
    def reset(self):
        shift = self.t2n(((self.height >> 1) - 1, (self.width >> 1) - 1))
        self.stone_exist = (0b11 << shift) + (0b11 << (shift + self.width))
        self.stone_black = (0b10 << shift) + (0b01 << (shift + self.width))
        self.turn = 1


    # 指定された 64 bit 整数の下から n bit 目の値を取得するためのメソッド
    def __getbit(self, name, n):
        return (getattr(self, name) >> n) & 1

    def getbit_stone_exist(self, n):
        return self.__getbit("stone_exist", n)

    def getbit_stone_black(self, n):
        return self.__getbit("stone_black", n)


    # 1 が立っているビットの数を取得
    @staticmethod
    def __bits_count(x):
        # 2 bit ごとにブロック分けして、それぞれのブロックにおいて１が立っているビット数を各ブロックの 2 bit で表現する
        x -= (x >> 1) & 0x5555555555555555

        # 4 bit ごとにブロック分けして、各ブロックに 上位 2 bit + 下位 2 bit を計算した値を入れる
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)

        # 8 bit ごとにブロック分けして、各ブロックに 上位 4 bit + 下位 4 bit を計算した値を入れる
        x += x >> 4
        x &= 0x0f0f0f0f0f0f0f0f

        # 以下、同様
        x += x >> 8
        x += x >> 16
        x += x >> 32

        # 0 ~ 64 を表現するために、下位 7 bit のみを取り出して出力とする
        return x & 0x7f

    @property
    def black_num(self):
        return self.__bits_count(self.stone_black)

    @property
    def white_num(self):
        return self.__bits_count(self.stone_exist ^ self.stone_black)


    # 石が存在するかどうかを示す変数、または存在する石が黒かどうかを示す変数を更新するためのメソッド
    def setbit_stone_exist(self, n):
        self.stone_exist |= 1 << n

    def setbit_stone_black(self, mask):
        self.stone_black ^= mask


    # 空きマスに自身の石を置けるかどうかの真偽値を取得するためのメソッド
    def is_placable(self, startpoint):
        for step, num in StepNumGenerator(startpoint).generator:
            n_gen = ElementNumRange(startpoint, step, num)
            try:
                n = next(n_gen)
            except StopIteration:
                continue

            if self.getbit_stone_exist(n) and (self.getbit_stone_black(n) ^ self.turn):
                for n in n_gen:
                    if self.getbit_stone_exist(n):
                        if self.getbit_stone_black(n) ^ self.turn:
                            continue
                        return True
                    break

        return False

    # 石を置ける箇所がどこかにあるかどうかの真偽値を取得するためのメソッド
    def somewhere_placable(self):
        for n in range(self.action_size):
            if not self.getbit_stone_exist(n) and self.is_placable(n):
                return True
        return False

    # エージェントが石を置ける箇所の番号をリストで取得するためのメソッド
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
    
    # nに駒を置く
    def __set(self, n):
        self.setbit_stone_exist(n)
        if self.turn == 1:
            self.setbit_stone_black(1 << n)
    
    # nに置いた時に返るマスを返す
    def __reverse(self, startpoint):
        for step, num in StepNumGenerator(startpoint).generator:
            n_gen = ElementNumRange(startpoint, step, num)
            try:
                n = next(n_gen)
            except StopIteration:
                continue

            if self.getbit_stone_exist(n) and (self.getbit_stone_black(n) ^ self.turn):
                mask = 1 << n
                for n in n_gen:
                    if self.getbit_stone_exist(n):
                        if self.getbit_stone_black(n) ^ self.turn:
                            mask |= 1 << n
                            continue
                        self.setbit_stone_black(mask)
                    break
    
    # nに駒を置き、返す
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
        tmp_board = copy.copy(self)
        if tmp_board.list_placable():
            return "true"
        else:
            tmp_board.turn ^= 1
            if tmp_board.list_placable():
                return "pass"
            else:
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




if __name__ == "__main__":
    def player(othello):
        while 1:
            try:
                n = int(input("enter n : "))
                if othello.is_placable(n):
                    return n
            except:
                print("error")
                continue

    def com_random(othello):
        return random.choice(othello.list_placable())


    board = Board()
    # それぞれのプレイヤーの戦略の関数をわたす
    # プレイヤー先行でゲーム開始
    #board.pre(player, com_random, 1)
    board.pre(player, player, 1)
    board.game()


    #詰み手順の確認
    #37, 43, 34, 29, 52, 45, 38, 44, 20