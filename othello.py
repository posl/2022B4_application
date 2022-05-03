import board as bd
import copy
import random


class Othello(bd.Board):
    # boardを継承
    def __init__(self):
        super().__init__()

    # 盤面リセット、戦略、先攻後攻(奇数:player1先行)を設定
    def pre(self, player1_plan, player2_plan, first_play):
        self.reset()
        self.player1_plan = player1_plan
        self.player2_plan = player2_plan
        self.player_turn = first_play % 2

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
        for step, num in bd.StepNumGenerator(startpoint).generator:
            n_gen = bd.ElementNumRange(startpoint, step, num)
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


    othello = Othello()
    # それぞれのプレイヤーの戦略の関数をわたす
    # プレイヤー先行でゲーム開始
    othello.pre(player, com_random, 1)
    othello.game()

