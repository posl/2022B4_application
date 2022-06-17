from random import choice, randrange
from time import sleep, time

from board import Board
from mc_tree_search import MonteCarloTreeSearch, RootPalallelMonteCarloTreeSearch
from mc_primitive import PrimitiveMonteCarlo
from gt_alpha_beta import AlphaBeta
from drl_rainbow import RainbowComputer
from drl_reinforce import ReinforceComputer
from drl_alphazero import AlphaZeroComputer, PolicyValueNet, play



class Human:
    def __init__(self):
        pass

    def __call__(self, board):
        return self.play(board)
    
    def play(self, board : Board):
        print("Where will you put? Enter [a-h][1-8] e.g. a4, h1.")
        while 1:
            s = input().lower()
            if len(s) == 2 and "a" <= s[0] <= "h" and "1" <= s[1] <= "8":
                i = int(s[1]) - 1
                j = ord(s[0]) - ord("a")
                n = 8 * i + j
                if n in board.list_placable():
                    return n
                else:
                    print("ERROR! you can't put at {}.".format(s))
            else:
                print("ERROR! Invalid Value. You should enter [a-h][1-8]")


class Random:
    def __init__(self):
        pass
    def __call__(self, board):
        return choice(board.list_placable())


class CuiBoard(Board):
    def __init__(self):
        super().__init__()

    def select_players(self, one_or_two):
        A = Board.action_size
        player_kind = [ "人間", 
                        "ランダム", 
                        "MC木探索", 
                        "MC木探索 + ルート並列化", 
                        "原始MC法", 
                        "AlphaBeta", 
                        "Reinforce + NegaAlpha", 
                        "RainBow + NegaAlpha", 
                        "AlphaZero" ]
        player_class = [ Human, 
                         Random, 
                         MonteCarloTreeSearch, 
                         RootPalallelMonteCarloTreeSearch, 
                         PrimitiveMonteCarlo, 
                         AlphaBeta, 
                         ReinforceComputer, 
                         RainbowComputer, 
                         AlphaZeroComputer ]
        player_diff = [ [ () ], 
                        [ () ], 
                        [ (1024, ), (4096, ), (16384, ) ], 
                        [ (5000, ), (10000, ), (20000, ) ], 
                        [ (256, ), (1024, ), (4096, ) ], 
                        [ (0, 5), (0, 6), (1, 6) ], 
                        [ (A, 1), (A, 3), (A, 6) ], 
                        [ (A, 1), (A, 3), (A, 6) ], 
                        [ (A, randrange(5), 50), (A, randrange(5, 10), 200), (A, 8) ] ]
        
        print("\nSelect player{}. Plz input 0 - {}".format(one_or_two, len(player_kind) - 1))
        
        while 1:
            for i in range(len(player_kind)):
                print("{} {}".format(i, player_kind[i]))
            
            try:
                player_id = int(input())
                if 0 <= player_id < len(player_kind):
                    break
                else:
                    print("\nOut-of-Range ERROR! you should input 0 - {}".format(len(player_kind) - 1))
            except:
                print("\nERROR! You should input number.")
        
        print("\nSelect CPU's difficualty. Plz input 1 - 3.")
        if player_id >= 2:
            while 1:
                try:
                    diff = int(input())
                    if 1 <= diff <= 3:
                        break
                    else:
                        print("Out-of-range ERROR! you should input 1 - 3")
                except:
                    print("ERROR! You should input number.")
        else:
            diff = 1

        print("\nOk. player{} is selected of {}, difficulty {}.".format(one_or_two, player_kind[player_id], diff))

        return player_class[player_id](*player_diff[player_id][diff - 1])


    def print_board(self):
        STR_NUM = "１２３４５６７８"
        STR_ABC = "　ＡＢＣＤＥＦＧＨ"
        black = self.stone_black
        white = self.stone_white
        placable = self.list_placable()
        
        print("\x1b[2J")

        print("BLACK : {}  WHITE : {}".format(self.black_num, self.white_num))
        if self.turn:
            print("turn : player1(⚫️)\n")
        else:
            print("turn : player2(⚪️)\n")


        print(STR_ABC)

        for i in range(self.height):
            s = STR_NUM[i]
            for j in range(self.width):
                if black & 1:
                    s += "⚫️"
                elif white & 1:
                    s += "⚪️"
                else:
                    if 8 * i + j in placable:
                        #s += "❌"
                        s += "＋"
                        #s += "＊" 
                    else:
                        s += "・"
                black >>= 1
                white >>= 1
            print(s)
        

    def game(self):
        flag = 1
        self.print_board()

        while flag:
            t = time()
            n = self.get_action()
            
            # 待ち時間を設ける
            while time() - t < 1:
                continue
            
            mask = self.put_stone(n)
            flag = self.can_continue()
            self.print_board()


    def play(self): 
        self.reset()
        self.clear_playlog()

        print("\x1b[2J")
        print("OTHELLO GAME!\n")


        player1 = self.select_players(1)
        player2 = self.select_players(2)
        
        self.set_plan(player1, player2)

        sleep(2)

        self.game()
        
        if self.black_num > self.white_num:
            print("BLACK WIN!!!")
        elif self.black_num < self.white_num:
            print("WHITE WIN!!!")
        else:
             print("DRAW!")       


def main():
    cui_board = CuiBoard()
    cui_board.play()

if __name__ == "__main__":
    main()










