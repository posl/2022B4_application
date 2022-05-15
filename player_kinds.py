import random

import mc_tree_search
import mc_primitive

class Human:
    def __init__(self, par):
        self.par = par  # par : MainWindow , main_loopを呼び出すために必要

    def player(self, board):
        placable = set(board.list_placable())
        while True:
            self.par.mainloop()
            n = board.click_attr
            board.click_attr = None
            if n in placable:
                break
        return n
    
    def cheat_player(self, board):
        t = board.turn
        bplace = board.black_positions
        wplace = board.white_positions
        n = 0
        while True:
            self.par.mainloop()
            n = board.click_attr
            board.click_attr = None
            break
        if t==1:
            if ((board.stone_white>>n) & 1):
                board.stone_white = board.stone_white ^ (1<<n)
            board.stone_black = board.stone_black ^ (1<<n)
        else:
            if ((board.stone_black>>n) & 1):
                board.stone_black = board.stone_black ^ (1<<n)
            board.stone_white = board.stone_white ^ (1<<n)
        if ((board.stone_black>>n) & 1) & ((board.stone_white>>n) & 1):
            board.stone_black = board.stone_black ^ (1<<n)
            board.stone_white = board.stone_white ^ (1<<n)
        return self.player(board)
    
    #本来はここに書くべきではなかろうが暫定的に
    def com_random(self, board):
        return random.choice(board.list_placable())

    def com_cheater1(self, board):
        t = board.turn
        bplace = board.black_positions
        wplace = board.white_positions
        if len(bplace)+len(wplace)>4:
            if t==1:
                p = random.choice(wplace)
                board.stone_white = board.stone_white ^ (1<<p)
                board.stone_black = board.stone_black ^ (1<<p)
            else:
                p = random.choice(bplace)
                board.stone_black = board.stone_black ^ (1<<p)
                board.stone_white = board.stone_white ^ (1<<p)
        return random.choice(board.list_placable())



class PlayerKinds:
    def __init__(self, par):
        self.kinds_name = [] # 名前（人間、ランダムなど）
        self.kinds_func = [] # どこに打つかを返す関数
        self.kinds_difficulty = [] # 難易度がいくつあるか(0からN-1) １以下なら難易度選択が非表示

        self.human = Human(par)
        self.kinds_name.append("人間")
        self.kinds_func.append(self.human.player)
        self.kinds_difficulty.append(1)

        self.human = Human(par)
        self.kinds_name.append("人間-チート1")
        self.kinds_func.append(self.human.cheat_player)
        self.kinds_difficulty.append(1)

        self.kinds_name.append("ランダム")
        self.kinds_func.append(self.human.com_random)
        self.kinds_difficulty.append(1)

        self.kinds_name.append("ランダム-チート1")
        self.kinds_func.append(self.human.com_cheater1)
        self.kinds_difficulty.append(1)

        self.kinds_name.append("MC木探索")
        self.kinds_func.append(mc_tree_search.MonteCarloTreeSearch())
        self.kinds_difficulty.append(1)

        self.kinds_name.append("原始MC探索")
        self.kinds_func.append(mc_primitive.PrimitiveMonteCarlo())
        self.kinds_difficulty.append(1)


    def get_num(self):
        return len(self.kinds_name)
    
    def get_name(self, id):
        if id<0 or id>=len(self.kinds_name):
            print("範囲外のIDが指定されました")
            exit()
        return self.kinds_name[id]

    def get_func(self, id):
        if id<0 or id>=len(self.kinds_func):
            print("範囲外のIDが指定されました")
            exit()
        return self.kinds_func[id]

    def get_difficulty(self, id):
        if id<0 or id>=len(self.kinds_difficulty):
            print("範囲外のIDが指定されました")
            exit()
        return self.kinds_difficulty[id]