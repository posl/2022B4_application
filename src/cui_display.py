from random import choice, randrange
from time import sleep, time
from collections import deque

from board import Board

from mc_tree_search import MonteCarloTreeSearch
from mc_primitive import PrimitiveMonteCarlo
from gt_alpha_beta import AlphaBeta
from drl_reinforce import ReinforceComputer
from drl_rainbow import RainbowComputer
from drl_alphazero import AlphaZeroComputer



class Human:
    def __init__(self):
        self.abc_dict = {chr(ord("a") + i) : i for i in range(Board.width)}
        self.abc_max = max(self.abc_dict)

    def __call__(self, board):
        abc_dict, abc_max = self.abc_dict, self.abc_max
        height = board.height
        placable = set(board.list_placable())

        print("\n次の形式で、着手箇所を指定してください。", end = "")
        print(f"([a-{abc_max}][1-{height}] e.g. {choice(list(abc_dict.keys()))}{randrange(height) + 1})\n")

        while True:
            try:
                print("\x1b[F\x1b[K", end = "", flush = True)
                S = input().lower()

                if len(S) == 2:
                    i = int(S[1]) - 1
                    if 0 <= i < height:
                        action = 8 * i + abc_dict[S[0]]
                        if action in placable:
                            return action
                        else:
                            print(f"\x1b[KERROR: \"{S}\" には置けません。", end = "", flush = True)
                            continue
            except KeyboardInterrupt as e:
                raise e
            except:
                pass
            print("ERROR: 指定された形式で入力してください。", end = "", flush = True)



class Random:
    def __call__(self, board):
        return choice(board.list_placable())




class CuiBoard(Board):
    def __init__(self):
        super().__init__()

        A = Board.action_size
        players = []
        self.players = players

        # (cls, init_args, name) の形式で格納する
        players.append([Human, ((), ), "Human"])
        players.append([Random, ((), ), "Random"])
        players.append([MonteCarloTreeSearch, ((1024, ), (4096, ), (16384, )), "MCTS"])
        players.append([PrimitiveMonteCarlo, ((256, ), (1024, ), (4096, )), "Primitive MC"])
        players.append([AlphaBeta, ((0, 4), (1, 4), (0, 6)), "AlphaBeta"])
        players.append([ReinforceComputer, ((A, 1), (A, 3), (A, 6)), "Reinforce"])
        players.append([RainbowComputer, ((A, 1), (A, 3), (A, 6)), "RainBow"])
        players.append([AlphaZeroComputer, ((A, 8, 50), (A, 8, 200), (A, 8, 800)), "AlphaZero"])

        # プレイヤー選択の都合上、名前だけ分離する
        self.names = [player.pop() for player in players]

        # y/n 判定に使う文字列と真偽値の対応を示す辞書
        self.yn_dict = {"yes": 1, "y": 1, "no": 0, "n": 0}


    def select_player(self, is_first):
        players, names = self.players, self.names
        repr = "先攻" if is_first else "後攻"

        N = len(players)
        M = N - 1

        desc = f"次の形式で、{repr}のプレイヤーを指定してください。([0-{M}] e.g. {randrange(N)})"
        player_id = self.input_additional(desc, names, N)
        (cls, args), name = players[player_id], names[player_id]

        L = len(args)
        K = L - 1

        if K > 0:
            desc = f"次の形式で、{repr} \"{name}\" のレベルを選択してください。([0-{K}] e.g. {randrange(L)})"
            player_levels = [f"Level {i}" for i in range(L)]
            level = self.input_additional(desc, player_levels, L)
            name += f" Lv.{level}"
        else:
            level = 0

        return cls(*args[level]), name


    # desc: 説明文,  lineup: 入力値の対応表,  lim: 入力値の上限,  yn_flag: Yes/No を答えさせるかどうか
    def input_additional(self, desc: str, lineup = None, lim = 0, yn_flag = False):
        if yn_flag:
            print(desc + " [y/n]", end = "\n\n")
        else:
            print(f"\x1b[2J{desc}", end = "\n\n")
            for i, value in enumerate(lineup):
                print(f"{i}: {value}")
            print(end = "\n\n")

        while True:
            try:
                print("\x1b[F\x1b[K", end = "", flush = True)
                i = input()
                i = self.yn_dict[i.lower()] if yn_flag else int(i)
                if 0 <= i < lim:
                    return i
            except KeyboardInterrupt as e:
                raise e
            except:
                pass
            print("ERROR: 指定された形式で入力してください。", end = "", flush = True)


    def play(self):
        # コンソールの初期化も兼ねた、ゲーム開始を表現する出力
        print("\x1b[2J")
        for __ in range(100):
            print("オセロゲーム", end = "", flush = True)
            sleep(0.015)
            print("\x1b[1K")

        play_flag = 1
        while play_flag:
            # コンソールの標準入力を用いた、プレイヤーの設定
            player1, name1 = self.select_player(is_first = True)
            player0, name0 = self.select_player(is_first = False)
            self.set_plan(player1, player0)

            # 画面表示で、先攻・後攻のプレイヤーを示す時に使う文字列を設定する
            maxlen = max(len(name1), len(name0))
            self.str_player1 = f"先攻(⚫️):  {name1.ljust(maxlen)}"
            self.str_player0 = f"後攻(⚪️):  {name0.ljust(maxlen)}"

            # オセロボードの初期化
            self.reset()
            self.print_board()
            sleep(0.1)

            # ゲーム本体
            game_flag = 1
            while game_flag:
                start = time()
                action = self.get_action()

                # ゲームが高速に終了しないように、待ち時間を設ける
                while (time() - start) < 0.5:
                    continue

                self.put_stone(action)
                game_flag = self.can_continue()
                self.print_board(pass_flag = (game_flag == 2))

            # 勝敗の表示
            diff = self.black_num - self.white_num
            if diff:
                if diff > 0:
                    print(f"\n先攻 \"{name1}\" の勝ち！")
                else:
                    print(f"\n後攻 \"{name0}\" の勝ち！")
            else:
                print("\n引き分け")

            # 継続するかを確認する
            desc = "続けますか？"
            play_flag = self.input_additional(desc, lim = 2, yn_flag = True)

        # コンソールを全クリアして、処理を終える
        print("\x1b[2J")


    def print_board(self, pass_flag = False):
        # 各プレイヤーと所有する石の数を表示する
        print("\x1b[2J")
        print(f"{self.str_player1}   {self.black_num}")
        print(f"{self.str_player0}   {self.white_num}")

        # パスが起こっていた場合は、それも画面に反映させる
        str_turn = ("⚫️" if self.turn else "⚪️")
        str_pass = ("( PASS )" if pass_flag else "")
        print(f"\n手番: {str_turn}  {str_pass}", end = "\n\n")

        width = self.width
        placable = deque(self.list_placable())
        stone_black, stone_white = self.stone_black, self.stone_white

        S = "　ＡＢＣＤＥＦＧＨ"
        NS = "１２３４５６７８"

        for action in range(self.action_size):
            i, j = divmod(action, width)
            if not j:
                S += f"\n{NS[i]}"

            if len(placable) and (action == placable[0]):
                del placable[0]
                S += "❌"
            elif (stone_black >> action) & 1:
                S += "⚫️"
            elif (stone_white >> action) & 1:
                S += "⚪️"
            else:
                S += "・"

        # 盤面を一気に表示する
        print(S)




def main():
    cui_board = CuiBoard()
    cui_board.play()


if __name__ == "__main__":
    main()