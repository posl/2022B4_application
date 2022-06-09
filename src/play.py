from argparse import ArgumentParser

from gui_network import OthelloServer
from gui_display import DisplayBoard



parser = ArgumentParser(description = "オセロゲーム")
parser.add_argument("--host", action = "store_true", help = "サーバを立てる")
parser.add_argument("--ip", default = "0", help = "通信用にサーバを立てた時、その出力である IP アドレスを入力する")
args = parser.parse_args()


if args.host:
    server = OthelloServer()
    server.mainloop()
else:
    displayboard = DisplayBoard(args.ip)
    displayboard.play()