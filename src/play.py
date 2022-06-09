from argparse import ArgumentParser

from gui_display import DisplayBoard



parser = ArgumentParser(description = "オセロゲーム")
parser.add_argument("--ip", default = "0", help = "通信用にサーバを立てた時、その出力である IP アドレスを指定する")
args = parser.parse_args()

displayboard = DisplayBoard(args.ip)
displayboard.play()