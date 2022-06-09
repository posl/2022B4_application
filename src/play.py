from argparse import ArgumentParser

from gui_display import DisplayBoard

parser = ArgumentParser(description="オセロゲーム")
parser.add_argument("--ip", default="0", help="クライアント側が指定する")

args = parser.parse_args()

ip = args.ip

displayboard = DisplayBoard()
displayboard.play(ip)