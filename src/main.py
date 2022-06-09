from argparse import ArgumentParser
import subprocess

from gui_display import DisplayBoard

parser = ArgumentParser(description="オセロゲーム")
parser.add_argument("--host", action="store_true", help="サーバー側が指定する")
parser.add_argument("--ip", default="0", help="クライアント側が指定する")

args = parser.parse_args()

host_flag = args.host
ip = args.ip


if host_flag:
    subprocess.run(["pipenv", "run", "python", "src/gui_network.py"])


displayboard = DisplayBoard()
displayboard.play(ip)