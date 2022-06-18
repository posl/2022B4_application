from argparse import ArgumentParser



parser = ArgumentParser(description = "オセロゲーム")
parser.add_argument("--host", action = "store_true", help = "サーバを立てる")
parser.add_argument("--console", action = "store_true", help = "GUI を使用したくない場合に指定する")
parser.add_argument("--ip", default = "0", help = "通信用にサーバを立てた時、その出力である IP アドレスを入力する")
args = parser.parse_args()


if args.host:
    from dp_network import main
elif args.console:
    from dp_cui import main
else:
    from dp_gui import main


try:
    main(args.ip)
except TypeError:
    main()