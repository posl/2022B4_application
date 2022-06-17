<!-- omit in toc -->
# B4 アプリ開発

1. [実行環境](#実行環境)
2. [実行環境構築の手順](#実行環境構築の手順)
3. [実行の手順](#実行の手順)
4. [担当](#担当)
5. [余談](#余談)


<br>


## 実行環境

- python 3.9.12
- tcl-tk 8.6.12


<br>


## 実行環境構築の手順

Docker (CUI でのみ実行可能) を使用する場合、以下の環境構築に関する記述は読み飛ばす。
<br>
<br>

Mac 環境で GUI を使う場合、tcl-tk が古いと正常に実行できない可能性があるので、先に以下の手順を踏む。

- brew が入っていない場合

  brew のインストールを行う。
  ```
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
  ```
  問題がなさそうか確認する。
  ```
  brew doctor
  ```
  問題がなさそうなら、アップデート。
  ```
  brew update
  ```

- tcl-tk 関連

  tcl-tk のバージョンを確認する。
  ```
  brew info tcl-tk
  ```
  バージョンが 8.6 以前の場合、アップグレード。
  ```
  brew upgrade tcl-tk
  ```


<br>


python での実行環境の準備は、以下の手順で行う。

- pipenv を使う場合

  環境を再現する。
  ```
  pipenv update
  ```
  Cython を使って、実行ファイルを生成する。

  (生成場所が変わるので、cd コマンドを推奨)
  ```
  cd ./src/pyx  &&  pipenv run python setup.py build_ext --inplace
  ```

- 自前の python 環境を使う場合

  パッケージをインストールする。
  ```
  pip install -U setuptools pip  &&  pip install -r requirements.txt
  ```
  Cython を使って、実行ファイルを生成する。

  (生成場所が変わるので、cd コマンドを推奨)
  ```
  cd ./src/pyx  &&  python setup.py build_ext --inplace
  ```


<br>


## 実行の手順

GUI で実行するための環境構築をした場合は、以下の手順で実行できる。

- 一人でプレイする場合

  いずれかの python 実行方法で、src/play.py を実行する。
  ```
  python src/play.py
  ```

- 二人でプレイする場合

  (プライベートな通信でのみ実行可能)
  <br>
  <br>

  片方が、いずれかの python 実行方法で、サーバを立てる。
  ```
  python src/play.py --host
  ```
  この時、画面に表示される IP アドレスは、下記の実行時に使用する。
  <br>
  <br>

  続けて、サーバを立てた側は、別のターミナルを開く。

  (サーバの役割を果たしているプロセスを終了させると、通信が行えなくなるため。)
  <br>
  <br>

  双方が、いずれかの python 実行方法で、以下に示すように src/play.py を実行する。
  ```
  python src/play.py --ip <上で表示された IP アドレス>
  ```

- 画面表示でエラーが発生した場合

  使用している Python 環境が古い方の tcl-tk を使い続けている可能性が高い。

  その場合は、Python 環境を一旦アンインストールして、再びインストールするとよい。


<br>
<br>


GUI の環境構築をしていない場合でも、コンソール入出力のみを使った実行方法を選択することができる。

- Docker を使う場合

  一切の環境構築を必要とせず、以下のコマンドのみを発行して、実行する。
  ```
  docker run -it --rm inadatsukasa/othello
  ```
  (指定したイメージは Docker Hub にて公開されており、サイズは 550 MB 、スキャンは未実行である。)


<br>


- pipenv または、自前の Python 環境を使う場合

  いずれかの python 実行方法で、コンソール入出力のフラグを立てつつ、src/play.py を実行する。
  ```
  python src/play.py --console
  ```


<br>


## 担当

- 若松
  - アルファ・ベータ法
- 東本
  - モンテカルロ木探索
- 稲田
  - 深層強化学習
- 友池
  - ゲーム画面表示


<br>


## 余談

Docker が GUI に対応していない理由は、デスクトップ上の画面表示のためのミドルウェアは、linux 系のものである必要があり、またデスクトップのそもそもの環境にインストールされている必要があるため、コンテナで環境ごと仮想化しているはずがそうなってない、といった状況になってしまうから。

ただ、linux 系 OS 上で実行する場合、GPU に対応させつつ、１コマンドで実行する方法はある。
(参考: https://towardsdatascience.com/empowering-docker-using-tkinter-gui-bf076d9e4974)

また、linux 系 OS 以外の上で実行する場合でも、XQuartz (X11) をデスクトップの環境にインストールし、適切な設定を施せば、実行できないことはない。
(参考: http://blog.eszett-design.com/2020/10/dockerpythontkinter.html)
