<!-- omit in toc -->
# B4 アプリ開発

1. [実行環境](#実行環境)
2. [環境構築](#環境構築)
3. [実行方法](#実行方法)
4. [担当分け](#担当分け)


<br>


## 実行環境

- python 3.9.12
- tcl-tk 8.6.12


<br>


## 環境構築

Mac の場合、tcl-tk のバージョンによって、正常に動作しない可能性があるため、先に以下の手順を踏む。

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
  Cython を使って、実行ファイルを生成する。 (生成場所が変わるので、cd コマンドを推奨)
  ```
  cd ./src/pyx  &&  pipenv run python setup.py build_ext --inplace
  ```

- 自前の python 環境を使う場合

  パッケージをインストールする。
  ```
  pip install -U setuptools pip  &&  pip install -r requirements.txt
  ```
  Cython を使って、実行ファイルを生成する。 (生成場所が変わるので、cd コマンドを推奨)
  ```
  cd ./src/pyx  &&  python setup.py build_ext --inplace
  ```


<br>


## 実行方法

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


<br>


## 担当分け

- 若松
  - アルファ・ベータ法
- 東本
  - モンテカルロ木探索
- 稲田
  - 深層強化学習
- 友池
  - ゲーム画面表示
