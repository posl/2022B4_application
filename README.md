<!-- omit in toc -->
# はじめに
1. [B4 アプリ開発](#b4-アプリ開発)
   1. [実行環境](#実行環境)
   2. [実行方法](#実行方法)
   3. [担当](#担当)
<br>


# B4 アプリ開発

## 実行環境
- python 3.9.12
- tcl-tk 8.6.12<br><br>


Mac ユーザーの場合、tcl-tk のバージョンによって、正常に動作しない可能性があるため、以下の手順を踏む。

- brew が入っていない場合<br><br>
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

- tcl-tk 関連<br><br>
  tcl-tk のバージョンを確認する。
  ```
  brew info tcl-tk
  ```
  バージョンが 8.6 以前の場合、アップグレード。
  ```
  brew upgrade tcl-tk
  ```
<br>


## 実行方法
- pipenv を使う場合
  環境を再現する。
  ```
  pipenv update
  ```
  Cython を使って、実行ファイルを生成する。 (実行ファイルの生成場所が変わるので、cd コマンドを推奨)
  ```
  cd ./src/pyx  &&  pipenv run python setup.py build_ext --inplace
  ```
  アプリケーションを実行する。
  ```
  cd ./src  &&  pipenv run python play.py
  ```
- 自前の python 環境を使う場合
  パッケージをインストールする。
  ```
  pip install -U setuptools pip  &&  pip install -r requirements.txt
  ```
  Cython を使って、実行ファイルを生成する。 (実行ファイルの生成場所が変わるので、cd コマンドを推奨)
  ```
  cd ./src/pyx  &&  python setup.py build_ext --inplace
  ```
  アプリケーションを実行する。
  ```
  cd ./src  &&  python play.py
  ```
- docker を使う場合
  鋭意、作成中。
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