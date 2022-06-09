<!-- omit in toc -->
# はじめに
1. [B4 アプリ開発](#b4-アプリ開発)
   1. [実行環境](#実行環境)
   2. [実行方法](#実行方法)
   3. [担当](#担当)



# B4 アプリ開発

## 実行環境
- python 3.9.12
- tcl-tk 8.6.12


<!-- omit in toc -->
### Mac ユーザーの場合、tcl-tk のバージョンによって、正常に動作しない可能性があるため、以下の手順を踏む。

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



## 実行方法
- pipenv を使う場合
  環境の再現
  ```
  pipenv update
  ```

- 自前の python 環境を使う場合
- docker を使う場合



## 担当
- 若松
  - アルファ・ベータ法
- 東本
  - モンテカルロ木探索
- 稲田
  - 深層強化学習
- 友池
  - ゲーム画面表示