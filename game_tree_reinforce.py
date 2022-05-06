from game_tree import AlphaBeta
from board import Board
import numpy as np
from random import random

# 動作確認未完了

class GTTDAgent:
    def __init__(self):
        self.__alpha = 0.01
        self.__data_size = Board.action_size
        self.__data_list = np.zeros(self.__data_size)

    def __creat_new_data(self):
        self.__new_data_list = list(map(lambda x : x + np.random.random() - 0.5), self.__data_list)

    def reset(self):
        self.__data_list = np.zeros(self.__data_size)
        self.__creat_new_data()

    def set_data(self, __data_list):
        if len(__data_list) != self.__data_size:
            return
        self.__data_list = __data_list
        self.__creat_new_data()

    def update(self, reward):
      self.set_data([(1 - self.__alpha) * self.__data_list + self.__alpha * reward * self.diff_data_list])

    def get_data(self):
        return self.__data_list  

    def get_new_data(self):
        return self.__new_data_list

class GTReinforce:
    def __init__(self):
        self.__agent = GTTDAgent()
        self.player1 = AlphaBeta(0)

    def reset(self):
        self.__agent.reset

    def set_player1(self, player):
        self.player1 = player

    def start(self, repeat_num = 100):
        board = Board()
        alphabeta = AlphaBeta()
        for i in range(repeat_num):
            board.reset()
            agent_alphabeta = AlphaBeta(self.__agent.get_new_data())
            board.set_plan(self.player1, agent_alphabeta, 1)
            #　機械学習用の試合のプログラムを作る
            # self.__agent.update(board.game())
        return self.__agent.get_data()
    
    def makefile(self, filename):
        f = open(filename, "w")
        f.write(self, self.__agent.get_data())
