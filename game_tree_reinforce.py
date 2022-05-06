from game_tree import AlphaBeta
from board import Board
import numpy as np
from random import random

class GTTDAgent:
	def __init__(self):
		self.alpha = 0.01
		self.data_size = Board.action_size
		self.data_list = np.zeros(self.data_size)
	
	def __creat_new_data(self):
		self.new_data_list = list(map(lambda x : x + np.random.random() - 0.5), self.data_list)

	def reset(self):
		self.data_list = np.zeros(self.data_size)
		self.__creat_new_data()

	def set_data(self, data_list):
		if len(data_list) != self.data_size:
			return
		self.data_list = data_list
		self.__creat_new_data()

	def update(self, reward):
		self.set_data([(1 - self.alpha) * self.data_list + self.alpha * reward * self.diff_data_list])

	def get_data(self):
		return self.data_list

	def get_new_data(self):
		return self.new_data_list

class GTReinforce:
	def __init__(self):
		self.agent = GTTDAgent()
		self.player1 = AlphaBeta(0)
	
	def reset(self):
		self.agent.reset()
	
	def set_player1(self, player):
		self.player1 = player

	def stard(self, repeat_num = 100):
		board = Board()
		alphabeta = AlphaBeta()
		for i in range(repeat_num):
			board.reset()
			agent_alphabeta = AlphaBeta(self.agent.get_new_data())
			board.set_plan(self.player1, agent_alphabeta, 1)
			#　機械学習用の試合のプログラムを作る
			# self.agent.update(board.game())
		return self.agent.get_data()
