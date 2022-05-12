from board import Board
import numpy as np

class AlphaBeta:
	__max_depth = 4

	def __init__(self, select_value = 0):
		self.set_value_list(select_value)

	# マスごとの評価値を決定する(未完)
	def set_value_list(self, select_value):
		if select_value == 1:
			self.__value_list = np.ones(Board.action_size) * -1
		else:
			self.__value_list = np.ones(Board.action_size)

		self.__max_value = np.abs(self.__value_list).sum()
		self.__min_value = - self.__max_value
		return

	# 次の手を示す
	def get_next_move(self, board : Board):
		self.turn = board.turn
		return self.__first_max_node(board, self.__min_value, self.__max_value)

	# 評価関数
	def __evaluate(self, board : Board):
		value_black = 0

		for i in range(Board.action_size):
			value_black += self.__value_list[i] * \
				board.getbit_stone_exist(i) * (board.getbit_stone_black(i) - (1^board.getbit_stone_black(i)))

		if self.turn == 1:
			return value_black
		else:
			return - value_black

	# 評価が最大値となる場所を求める
	def __first_max_node(self, board : Board, alpha , beta):

		value = self.__min_value
		place_list = board.list_placable()
		if not place_list:
			print("error")
			return -1
		place_max = place_list[0]

		for i in place_list:
			with board.log_runtime(i):
				tmp_value = self.__node(board, 1, alpha, beta)
			if tmp_value >= beta:
				return tmp_value
			if tmp_value > alpha:
				alpha = tmp_value
				place_max = i

		return place_max

	# 評価を求める
	def __node(self, board : Board, depth, alpha, beta):
		if depth == AlphaBeta.__max_depth:
			return self.__evaluate(board)

		# 求める評価値が最大か最小か決定する
		ismax = (depth + 1) % 2

		if ismax:
			value = self.__min_value
		else:
			value = self.__max_value
		
		place_list = board.list_placable()
		if not place_list:
			board.turn_change()
			# 自分が打てないが相手が打てる場合
			if board.list_placable():
				value = self.__node(board, depth + 1, alpha, beta)
				board.turn_change()
				return value

			#ゲームが終了している場合
			else:
				board.turn_change()
				return self.__evaluate(board)

		for i in place_list:
			with board.log_runtime(i):
				tmp_value = self.__node(board, depth + 1, alpha, beta)

			if ismax:
				if tmp_value >= beta:
					return tmp_value
				if tmp_value > value:
					value = tmp_value
					if value > alpha:
						alpha = value
			else:
				if tmp_value <= alpha:
					return tmp_value
				if tmp_value < value:
					value = tmp_value
					if value < beta:
						beta = value

		return value




if __name__ == "__main__":
	def player(board):
		while 1:
			try:
				n = int(input("enter n : "))
				if board.is_placable(n):
					return n
			except:
				print("error")
				continue

	board = Board()
	# それぞれのプレイヤーの戦略の関数をわたす
	# プレイヤー先行でゲーム開始
	ab0 = AlphaBeta(0)
	ab1 = AlphaBeta(1)
	board.reset()
	board.set_plan(ab0.get_next_move, ab1.get_next_move)
	board.game()
