from matplotlib.pyplot import get
from board import Board
import numpy as np

class GTValue:
	def __init__(self, select_value = 1):
		self.corner = select_value
		self.around_corner = select_value
		self.edge = select_value
		self.around_edge = select_value
		self.others = select_value
		self.place = select_value

	def reset(self):
		self.corner = 0
		self.around_corner = 0
		self.edge = 0
		self.around_edge = 0
		self.others = 0
		self.place = 0

	#評価に必要とする変数をリストとして返す
	def get_raw_value_list(self):
		return [self.corner, self.around_corner, self.edge, self.around_edge, self.others, self.place]
	
	#評価に必要とする変数を受け取る
	def set_raw_value_list(self, value_list):
		try:
			self.corner, self.around_corner, self.edge, self.around_edge, self.others, self.place\
			= value_list
		except:
			pass
	
	#盤面の指定されたのマスの評価値を返す
	def get_board_value(self, board_index):
		x, y = divmod(board_index, Board.width)
		x = min(x, Board.height - x)
		y = min(y, Board.width - y)
		if x == 0  and y == 0:
			return self.corner
		if x + y == 1:
			return self.around_corner
		if x == 0 or y == 0:
			return self.edge
		if x == 1 or y == 1:
			return self.around_edge
		return self.others

	#黒側の評価値を返す
	def evaluate_black(self, board : Board):
		#石の数の評価
		value_black = 0
		for i in range(Board.action_size):
			value_black += self.get_board_value(i) * \
				board.getbit_stone_exist(i) * (board.getbit_stone_black(i) - (1 ^ board.getbit_stone_black(i)))
	
		#置ける場所の数の評価	
		tmp_turn = board.turn
		board.turn = 1
		value_black += self.place * len(board.list_placable())
		board.turn = 0
		value_black -= self.place * len(board.list_placable())
		board.turn = tmp_turn
		return value_black


class AlphaBeta:
	def __init__(self, select_value = 0):
		if type(select_value) == type(int()):
			self.value = GTValue(select_value)
		else:
			self.value = select_value
		self.__min_value = -999
		self.__max_value = 999
		self.set_depth(2)

	def __call__(self, board : Board):
		return self.get_next_move(board)

	def reset(self, select_value = 0):
		self.value = GTValue(select_value)
		self.set_depth(2)

	def set_depth(self, depth):
		self.__max_depth = depth

	# 次の手を示す
	def get_next_move(self, board : Board):
		self.turn = board.turn
		return int(self.__first_max_node(board, self.__min_value, self.__max_value))

	# 評価関数
	def __evaluate(self, board : Board):
		if self.turn == 1:
			return self.value.evaluate_black(board)
		else:
			return - self.value.evaluate_black(board)

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
		if depth == self.__max_depth:
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
	board.set_plan(ab0, ab1)
	board.game()
