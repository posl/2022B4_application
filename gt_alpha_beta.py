from matplotlib.pyplot import get
from board import Board
import numpy as np

class GTValue:
	def __init__(self, select_value = 0):
		self.corner = select_value
		self.around_corner = select_value
		self.edge = select_value
		self.around_edge = select_value
		self.others = select_value
		self.place = select_value

		self.set_block_points()

		if False:
			self.set_data_dir("./data/gt/current/")

		if True:
			self.set_data_dir("./data/gt/valid/")

		if select_value == 0:
			self.reset()

	def reset(self):
		self.read_value_list("default_data")

	#評価に必要とする変数をリストとして返す
	def get_raw_value_list(self):
		return [self.corner, self.around_corner, self.edge, self.around_edge, self.others, self.place]
	
	#評価に必要とする変数を受け取る
	def set_raw_value_list(self, value_list):
		try:
			self.corner, self.around_corner, self.edge, self.around_edge, self.others, self.place\
			= value_list
			self.set_block_points()
		except:
			pass

	def set_data_dir(self, dir_name):
		self.dir_name = dir_name

	def write_value_list(self, file_name = "tmp_data"):
		if not "/" in file_name:
			file_name = self.dir_name + file_name
		with open(file_name, mode = "w") as f:
			f.write(" ".join(map(str, self.get_raw_value_list())))

	def read_value_list(self, file_name = "default_data"):
		if not "/" in file_name:
			file_name = self.dir_name + file_name
		with open(file_name, mode = "r") as f:
			tmp_value_list = list(map(float, f.read().split()))
		self.set_raw_value_list(tmp_value_list)
	
	#盤面の指定されたのマスの評価値を返す
	# def get_board_value(self, board_index):
	# 	x, y = divmod(board_index, Board.width)
	# 	x = min(x, Board.height - x - 1)
	# 	y = min(y, Board.width - y - 1)
	# 	if x == 0  and y == 0:
	# 		return self.corner
	# 	if x + y == 1:
	# 		return self.around_corner
	# 	if x == 0 or y == 0:
	# 		return self.edge
	# 	if x == 1 or y == 1:
	# 		return self.around_edge
	# 	return self.others

	def set_block_points(self):
		block_points = {}
		self.block_points = block_points

		for i in range(Board.action_size):
			x, y = divmod(i, Board.width)
			x = min(x, Board.height - x - 1)
			y = min(y, Board.width - y - 1)

			if x == 0  and y == 0:
				block_point = self.corner
			elif x + y == 1:
				block_point = self.around_corner
			elif x == 0 or y == 0:
				block_point = self.edge
			elif x == 1 or y == 1:
				block_point = self.around_edge
			else:
				block_point = self.others

			block_points[i] = block_point


	#黒側の評価値を返す
	def evaluate_black(self, board : Board, flag):
		#石の数の評価
		value_black = 0
		for i in range(board.action_size):
			if (board.stone_black >> i) & 1:
				value_black += self.block_points[i]
			elif (board.stone_white >> i) & 1:
				value_black -= self.block_points[i]
	
		#置ける場所の数の評価
		if False:
			value_black += (board.turn - (not board.turn)) * self.__eval_current_place(board)
		if True:
			if flag:
				flag = 2
			value_black += (board.turn - (not board.turn)) * self.__eval_valid_actions(board, flag)

		return value_black

	def __eval_current_place(self, board : Board):
		value = self.place * len(board.list_placable())
		board.turn = not board.turn
		value -= self.place * len(board.list_placable())
		board.turn = not board.turn
		return value

	# 置けるマスの数による評価
	def __eval_valid_actions(self, board: Board, flag = 2):
		# ゲームが終了した場合、置けるマスは０
		if not flag:
			return 0

		p_list = board.list_placable()
		p_length = len(p_list)

		# 手番が交代した場合は、相手の置けるマスの数にマイナスを掛けたものを評価点に使う
		if flag == 1:
			return -p_length

		p_value = 0
		for i in p_list:
			with board.log_runtime():
				board.put_stone(i)
				flag = board.can_continue()
				p_value += self.__eval_valid_actions(board, flag)

		# 置けるマスの数を評価点とし、実際に石を置いた後の、置けるマスの数による評価点の平均をそれに加算して、出力とする
		if p_length:
			p_length += p_value / p_length

		return p_length



class AlphaBeta:
	def __init__(self, select_value = 0):
		if type(select_value) == type(int()):
			self.value = GTValue(0)
		else:
			self.value = select_value
		self.__min_value = -999
		self.__max_value = 999
		self.set_depth(6)

	def __call__(self, board : Board):
		return self.get_next_move(board)

	def reset(self, select_value = 0):
		self.value = GTValue(select_value)
		self.set_depth(6)

	def set_depth(self, depth):
		self.__max_depth = depth

	# 次の手を示す
	def get_next_move(self, board : Board):
		self.turn = board.turn
		return int(self.__first_max_node(board, self.__min_value, self.__max_value))

	# 評価関数
	def __evaluate(self, board : Board, flag):
		if self.turn == 1:
			return self.value.evaluate_black(board, flag)
		else:
			return - self.value.evaluate_black(board, flag)

	# 評価が最大値となる場所を求める
	def __first_max_node(self, board : Board, alpha , beta):

		value = self.__min_value
		place_list = board.list_placable()

		place_max = place_list[0]

		for i in place_list:
			with board.log_runtime():
				board.put_stone(i)
				flag = board.can_continue()
				tmp_value = self.__node(board, 1, alpha, beta)
			if tmp_value > alpha:
				alpha = tmp_value
				place_max = i

		return place_max

	# 評価を求める
	def __node(self, board : Board, depth, alpha, beta, flag = 1):
		if depth == self.__max_depth:
			return self.__evaluate(board, flag)

		if flag == 0:
			return self.__evaluate(board, flag)

		# 求める評価値が最大か最小か決定する
		ismax = not (self.turn ^ board.turn)

		if ismax:
			value = self.__min_value
		else:
			value = self.__max_value

		place_list = board.list_placable()

		for i in place_list:
			with board.log_runtime():
				board.put_stone(i)
				flag = board.can_continue()
				tmp_value = self.__node(board, depth + 1, alpha, beta, flag)

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
	# ab0.value.read_value_list("./data/gt/self_match2")
	ab0.value.read_value_list("default_data")
	ab1.value.read_value_list("past_data")
	board.reset()
	# board.set_plan(ab0, ab1)
	# board.game()
	board.debug_game(ab0, ab1)
