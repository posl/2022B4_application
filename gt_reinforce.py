from gt_alpha_beta import AlphaBeta
from gt_alpha_beta import GTValue
from board import Board
import numpy as np
from random import random
from drl_selfmatch import corners_plan as simple_plan

# 動作確認未完了


class GTTDAgent:
    def __init__(self):
        self.__alpha = 0.03
        self.data = GTValue(1)
        self.data.reset()
        self.new_data = GTValue()
        self.__creat_new_data()

    def __creat_new_data(self):
        self.new_data.set_raw_value_list([i + np.random.random() - 0.5 for i in self.data.get_raw_value_list()])

    def reset(self):
        self.data.reset()
        self.__creat_new_data()

    def update(self, reward):
        data_list = self.data.get_raw_value_list()
        new_data_list = self.new_data.get_raw_value_list()
        self.data.set_raw_value_list([data_list[i] + self.__alpha * reward * (new_data_list[i] - data_list[i]) for i in range(len(data_list))])
        self.__creat_new_data()

    def get_data(self):
        return self.data.get_raw_value_list()

    def get_new_data(self):
        return self.new_data.get_raw_value_list()

class GTReinforce:
    def __init__(self):
        self.__agent = GTTDAgent()
        self.player1 = AlphaBeta(0)

    def reset(self):
        self.__agent.reset()

    def set_player1(self, player):
        self.player1 = player

    def start(self, repeat_num = 20):
        board = Board()
        sum_reward = 0
        for i in range(repeat_num):
            board.reset()
            agent_alphabeta = AlphaBeta(self.__agent.new_data)
            # board.set_plan(agent_alphabeta.get_next_move, self.player1)
            board.set_plan(self.player1, agent_alphabeta.get_next_move)
            board.game()
            diff = board.black_num - board.white_num
            if diff:
                if (diff > 0):
                  reward =  1
                else:
                    reward = -1
            else:
                reward = 0

            self.__agent.update(-reward)
            sum_reward += -reward
        print(sum_reward)
        return self.__agent.get_data()
    
    def makefile(self, filename):
        f = open(filename, "w")
        f.write(self, self.__agent.get_data())

if __name__ == "__main__":
    gtr = GTReinforce()
    gtr.reset()
    # ab = AlphaBeta(0)
    # gtr.set_player1(ab.get_next_move)
    gtr.set_player1(simple_plan)
    while 1:
        print(gtr.start(50))
