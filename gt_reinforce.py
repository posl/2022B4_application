from gt_alpha_beta import AlphaBeta
from gt_alpha_beta import GTValue
from board import Board
import numpy as np
from random import random



class GTTDAgent:
    def __init__(self, select_place_func = 0):
        self.__alpha = 0.03
        self.data = GTValue(select_place_func)
        self.data.reset()
        self.new_data = GTValue(select_place_func)
        self.__creat_new_data()

    def __creat_new_data(self):
        self.new_data.set_raw_value_list([i + min(0.3 , 1 - i, i - (-1)) * (np.random.random() * 2 - 1) for i in self.data.get_raw_value_list()])

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
    def __init__(self, place_func = 0):
        self.place_func = place_func
        self.agent = GTTDAgent(place_func)
        self.player1 = AlphaBeta(0)
        self.depth = 6  

    def reset(self):
        self.agent.reset()

    def set_player1(self, player):
        self.player1 = player

    def set_depth(self, depth):
        self.depth = depth

    def start(self, repeat_num = 20):
        board = Board()
        sum_reward = 0
        for i in range(repeat_num):
            board.reset()
            agent_alphabeta = AlphaBeta(self.place_func, self.agent.new_data)
            agent_alphabeta.set_depth(self.depth)
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
            print(-reward)
            self.agent.update(-reward)
            sum_reward += -reward

            board.reset()
            agent_alphabeta = AlphaBeta(self.agent.new_data)
            agent_alphabeta.set_depth(self.depth)
            board.set_plan(agent_alphabeta.get_next_move, self.player1)
            board.game()
            diff = board.black_num - board.white_num
            if diff:
                if (diff > 0):
                  reward =  1
                else:
                    reward = -1
            else:
                reward = 0

            self.agent.update(reward)
            sum_reward += reward




        print(sum_reward)
        return self.agent.get_data()


if __name__ == "__main__":
    # def random_plan(board : Board):
    #     place_list = board.list_placable()
    #     num = int(np.random.random()  * len(place_list))
    #     if num == len(place_list):
    #         num = 0
    #     return place_list[num]
        

    # gtr = GTReinforce()
    # gtr.reset()
    # gtr.set_player1(random_plan)
    # while 1:
    #     print(gtr.start(1))
    #     gtr.agent.data.write_value_list()
    place_func = 1

    gtr1 = GTReinforce(place_func)
    gtr1.reset()
    gtr1.set_depth(6)
    gtr1.agent.data.read_value_list("self_match1")
    gtr1.agent.update(0)
    gtr1_file_path = "self_match1"
    gtr2 = GTReinforce(place_func)
    gtr2.reset()
    gtr2.set_depth(6)
    gtr2.agent.data.read_value_list("self_match2")
    gtr2.agent.update(0)
    gtr2_file_path = "self_match2"

    
    while 1:
        tmp_gtr2_ab = AlphaBeta(place_func, gtr2.agent.data)
        tmp_gtr2_ab.set_depth(6)
        gtr1.set_player1(tmp_gtr2_ab)
        print(gtr1.start(1))
        gtr1.agent.data.write_value_list(gtr1_file_path)
        gtr1, gtr2 = gtr2, gtr1
        gtr1_file_path, gtr2_file_path = gtr2_file_path, gtr1_file_path

