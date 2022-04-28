import numpy as np


class Board:
    def __init__(self):
        self.stone_exist = 0
        self.stone_black = 0
        self.turn = False

        assert not self.height % 2
        assert not self.width % 2

    @property
    def height(self):
        return 8

    @property
    def width(self):
        return 8

    @property
    def action_size(self):
        return self.height * self.width

    @property
    def state(self):
        return np.array([self.stone_exist, self.stone_black])

    def t2n(self, t):
        return self.width * t[0] + t[1]

    def reset(self):
        shift = self.t2n((self.height // 2 - 1, self.width // 2 - 1))
        self.stone_exist = (0b11 << shift) + (0b11 << (shift + self.width))
        self.stone_black = (0b01 << shift) + (0b10 << (shift + self.width))
        self.turn = False

    def __getbit(self, x, t):
        n = self.t2n(t)
        x = x >> n
        return x & 1

    def getbit_stone_exist(self, t):
        return self.__getbit(self.stone_exist, t)

    def getbit_stone_black(self, t):
        return self.__getbit(self.stone_black, t)

    def setbit_stone_exist(self, n):
        self.stone_exist |= 1 << n

    def setbit_stone_black(self, mask):
        self.stone_black ^= mask