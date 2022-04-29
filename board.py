import numpy as np


class StepGenerator:
    def __init__(self, limit, startpoint, step):
        self.count = 0
        self.limit = limit
        self.value = startpoint
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.limit:
            raise StopIteration()

        self.count += 1
        self.value += self.step
        return self.value



class Board:
    def __init__(self):
        self.stone_exist = 0
        self.stone_black = 0
        self.turn = 1

        assert not self.height % 2 and self.height < 9
        assert not self.width % 2 and self.width < 9

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
        self.stone_black = (0b10 << shift) + (0b01 << (shift + self.width))
        self.turn = 1


    def __getbit(self, x, n):
        x = x >> n
        return x & 1

    def getbit_stone_exist(self, n):
        return self.__getbit(self.stone_exist, n)

    def getbit_stone_black(self, n):
        return self.__getbit(self.stone_black, n)

    def setbit_stone_exist(self, n):
        self.stone_exist |= 1 << n

    def setbit_stone_black(self, mask):
        self.stone_black ^= mask


    def is_placable(self, startpoint):
        up, left = divmod(startpoint, self.width)
        right = self.width - 1 - left
        down = self.height - 1 - up

        tuples = set()
        tuples.add((min(left, up), -9))
        tuples.add((up, -8))
        tuples.add((min(up, right), -7))
        tuples.add((left, -1))
        tuples.add((right, 1))
        tuples.add((min(down, left), 7))
        tuples.add((down, 8))
        tuples.add((min(right, down), 9))

        for limit, step in tuples:
            flag = False
            for n in StepGenerator(limit, startpoint, step):
                if self.getbit_stone_exist(n):
                    if self.getbit_stone_black(n) ^ self.turn:
                        flag = True
                        continue
                    if flag:
                        return True
                    else:
                        break
                else:
                    break
        return False

    def list_placable(self):
        p_list = []

        for n in range(self.action_size):
            if self.getbit_stone_exist(n):
                continue

            if self.is_placable(n):
                p_list.append(n)

        return p_list