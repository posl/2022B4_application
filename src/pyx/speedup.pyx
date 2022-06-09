# distutils: language = c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level = 3, boundscheck = False, wraparound = False
# cython: cdivision = True

from libcpp.vector cimport vector
from libc.time cimport time, time_t

import numpy as np
cimport numpy as cnp
cimport cython
from cython import boundscheck, wraparound


ctypedef unsigned long long uint

ctypedef cnp.float32_t F32
ctypedef cnp.float64_t F64



# =============================================================================
# オセロボードの機能の高速化
# =============================================================================

# 合法手の箇所だけ１が立った、ボードを表す 64 bit 符号無し整数を返す (ブラックボックス化するため、反復処理は展開している)
# 引数は手番または、相手プレイヤーの石が置かれた箇所だけ１が立った、ボードを表す 64 bit 符号無し整数
cdef inline uint __get_legal_board(uint move_player, uint opposition_player):
    # ちゃんと型宣言をした方が速い
    cdef uint mask, legal

    # 左右方向の探索に使う、オセロ盤の水平方向の両端以外で相手の石が置かれている箇所に１が立ったマスク
    mask = 0x7e7e7e7e7e7e7e7e
    mask &= opposition_player
    legal = search_upper_legal(move_player, 1, mask)
    legal |= search_lower_legal(move_player, 1, mask)

    # 上下方向の探索に使う、オセロ盤の垂直方向の両端以外で相手の石が置かれている箇所に１が立ったマスク
    mask = 0x00ffffffffffff00
    mask &= opposition_player
    legal |= search_upper_legal(move_player, 8, mask)
    legal |= search_lower_legal(move_player, 8, mask)

    # 斜め方向の探索に使う、オセロ盤の全端辺以外で相手の石が置かれている箇所に１が立ったマスク
    mask = 0x007e7e7e7e7e7e00
    mask &= opposition_player
    legal |= search_upper_legal(move_player, 9, mask)
    legal |= search_lower_legal(move_player, 9, mask)
    legal |= search_upper_legal(move_player, 7, mask)
    legal |= search_lower_legal(move_player, 7, mask)

    # 空白箇所だけに絞って、出力を得る
    return legal & (~(move_player | opposition_player))

def get_legal_list(uint move_player, uint opposition_player):
    return __get_stand_bits(__get_legal_board(move_player, opposition_player))




# 反転するマスだけ１が立った、ボードを表す 64 bit 符号無し整数を返す
# 第１引数は手番プレイヤーによって新たに石が置かれた箇所だけ１が立った、ボードを表す 64 bit 符号無し整数
cdef inline uint __get_reverse_board(uint set_position, uint move_player, uint opposition_player):
    cdef uint mask, tmp, reverse
    reverse = 0

    # 左右方向
    mask = 0x7e7e7e7e7e7e7e7e
    mask &= opposition_player
    tmp = search_upper(set_position, 1, mask)
    if move_player & (tmp >> 1):
        reverse |= tmp
    tmp = search_lower(set_position, 1, mask)
    if move_player & (tmp << 1):
        reverse |= tmp

    # 上下方向
    mask = 0x00ffffffffffff00
    mask &= opposition_player
    tmp = search_upper(set_position, 8, mask)
    if move_player & (tmp >> 8):
        reverse |= tmp
    tmp = search_lower(set_position, 8, mask)
    if move_player & (tmp << 8):
        reverse |= tmp

    # 斜め方向
    mask = 0x007e7e7e7e7e7e00
    mask &= opposition_player
    tmp = search_upper(set_position, 9, mask)
    if move_player & (tmp >> 9):
        reverse |= tmp
    tmp = search_lower(set_position, 9, mask)
    if move_player & (tmp << 9):
        reverse |= tmp
    tmp = search_upper(set_position, 7, mask)
    if move_player & (tmp >> 7):
        reverse |= tmp
    tmp = search_lower(set_position, 7, mask)
    if move_player & (tmp << 7):
        reverse |= tmp

    return reverse

def get_reverse_board(int startpoint, uint move_player, uint opposition_player):
    return __get_reverse_board(<uint>1 << startpoint, move_player, opposition_player)




# 手番プレイヤーの石から見て、指定方向に相手の石が連続していた場合、その終端のもう１つ先が手番にとっての合法手である
cdef inline uint search_upper_legal(uint tmp, int n, uint mask):
    tmp = search_upper(tmp, n, mask)
    return tmp >> n

cdef inline uint search_lower_legal(uint tmp, int n, uint mask):
    tmp = search_lower(tmp, n, mask)
    return tmp << n


# ボードを表す整数を手前にシフトする方向に探索する
cdef inline uint search_upper(uint tmp, int n, uint mask):
    # 手番プレイヤーの石から見て、指定方向に連続する相手の石の箇所を得る
    tmp = mask & (tmp >> n)
    tmp |= mask & (tmp >> n)
    tmp |= mask & (tmp >> n)
    tmp |= mask & (tmp >> n)
    tmp |= mask & (tmp >> n)
    tmp |= mask & (tmp >> n)
    return tmp

# ボードを表す整数を奥にシフトする方向に探索する
cdef inline uint search_lower(uint tmp, int n, uint mask):
    tmp = mask & (tmp << n)
    tmp |= mask & (tmp << n)
    tmp |= mask & (tmp << n)
    tmp |= mask & (tmp << n)
    tmp |= mask & (tmp << n)
    tmp |= mask & (tmp << n)
    return tmp




# １が立っているビット位置のリストを取得する
cdef inline vector[int] __get_stand_bits(uint x):
    cdef:
        int n = 0
        vector[int] l

    while x:
        if x & 1:
            l.push_back(n)
        n += 1
        x >>= 1

    return l

def get_stand_bits(uint x):
    return __get_stand_bits(x)




# １が立っているビットの数を数える
cdef inline int __count_stand_bits(uint n):
    cdef uint mask

    # 2 bit ごとにブロック分けして、それぞれのブロックにおいて１が立っているビット数を各ブロックの 2 bit で表現する
    mask = 0x5555_5555_5555_5555
    n -= (n >> 1) & mask

    # 4 bit ごとにブロック分けして、各ブロックに 上位 2 bit + 下位 2 bit を計算した値を入れる
    mask = 0x3333_3333_3333_3333
    n = (n & mask) + ((n >> 2) & mask)

    # 8 bit ごとにブロック分けして、各ブロックに 上位 4 bit + 下位 4 bit を計算した値を入れる
    mask = 0x0f0f_0f0f_0f0f_0f0f
    n += n >> 4
    n &= mask

    # 以下、同様
    n += n >> 8
    n += n >> 16
    n += n >> 32

    # 0 ~ 64 を表現するために、下位 7 bit のみを取り出して出力とする
    mask = 0x7f
    return n & mask

def count_stand_bits(uint n):
    return __count_stand_bits(n)




# =============================================================================
# nega_alpha (nega_max + 枝刈り)
# =============================================================================

# flag -1: 初回呼び出し、0: 正常、１: パスした
cdef inline int __nega_alpha(uint move_player, uint opposition_player, int flag, time_t limit_time):
    cdef:
        uint mask, legal_board, put
        int value, n = 0

    legal_board = __get_legal_board(move_player, opposition_player)

    if legal_board == 0:
        if flag:
            return  __count_stand_bits(move_player) - __count_stand_bits(opposition_player)

        else:
            return - __nega_alpha(opposition_player, move_player, 1, limit_time)

    while legal_board:
        if time(NULL) > limit_time:
            return -100

        if legal_board & 1:
            put =  <uint>1 << n
            mask = __get_reverse_board(put, move_player, opposition_player)

            value = - __nega_alpha(opposition_player ^ mask, move_player ^ mask ^ put, 0, limit_time)

            if value == 100:
                return -100

            if value > 0:
                if flag == -1:
                    return n
                else:
                    return value

        n += 1
        legal_board >>= 1

    return -1

def nega_alpha(uint move_player, uint opposition_player, time_t limit_time):
    return __nega_alpha(move_player, opposition_player, -1, time(NULL) + limit_time)




# =============================================================================
# 深層強化学習で使用する関数・メソッドの高速化
# =============================================================================

cpdef inline cnp.ndarray[F32, ndim = 3] get_board_img(uint move, uint opposition, int height, int width):
    cdef:
        cnp.ndarray[F32, ndim = 3] A
        int i, j, n

    with boundscheck(False), wraparound(False):
        A = np.empty((2, height, width), dtype = np.float32)

        for i in range(height):
            for j in range(width):
                n = width * i + j
                A[0, i, j] = (move >> n) & 1
                A[1, i, j] = (opposition >> n) & 1

    return A




cpdef inline int get_qmax_action(cnp.ndarray[F32, ndim = 1] qs, vector[int] placable):
    cdef:
        float q, qmax
        int action, qmax_action

    qmax = -float("inf")
    qmax_action = 0

    with boundscheck(False), wraparound(False):
        for action in placable:
            q = qs[action]
            if qmax < q:
                qmax = q
                qmax_action = action

    return qmax_action




cpdef inline void update_sumtree(cnp.ndarray[F64, ndim = 1] tree, int index, float value):
    cdef int left_child

    with boundscheck(False), wraparound(False):
        index += len(tree) >> 1
        tree[index] = value

        # 親ノードに２つの子ノードの和が格納されている状態を保つように更新する (インデックス１が最上位の親ノード)
        while index > 1:
            index >>= 1
            left_child = index << 1
            tree[index] = tree[left_child] + tree[left_child + 1]




cpdef inline vector[int] weighted_sampling(cnp.ndarray[F64, ndim = 1] tree, cnp.ndarray[F64, ndim = 1] zs):
    cdef:
        int capacity, i, index, left_child
        float z, left_value
        vector[int] l

    capacity = len(tree) >> 1

    with boundscheck(False), wraparound(False):
        for i in range(len(zs)):
            index = 1
            z = zs[i]

            while index < capacity:
                left_child = index << 1

                left_value = tree[left_child]
                if z > left_value:
                    index = left_child + 1
                    z -= left_value
                else:
                    index = left_child

            l.push_back(index - capacity)

    return l