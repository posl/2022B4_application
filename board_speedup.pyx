# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True

from libcpp.vector cimport vector
from libc.time cimport time, time_t

ctypedef unsigned long long uint



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

def get_legal_board(uint move_player, uint opposition_player):
    return __get_legal_board(move_player, opposition_player)




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

def get_reverse_board(uint set_position, uint move_player, uint opposition_player):
    return __get_reverse_board(set_position, move_player, opposition_player)




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




# １が立っているビットの数を数える
cdef inline uint __count_stand_bits(uint n):
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




# １が立っているビット位置のリストを取得する
def get_stand_bits(int num, uint x):
    cdef:
        int n
        vector[int] l = []

    for n in range(num):
        if x & 1:
            l.push_back(n)
        x >>= 1

    return l




cdef inline uint __alpha_beta(uint move_player, uint opposition_player, int depth, time_t limit_time):
    cdef:
        uint mask, legal_board, put
        int value, n

    legal_board = __get_legal_board(move_player, opposition_player)

    if legal_board == 0:
        if __get_legal_board(opposition_player, move_player) == 0:
            return __count_stand_bits(move_player) - __count_stand_bits(opposition_player)
        else:
            return - __alpha_beta(opposition_player, move_player, 1, limit_time)

    for n in range(64):
        if time(NULL) > limit_time:
            return -100

        if (legal_board >> n) & 1:
            put =  <uint> 1 << n

            mask = __get_reverse_board(put, move_player, opposition_player)

            value = - __alpha_beta(opposition_player ^ mask, move_player ^ mask ^ put, 1, limit_time)

            if value == 100:
                return -100

            if value > 0:
                if depth:
                    return value
                else:
                    return n

    return -1

def alpha_beta(uint move_player, uint opposition_player, time_t limit_time):
    return __alpha_beta(move_player, opposition_player, 0, time(NULL) + limit_time)