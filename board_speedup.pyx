# distutils: language = c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level = 3, boundscheck = False, wraparound = False
# cython: cdivision = True

from libcpp.vector cimport vector

ctypedef unsigned long long uint



# 引数は手番または、相手プレイヤーの石が置かれた箇所だけ１が立った、ボードを表す 64 bit 符号無し整数
def get_legal_board(uint move_player, uint opposition_player):
    return __get_legal_board(move_player, opposition_player)


# 合法手の箇所だけ１が立った、ボードを表す 64 bit 符号無し整数を返す (ブラックボックス化するため、反復処理は展開している)
cdef inline uint __get_legal_board(uint move_player, uint opposition_player):
    # ちゃんと型宣言をした方が速い
    cdef uint mask, legal

    # 左右方向の探索に使う、オセロ盤の水平方向の両端以外で相手の石が置かれている箇所に１が立ったマスク
    mask =  0x7e7e7e7e7e7e7e7e
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




# 第１引数は手番プレイヤーによって新たに石が置かれた箇所だけ１が立った、ボードを表す 64 bit 符号無し整数
def get_reverse_board(uint set_position, uint move_player, uint opposition_player):
    return __get_reverse_board(set_position, move_player, opposition_player)


# 反転するマスだけ１が立った、ボードを表す 64 bit 符号無し整数を返す
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
    mask =  0x007e7e7e7e7e7e00
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
def get_stand_bits(int num, uint x):
    cdef:
        int n
        vector[int] l = []

    for n in range(num):
        if x & 1:
            l.push_back(n)
        x >>= 1

    return l