

# =============================================================================
# ndarray の機能のために必要となる関数
# =============================================================================

# 入力データを指定された形状になるように集約して返す関数
def xp_sum_to(x, shape):
    # 次元数の違いは、先頭の次元から集約していくことで整合性を保つ (ブロードキャストのルールに基づく)
    lead = x.ndim - len(shape)
    lead_axis = tuple(range(lead))

    # 指定された形状の中でサイズが１である次元も集約対象とする
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(axis = lead_axis + axis, keepdims = True)

    # 次元数の違いによって集約された先頭の次元列を削除する (np.squeeze : 指定されたサイズが１の次元を削除)
    if lead:
        y = y.squeeze(lead_axis)
    return y


# DeZero の sum, average, max, min 関数の逆伝播の際に使われる
def reshape_for_broadcast(gy, in_shape, axis, keepdims):
    # そのままブロードキャストが使えるなら、何も手を加えずに返す
    ndim = len(in_shape)
    if not ndim or axis is None or keepdims:
        return gy

    if isinstance(axis, int):
        axis = (axis, )

    # sum 関数で集約され、順伝播時に消えてしまった次元をサイズ１として復元する (負の次元番号にも対応)
    actual_axis = [a if a >= 0 else a + ndim for a in axis]
    actual_shape = [1 if ax in actual_axis else s for ax, s in enumerate(in_shape)]

    return gy.reshape(actual_shape)




# =============================================================================
# CNN に使う機能のための関数
# =============================================================================

def get_conv_outsize(insize, filter_size, stride, padding):
    return (insize + 2 * padding - filter_size) // stride + 1


def get_deconv_outsize(insize, filter_size, stride, padding):
    OH, OW = insize
    FH, FW = filter_size
    SH, SW = pair(stride)
    PH, PW = pair(padding)
    return (SH * (OH - 1) + FH - 2 * PH, SW * (OW - 1) + FW - 2 * PW)


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, (tuple, list)):
        assert len(x) == 2
        return x
    else:
        raise TypeError("{} cannot be interpreted as pair".format(type(x)))