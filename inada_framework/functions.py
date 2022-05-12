from inada_framework import Function, cuda, as_array, as_variable, Variable, Config
import numpy as np
from inada_framework.utilitys import xp_sum_to, reshape_for_broadcast
try:
    import cupyx
except ImportError:
    pass


# =============================================================================
# 算術演算関数 (ブロードキャストに対応可能)
# =============================================================================

# 以降の関数にも共通するが、順伝播の引数は ndarray、逆伝播の引数は Variable であることに注意
class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy, gy
        if x0.shape != x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

def add(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Add()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy, -gy
        if x0.shape != x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

def sub(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy * x1, gy * x0
        if x0.shape != x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

def mul(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 / x1)
        if x0.shape != x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def forward(self, x0, x1):
        return x0 ** x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * (x1 * x0 ** (x1 - 1.0))
        gx1 = gy * (x0 ** x1 * log(x0))
        if x0.shape != x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

def pow(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Pow()(x0, x1)

def rpow(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0.data))
    return Pow()(x1, x0)




# =============================================================================
# 数学関数
# =============================================================================

class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.sin(x)

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.cos(x)

    def backward(self, gy):
        x, = self.inputs
        return gy * -sin(x)

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y * y)

def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y

def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.log(x)

    def backward(self, gy):
        x, = self.inputs
        return gy / x

def log(x):
    return Log()(x)




# =============================================================================
# テンソル操作
# =============================================================================

# スライス表記で Variable インスタンスから要素を取り出すときに使う Function インスタンス
class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        return x[self.slices]

    def backward(self, gy):
        return GetItemGrad(self.slices, self.inputs[0].shape)(gy)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    # 順伝播時に取り出した要素に対応する勾配だけを次へ流す
    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype = gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            cupyx.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return GetItem(self.slices)(ggx)

def get_item(x, slices):
    return GetItem(slices)(x)


class Reshape(Function):
    def __init__(self, shape):
        self.out_shape = shape

    def forward(self, x):
        return x.reshape(self.out_shape)

    def backward(self, gy):
        return reshape(gy, self.inputs[0].shape)

def reshape(x, *shape):
    # 引数はタプル・リスト・可変引数に対応する
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    # Function インスタンスを経由しないので、Variable インスタンスに変換する
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def flatten(x):
    # バッチ方向には影響を及ぼさないように平板化する
    return reshape(x, (len(x), -1))

# 要素数が１の指定した次元を追加する関数
def expand_dims(x, axis):
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, shape)


class Transpose(Function):
    def __init__(self, axes):
        self.axes = axes

    def forward(self, x):
        return x.transpose(self.axes)

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        # 次元番号の配列をソートした時のインデックスによって、順伝播時と逆の操作を行う
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

def transpose(x, *axes):
    if not len(axes):
        return Transpose(None)(x)
    if len(axes) == 1 and (isinstance(axes[0], (tuple, list)) or axes[0] is None):
        axes = axes[0]
    return Transpose(axes)(x)


class Concatenate(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, *arrays):
        xp = cuda.get_array_module(arrays[0])
        return xp.concatenate(arrays, self.axis)

    def backward(self, gy):
        arrays = self.inputs
        N = len(arrays) - 1

        # 配列が指定された軸のどこで分割されるかを表すインデックスの配列を作る
        indices = [0] * N
        index_count = 0
        for i in range(N):
            index_count += arrays[i].shape[self.axis]
            indices[i] = index_count

        return Split(indices, self.axis)(gy)

# arrays は連結したい配列のリスト
def concatenate(arrays, axis = 0):
    if axis < 0:
        axis += arrays[0].ndim
    return Concatenate(axis)(*arrays)


class Split(Function):
    def __init__(self, indices, axis):
        self.indices = indices
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.split(x, self.indices, self.axis)

    def backward(self, *garrays):
        return Concatenate(self.axis)(*garrays)

def split(x, indices_or_sections, axis = 0):
    # 分割後の配列のセクション数が与えられたときは、インデックスの配列に変換する
    if isinstance(indices_or_sections, int):
        N = x.shape[axis]
        q, r = divmod(N, indices_or_sections)
        assert not r
        indices_or_sections = np.arange(q, N, q, dtype = np.int64)

    # Function インスタンスの出力の仕様によって、順伝播後に長さ１のリストは返せないので必要と分岐
    if not len(indices_or_sections):
        return [as_variable(x)]
    if axis < 0:
        axis += x.ndim
    return Split(indices_or_sections, axis)(x)




# =============================================================================
# テンソル演算
# =============================================================================

class BroadcastTo(Function):
    def __init__(self, shape):
        self.out_shape = shape

    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.broadcast_to(x, self.out_shape)

    def backward(self, gy):
        return sum_to(gy, self.inputs[0].shape)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.out_shape = shape

    def forward(self, x):
        return xp_sum_to(x, self.out_shape)

    def backward(self, gy):
        return broadcast_to(gy, self.inputs[0].shape)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return x.sum(axis = self.axis, keepdims = self.keepdims)

    def backward(self, gy):
        # 必要ならば、ブロードキャストできるように変形する
        in_shape = self.inputs[0].shape
        gy = reshape_for_broadcast(gy, in_shape, self.axis, self.keepdims)
        return broadcast_to(gy, in_shape)

def sum(x, axis = None, keepdims = False):
    return Sum(axis, keepdims)(x)


class Average(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return x.mean(axis = self.axis, keepdims = self.keepdims)

    def backward(self, gy):
        x = self.inputs[0].data
        gy *= (gy.size / x.size)

        gy = reshape_for_broadcast(gy, x.shape, self.axis, self.keepdims)
        return broadcast_to(gy, x.shape)

def average(x, axis = None, keepdims = False):
    return Average(axis, keepdims)(x)

mean = average


class Max(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return x.max(axis = self.axis, keepdims = self.keepdims)

    def backward(self, gy):
        x = self.inputs[0].data
        y = self.outputs[0]().data

        # 必要ならば、ブロードキャストできるように変形する (gy と y は同じ形状)
        gy = reshape_for_broadcast(gy, x.shape, self.axis, self.keepdims)
        y = y.reshape(gy.shape)

        gy = broadcast_to(gy, x.shape)
        return gy * (x == y)

def max(x, axis = None, keepdims = False):
    return Max(axis, keepdims)(x)


class Min(Max):
    def forward(self, x):
        return x.min(axis = self.axis, keepdims = self.keepdims)

def min(x, axis = None, keepdims = False):
    return Min(axis, keepdims)(x)


# x_min 以下であれば x_min に、x_max 以上であれば x_max にして、全要素を [x_min, x_max] の範囲に収める
class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.clip(x, self.x_min, self.x_max)

    def backward(self, gy):
        x = self.inputs[0].data
        return gy * (x >= self.x_min and x <= self.x_max)

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)


class Affine(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def affine(x, W, b = None):
    return Affine()(x, W, b)




# =============================================================================
# 活性化関数 (出力層のものも含む)
# =============================================================================

# tanh を使う実装の方が速い
class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.tanh(x * 0.5) * 0.5 + 0.5

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y * (1 - y)

def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.maximum(x, 0.0)

    def backward(self, gy):
        x, = self.inputs
        return gy * (x.data > 0)

def relu(x):
    return ReLU()(x)


# 0 以下の要素も少しだけ流すようにした ReLU 関数 (leak : 漏れる)
class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        return gy * mask

def leaky_relu(x, slope = 0.2):
    return LeakyReLU(slope)(x)


def abs(x):
    return LeakyReLU(-1)(x)


class Softmax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)

        # オーバーフロー対策に最大要素との差を取っている (逆伝播には影響しない)
        y = x - xp.max(x, axis = self.axis, keepdims = True)
        y = xp.exp(y)
        y /= xp.sum(y, axis = self.axis, keepdims = True)
        return y

    # 計算グラフを辿れば納得できる
    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = sum(gx, axis = self.axis, keepdims = True)
        gx -= y * sumdx
        return gx

def softmax(x, axis = 1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis):
        self.axis = axis

    # axis 方向で対数尤度を考えたときの、各項をその軸に配置して返す
    def forward(self, x):
        xp = cuda.get_array_module(x)

        # オーバーフロー対策 (この辻褄合わせとして、後で x_max を足す (式変形で導出できる))
        x_max = xp.max(x, axis = self.axis, keepdims = True)
        y = x - x_max

        # log(sum(exp(y)))
        y = xp.exp(y)
        y = xp.sum(y, axis = self.axis, keepdims = True)
        y = xp.log(y)
        y += x_max

        return x - y

    # exp(y) を式変形すれば納得できる (オーバーフロー対策の部分は考慮しなくてよい)
    def backward(self, gy):
        y = self.outputs[0]()
        return gy - exp(y) * sum(gy, axis = self.axis, keepdims = True)

def log_softmax(x, axis = 1):
    return LogSoftmax(axis)(x)




# =============================================================================
# 損失関数
# =============================================================================

class MeanSquaredError(Function):
    def forward(self, x, t):
        t = t.reshape(x.shape)
        diff = x - t
        return 0.5 * (diff * diff).sum() / len(diff)

    def backward(self, gy):
        x, t = self.inputs
        x, t = x.data, t.data
        t = t.reshape(x.shape)
        diff = x - t
        gdiff = gy * (diff / len(diff))
        return gdiff, -gdiff

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        xp = cuda.get_array_module(x)

        # Softmax
        y = x - xp.max(x, axis = 1, keepdims = True)
        y = xp.exp(y)
        self.y = y / xp.sum(y, axis = 1, keepdims = True)

        # Cross-Entropy-Error
        N = len(x)
        y = self.y[np.arange(N), t.ravel()]
        y = xp.clip(y, a_min = 1e-15, a_max = None)
        y = xp.log(y)
        return -xp.sum(y) / N

    def backward(self, gy):
        x, t = self.inputs
        N, O = x.shape

        # GPU を活かすために、One-Hot-Vector 型に変換してから計算する (eye : 単位行列を生成する)
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(O, dtype = t.dtype)[t.data.ravel()]
        gx = gy * ((self.y - t_onehot) / N)

        # 不要となった配列をメモリから消去する
        self.y = None
        return gx

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


# ２値分類問題の Cross-Entropy-Error
def binary_cross_entropy(p, t):
    if isinstance(t, Variable):
        t = t.data
    t = t.reshape(p.shape)
    p = clip(p, 1e-15, 1.0 - 1e-15)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    return -sum(tlog_p) / len(p)


def sigmoid_cross_entropy(x, t):
    return binary_cross_entropy(sigmoid(x), t)




# =============================================================================
# ニューラルネットワークの学習上のテクニック
# =============================================================================

class Dropout(Function):
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.scale = None

    def forward(self, x):
        if Config.train_flag:
            xp = cuda.get_array_module(x)
            mask = (xp.random.rand(*x.shape) > self.dropout_ratio)
            self.scale = mask / xp.array(1.0 - self.dropout_ratio).astype(x.dtype)
            return x * self.scale
        return x

    def backward(self, gy):
        if self.scale is None:
            return gy
        return gy * self.scale

def dropout(x, dropout_ratio = 0.5):
    return Dropout(dropout_ratio)(x)


class BatchNorm(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps

        self.inv_std = None

    def forward(self, x, gamma, beta):
        x_ndim = x.ndim
        assert x_ndim in (2, 4)

        xp = cuda.get_array_module(x)

        tensor_flag = (x_ndim == 4)
        if tensor_flag:
            # (N, C, H, W) -> (N * H * W, C)
            N, C, H, W = x.shape
            x = xp.reshape(xp.moveaxis(x, source = 1, destination = 3), (-1, C))

        if Config.train_flag:
            # バッチ方向の平均・分散・標準偏差の逆数
            mean = xp.mean(x, axis = 0)
            var = xp.mean((x - mean) ** 2, axis = 0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            # テスト時に使う平均・分散
            m = x.size // gamma.size
            s = m - 1.0 if m > 2 else 1.0
            self.avg_mean += (1 - self.decay) * (mean - self.avg_mean)
            self.avg_var += (1 - self.decay) * ((m / s) * var - self.avg_var)

            # 逆伝播時に使うデータ
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std

        # 正規化とスケール・シフト変換を行ったデータを出力する
        y = gamma * xc + beta

        if tensor_flag:
            # (N * H * W, C) -> (N, C, H, W)
            y = xp.moveaxis(xp.reshape(y, (N, H, W, C)), source = 3, destination = 1)
        return y

    def backward(self, gy):
        x, gamma, _ = self.inputs
        x, gamma = x.data, gamma.data

        tensor_flag = (gy.ndim == 4)
        if tensor_flag:
            N, C, H, W = gy.shape
            gy = reshape(transpose(gy, (0, 2, 3, 1)), (-1, C))
            x = x.transpose((0, 2, 3, 1)).reshape(-1, C)

        # メモリの消費量を抑えるために、正規化は再度計算する (標準偏差は計算量を鑑みて、保存しておく)
        xc = (x - x.mean(axis = 0)) * self.inv_std

        gbeta = sum(gy, axis = 0)
        ggamma = sum(xc * gy, axis = 0)
        gx = gy - (gbeta + xc * ggamma) * (1 / len(gy))
        gx *= gamma * self.inv_std

        if tensor_flag:
            gx = transpose(reshape(gx, (N, H, W, C)), (0, 3, 1, 2))
        return gx, ggamma, gbeta


def batch_nrom(x, gamma, beta, mean, var, decay = 0.9, eps = 2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)




# =============================================================================
# その他
# =============================================================================

# inada
#_framework.functions から cnn, rnn 用の関数もインポートできるようにする
from inada_framework.functions_cnn import conv2d
from inada_framework.functions_cnn import deconv2d
from inada_framework.functions_cnn import conv2d_1x1filter
from inada_framework.functions_cnn import pooling
from inada_framework.functions_cnn import average_pooling
from inada_framework.functions_cnn import global_average_pooling
from inada_framework.functions_cnn import im2col
from inada_framework.functions_cnn import col2im

from inada_framework.functions_rnn import lstm
from inada_framework.functions_rnn import gru
