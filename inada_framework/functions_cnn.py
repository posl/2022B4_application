from inada_framework import Function, cuda
from inada_framework.utilitys import get_deconv_outsize, pair, get_conv_outsize
from inada_framework.functions import broadcast_to
import numpy as np


# =============================================================================
# 畳み込み層 (conv2d, conv2d_1x1filter)
# =============================================================================

class Conv2d(Function):
    def __init__(self, stride, padding):
        self.stride = stride
        self.padding = padding

    def forward(self, x, W, b = None):
        xp = cuda.get_array_module(x)

        # x : (N, C, H, W)  ->  col : (N, C, FH, FW, OH, OW)
        col = im2col_ndarray(x, W.shape[2:], self.stride, self.padding)

        # col : (N, C, FH, FW, OH, OW)  *  W : (OC, C, FH, FW)  +  b : (OC)  ->  y : (N, OH, OW, OC)
        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b

        # xp.transpose(y, (0, 3, 1, 2)) と等価
        return xp.moveaxis(y, source = 3, destination = 1)

    def backward(self, gy):
        x, W, b = self.inputs

        # Deconv2d の順伝播が Conv2d の逆伝播の入力画像データの勾配計算部分
        gx = Deconv2d(self.stride, self.padding, img_shape = x.shape)(gy, W)
        gW = Conv2dGradW(self.stride, self.padding, filter_size = W.shape[2:])(x, gy)

        if b.data is None:
            gb = None
        else:
            gb = gy.sum(axis = (0, 2, 3))
        return gx, gW, gb


class Deconv2d(Function):
    def __init__(self, stride, padding, img_shape):
        self.stride = stride
        self.padding = padding
        self.img_shape = img_shape

    # Conv2d の逆伝播から呼ばれるため、テンソル積の後に和を取るバイアスは無い
    def forward(self, gy, W, b = None):
        xp = cuda.get_array_module(gy)

        # W : (OC, C, FH, FW)  *  gy : (N, OC, OH, OW)  ->  gcol : (C, FH, FW, N, OH, OW)
        gcol = xp.tensordot(W, gy, (0, 1))
        gcol = xp.moveaxis(gcol, source = 3, destination = 0)

        # gcol : (N, C, FH, FW, OH, OW)  ->  gx : (N, C, H, W)
        gx = col2im_ndarray(gcol, self.img_shape, W.shape[2:], self.stride, self.padding)

        if b is not None:
            gx += xp.reshape(b, (1, len(b), 1, 1))
        return gx

    def backward(self, ggx):
        gy, W, b = self.inputs

        # Conv2d の順伝播が Deconv2d の逆伝播の入力画像データの勾配計算部分
        ggy = Conv2d(self.stride, self.padding)(ggx, W)
        gW = Conv2dGradW(self.stride, self.padding, filter_size = W.shape[2:])(ggx, gy)

        if b.data is None:
            gb = None
        else:
            gb = ggx.sum(axis = (0, 2, 3))
        return ggy, gW, gb


# Conv2d と Deconv2d の逆伝播で共通して使用する、重みの勾配を求める関数
class Conv2dGradW(Function):
    def __init__(self, stride, padding, filter_size):
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)

        # x : (N, C, H, W)  ->  col : (N, C, FH, FW, OH, OW)
        col = im2col_ndarray(x, self.filter_size, self.stride, self.padding)

        # gy : (N, OC, OH, OW)  *  col : (N, C, FH, FW, OH, OW)  ->  W : (OC, C, FH, FW)
        return xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))

    def backward(self, gW):
        x, gy = self.inputs

        gx = Deconv2d(self.stride, self.padding, img_shape = x.shape)(gy, gW)
        ggy = Conv2d(self.stride, self.padding)(x, gW)
        return gx, ggy


def conv2d(x, W, b = None, stride = 1, padding = 0):
    return Conv2d(stride, padding)(x, W, b)


def deconv2d(x, W, b = None, stride = 1, padding = 0, outsize = None):
    N, OC, OH, OW = x.shape
    OC, C, FH, FW = W.shape
    if outsize is None:
        outsize = get_deconv_outsize((OH, OW), (FH, FW), stride, padding)
    return Deconv2d(stride, padding, img_shape = (N, C) + outsize)(x, W, b)



# Conv2d1x1 と Deconv2d1x1 を１つにまとめた実装
class Conv2d1x1(Function):
    def __init__(self, tensordot_axis = 1):
        self.tensordot_axis = tensordot_axis

    def forward(self, x, W):
        xp = cuda.get_array_module(x)

        # W : (OC, C)  *  x : (N, C, H, W)  ->  y : (OC, N, H, W)  (tensordot_axis == 1  :  Conv2d1x1)
        # W : (OC, C)  *  x : (N, OC, H, W)  ->  y : (C, N, H, W)  (tensordot_axis == 0  :  Deconv2d1x1)
        y = xp.tensordot(W, x, (self.tensordot_axis, 1))
        return xp.moveaxis(y, source = 1, destination = 0)

    def backward(self, gy):
        x, W = self.inputs

        # 設定される tensordot_axis が 1, 0 で交互に変わる
        gx = Conv2d1x1((self.tensordot_axis + 1) % 2)(gy, W)

        if self.tensordot_axis:
            gW = Conv2d1x1GradW()(gy, x)
        else:
            gW = Conv2d1x1GradW()(x, gy)
        return gx, gW


class Conv2d1x1GradW(Function):
    def forward(self, gy, x):
        xp = cuda.get_array_module(x)

        # gy : (N, OC, H, W)  *  x : (N, C, H, W)  ->  W : (OC, C)
        return xp.tensordot(gy, x, ((0, 2, 3), (0, 2, 3)))

    def backward(self, gW):
        gy, x = self.inputs

        ggy = Conv2d1x1(1)(x, gW)
        gx = Conv2d1x1(0)(gy, gW)
        return ggy, gx


def conv2d_1x1filter(x, W):
    return Conv2d1x1()(x, W)




# =============================================================================
# プーリング層 (max_pooling, average_pooling, global_average_pooling)
# =============================================================================

class Pooling(Function):
    def __init__(self, filter_size, stride, padding):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        col = im2col_ndarray(x, self.filter_size, self.stride, self.padding)
        return col.max(axis = (2, 3))

    def backward(self, gy):
        x = self.inputs[0].data
        y = self.outputs[0]().data

        # 順伝播時の max の引数 col は、メモリ節約のために再度計算する
        col = im2col_ndarray(x, self.filter_size, self.stride, self.padding)
        N, C, FH, FW, OH, OW = col.shape
        col = col.reshape((N, C, FH, FW, OH, OW))

        # gy は Variable インスタンス、y は ndarray インスタンス
        gy = gy.reshape((N, C, 1, 1, OH, OW))
        y = y.reshape((N, C, 1, 1, OH, OW))

        gcol = gy * (col == y)
        return col2im(gcol, x.shape, self.filter_size, self.stride, self.padding)

def pooling(x, filter_size, stride = 1, padding = 0):
    return Pooling(filter_size, stride, padding)(x)


class AveragePooling(Function):
    def __init__(self, filter_size, stride, padding):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        col = im2col_ndarray(x, self.filter_size, self.stride, self.padding)
        return col.mean(axis = (2, 3))

    def backward(self, gy):
        N, C, OH, OW = gy.shape
        FW, FH = pair(self.filter_size)
        gy /= FW * FH
        gy = gy.reshape((N, C, 1, 1, OH, OW))
        gcol = broadcast_to(gy, (N, C, FW, FH, OH, OW))
        return col2im(gcol, self.inputs[0].shape, self.filter_size, self.stride, self.padding)

def average_pooling(x, filter_size, stride = 1, padding = 0):
    return AveragePooling(filter_size, stride, padding)(x)


def global_average_pooling(x):
    return x.mean(axis = (2, 3))




# =============================================================================
# im2col, col2im  (Function インスタンス)
# =============================================================================

class Im2col(Function):
    def __init__(self, filter_size, stride, padding, to_matrix):
        super().__init__()
        self.x_shape = None
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.to_matrix = to_matrix

    def forward(self, x):
        self.x_shape = x.shape
        return im2col_ndarray(x, self.filter_size, self.stride, self.padding, self.to_matrix)

    def backward(self, gy):
        return col2im(gy, self.x_shape, self.filter_size, self.stride, self.padding, self.to_matrix)

def im2col(x, filter_size, stride = 1, padding = 0, to_matrix = False):
    return Im2col(filter_size, stride, padding, to_matrix)(x)


class Col2im(Function):
    def __init__(self, x_shape, filter_size, stride, padding, to_matrix):
        super().__init__()
        self.x_shape = x_shape
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.to_matrix = to_matrix

    def forward(self, x):
        return col2im_ndarray(x, self.x_shape, self.filter_size, self.stride, self.padding, self.to_matrix)

    def backward(self, gy):
        return im2col(gy, self.filter_size, self.stride, self.padding, self.to_matrix)

def col2im(x, input_shape, filter_size, stride = 1, padding = 0, to_matrix = False):
    return Col2im(input_shape, filter_size, stride, padding, to_matrix)(x)




# =============================================================================
# im2col, col2im  (ndarray インスタンス)
# =============================================================================

def im2col_ndarray(img, filter_size, stride, padding, to_matrix = False):
    N, C, H, W = img.shape
    FH, FW = pair(filter_size)
    SH, SW = pair(stride)
    PH, PW = pair(padding)
    OH, OW = get_conv_outsize(H, FH, SH, PH), get_conv_outsize(W, FW, SW, PW)

    if cuda.gpu_enable:
        col = cuda.cp.empty((N, C, FH, FW, OH, OW), dtype = img.dtype)
        dy, dx = 1, 1

        # 参考元 : Chainer (https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py)
        # ElementwiseKernel : 並列計算を行う C/C++ 言語プログラムを渡して、GPU のカーネルに実行してもらう
        cuda.cp.ElementwiseKernel(
            "raw T img, int32 h, int32 w, int32 out_h, int32 out_w,"
            "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
            "int32 dy, int32 dx",
            "T col",
            """
                int c0 = i / (kh * kw * out_h * out_w);
                int ky = i / (kw * out_h * out_w) % kh;
                int kx = i / (out_h * out_w) % kw;
                int out_y = i / out_w % out_h;
                int out_x = i % out_w;
                int in_y = ky * dy + out_y * sy - ph;
                int in_x = kx * dx + out_x * sx - pw;
                if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
                    col = img[in_x + w * (in_y + h * c0)];
                } else {
                    col = 0;    /* set zero at padding part */
                }
            """,
            "im2col")(img.reduced_view(), H, W, OH, OW, FH, FW, SH, SW, PH, PW, dy, dx, col)

    else:
        img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), mode = "constant")
        col = np.empty((N, C, FH, FW, OH, OW), dtype = img.dtype)

        # 同じフィルタ要素が適用される画像データ要素を集めて、col の４・５次元目に行列形式で並べる
        for i in range(FH):
            i_lim = i + SH * OH
            for j in range(FW):
                j_lim = j + SW * OW
                col[:, :, i, j, :, :] = img[:, :, i:i_lim:SH, j:j_lim:SW]

    # to_matrix によって出力の形状が異なる (True -> (N*OH*OW, C*FH*FW), False -> (N, C, FH, FW, OH, OW))
    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))
    return col


def col2im_ndarray(col, img_shape, filter_size, stride, padding, to_matrix = False):
    N, C, H, W = img_shape
    FH, FW = pair(filter_size)
    SH, SW = pair(stride)
    PH, PW = pair(padding)
    OH, OW = get_conv_outsize(H, FH, SH, PH), get_conv_outsize(W, FW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    if cuda.gpu_enable:
        img = cuda.cp.empty((N, C, H, W), dtype = col.dtype)
        dx, dy = 1, 1

        # 参考元 : Chainer (https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py)
        cuda.cp.ElementwiseKernel(
            "raw T col, int32 h, int32 w, int32 out_h, int32 out_w,"
            "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
            "int32 dx, int32 dy",
            "T img",
            """
                int c0 = i / (h * w);
                int y  = i / w % h;
                int x  = i % w;
                T val = 0;
                for (int ky = 0; ky < kh; ++ky) {
                    int out_y = (y + ph - ky * dy);
                    if (0 > out_y || out_y >= out_h * sy) continue;    /* skip padding part */
                    if (out_y % sy != 0) continue;    /* consider stride */
                    out_y /= sy;
                    for (int kx = 0; kx < kw; ++kx) {
                        int out_x = (x + pw - kx * dx);
                        if (0 > out_x || out_x >= out_w * sx) continue;    /* skip padding part */
                        if (out_x % sx != 0) continue;    /* consider stride */
                        out_x /= sx;
                        int k = out_y + out_h * (kx + kw * (ky + kh * c0));
                        val = val + col[out_x + out_w * k];
                    }
                }
                img = val;
            """,
            "col2im")(col.reduced_view(), H, W, OH, OW, FH, FW, SH, SW, PH, PW, dx, dy, img)
        return img

    # np.pad による形状の変化も考慮して、出力に使う箱を用意する
    img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype = col.dtype)

    # 順伝播時に画像データの同じ部分が複数回コピーされているから、逆伝播では和を取る
    for i in range(FH):
        i_lim = i + SH * OH
        for j in range(FW):
            j_lim = j + SW * OW
            img[:, :, i:i_lim:SH, j:j_lim:SW] += col[:, :, i, j, :, :]

    # パディング前の入力データと形状が同じデータを返す
    return img[:, :, PH:H + PH, PW:W + PW]