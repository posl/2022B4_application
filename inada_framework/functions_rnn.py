from inada_framework import Function, cuda
import inada_framework.functions as dzf



# =============================================================================
# RNN 層 (LSTM, GRU)
# =============================================================================

class LSTM(Function):
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, x, h_prev, c_prev, Wx, Wh, b):
        xp = cuda.get_array_module(x)
        A = xp.dot(x, Wx) + xp.dot(h_prev, Wh)
        if b is not None:
            A += b

        H = len(Wh)
        H2 = H + H

        # tanh 関数・sigmoid 関数
        self.get = xp.tanh(A[:, :H])
        gate = xp.tanh(A[:, H:] * 0.5) * 0.5 + 0.5

        # ゲートは前から順に forget, input, output
        self.forget = gate[:, :H]
        self.input = gate[:, H:H2]
        output = gate[:, H2:]

        c = self.forget * c_prev + self.input * self.get
        h = output * xp.tanh(c)

        if self.dropout_ratio is None:
            return h, h, c

        mask = (xp.random.rand(*h.shape) > self.dropout_ratio)
        self.scale = mask / xp.array(1.0 - self.dropout_ratio).astype(h.dtype)
        x = h * self.scale
        return x, h, c

    def backward(self, gx, gh, gc):
        x, h_prev, c_prev, Wx, Wh, b = self.inputs
        _, h, c = self.outputs

        # メモリ節約のために、tanh(c) と output ゲートは再計算する
        c = c().data
        xp = cuda.get_array_module(c)
        tanhc = xp.tanh(c)
        output = h().data / (tanhc + 1e-7)

        # gh, gc に他のレイヤからの勾配が直接伝わってくることはない
        if gh is None:
            gh = 0
        if gc is None:
            gc = 0

        if self.dropout_ratio is not None:
            gh += gx * self.scale
        else:
            gh += gx

        gc += (gh * output) * (1.0 - tanhc ** 2)

        gc_prev = gc * self.forget
        gget = gc * self.input
        gforget = gc * c_prev
        ginput = gc * self.get
        goutput = gh * tanhc

        ggate = dzf.concatenate([gforget, ginput, goutput], axis = 1)
        gate = xp.concatenate([self.forget, self.input, output], axis = 1)
        gA1 = ggate * gate * (1.0 - gate)
        gA0 = gget * (1.0 - self.get ** 2)
        gA = dzf.concatenate([gA0, gA1], axis = 1)

        gx = dzf.matmul(gA, Wx.T)
        gWx = dzf.matmul(x.T, gA)
        gh_prev = dzf.matmul(gA, Wh.T)
        gWh = dzf.matmul(h_prev.T, gA)

        if b.data is None:
            gb = None
        else:
            gb = dzf.sum(gA, axis = 0)
        return gx, gh_prev, gc_prev, gWx, gWh, gb


def lstm(x, h_prev, c_prev, Wx, Wh, b = None, dropout_ratio = None):
    return LSTM(dropout_ratio)(x, h_prev, c_prev, Wx, Wh, b)



class GRU(Function):
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, x, h_prev, Wx, Wh, b):
        xp = cuda.get_array_module(x)
        H = len(Wh)

        # 行列積はなるべくまとめて行う
        A = xp.dot(x, Wx) + b
        B = xp.dot(h_prev, Wh[:, H:])
        C = A[:, H:] + B

        # z が (forget と input を担う) update ゲートで、r が reset ゲート
        self.gate = xp.tanh(C * 0.5) * 0.5 + 0.5
        z = self.gate[:, :H]
        r = self.gate[:, H:]

        self.h_hat = xp.tanh(A[:, :H] + xp.dot(r * h_prev, Wh[:, :H]))
        h = (1 - z) * h_prev + z * self.h_hat

        if self.dropout_ratio is None:
            return h, h

        mask = (xp.random.rand(*h.shape) > self.dropout_ratio)
        self.scale = mask / xp.array(1.0 - self.dropout_ratio).astype(h.dtype)
        x = h * self.scale
        return x, h

    def backward(self, gx, gh):
        x, h_prev, Wx, Wh, b = self.inputs
        H = len(Wh)

        # gh に他のレイヤからの勾配が直接伝わってくることはない
        if gh is None:
            gh = 0

        if self.dropout_ratio is not None:
            gh += gx * self.scale
        else:
            gh += gx

        z = self.gate[:, :H]
        r = self.gate[:, H:]

        # h の逆伝播
        gh_prev = gh * (1 - z)
        gh_hat = gh * z
        gz = gh * self.h_hat - gh * h_prev

        # h_hat の逆伝播
        gA0 = gh_hat * (1.0 - self.h_hat ** 2)
        grh_prev = dzf.matmul(gA0, Wh[:, :H].T)
        gr = grh_prev * h_prev
        gh_prev += grh_prev * r
        gWh0 = dzf.matmul((r * h_prev).T, gA0)

        # reset gate, update gate の逆伝播
        ggate = dzf.concatenate([gz, gr], axis = 1)
        gC = ggate * self.gate * (1.0 - self.gate)

        # 行列積の逆伝播
        gh_prev += dzf.matmul(gC, Wh[:, H:].T)
        gWh1 = dzf.matmul(h_prev.T, gC)
        gWh = dzf.concatenate([gWh0, gWh1], axis = 1)

        gA = dzf.concatenate([gA0, gC], axis = 1)
        gx = dzf.matmul(gA, Wx.T)
        gWx = dzf.matmul(x.T, gA)

        if b.data is None:
            gb = None
        else:
            gb = dzf.sum(gA, axis = 0)
        return gx, gh_prev, gWx, gWh, gb


def gru(x, h_prev, Wx, Wh, b = None, dropout_ratio = None):
    return GRU(dropout_ratio)(x, h_prev, Wx, Wh, b)