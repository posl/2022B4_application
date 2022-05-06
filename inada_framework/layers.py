from inada_framework.utilitys import pair
import os
from inada_framework import Parameter, cuda
import numpy as np
import inada_framework.functions as dzf


# =============================================================================
# レイヤ・モデルの基底クラス
# =============================================================================

# 学習するパラメータを持つクラスがレイヤであるとする
parameters_dir = os.path.join(os.path.dirname(__file__), "..", "parameters")

class Layer:
    def __init__(self):
        # パラメータは集合で保持する (同じオブジェクト ID の要素は含まれない)
        self._params = set()

    # インスタンス変数 (属性) を設定する際に呼び出せれる特殊メソッド
    def __setattr__(self, name, value):
        # パラメータの管理を自動化するために、パラメータの名前・入れ子になっているレイヤの名前を取っておく
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def forward(self, inputs):
        raise NotImplementedError()

    def __call__(self, *inputs):
        return self.forward(*inputs)

    # yield 文は関数の進行具合を保存した状態で値を返す (イテレータとして利用できる)
    def params(self):
        for name in self._params:
            # __dict__ から辞書形式で内部の属性情報に直接アクセスできる
            object = self.__dict__[name]

            if isinstance(object, Layer):
                # yield from 文は再帰的なジェネレータを作るために必要
                yield from object.params()
            elif isinstance(object, Parameter):
                yield object

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    # パラメータを保持するインスタンスの辞書を平板化して取得する
    def flatten_params(self, params_dict, parent_key = ""):
        for name in self._params:
            object = self.__dict__[name]
            key = parent_key + name

            if isinstance(object, Layer):
                # Layer が入れ子構造になっている場合は、再帰的にこの関数を呼び出す
                object.flatten_params(params_dict, key + "/")
            elif isinstance(object, Parameter):
                params_dict[key] = object

    def save_weights(self, file_name = "params.npz"):
        # パラメータを保存するときは必ずデータが主記憶上にあるようにする
        if cuda.gpu_enable:
            self.to_cpu()

        params_dict = {}
        self.flatten_params(params_dict)
        array_dict = {key : param.data for key, param in params_dict.items()}

        try:
            if not os.path.exists(parameters_dir):
                os.mkdir(parameters_dir)
            file_path = os.path.join(parameters_dir, file_name)
            np.savez_compressed(file_path, **array_dict)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
        else:
            if cuda.gpu_enable:
                self.to_gpu()

    def load_weights(self, file_name = "params.npz"):
        file_path = os.path.join(parameters_dir, file_name)

        # 指定されたファイルが無ければ、何もせずに戻る
        if os.path.exists(file_path):
            npz = np.load(file_path)

            params_dict = {}
            self.flatten_params(params_dict)
            for key, param in params_dict.items():
                param.data = npz[key]
        else:
            print("parameters file could not be found.")

    # 同じレイヤクラスのインスタンス同士でパラメータを同期させる
    def copy_weights(self, layer):
        assert isinstance(layer, self.__class__)

        passed_params_dict = {}
        layer.flatten_params(passed_params_dict)

        params_dict = {}
        self.flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = passed_params_dict[key].data


class Model(Layer):
    pass




# =============================================================================
# 汎用レイヤ
# =============================================================================

class Affine(Layer):
    # 入力サイズは流れてきたデータから分かるので、初期化時には指定しない
    def __init__(self, out_size, W_mode = "he", nobias = False):
        super().__init__()
        self.out_size = out_size

        # 重みの初期化に He の初期値を使うか、Xavier の初期値を使うかを決める
        self.W_mode = 2.0 if W_mode.lower() == "he" else 1.0

        # 重みは学習が開始したときに動的に生成する
        self.W = Parameter(None, name = "W")

        # バイアスは使用するか否か選ぶことができる
        self.b = None if nobias else Parameter(np.zeros(out_size, dtype = np.float32), name = "b")

    def forward(self, x):
        if self.W.data is None:
            self.W.data = self.init_W(x)
        return dzf.affine(x, self.W, self.b)

    # 重みは Xavier の初期値
    def init_W(self, x):
        I = x.shape[1]
        xp = cuda.get_array_module(x)
        return xp.random.randn(I, self.out_size).astype(np.float32) * np.sqrt(self.W_mode / I)


class BatchNorm(Layer):
    # avg_mean, avg_var も順伝播時に変化するので、重みと一緒に保存できるように Parameter インスタンスとする
    def __init__(self):
        super().__init__()
        self.gamma = Parameter(None, name = "gamma")
        self.beta = Parameter(None, name = "beta")
        self.avg_mean = Parameter(None, name = "avg_mean")
        self.avg_var = Parameter(None, name = "avg_var")

    def forward(self, x):
        if self.gamma.data is None:
            self.init_params(x)
        return dzf.batch_nrom(x, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data)

    def init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        self.gamma.data = xp.ones(D, dtype = x.dtype)
        self.beta.data = xp.zeros(D, dtype = x.dtype)
        self.avg_mean.data = xp.zeros(D, dtype = x.dtype)
        self.avg_var.data = xp.ones(D, dtype = x.dtype)




# =============================================================================
# CNN レイヤ
# =============================================================================

class Conv2d(Layer):
    # 入力データのチャネル数は流れてきたデータから分かるので、初期化時には指定しない
    def __init__(self, out_channels, filter_size, stride = 1, padding = 0, W_mode = "he", nobias = False):
        super().__init__()
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.W_mode = 2.0 if W_mode.lower() == "he" else 1.0
        self.W = Parameter(None, name = "W")
        self.b = None if nobias else Parameter(np.zeros(out_channels, dtype = np.float32), name = "b")

    def forward(self, x, dzf_func = dzf.conv2d):
        if self.W.data is None:
            self.W.data = self.init_W(x)
        return dzf_func(x, self.W, self.b, self.stride, self.padding)

    def init_W(self, x):
        C, OC = x.shape[1], self.out_channels
        KH, KW = pair(self.filter_size)
        xp = cuda.get_array_module(x)
        return xp.random.randn(OC, C, KH, KW).astype(np.float32) * np.sqrt(self.W_mode / (C * KH * KW))


class Deconv2d(Conv2d):
    def forward(self, x):
        return super().forward(x, dzf.deconv2d)

    def init_W(self, x):
        return super().init_W(x).transpose(1, 0, 2, 3)


# 1×1 の畳み込み層 (チャネル数だけが変化する)
class Conv2d1x1(Layer):
    def __init__(self, out_channels, W_mode = "he"):
        super().__init__()
        self.out_channels = out_channels
        self.W_mode = 2.0 if W_mode.lower() == "he" else 1.0
        self.W = Parameter(None, name = "W")

    def forward(self, x):
        if self.W.data is None:
            self.W.data = self.init_W(x)

        # ImageNet で学習済みのパラメータの次元数が 4 であるために必要な処理
        elif self.W.data.ndim == 4:
            W = self.W.data
            self.W.data = W.reshape(W.shape[:2])

        return dzf.conv2d_1x1filter(x, self.W)

    def init_W(self, x):
        C, OC = x.shape[1], self.out_channels
        xp = cuda.get_array_module(x)
        return xp.random.randn(OC, C).astype(np.float32) * np.sqrt(self.W_mode / C)




# =============================================================================
# RNN レイヤ
# =============================================================================

class Embedding(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name = "W")

    def __call__(self, x):
        return self.W[x]


class LSTM(Layer):
    def __init__(self, hidden_size, dropout_ratio = None, nobias = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.Wx = Parameter(None, name = "Wx")

        horizon_size = hidden_size * 4
        Wh = np.random.randn(hidden_size, horizon_size).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.Wh = Parameter(Wh, name = "Wh")
        self.b = None if nobias else Parameter(np.zeros(horizon_size, dtype = np.float32), name = "b")

        self.reset_state()

    # エポックごとに状態をリセットする
    def reset_state(self):
        self.h = None
        self.c = None

    # 記憶セルは各々の、このレイヤのインスタンス内だけで完結しているので、外から設定することはない
    def set_state(self, h):
        self.h = h

    # RNN の学習の際、時系列方向のつながりを断ち切るメソッド
    def unchain(self):
        self.h.unchain()
        self.c.unchain()

    def forward(self, x):
        # 隠れ状態ベクトルだけが単独で Nonw になることはない
        if self.c is None:
            xp = cuda.get_array_module(x)
            self.h = xp.zeros((len(x), self.hidden_size), dtype = np.float32)
            self.c = xp.zeros((len(x), self.hidden_size), dtype = np.float32)

            # この条件式が真になるのは学習を開始した直後の順伝播時だけ
            if self.Wx.data is None:
                self.Wx.data = self.init_Wx(x.shape[1], xp)

        x, self.h, self.c = dzf.lstm(x, self.h, self.c, self.Wx, self.Wh, self.b, self.dropout_ratio)
        return x

    def init_Wx(self, in_size, xp):
        return xp.random.randn(in_size, self.hidden_size * 4).astype(np.float32) * np.sqrt(1 / in_size)


class GRU(Layer):
    def __init__(self, hidden_size, dropout_ratio = None, nobias = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.Wx = Parameter(None, name = "Wx")

        horizon_size = hidden_size * 3
        Wh = np.random.randn(hidden_size, horizon_size).astype(np.float32) * np.sqrt(1 / hidden_size)
        self.Wh = Parameter(Wh, name = "Wh")
        self.b = None if nobias else Parameter(np.zeros(horizon_size, dtype = np.float32), name = "b")

        self.reset_state()

    # エポックごとに状態をリセットする
    def reset_state(self):
        self.h = None

    # 記憶セルは各々の、このレイヤのインスタンス内だけで完結しているので、外から設定することはない
    def set_state(self, h):
        self.h = h

    # RNN の学習の際、時系列方向のつながりを断ち切るメソッド
    def unchain(self):
        self.h.unchain()

    def forward(self, x):
        if self.h is None:
            xp = cuda.get_array_module(x)
            self.h = xp.zeros((len(x), self.hidden_size), dtype = np.float32)

            # この条件式が真になるのは学習を開始した直後の順伝播時だけ
            if self.Wx.data is None:
                self.Wx.data = self.init_Wx(x.shape[1], xp)

        x, self.h = dzf.gru(x, self.h, self.Wx, self.Wh, self.b, self.dropout_ratio)
        return x

    def init_Wx(self, in_size, xp):
        return xp.random.randn(in_size, self.hidden_size * 3).astype(np.float32) * np.sqrt(1 / in_size)




class TimeRNN(Layer):
    def __init__(self, rnn_num, hidden_size, dropout_ratio = None, embed_args = None, stateful = True):
        super().__init__()
        assert isinstance(rnn_num, int) and rnn_num > 0

        self._forward = []
        for i in range(rnn_num):
            layer = self.layer_class(hidden_size, dropout_ratio)
            name = layer.__class__.__name__.lower() + str(i)
            setattr(self, name, layer)
            self._forward.append(name)

        # Embedding レイヤを使用する場合に、embed_args にはその引数がタプルとして渡される
        self.embed = Embedding(*embed_args) if embed_args is not None else None
        self.dropout_ratio = dropout_ratio
        self.stateful = stateful

    def reset_state(self):
        for name in self._forward:
            layer = getattr(self, name)
            layer.reset_state()

    def set_state(self, h):
        layer = getattr(self, self._forward[0])
        layer.set_state(h)

    def unchain(self):
        for name in self._forward:
            layer = getattr(self, name)
            layer.unchain()

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        if self.dropout_ratio is not None:
            x = dzf.dropout(x, self.dropout_ratio)

        # split, concatenate は次元をそのままに保つので、次元を落としてから次元０方向に T 分割する
        T, N, D = x.shape
        R = T * N
        x_list = dzf.split(dzf.reshape(x, (R, D)), np.arange(N, R, N))

        for name in self._forward:
            layer = getattr(self, name)
            if not self.stateful:
                layer.reset_state()

            for t in range(T):
                x_list[t] = layer(x_list[t])

        # ２次元の配列として次のレイヤへ出力する
        return dzf.concatenate(x_list)


class TimeLSTM(TimeRNN):
    layer_class = LSTM

class TimeGRU(TimeRNN):
    layer_class = GRU