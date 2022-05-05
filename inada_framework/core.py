import contextlib
import numpy as np
import inada_framework
import heapq
import weakref


# =============================================================================
# 設定情報 (クラス属性のみを持つ)
# =============================================================================

class Config:
    enable_backprop = True
    train_flag = True


@contextlib.contextmanager
def using_config(name, value):
    # Config のクラス属性に関する処理 (attribute : 属性)
    old_value = getattr(Config, name)
    setattr(Config, name, value)

    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config("enable_backprop", False)


# テストモードの時は計算グラフを作る必要もないので、新たな関数として test_mode を定義する
@contextlib.contextmanager
def test_mode():
    setattr(Config, "enable_backprop", False)
    setattr(Config, "train_flag", False)
    try:
        yield
    finally:
        setattr(Config, "enable_backprop", True)
        setattr(Config, "train_flag", True)




# =============================================================================
# 変数・パラメータの基本クラス
# =============================================================================

try:
    import cupy as cp
    array_types = (np.ndarray, cp.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Variable:
    # 二項演算子のオーバーロードにおいて、ndarray オブジェクトよりも優先されるようにする
    __array_priority__ = 200

    def __init__(self, data, name = None):
        # 扱うデータ型を None と ndarray に限定する
        if not (data is None or isinstance(data, array_types)):
            raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.grad = None
        self.name = name

        # 自分を生み出した関数とのつながりを動的に構築するためのメンバ変数 (Define-by-Run)
        self.creator = None
        self.generation = 0

    # プロパティ名 (この例では "shape") を持つメンバ変数としてアクセスできるようになる
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()

    # Python の組み込み関数 (標準搭載の関数) である len() で呼び出されるメソッド
    def __len__(self):
        return len(self.data)

    # オブジェクトの文字列表記を返す特殊メソッド
    def __repr__(self):
        if self.data is None:
            return "Variable(None)"

        # 複数行に渡って出力する場合に整形して出力するように改行文字を置き換える
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "Variable(" + p + ")"

    def set_creator(self, func):
        # heapq モジュールを使う都合上、世代は進むにつれて小さくなるようにする
        self.generation = func.generation - 1
        self.creator = func

    # RNN で時系列方向のつながりを断ち切るためのメソッド
    def unchain(self):
        self.creator = None

    def clear_grad(self):
        self.grad = None

    # 動的に構築された関数とのつながりから自身の１つ前の変数の勾配を求めるという処理を繰り返す
    def backward(self, retain_grad = False, create_graph = False):
        # 高階微分を行うために、grad は Variable インスタンスとする
        if self.grad is None:
            xp = inada_framework.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        # 関数のリストにはヒープ、既視かどうかの判断には集合を使う
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs, f)
                seen_set.add(f)

        add_func(self.creator)

        # この逆伝播に対する逆伝播を行うかどうかを指定する
        with using_config("enable_backprop", create_graph):
            while funcs:
                # 世代が最も新しい関数をリストから取り出し、逆伝播を実行する
                f = heapq.heappop(funcs)
                gys = [output().grad for output in f.outputs]
                gxs = f.backward(*gys)

                # 逆伝播メソッドの出力がタプルでない場合、次の処理のためにタプルに変換する
                if not isinstance(gxs, (tuple, list)):
                    gxs = (gxs, )

                for input, gx in zip(f.inputs, gxs):
                    # １回目に勾配が伝播してきた場合は代入、それ以降は加算を行う
                    if input.grad is None:
                        input.grad = gx
                    else:
                        # Variable インスタンスの累計算術演算子と見なされるため、in-place 演算ではない
                        input.grad += gx

                    # 計算グラフに続きがある場合は、リストに１つ前の変数の生みの親である関数を追加する
                    if input.creator is not None:
                        add_func(input.creator)

                # 計算グラフにおいて末端の変数以外の勾配を保持するかどうかを指定する
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None

    def to_cpu(self):
        if self.data is not None:
            self.data = inada_framework.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = inada_framework.cuda.as_cupy(self.data)


# 各種レイヤのパラメータと、ただの Variable インスタンスを区別するためのクラス
class Parameter(Variable):
    pass




# =============================================================================
# 関数の基底クラス
# =============================================================================

class Function:
    # Python の特殊メソッドの１つで、インスタンス (...) で __call__ メソッドを呼び出す
    def __call__(self, *inputs):
        # "*" によって、引数を packing して受け取り、unpacking して渡して、可変長引数に対応
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        # 順伝播メソッドの出力がタプルでない場合、次の処理のためにタプルに変換する
        if not isinstance(ys, (tuple, list)):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        # 逆伝播しないモードのときは以下の処理が不要なので、それを行わないようにする
        if Config.enable_backprop:
            # 正しい順序で逆伝播を行うための世代を関数に設定し、出力変数に生みの親である関数を覚えさせる
            self.generation = min([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            # creator との循環参照を避けるために、outputs は弱参照とする (参照先が無くなると消える参照)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        # 出力の要素数が１の場合は、最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]

    # heapq で用いる比較基準に世代を用いる
    def __lt__(self, other):
        return self.generation < other.generation

    # 実装をされていないことを示すエラーを出力する (基底クラスなので、このまま使われることを想定していない)
    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# Function サブクラスの引数が Variable クラスのみを扱うという仕様をクリアするための関数
def as_variable(object):
    if isinstance(object, Variable):
        return object
    return Variable(object)


# Variable が ndarray のみを扱うという仕様をクリアするための関数
def as_array(x, array_module = np):
    if np.isscalar(x):
        return array_module.array(x)
    return x




# =============================================================================
# 二項演算子のオーバーロード・特殊メソッド・クラスメソッド
# =============================================================================

# 循環インポートを避けるために、import mydezero.functions as dzf のようには利用しない
def setup_variable():
    dzf = inada_framework.functions

    Variable.__add__ = dzf.add
    Variable.__radd__ = dzf.add
    Variable.__iadd__ = dzf.add
    Variable.__neg__ = dzf.neg
    Variable.__sub__ = dzf.sub
    Variable.__rsub__ = dzf.rsub
    Variable.__isub__ = dzf.sub
    Variable.__mul__ = dzf.mul
    Variable.__rmul__ = dzf.mul
    Variable.__imul__ = dzf.mul
    Variable.__truediv__ = dzf.div
    Variable.__rtruediv__ = dzf.rdiv
    Variable.__itruediv__ = dzf.div
    Variable.__pow__ = dzf.pow
    Variable.__rpow__ = dzf.rpow
    Variable.__ipow__ = dzf.pow

    Variable.__abs__ = dzf.abs
    Variable.__getitem__ = dzf.get_item

    Variable.reshape = dzf.reshape
    Variable.flatten = dzf.flatten
    Variable.transpose = dzf.transpose
    Variable.sum = dzf.sum
    Variable.mean = dzf.mean
    Variable.max = dzf.max
    Variable.min = dzf.min
    Variable.matmul = dzf.matmul
    Variable.dot = dzf.matmul