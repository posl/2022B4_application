import math
from inada_framework import Parameter, cuda


# =============================================================================
# Optimizer の基底クラス
# =============================================================================

class Optimizer:
    def __init__(self):
        self.model = None
        self.hooks = []

    def setup(self, target):
        self.model = target
        return self

    def update(self):
        # 勾配が設定されているパラメータだけ取り出す
        params = [p for p in self.model.params() if p.grad is not None]

        # 更新作業前にパラメータに対して、事前に追加されている前処理を行う
        for f in self.hooks:
            f(params)

        for param in params:
            self._update(param)

    def _update(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)




# =============================================================================
# フック関数
# =============================================================================

class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data


# 勾配の L2 ノルムが規定値を超えたら、クリッピングする
class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data ** 2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate


# 更新しないパラメータを更新対象パラメータのリストから除外する
class FreezeParam:
    def __init__(self, *params_or_layers):
        self.freeze_params = []
        for object in params_or_layers:
            if isinstance(object, Parameter):
                self.freeze_params.append(object)
            else:
                for p in object.params():
                    self.freeze_params.append(p)

    def __call__(self, params):
        for p in self.freeze_params:
            params.remove(p)




# =============================================================================
# 具体的な最適化手法
# =============================================================================

# Stochastic Gradient Descent : 確率的勾配降下法
class SGD(Optimizer):
    def __init__(self, lr = 0.01):
        super().__init__()
        self.lr = lr

    def _update(self, param):
        param.data -= self.lr * param.grad.data


# Momentum : 慣性
class Momentum(Optimizer):
    def __init__(self, lr = 0.01, momentum = 0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def _update(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        # v にポインタをコピーし、in-place 演算を行っている
        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


# １つ前の速度でパラメータを更新することで、Momentum に比べて慣性をより再現したような実装
class Nesterov(Optimizer):
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def _update(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        grad = param.grad.data
        param.data += (self.momentum ** 2) * v - (1 + self.momentum) * self.lr * grad
        v *= self.momentum
        v -= self.lr * grad


# Adaptive Gradient : (学習の進行具合に) 適用性のある勾配
class AdaGrad(Optimizer):
    def __init__(self, lr = 0.001, eps = 1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def _update(self, param):
        xp = cuda.get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        # 勾配の２乗和で学習の進行具合を表す
        h = self.hs[h_key]
        grad = param.grad.data
        h += grad * grad
        param.data -= self.lr * grad / (xp.sqrt(h) + self.eps)


# 初期学習係数のパラメータを無くすように改良した AdaGrad
class AdaDelta(Optimizer):
    def __init__(self, lr = 0.95, eps = 1e-6):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def _update(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)
            self.msdx[key] = xp.zeros_like(param.data)

        msg, msdx = self.msg[key], self.msdx[key]
        lr, eps = self.lr, self.eps
        grad = param.grad.data

        msg += (1 - lr) * (grad * grad - msg)
        dx = xp.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx += (1 - lr) * (dx * dx - msdx)
        param.data -= dx


# 学習が進めば進むほど更新が行われなくなるということが無いように、過去の勾配を徐々に忘れていくようにした AdaGrad
class RMSprop:
    def __init__(self, lr = 0.01, decay_rate = 0.99, eps = 1e-7):
        self.lr = lr
        self.decay_rate = decay_rate
        self.eps = eps
        self.hs = {}

    def _update(self, param):
        xp = cuda.get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        h = self.h[h_key]
        grad = param.grad.data
        h += (1 - self.decay_rate) * (grad ** 2 - h)
        param.data -= self.lr * grad / (xp.sqrt(h) + self.eps)


# Adaptive moment estimation (beta1 が momentum, beta2 が decay_rate に対応する)
class Adam(Optimizer):
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.product1 = 1
        self.product2 = 1
        self.ms = {}
        self.vs = {}

    # 学習率補正値は 12 回目まで約 0.3 ~ 0.15 の値域で単調減少、それ以降は対数関数的に 1.0 に近づく
    @property
    def biased_lr(self):
        self.product1 *= self.beta1
        self.product2 *= self.beta2
        return self.lr * math.sqrt(1.0 - self.product2) / (1.0 - self.product1)

    def _update(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        grad = param.grad.data

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        param.data -= self.biased_lr * m / (xp.sqrt(v) + self.eps)