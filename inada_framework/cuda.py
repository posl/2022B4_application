import numpy as np
try:
    import cupy as cp
except ImportError:
    gpu_enable = False
else:
    gpu_enable = True

from inada_framework import Variable


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np

    # 引数の型に応じて、numpy か cupy を返す
    return cp.get_array_module(x)


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    if isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data
    return cp.asarray(x)