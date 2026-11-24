import numpy as np

def to_categorical(x, n_col=None):
    n_col = np.max(x) + 1 if n_col is None else n_col

    one_hot_encoder = np.zeros((len(x), n_col), dtype=int)
    one_hot_encoder[np.arange(len(x)), x] = 1

    return one_hot_encoder