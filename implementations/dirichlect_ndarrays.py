import math

import numpy as np

from implementations.helpers import N_l


# Dirichlet Randwertproblem
def dirichlect_randwert_a_l(stufenindex_l: int) -> np.ndarray:
    assert stufenindex_l >= 0
    n_l = N_l(stufenindex_l)
    a_l = (n_l + 1) * (n_l + 1) * (np.diag([-1. for _ in range(n_l - 1)], 1)
                                   + 2 * np.eye(n_l, dtype=np.float64)
                                   + np.diag([-1. for _ in range(n_l - 1)], -1))
    return a_l


def fourier_mode(stufenindex_l: int, mode: int, scale: bool = False) -> np.ndarray:
    assert stufenindex_l >= 0
    n_l = N_l(stufenindex_l)
    h_l = 1 / (n_l + 1)
    assert mode > 0
    assert mode <= n_l
    x = np.sin(mode * np.pi * h_l * np.arange(1, n_l + 1, dtype=np.float64))
    if not scale:
        x *= math.sqrt(2 * h_l)
    return x


def eigenvalues(stufenindex_l: int, mode: int, w: float) -> np.ndarray:
    assert stufenindex_l >= 0
    n_l = N_l(stufenindex_l)
    assert mode > 0
    assert mode <= n_l
    h_l = 1 / (n_l + 1)
    return 1 - 4 * w * np.sin(mode * np.pi * h_l / 2) ** 2