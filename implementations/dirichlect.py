import math
from typing import Callable

import numpy as np

from implementations.gaussseidel import gauss_seidel_steps
from implementations.jacobi import jacobi_steps

MATRIXFOLGENFUNKTION = Callable[[int], np.ndarray]


# Dirichlet Randwertproblem

def N_l(stufenindex_l: int) -> int:
    return (2 ** (stufenindex_l + 1)) - 1


def dirichlect_randwert_a_l(stufenindex_l: int) -> np.ndarray:
    assert stufenindex_l >= 0
    n_l = N_l(stufenindex_l)
    a_l = (n_l + 1) * (n_l + 1) * (np.diag([-1 for _ in range(n_l - 1)], 1)
                                   + 2 * np.eye(n_l, dtype=np.float64)
                                   + np.diag([-1 for _ in range(n_l - 1)], -1))
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


def get_dirichlect_generator(stufenindex_l: int, mode: int, w: float, startwith: np.ndarray = None, iterator: str = "jacobi"):
    if startwith is None:
        e = fourier_mode(stufenindex_l, mode)
    else:
        e = startwith
    a = dirichlect_randwert_a_l(stufenindex_l)
    yield e
    if iterator == "jacobi":
        gen = jacobi_steps(a, e, np.zeros_like(e), 2 * w)
    else:
        gen = gauss_seidel_steps(a, e, np.zeros_like(e), 2 * w)
    yield from gen
