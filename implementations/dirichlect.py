# This needs its own file to avoid circular imports.
import numpy as np

from implementations.dirichlect_ndarrays import fourier_mode, dirichlect_randwert_a_l
from implementations.gaussseidel import gauss_seidel_steps, _gauss_seidel_matrices
from implementations.jacobi import jacobi_steps, jacobi_matrices
from implementations.zweigitter import zweigitter_steps


def get_dirichlect_generator(stufenindex_l: int, mode: int, w: float, startwith: np.ndarray = None,
                             iterator: str = "jacobi"):
    if startwith is None:
        e = fourier_mode(stufenindex_l, mode)
    else:
        e = startwith
    a = dirichlect_randwert_a_l(stufenindex_l)
    yield e
    if iterator == "jacobi":
        gen = jacobi_steps(a, e, np.zeros_like(e), 2 * w)
    elif iterator == "gaussseidel":
        gen = gauss_seidel_steps(a, e, np.zeros_like(e), 2 * w)
    elif iterator == "zweigitter-jacobi":
        gen = zweigitter_steps(stufenindex_l, 2, 2, e, np.zeros_like(e), jacobi_matrices, w1=2 * w, w2=2 * w)
    else:
        gen = zweigitter_steps(stufenindex_l, 2, 2, e, np.zeros_like(e), _gauss_seidel_matrices, w1=2 * w, w2=2 * w)

    yield from gen