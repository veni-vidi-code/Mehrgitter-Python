from __future__ import annotations  # back support for python 3.8 & 3.9

import numpy as np

from typing import Callable, Tuple, Generator, Any

from numpy import ndarray

MATRIXFOLGENFUNKTION = Callable[[int], np.ndarray]


def N_l(stufenindex_l: int) -> int:
    return (2 ** (stufenindex_l + 1)) - 1


def iter_step(m: np.ndarray, nb: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Performs M*x + nb

    (e.g. jacobi or gauss_seidel step)
    """
    return np.dot(m, x) + nb


def iter_tests(a: np.ndarray, diagonals: np.ndarray, x: np.ndarray):
    assert a.ndim == 2  # a must be a matrix
    assert diagonals.ndim == 1  # diagonals must be a vector
    assert x.ndim == 1  # x must be a vector
    assert np.all(diagonals)  # diagonals must not be zero


def iter_steps_generatordef(matrices: Callable[[np.ndarray, np.ndarray, np.ndarray, float],
                                               Tuple[np.ndarray, np.ndarray]] | Tuple[np.ndarray, np.ndarray],
                            a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1) \
        -> Generator[np.ndarray, Any, Tuple[ndarray | None, int]]:
    """
    Generator der viele Schritte der Iteration durchführt.
    Diese Funktion dient als Wrapper für iter_step, um die Iteration zu vereinfachen.

    :param matrices: Funktion, die die Matrizen M und nb berechnet
    :param a: Matrix A
    :param x: Startvektor
    :param b: Vektor b
    :param w: Relaxationsfaktor
    :return: Generator, der bei jedem Aufruf einen Schritt der Iteration durchführt
    """
    set_matrices = isinstance(matrices, tuple)

    m, nb = matrices(a, b, x, w) if not set_matrices else matrices
    x = x.copy()
    total_steps = 0
    try:
        while True:
            x = iter_step(m, nb, x)
            total_steps += 1
            new_w = yield x
            if new_w is not None:
                if set_matrices:
                    raise ValueError("Cannot change w if matrices are given")
                else:
                    m, nb = matrices(a, b, x, new_w)
    except GeneratorExit:
        return x, total_steps


def n_steps_of_generator(generator, n: int) -> np.ndarray:
    """
    Führt n Schritte des Generators durch, schließt diesen und gibt das Ergebnis zurück.
    """
    for _ in range(n - 1):
        next(generator)
    y = next(generator)
    generator.close()
    return y
