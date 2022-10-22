import numpy as np

from typing import Callable, Tuple


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
                                               Tuple[np.ndarray, np.ndarray]],
                            a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1):
    """
    Generator that performs many steps
    """
    m, nb = matrices(a, b, x, w)
    y = x.copy()
    total_steps = 0
    try:
        while True:
            y = iter_step(m, nb, y)
            total_steps += 1
            new_w = yield y
            if new_w is not None:
                m, nb = matrices(a, b, x, new_w)
    except GeneratorExit:
        return y, total_steps


def n_steps_of_generator(generator, n):
    """
    Performs n steps of a generator, returns the last result and closes the generator
    """
    for _ in range(n - 1):
        next(generator)
    y = next(generator)
    generator.close()
    return y
