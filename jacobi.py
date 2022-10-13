import numpy as np


def jacobi_test(a: np.ndarray, diagonals: np.ndarray, x: np.ndarray):
    assert a.ndim == 2
    assert diagonals.ndim == 1
    assert x.ndim == 1
    assert np.all(diagonals)


def _jacobi_step(m: np.ndarray, nb: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Performs jacobi step without tests

    :param m: D^-1(D-A)
    :param n: D^-1
    :param x: x_{m}
    :return: x_{m+1}
    """
    return np.dot(m, x) + nb


def jacobi_step(a: np.ndarray, x: np.ndarray, b: np.ndarray):
    """
    Performs one Jacobi Step

    :param b:
    :param a:
    :param x:
    :return:
    """
    diagonals = np.diag(a)  # depending on the version used this might be a view. do not write to this!
    d = np.diag(diagonals)  # same here
    jacobi_test(a, diagonals, x)
    n = (1 / d)
    m = n * (d - a)
    return _jacobi_step(m, np.dot(n, b), x)


def jacobi_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray):
    """
    Generator to perform many jacobi steps

    :param b:
    :param a:
    :param x:
    :return:
    """
    diagonals = np.diag(a)  # depending on the version used this might be a view. do not write to this!
    d = np.diag(diagonals)  # same here
    jacobi_test(a, diagonals, x)
    n = np.diag((1 / diagonals))
    m = np.dot(n, (d - a))
    n = np.dot(n, b)
    y = x.copy()
    total_steps = 0
    try:
        while True:
            y = _jacobi_step(m, n, y)
            total_steps += 1
            yield y
    except GeneratorExit:
        return y, total_steps


def n_jacobi_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, n: int):
    generator = jacobi_steps(a, x, b)
    for _ in range(n - 1):
        next(generator)
    y = next(generator)
    generator.close()
    return y
