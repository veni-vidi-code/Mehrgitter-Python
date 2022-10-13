import numpy as np


def gauss_seidel_test(a: np.ndarray, diagonals: np.ndarray, x: np.ndarray):
    assert a.ndim == 2
    assert diagonals.ndim == 1
    assert x.ndim == 1
    assert np.all(diagonals)


def _gauss_seidel_matrices(a: np.ndarray, b: np.ndarray, x: np.ndarray):
    gauss_seidel_test(a, np.diag(a), x)
    n = np.linalg.inv(np.tril(a))
    m = - np.dot(n, np.triu(a, 1))
    return m, np.dot(n, b)


def _gauss_seidel_step(m: np.ndarray, nb: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Performs gauss_seidel step without tests

    :param m: D^-1(D-A)
    :param n: D^-1
    :param x: x_{m}
    :return: x_{m+1}
    """
    return np.dot(m, x) + nb


def gauss_seidel_step(a: np.ndarray, x: np.ndarray, b: np.ndarray):
    """
    Performs one gauss_seidel Step

    :param b:
    :param a:
    :param x:
    :return:
    """
    m, nb = _gauss_seidel_matrices(a, b, x)
    return _gauss_seidel_step(m, nb, x)


def gauss_seidel_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray):
    """
    Generator to perform many gauss_seidel steps

    :param b:
    :param a:
    :param x:
    :return:
    """
    m, nb = _gauss_seidel_matrices(a, b, x)
    y = x.copy()
    total_steps = 0
    try:
        while True:
            y = _gauss_seidel_step(m, nb, y)
            total_steps += 1
            yield y
    except GeneratorExit:
        return y, total_steps


def n_gauss_seidel_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, n: int):
    generator = gauss_seidel_steps(a, x, b)
    for _ in range(n - 1):
        next(generator)
    y = next(generator)
    generator.close()
    return y


if __name__ == "__main__":
    # Example from the book
    A = np.array([[0.7, -0.4], [-0.2, 0.5]], np.float64)
    X = np.array([21, -19], np.float64)
    B = np.array([0.3, 0.3], np.float64)
    M, NB = _gauss_seidel_matrices(A, B, X)
    print(M)
    generator = gauss_seidel_steps(A, X, B)
    y = X
    for i in range(5):
        for j in range(5):
            y = next(generator)
        print((i + 1) * 5, ": ", y)
