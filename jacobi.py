import numpy as np


def jacobi_test(a: np.ndarray, diagonals: np.ndarray, x: np.ndarray):
    assert a.ndim == 2
    assert diagonals.ndim == 1
    assert x.ndim == 1
    assert np.all(diagonals)


def _jacobi_matrices(a: np.ndarray, b: np.ndarray, x: np.ndarray):
    diagonals = np.diag(a)  # depending on the version used this might be a view. do not write to this!
    d = np.diag(diagonals)  # same here
    jacobi_test(a, diagonals, x)
    n = np.diag((1 / diagonals))
    m = np.dot(n, (d - a))
    nb = np.dot(n, b)
    return m, nb


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
    m, nb = _jacobi_matrices(a, b, x)
    return _jacobi_step(m, nb, x)


def jacobi_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray):
    """
    Generator to perform many jacobi steps

    :param b:
    :param a:
    :param x:
    :return:
    """
    m, nb = _jacobi_matrices(a, b, x)
    y = x.copy()
    total_steps = 0
    try:
        while True:
            y = _jacobi_step(m, nb, y)
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


if __name__ == "__main__":
    # Example from the book
    A = np.array([[0.7, -0.4], [-0.2, 0.5]])
    X = np.array([21, -19])
    B = np.array([0.3, 0.3])
    M, NB = _jacobi_matrices(A, B, X)
    print(M)
    generator = jacobi_steps(A, X, B)
    y = X
    for i in range(4):
        for j in range(15):
            y = next(generator)
        print((i + 1) * 15, ": ", y)
