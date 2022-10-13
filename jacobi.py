import numpy as np


def jacobi_test(a: np.ndarray, diagonals: np.ndarray, x: np.ndarray):
    assert a.ndim == 2
    assert diagonals.ndim == 1
    assert x.ndim == 1
    assert np.all(diagonals)


def _jacobi_matrices(a: np.ndarray, b: np.ndarray, x: np.ndarray, w=1):
    diagonals = np.diag(a)  # depending on the version used this might be a view. do not write to this!
    jacobi_test(a, diagonals, x)
    n = w * np.diag((1 / diagonals))
    m = np.identity(a.shape[0], dtype=a.dtype) - np.dot(n, a)
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


def jacobi_step(a: np.ndarray, x: np.ndarray, b: np.ndarray, w=1):
    """
    Performs one Jacobi Step
    """
    m, nb = _jacobi_matrices(a, b, x, w)
    return _jacobi_step(m, nb, x)


def jacobi_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, w=1):
    """
    Generator to perform many jacobi steps
    """
    m, nb = _jacobi_matrices(a, b, x, w)
    y = x.copy()
    total_steps = 0
    try:
        while True:
            y = _jacobi_step(m, nb, y)
            total_steps += 1
            new_w = yield y
            if new_w is not None:
                m, nb = _jacobi_matrices(a, b, x, new_w)
    except GeneratorExit:
        return y, total_steps


def n_jacobi_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, n: int, w=1):
    generator = jacobi_steps(a, x, b, w)
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
    M, NB = _jacobi_matrices(A, B, X)
    print(M)
    generator = jacobi_steps(A, X, B)
    y = X
    for i in range(4):
        for j in range(15):
            y = next(generator)
        print((i + 1) * 15, ": ", y)
