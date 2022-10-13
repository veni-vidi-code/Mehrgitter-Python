import numpy as np


def gauss_seidel_test(a: np.ndarray, diagonals: np.ndarray, x: np.ndarray):
    assert a.ndim == 2
    assert diagonals.ndim == 1
    assert x.ndim == 1
    assert np.all(diagonals)


def _gauss_seidel_matrices(a: np.ndarray, b: np.ndarray, x: np.ndarray, w: float = 1):
    diagonals = np.diag(a)  # depending on the version used this might be a view. do not write to this!
    d = np.diag(diagonals)  # same here
    gauss_seidel_test(a, diagonals, x)
    n = np.linalg.inv(d + w * np.tril(a, -1))
    m = np.dot(n, ((1 - w) * d) - w * np.triu(a, 1))
    return m, w * np.dot(n, b)


def _gauss_seidel_step(m: np.ndarray, nb: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Performs gauss_seidel step without tests

    :param m: D^-1(D-A)
    :param n: D^-1
    :param x: x_{m}
    :return: x_{m+1}
    """
    return np.dot(m, x) + nb


def gauss_seidel_step(a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1):
    """
    Performs one gauss_seidel Step
    """
    m, nb = _gauss_seidel_matrices(a, b, x, w)
    return _gauss_seidel_step(m, nb, x)


def gauss_seidel_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1):
    """
    Generator to perform many gauss_seidel steps
    """
    m, nb = _gauss_seidel_matrices(a, b, x, w)
    y = x.copy()
    total_steps = 0
    try:
        while True:
            y = _gauss_seidel_step(m, nb, y)
            total_steps += 1
            new_w = yield y
            if new_w is not None:
                m, nb = _gauss_seidel_matrices(a, b, x, new_w)
    except GeneratorExit:
        return y, total_steps


def n_gauss_seidel_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, n: int, w: float = 1):
    generator = gauss_seidel_steps(a, x, b, w)
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

    # example for Relaxationsverfahren
    print("--------------------------------------------------")
    w = 1.0648
    generator = gauss_seidel_steps(A, X, B, w)
    y = X
    for i in range(3):
        for j in range(5):
            y = next(generator)
        print((i + 1) * 5, ": ", y)