import numpy as np

from implementations.helpers import iter_step, iter_tests, iter_steps_generatordef, n_steps_of_generator


def gauss_seidel_matrices(a: np.ndarray, b: np.ndarray, x: np.ndarray, w: float = 1, nb: bool = True):
    diagonals = np.diag(a)  # depending on the version used this might be a view. do not write to this!
    d = np.diag(diagonals)  # same here
    iter_tests(a, diagonals, x)
    n = np.linalg.inv(d + w * np.tril(a, -1))
    m = np.dot(n, ((1 - w) * d) - w * np.triu(a, 1))
    if nb:
        return m, w * np.dot(n, b)
    else:
        return m, w * n


def gauss_seidel_step(a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1):
    """
    Performs one gauss_seidel Step
    """
    m, nb = gauss_seidel_matrices(a, b, x, w)
    return iter_step(m, nb, x)


def gauss_seidel_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1):
    """
    Generator to perform many gauss_seidel steps
    """
    return iter_steps_generatordef(gauss_seidel_matrices, a, x, b, w)


def n_gauss_seidel_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, n: int, w: float = 1):
    generator = gauss_seidel_steps(a, x, b, w)
    return n_steps_of_generator(generator, n)


if __name__ == "__main__":
    # Example from the book
    A = np.array([[0.7, -0.4], [-0.2, 0.5]], np.float64)
    X = np.array([21, -19], np.float64)
    B = np.array([0.3, 0.3], np.float64)
    M, NB = gauss_seidel_matrices(A, B, X)
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
