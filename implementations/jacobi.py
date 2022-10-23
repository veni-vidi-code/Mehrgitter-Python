import numpy as np

from implementations.helpers import iter_step, iter_tests, iter_steps_generatordef, n_steps_of_generator


def jacobi_matrices(a: np.ndarray, b: np.ndarray, x: np.ndarray, w: float = 1, nb: bool = True):
    diagonals = np.diag(a)  # depending on the version used this might be a view. do not write to this!
    iter_tests(a, diagonals, x)
    n = w * np.diag((1 / diagonals))
    m = np.identity(a.shape[0], dtype=a.dtype) - np.dot(n, a)
    if nb:
        nb = np.dot(n, b)
    return m, nb


def jacobi_step(a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1):
    """
    Performs one Jacobi Step
    """
    m, nb = jacobi_matrices(a, b, x, w)
    return iter_step(m, nb, x)


def jacobi_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1):
    """
    Generator to perform many jacobi steps
    """
    return iter_steps_generatordef(jacobi_matrices, a, x, b, w)


def n_jacobi_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, n: int, w: float = 1):
    generator = jacobi_steps(a, x, b, w)
    return n_steps_of_generator(generator, n)


if __name__ == "__main__":
    # Example from the book
    A = np.array([[0.7, -0.4], [-0.2, 0.5]], np.float64)
    X = np.array([21, -19], np.float64)
    B = np.array([0.3, 0.3], np.float64)
    M, NB = jacobi_matrices(A, B, X)
    print(M)
    generator = jacobi_steps(A, X, B)
    y = X
    for i in range(4):
        for j in range(15):
            y = next(generator)
        print((i + 1) * 15, ": ", y)
