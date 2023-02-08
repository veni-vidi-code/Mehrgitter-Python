from typing import Generator, Any, Tuple

import numpy as np

from implementations.helpers import iter_step, iter_tests, iter_steps_generatordef, n_steps_of_generator


def gauss_seidel_matrices(a: np.ndarray, b: np.ndarray, x: np.ndarray, w: float = 1, nb: bool = True) \
        -> (np.ndarray, np.ndarray):
    """
    Berechnet M_\ell(w) und N_\ell(w)b für Gauss-Seidel

    :param a: Matrix A
    :param b: Ax = b
    :param x: Näherungslösung. Wird nicht benötigt, aber wird hier auf korrekte Dimension geprüft
    :param w: Relaxationsparameter
    :param nb: Setze auf False, wenn N statt NB zurückgegeben werden soll (z.B. bei wechselnden b)
    :return: M_\ell(w), N_\ell(w)b
    """
    diagonals = np.diag(a)  # depending on the version used this might be a view. do not write to this!
    d = np.diag(diagonals)  # same here
    iter_tests(a, diagonals, x)
    n = np.linalg.inv(d + w * np.tril(a, -1))
    m = np.dot(n, ((1 - w) * d) - w * np.triu(a, 1))
    if nb:
        return m, w * np.dot(n, b)
    else:
        return m, w * n


def gauss_seidel_step(a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1) -> np.ndarray:
    """
    Führt einen Gauss-Seidel-Schritt aus

    :param a: Matrix A
    :param b: Ax = b
    :param x: Näherungslösung. Wird nicht benötigt, aber wird hier auf korrekte Dimension geprüft
    :param w: Relaxationsparameter
    """
    m, nb = gauss_seidel_matrices(a, b, x, w)
    return iter_step(m, nb, x)


def gauss_seidel_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, w: float = 1) \
        -> Generator[np.ndarray, Any, Tuple[np.ndarray | None, int]]:
    """
    Generator um mehrere Gauss-Seidel-Schritte auszuführen

    :param a: Matrix A
    :param b: Ax = b
    :param x: Näherungslösung. Wird nicht benötigt, aber wird hier auf korrekte Dimension geprüft
    :param w: Relaxationsparameter
    """
    return iter_steps_generatordef(gauss_seidel_matrices, a, x, b, w)


def n_gauss_seidel_steps(a: np.ndarray, x: np.ndarray, b: np.ndarray, n: int, w: float = 1) -> np.ndarray:
    """
    Führt n Gauss-Seidel-Schritte aus

    :param a: Matrix A
    :param b: Ax = b
    :param x: Näherungslösung. Wird nicht benötigt, aber wird hier auf korrekte Dimension geprüft
    :param n: Anzahl Schritte
    :param w: Relaxationsparameter
    """
    generator = gauss_seidel_steps(a, x, b, w)
    return n_steps_of_generator(generator, n)


if __name__ == "__main__":
    """
    Der Code in diesem Block wird nur ausgeführt, wenn die Datei direkt ausgeführt wird.
    Wird sie importiert, wird der Code nicht ausgeführt.
    
    Er dient zum Testen der Funktionen anhand von bekannten Beispielen.
    """
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
