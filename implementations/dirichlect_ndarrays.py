import math

import numpy as np

from implementations.helpers import N_l


# Dirichlet Randwertproblem
def dirichlect_randwert_a_l(stufenindex_l: int) -> np.ndarray:
    """
    Generiert die Matrix A_\\ell für das Dirichlet Randwertproblem auf einem homogenen Gitter

    :param stufenindex_l: Gitterstufe
    :return: A_\\ell
    """
    assert stufenindex_l >= 0
    n_l = N_l(stufenindex_l)
    a_l = (n_l + 1) * (n_l + 1) * (np.diag([-1. for _ in range(n_l - 1)], 1)
                                   + 2 * np.eye(n_l, dtype=np.float64)
                                   + np.diag([-1. for _ in range(n_l - 1)], -1))
    return a_l


def fourier_mode(stufenindex_l: int, mode: int, scale: bool = False) -> np.ndarray:
    """
    Gibt Eigenvektoren der Matrix A_\\ell an

    :param stufenindex_l: Gitterstufe
    :param mode: j
    :param scale: Gibt an ob der Vektor normiert werden soll
    :return: e^{\\ell, j}
    """
    assert stufenindex_l >= 0
    n_l = N_l(stufenindex_l)
    h_l = 1 / (n_l + 1)
    assert mode > 0
    assert mode <= n_l
    x = np.sin(mode * np.pi * h_l * np.arange(1, n_l + 1, dtype=np.float64))
    if not scale:
        x *= math.sqrt(2 * h_l)
    return x


def eigenvalues(stufenindex_l: int, mode: int, w: float) -> np.ndarray:
    """
    Gibt die Eigenwerte der Iterationsmatrix des gedämpften Jacobi Verfahrens an

    :param stufenindex_l: Gitterstufe
    :param mode: j
    :param w: Dämpfungsparameter
    :return: EW
    """
    assert stufenindex_l >= 0
    n_l = N_l(stufenindex_l)
    assert mode > 0
    assert mode <= n_l
    h_l = 1 / (n_l + 1)
    return 1 - 4 * w * np.sin(mode * np.pi * h_l / 2) ** 2
