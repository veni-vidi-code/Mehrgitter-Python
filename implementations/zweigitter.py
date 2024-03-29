from typing import Callable, Tuple, Optional, Generator, Any

import numpy as np

from implementations.Gitter import linear_prolongation, linear_restriction
from implementations.dirichlect_ndarrays import dirichlect_randwert_a_l
from implementations.helpers import iter_steps_generatordef, n_steps_of_generator
from implementations.ggk import ggk_matrices


def zweigitter_steps(stufenindex_l: int, v1: int, v2: int, x: np.ndarray, y: np.ndarray,
                     psi_vor_matrice: Callable[[np.ndarray, np.ndarray, np.ndarray, float, bool],
                                               Tuple[np.ndarray, np.ndarray]],
                     psi_nach_matrice: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float, bool],
                                                         Tuple[np.ndarray, np.ndarray]]] = None,
                     w1: float = 1, w2: float = 1, *,
                     a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
                     prolongation: Callable[[int], np.ndarray] = linear_prolongation,
                     restriction: Callable[[int], np.ndarray] = linear_restriction)\
        -> Generator[np.ndarray, Any, tuple[Any, int]]:
    """
    Generator um mehrere Zweigitter-Schritte auszuführen

    :param stufenindex_l: Gitterstufe (Anfang)
    :param v1: Anzahl Vorglätterungsschritte
    :param v2: Anzahl Nachglätterungsschritte
    :param psi_vor_matrice: Vorglätter
    :param psi_nach_matrice: Nachglätter
    :param w1: Vorglätterrelaxionsparameter
    :param w2: Nachglätterrelaxionsparameter
    :param a_func: A Matrix, default: dirichlet
    :param prolongation: prolongation matrix, default: linear
    :param restriction: restriction matrix, default: linear
    :param x: Näherungslösung. Wird nicht benötigt, aber wird hier auf korrekte Dimension geprüft
    """
    a_l = a_func(stufenindex_l)
    if psi_nach_matrice is None:
        psi_nach_matrice = psi_vor_matrice
    psi_1 = psi_vor_matrice(a_l, y, x, w1, False)
    psi_2 = psi_nach_matrice(a_l, y, x, w2, False)
    ggk = ggk_matrices(stufenindex_l, a_func=a_func, prolongation=prolongation, restriction=restriction)
    ggk = ggk(None, y, None, 1, False)
    m = np.linalg.matrix_power(psi_1[0], v1) @ ggk[0] @ np.linalg.matrix_power(psi_2[0], v2)

    n = np.zeros_like(ggk[1])
    for _ in range(v1):
        n = psi_1[0] @ n + psi_1[1]
    n = ggk[0] @ n + ggk[1]
    for _ in range(v2):
        n = psi_2[0] @ n + psi_2[1]
    nb = np.dot(n, y)
    return iter_steps_generatordef((m, nb), a_l, x, y)


def n_zweigitter_steps(stufenindex_l: int, v1: int, v2: int, x: np.ndarray, y: np.ndarray, n: int,
                       psi_vor_matrice: Callable[[np.ndarray, np.ndarray, np.ndarray, float],
                                                 Tuple[np.ndarray, np.ndarray]],
                       psi_nach_matrice: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float],
                                                           Tuple[np.ndarray, np.ndarray]]] = None,
                       w1: float = 1, w2: float = 1, *,
                       a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
                       prolongation: Callable[[int], np.ndarray] = linear_prolongation,
                       restriction: Callable[[int], np.ndarray] = linear_restriction):
    """
    Führt n Zweigitter-Schritte aus

    :param stufenindex_l: Gitterstufe (Anfang)
    :param v1: Anzahl Vorglätterungsschritte
    :param v2: Anzahl Nachglätterungsschritte
    :param psi_vor_matrice: Vorglätter
    :param psi_nach_matrice: Nachglätter
    :param w1: Vorglätterrelaxionsparameter
    :param w2: Nachglätterrelaxionsparameter
    :param a_func: A Matrix, default: dirichlet
    :param prolongation: prolongation matrix, default: linear
    :param restriction: restriction matrix, default: linear
    :param x: Näherungslösung. Wird nicht benötigt, aber wird hier auf korrekte Dimension geprüft
    """
    generator = zweigitter_steps(stufenindex_l, v1, v2, x, y, psi_vor_matrice, psi_nach_matrice, w1, w2,
                                 a_func=a_func, prolongation=prolongation, restriction=restriction)
    return n_steps_of_generator(generator, n)


def zweigitter_step(stufenindex_l: int, v1: int, v2: int, x: np.ndarray, y: np.ndarray,
                    psi_vor_matrice: Callable[[np.ndarray, np.ndarray, np.ndarray, float],
                                              Tuple[np.ndarray, np.ndarray]],
                    psi_nach_matrice: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float],
                                                        Tuple[np.ndarray, np.ndarray]]] = None,
                    w1: float = 1, w2: float = 1, *,
                    a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
                    prolongation: Callable[[int], np.ndarray] = linear_prolongation,
                    restriction: Callable[[int], np.ndarray] = linear_restriction):
    """
    Hierbei handelt es sich um einen Wrapper um n_zweigitter_steps, um die gleiche Schnittstelle wie
    die anderen Methoden zu bieten
    """
    return n_zweigitter_steps(stufenindex_l, v1, v2, x, y, 1, psi_vor_matrice, psi_nach_matrice, w1, w2,
                              a_func=a_func, prolongation=prolongation, restriction=restriction)
