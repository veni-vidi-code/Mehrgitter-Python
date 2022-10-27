from typing import Callable, Optional, Tuple

import numpy as np

from implementations.Gitter import linear_prolongation, linear_restriction
from implementations.dirichlect_ndarrays import dirichlect_randwert_a_l
from implementations.helpers import iter_steps_generatordef, n_steps_of_generator

import sys

rec_limit = sys.getrecursionlimit() - 1


def mehrgitterverfahren_rekursiv(stufenindex_l: int, v1: int, v2: int, u: np.ndarray, f: np.ndarray,
                                 psi_vor_matrice: Callable[[np.ndarray, np.ndarray, np.ndarray, float, bool],
                                                           Tuple[np.ndarray, np.ndarray]],
                                 psi_nach_matrice: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float, bool],
                                                                     Tuple[np.ndarray, np.ndarray]]] = None,
                                 w1: float = 1, w2: float = 1, gamma: int = 1, *,
                                 a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
                                 prolongation: Callable[[int], np.ndarray] = linear_prolongation,
                                 restriction: Callable[[int], np.ndarray] = linear_restriction) -> np.ndarray:
    if stufenindex_l >= rec_limit:
        raise RecursionError("Recursion limit exceeded")  # We know it will reach this anyway, so we can save some time
    if stufenindex_l == 0:
        return 1 / (a_func(0)[0]) * f
    else:
        a_l = a_func(stufenindex_l)
        if psi_nach_matrice is None:
            psi_nach_matrice = psi_vor_matrice
        psi_1_gen = iter_steps_generatordef(psi_vor_matrice, a_l, u, f, w1)
        u = n_steps_of_generator(psi_1_gen, v1)
        d = restriction(stufenindex_l) @ (a_l @ u - f)
        e = np.zeros_like(d)
        for _ in range(gamma):
            e = mehrgitterverfahren_rekursiv(stufenindex_l - 1, v1, v2, e, d, psi_vor_matrice, psi_nach_matrice, w1, w2,
                                             gamma, a_func=a_func, prolongation=prolongation, restriction=restriction)
        u = u - prolongation(stufenindex_l) @ e
        psi_2_gen = iter_steps_generatordef(psi_nach_matrice, a_l, u, f, w2)
        u = n_steps_of_generator(psi_2_gen, v2)
        return u


def mehgitterverfaren_steps(stufenindex_l: int, v1: int, v2: int, u: np.ndarray, f: np.ndarray,
                                psi_vor_matrice: Callable[[np.ndarray, np.ndarray, np.ndarray, float, bool],
                                                          Tuple[np.ndarray, np.ndarray]],
                                psi_nach_matrice: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float, bool],
                                                                    Tuple[np.ndarray, np.ndarray]]] = None,
                                w1: float = 1, w2: float = 1, gamma: int = 1, *,
                                a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
                                prolongation: Callable[[int], np.ndarray] = linear_prolongation,
                                restriction: Callable[[int], np.ndarray] = linear_restriction):
    x = u.copy()
    total_steps = 0
    try:
        while True:
            x = mehrgitterverfahren_rekursiv(stufenindex_l, v1, v2, x, f, psi_vor_matrice, psi_nach_matrice, w1, w2,
                                             gamma, a_func=a_func, prolongation=prolongation, restriction=restriction)
            total_steps += 1
            yield x
    except GeneratorExit:
        return x, total_steps

