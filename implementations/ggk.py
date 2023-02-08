# grobgitterkorrekturverfahren
from typing import Callable, Any, Tuple

import numpy as np

from implementations.dirichlect_ndarrays import dirichlect_randwert_a_l
from implementations.helpers import iter_step, iter_steps_generatordef, n_steps_of_generator

from implementations.Gitter import linear_prolongation, linear_restriction


def ggk_matrices(stufenindex_l: int, *, a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
                 prolongation: Callable[[int], np.ndarray] = linear_prolongation,
                 restriction: Callable[[int], np.ndarray] = linear_restriction):
    """
    Gibt Funktion zurück, die ggk-matrizen berechnet

    :param stufenindex_l: Gitterstufe
    :param a_func: Funktion die a_l berechnet
    :param prolongation: Funktion die Prolongationsmatrix angibt
    :param restriction: Funktion die Restriktionsmatrix angibt
    """
    p_l = prolongation(stufenindex_l)
    r_l = restriction(stufenindex_l)
    a_l_minus_1 = a_func(stufenindex_l - 1)
    a_l = a_func(stufenindex_l)

    def _ggk_matrices(a: Any, f: np.ndarray, x: Any, w: float = 1, nb: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Note: a, w, x will get ignored. They are only there to make the interface
        compatible with iter_steps_generatordef
        """
        if w != 1:
            raise ValueError("GGK does not support relaxation")
        n = np.dot(p_l, np.dot(np.linalg.inv(a_l_minus_1), r_l))
        m = np.identity(a_l.shape[0], dtype=np.float64) - np.dot(n, a_l)
        if nb:
            return m, np.dot(n, f)
        return m, n

    return _ggk_matrices


def ggk_step(stufenindex_l: int, u: np.ndarray, f: np.ndarray, *,
             a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
             prolongation: Callable[[int], np.ndarray] = linear_prolongation,
             restriction: Callable[[int], np.ndarray] = linear_restriction):
    """
    Führt einen ggk-Schritt durch
    Au=f

    :param stufenindex_l: Gitterstufe
    :param a_func: Funktion die A_l berechnet
    :param prolongation: Funktion die Prolongationsmatrix angibt
    :param restriction: Funktion die Restriktionsmatrix angibt
    :return: Ergebnis des ggk-Schritts
    """
    fun = ggk_matrices(stufenindex_l, a_func=a_func, prolongation=prolongation, restriction=restriction)
    m, nb = fun(None, f, None)
    return iter_step(m, nb, u)


def ggk_steps(stufenindex_l: int, u: np.ndarray, f: np.ndarray, *,
              a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l):
    """
    Generator um mehrere ggk-Schritte durchzuführen
    (Sie sollten im allgemeinen nicht mehr als einen Schritt durchführen)
    """
    return iter_steps_generatordef((ggk_matrices(stufenindex_l, a_func=a_func)), None, u, f, 1)  # type: ignore


def n_ggk_steps(stufenindex_l: int,
                u: np.ndarray,
                f: np.ndarray,
                n: int, *,
                a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l):
    """
    (Sie sollten im allgemeinen nicht mehr als einen Schritt durchführen)
    """
    generator = ggk_steps(stufenindex_l, u, f, a_func=a_func)
    return n_steps_of_generator(generator, n)


def ggk_Psi_l(stufenindex_l: int, e: np.ndarray, *,
              a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
              prolongation: Callable[[int], np.ndarray] = linear_prolongation,
              restriction: Callable[[int], np.ndarray] = linear_restriction):
    """
    Gibt die GGK Korrektur für die Lösung u bzw e zurück (f = 0)
    """
    return ggk_step(stufenindex_l, e, np.zeros_like(e), a_func=a_func, prolongation=prolongation,
                    restriction=restriction)
