# grobgitterkorrekturverfahren
from typing import Callable, Any, Tuple

import numpy as np

from implementations.dirichlect import dirichlect_randwert_a_l
from implementations.helpers import iter_step, iter_steps_generatordef, n_steps_of_generator

from implementations.Gitter import linear_prolongation, linear_restriction


def ggk_matrices(stufenindex_l: int, *, a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
                 prolongation: Callable[[int], np.ndarray] = linear_prolongation,
                 restriction: Callable[[int], np.ndarray] = linear_restriction):
    p_l = prolongation(stufenindex_l)
    r_l = restriction(stufenindex_l)
    a_l_minus_1 = a_func(stufenindex_l - 1)
    a_l = a_func(stufenindex_l)

    def _ggk_matrices(a: Any, f: np.ndarray, x: Any, w: float = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Note: a, w, x will get ignored. They are only there to make the interface
        compatible with iter_steps_generatordef
        """
        if w != 1:
            raise ValueError("GGK does not support relaxation")
        n = np.dot(p_l, np.dot(np.linalg.inv(a_l_minus_1), r_l))
        m = np.identity(a_l.shape[0], dtype=a_l.dtype) - np.dot(n, a_l)
        nb = np.dot(n, f)
        return m, nb

    return _ggk_matrices


def ggk_step(stufenindex_l: int, u: np.ndarray, f: np.ndarray, *,
             a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
             prolongation: Callable[[int], np.ndarray] = linear_prolongation,
             restriction: Callable[[int], np.ndarray] = linear_restriction):
    fun = ggk_matrices(stufenindex_l, a_func=a_func, prolongation=prolongation, restriction=restriction)
    m, nb = fun(None, f, None)
    return iter_step(m, nb, u)


def ggk_steps(stufenindex_l: int, u: np.ndarray, f: np.ndarray, *,
              a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l):
    """
    Generator to perform many ggk steps
    (You should generrally not do more than one step)
    """
    return iter_steps_generatordef((ggk_matrices(stufenindex_l, a_func=a_func)), None, u, f, 1)  # type: ignore


def n_ggk_steps(stufenindex_l: int,
                u: np.ndarray,
                f: np.ndarray,
                n: int, *,
                a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l):
    """
    (You should generrally not do more than one step)
    """
    generator = ggk_steps(stufenindex_l, u, f, a_func=a_func)
    return n_steps_of_generator(generator, n)


def ggk_Psi_l(stufenindex_l: int, e: np.ndarray, *,
              a_func: Callable[[int], np.ndarray] = dirichlect_randwert_a_l,
              prolongation: Callable[[int], np.ndarray] = linear_prolongation,
              restriction: Callable[[int], np.ndarray] = linear_restriction):
    """
    Returns the ggk correction for the solution u
    """
    return ggk_step(stufenindex_l, e, np.zeros_like(e), a_func=a_func, prolongation=prolongation,
                    restriction=restriction)
