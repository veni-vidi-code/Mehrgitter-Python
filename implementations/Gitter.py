import numpy as np

from implementations.helpers import MATRIXFOLGENFUNKTION, N_l


def standard_schrittweitenfolge(stufenindex_l: int) -> np.ndarray:
    assert stufenindex_l >= 0
    h = 2 ** (-(stufenindex_l + 1))
    return np.arange(start=h, stop=1, step=h, dtype=np.float64)


class Gitterhierachie:
    """
    Class which should construct the Omega l's and gives the Prolongation and Restriktion
    """

    def __init__(self, gitterfolgen: MATRIXFOLGENFUNKTION,
                 restriktionsmatrix: MATRIXFOLGENFUNKTION,
                 prolongationsmatrix: MATRIXFOLGENFUNKTION):
        self.get_gitterfolge = gitterfolgen
        self.get_restriktionsmatrix = restriktionsmatrix
        self.get_prolongationsmatrix = prolongationsmatrix


def _three_elem_restriction(elems: list[float]):
    assert len(elems) == 3

    def restriction(stufenindex_l: int) -> np.ndarray:
        assert stufenindex_l >= 0
        n_l = N_l(stufenindex_l)
        n_l_minus_1 = N_l(stufenindex_l - 1)
        x = np.zeros((n_l_minus_1, n_l))
        for i in range(n_l_minus_1):
            for j in range(3):
                x[i][2 * i + j] = elems[j]
        return x

    return restriction


def _three_elem_prolongation(elems: list[float]):
    assert len(elems) == 3

    def prolongation(stufenindex_l: int) -> np.ndarray:
        assert stufenindex_l >= 0
        n_l = N_l(stufenindex_l)
        n_l_minus_1 = N_l(stufenindex_l - 1)
        x = np.zeros((n_l, n_l_minus_1))
        for i in range(n_l_minus_1):
            for j in range(3):
                x[2 * i + j][i] = elems[j]
        return x

    return prolongation


linear_restriction = _three_elem_restriction([0.25, 0.5, 0.25])
trivial_restriction = _three_elem_restriction([0, 1, 0])
linear_prolongation = _three_elem_prolongation([0.5, 1, 0.5])
trivial_prolongation = _three_elem_prolongation([0, 1, 0])

LINEAR_GITTERHIERACHIE = Gitterhierachie(standard_schrittweitenfolge, linear_restriction, linear_prolongation)
TRIVIAL_GITTERHIERACHIE = Gitterhierachie(standard_schrittweitenfolge, trivial_restriction, trivial_prolongation)
