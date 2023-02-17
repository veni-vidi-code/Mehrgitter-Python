from enum import Enum


class MG_VIS_Matrizes(Enum):
    E = 1  # Einheitsmatrix
    P = 2  # Prolongation
    R = 3  # Restrikion
    G_v1 = 4  # Vorglättung
    G_v2 = 5  # Nachglättung

    def __str__(self):
        if self is MG_VIS_Matrizes.G_v1:
            return "$$G^{v_1}$$"
        elif self is MG_VIS_Matrizes.G_v2:
            return "$$G^{v_2}$$"
        else:
            return f"$${self.name}$$"


def mehrgitterverfahren_visualization(stufenindex_l: int, gamma: int, rec=None) \
        -> tuple[list[int], list[MG_VIS_Matrizes]]:
    """
    Gibt die Reihenfolge der Gitterstufen und die Matrizen, die auf diesen angewendet werden, zurück
    (Mehrgitterverfahren)

    :param stufenindex_l: Gitterstufe
    :param gamma: \\gamma
    :param rec: Erlaubt die Nutzung einer gecachten Version der Funktion
    """
    if stufenindex_l == 0:
        return [0, 0], [MG_VIS_Matrizes.E]
    else:
        res_levels = [stufenindex_l, stufenindex_l]
        res_matrizes = [MG_VIS_Matrizes.G_v1, MG_VIS_Matrizes.R]
        if stufenindex_l == 1:
            gamma = min(gamma, 1)
        for i in range(gamma):
            if rec is None:
                a, b = mehrgitterverfahren_visualization(stufenindex_l - 1, gamma)
            else:
                a, b = rec(stufenindex_l - 1, gamma)
            if i != 0:
                a.pop(0)
            res_levels.extend(a)
            res_matrizes.extend(b)
        res_levels.extend([stufenindex_l, stufenindex_l])
        res_matrizes.extend([MG_VIS_Matrizes.P, MG_VIS_Matrizes.G_v2])
        return res_levels, res_matrizes


def vollstaendiges_mehrgitterverfahren_visualization(stufenindex_l: int, gamma: int,
                                                     mehrgitter_visualization=mehrgitterverfahren_visualization) \
        -> tuple[list[int], list[MG_VIS_Matrizes]]:
    """
    Gibt die Reihenfolge der Gitterstufen und die Matrizen, die auf diesen angewendet werden, zurück
    (Vollständiges Mehrgitterverfahren)

    :param stufenindex_l: Gitterstufe
    :param gamma: \\gamma
    :param mehrgitter_visualization: Optional: Vis. für das Mehrgitterverfahren
    :return:
    """
    if stufenindex_l == 0:
        return [0, 0], [MG_VIS_Matrizes.E]
    else:
        res_levels = [0, 0]
        res_matrizes = [MG_VIS_Matrizes.E]
        for current in range(1, stufenindex_l + 1):
            res_matrizes.extend([MG_VIS_Matrizes.P, MG_VIS_Matrizes.G_v2])
            res_levels.extend([current, current])
        a, b = mehrgitter_visualization(stufenindex_l, gamma)
        a.pop(0)
        res_levels.extend(a)
        res_matrizes.extend(b)
        return res_levels, res_matrizes
