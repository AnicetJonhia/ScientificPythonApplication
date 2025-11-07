
from typing import List, Tuple
try:
    import pulp
except Exception:
    pulp = None


def solve_lp(c: List[float], A: List[List[float]], b: List[float], maximize=True) -> Tuple[dict, float]:


    """
    desc:
        Résout un problème de programmation linéaire (LP) du type :
        - Maximiser ou minimiser cᵀx
        - Sous les contraintes Ax ≤ b et x ≥ 0
        Utilise la bibliothèque PuLP pour la modélisation et la résolution.

    params:
        c : Liste des coefficients de la fonction objectif (1D).
        A : Matrice des coefficients des contraintes (2D).
        b : Liste des bornes supérieures du système de contraintes (1D).
        maximize : Booléen indiquant s’il faut maximiser (True) ou minimiser (False) la fonction objectif.

    return:
        Un tuple contenant :
            - Un dictionnaire associant chaque variable à sa valeur optimale.
            - La valeur optimale de la fonction objectif.
    """

    
    if pulp is None:
        raise RuntimeError("PuLP n'est pas installé.")

    # basic validation
    if not isinstance(c, (list, tuple)) or len(c) == 0:
        raise ValueError("Le vecteur c doit être une liste non vide de coefficients.")
    n = len(c)
    if any(len(row) != n for row in A):
        raise ValueError("Toutes les lignes de A doivent avoir la même longueur que c.")
    if len(b) != len(A):
        raise ValueError("La longueur de b doit correspondre au nombre de contraintes (lignes de A).")

    prob = pulp.LpProblem("LP_problem", pulp.LpMaximize if maximize else pulp.LpMinimize)
    # variables non négatives par défaut
    vars = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(n)]

    # objectif
    prob += pulp.lpDot(c, vars)

    # contraintes Ax <= b
    for row, bi in zip(A, b):
        prob += pulp.lpDot(row, vars) <= bi

    status = prob.solve()

    status_str = pulp.LpStatus.get(status, str(status)) if hasattr(pulp, 'LpStatus') else str(status)
    solution = {f"x{i}": (pulp.value(var) if pulp.value(var) is not None else float('nan')) for i, var in enumerate(vars)}
    opt = pulp.value(prob.objective)
    return solution, opt