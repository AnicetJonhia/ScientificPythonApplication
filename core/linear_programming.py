
from typing import List, Tuple
try:
    import pulp
except Exception:
    pulp = None


def solve_lp(c: List[float], A: List[List[float]], b: List[float], maximize=True) -> Tuple[dict, float]:
    """Résout un problème LP simple : max/min c^T x s.t. Ax <= b, x >= 0.
    Retourne un dictionnaire {i: valeur} et la valeur optimale.
    Nécessite pulp.
    """
    if pulp is None:
        raise RuntimeError("PuLP n'est pas installé.")

    n = len(c)
    prob = pulp.LpProblem("LP_problem", pulp.LpMaximize if maximize else pulp.LpMinimize)
    vars = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(n)]

    # objectif
    prob += pulp.lpDot(c, vars)

    # contraintes Ax <= b
    for row, bi in zip(A, b):
        prob += pulp.lpDot(row, vars) <= bi

    status = prob.solve()

    solution = {f"x{i}": pulp.value(var) for i, var in enumerate(vars)}
    opt = pulp.value(prob.objective)
    return solution, opt