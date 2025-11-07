
import numpy as np

def solve_linear_system(A, b):
    
    """
    desc:
        Résout un système linéaire de la forme A·X = b.  
        Si la matrice A est singulière (non inversible), utilise la méthode des moindres carrés en repli.

    params:
        A : Matrice carrée (liste de listes ou tableau numpy) représentant les coefficients du système.  
        b : Vecteur ou tableau numpy représentant les constantes du système.

    return:
        x : Vecteur solution du système sous forme de tableau numpy.
             Si A est singulière, retourne la solution approchée en moindres carrés.
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if A.ndim != 2:
        raise ValueError("A doit être une matrice 2D.")
    if b.ndim > 1 and b.shape[1] != 1:
        raise ValueError("b doit être un vecteur 1D ou une colonne 2D.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Le nombre de lignes de A doit correspondre à la taille de b.")

    # essayer la résolution directe
    try:
        x = np.linalg.solve(A, b)
        return x
    except np.linalg.LinAlgError:
        # fallback : moindres carrés
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return x