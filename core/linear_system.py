
import numpy as np

def solve_linear_system(A, b):
    """Résout AX = b. A et b doivent être des arrays numpy.
    Renvoie le vecteur solution ou lève une exception si singulier."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    # essayer la résolution directe
    try:
        x = np.linalg.solve(A, b)
        return x
    except np.linalg.LinAlgError:
        # fallback : moindres carrés
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x