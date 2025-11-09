
import numpy as np


def _forward_sub(L, b):
    """Forward substitution for lower-triangular L with unit/non-unit diagonal."""
    n = L.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        denom = L[i, i] if L[i, i] != 0 else 1.0
        y[i] = y[i] / denom
    return y


def _backward_sub(U, y):
    """Backward substitution for upper-triangular U."""
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:]))
        denom = U[i, i]
        if denom == 0:
            raise np.linalg.LinAlgError("Matrice singulière détectée lors de la substitution arrière.")
        x[i] = x[i] / denom
    return x


def lu_decompose(A):
    """Compute LU decomposition using Doolittle's method.

    Returns L, U where A = L @ U. Raises LinAlgError if decomposition fails.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)
    for i in range(n):
        # U row
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))
        # L column
        if U[i, i] == 0:
            raise np.linalg.LinAlgError("Pivot nul lors de la décomposition LU.")
        L[i, i] = 1.0
        for k in range(i + 1, n):
            L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]
    return L, U


def lu_solve(A, b):
    """Solve Ax=b using LU decomposition (no pivoting)."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    L, U = lu_decompose(A)
    y = _forward_sub(L, b)
    x = _backward_sub(U, y)
    return x


def gaussian_elimination(A, b):
    """Simple Gaussian elimination with partial pivoting returning solution x.

    This implementation is educational (returns only the final solution).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).astype(float).flatten()
    n = A.shape[0]
    # Augmented matrix
    M = np.hstack((A.copy(), b.reshape(-1, 1)))
    for k in range(n):
        # partial pivot
        max_row = np.argmax(np.abs(M[k:, k])) + k
        if M[max_row, k] == 0:
            continue
        if max_row != k:
            M[[k, max_row]] = M[[max_row, k]]
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] = M[i, k:] - factor * M[k, k:]
    # back substitution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if M[i, i] == 0:
            raise np.linalg.LinAlgError('Système singulier ou indéterminé (pivot nul).')
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])) / M[i, i]
    return x


def solve_linear_system(A, b, method='auto'):
    """
    Résout un système linéaire A x = b.

    Parameters
    ----------
    A : array-like, shape (n,n)
        Coefficient matrix.
    b : array-like, shape (n,) or (n,1)
        Right-hand side vector.
    method : {'auto','direct','gauss','lu','lstsq'}
        Méthode de résolution à utiliser:
        - 'auto': essaie numpy.linalg.solve puis fallback vers lstsq
        - 'direct': numpy.linalg.solve (lève erreur si singulière)
        - 'gauss': gaussian elimination (partial pivot)
        - 'lu': LU decomposition + forward/back substitution
        - 'lstsq': least-squares solution

    Returns
    -------
    x : ndarray
        Solution vector.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if A.ndim != 2:
        raise ValueError("A doit être une matrice 2D.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A doit être carrée pour les méthodes directes/lu/gauss.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Le nombre de lignes de A doit correspondre à la taille de b.")

    if method == 'auto' or method == 'direct':
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            if method == 'direct':
                raise
            # else continue to fallback

    if method == 'gauss':
        return gaussian_elimination(A, b)
    if method == 'lu':
        return lu_solve(A, b)
    if method == 'lstsq':
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return x

    # fallback default: least squares
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x