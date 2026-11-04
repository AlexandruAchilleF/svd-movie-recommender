"""Numerical linear algebra implemented from scratch.

Only basic NumPy array arithmetic is used here; no ``numpy.linalg``
routines are called, so every factorization is computed by our own code.
"""

import numpy as np

# Values below this threshold are treated as zero.
EPSILON = 1e-15


def vector_norm(x: np.ndarray) -> float:
    """Return the Euclidean (L2) norm of a vector."""
    return float(np.sqrt(np.sum(x**2)))


def frobenius_norm(a: np.ndarray) -> float:
    """Return the Frobenius norm of a matrix."""
    return float(np.sqrt(np.sum(a**2)))


def qr_householder(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the QR factorization ``A = Q R`` using Householder reflections.

    Args:
        a: Matrix of shape ``(m, n)``.

    Returns:
        ``(Q, R)`` where ``Q`` is an ``(m, m)`` orthogonal matrix and ``R`` is
        an ``(m, n)`` upper triangular matrix.
    """
    r = np.array(a, dtype=float)
    m, n = r.shape
    q = np.eye(m)

    for k in range(min(m, n)):
        x = r[k:, k]
        norm_x = vector_norm(x)
        if len(x) <= 1 or norm_x < EPSILON:
            continue

        # Reflection vector v = x + sign(x_0) * ||x|| * e_1.
        # Choosing the sign of x_0 avoids catastrophic cancellation.
        v = x.copy()
        v[0] += (1.0 if v[0] >= 0 else -1.0) * norm_x
        v_norm_sq = float(np.sum(v**2))
        if v_norm_sq < EPSILON:
            continue

        # H = I - 2 v v^T / (v^T v), applied only to the trailing block.
        h = np.eye(m)
        h[k:, k:] -= 2.0 * np.outer(v, v) / v_norm_sq

        r = h @ r
        q = q @ h

    return q, r


def svd(
    a: np.ndarray,
    k: int | None = None,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a truncated SVD ``A ≈ U Σ Vᵀ`` via QR iteration on ``AᵀA``.

    The eigen-decomposition ``AᵀA = V Λ Vᵀ`` is obtained with the unshifted
    QR algorithm; singular values are ``σ_i = sqrt(λ_i)`` and the left
    singular vectors are recovered as ``u_i = A v_i / σ_i``.

    Args:
        a: Matrix of shape ``(m, n)``.
        k: Number of components to keep. Defaults to ``min(m, n)``.
        max_iter: Maximum number of QR iterations.
        tol: Convergence threshold on the Frobenius norm of the strictly
            lower triangular part of the iterated matrix.

    Returns:
        ``(U, Sigma, Vt)`` with shapes ``(m, k)``, ``(k, k)`` and ``(k, n)``,
        singular values sorted in descending order.
    """
    a = np.array(a, dtype=float)
    m, n = a.shape
    if k is None:
        k = min(m, n)

    t = a.T @ a
    v = np.eye(n)

    for _ in range(max_iter):
        q, r = qr_householder(t)
        t = r @ q
        # Re-symmetrize to stop floating-point drift from accumulating.
        t = (t + t.T) / 2.0
        v = v @ q
        if frobenius_norm(np.tril(t, -1)) < tol:
            break

    # Tiny negative eigenvalues can appear from rounding; clamp them to zero.
    singular_values = np.sqrt(np.maximum(np.diag(t), 0.0))

    order = np.argsort(singular_values)[::-1][:k]
    singular_values = singular_values[order]
    v = v[:, order]

    u = np.zeros((m, k))
    nonzero = singular_values > 1e-12
    u[:, nonzero] = (a @ v[:, nonzero]) / singular_values[nonzero]

    return u, np.diag(singular_values), v.T
