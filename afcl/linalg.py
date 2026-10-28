"""Numerical linear-algebra helpers for AFCL.

Everything in AFCL reduces to solving symmetric positive-(semi)definite systems
of the form ``(F^T F + gamma I) X = B``.  We go through Cholesky where possible
and degrade gracefully to an eigenvalue-truncated pseudo-inverse when the
system is singular (which can happen when ``gamma = 0`` and ``N_k < l_e``).

All AFCL state is held in float64.  The recursion in Eq. (8)-(9) of the paper
chains several explicit matrix inverses, and float32 loses the equivalence to
centralised joint learning (Theorem 1) after a handful of rounds.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

__all__ = ["spd_solve", "spd_inv", "symmetrize"]

_JITTER = 1e-10


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Force exact symmetry, undoing float round-off in ``F.T @ F``."""
    return 0.5 * (A + A.T)


def spd_solve(A: np.ndarray, B: np.ndarray, assume_spd: bool = True) -> np.ndarray:
    """Return ``A^{-1} B`` for symmetric positive-(semi)definite ``A``.

    Parameters
    ----------
    A : (n, n) array
    B : (n, m) array -- ``m == 0`` is allowed and returns an (n, 0) array.
    assume_spd : try Cholesky first.  Set False to go straight to the
        eigendecomposition path.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if B.ndim == 1:
        B = B[:, None]
    if B.shape[1] == 0:  # zero-width blocks appear whenever a client adds no new class
        return np.zeros((A.shape[0], 0), dtype=np.float64)

    if assume_spd:
        try:
            c, low = cho_factor(symmetrize(A), lower=True, check_finite=False)
            return cho_solve((c, low), B, check_finite=False)
        except Exception:
            pass

    # Cholesky failed => A is singular or indefinite.  Use a truncated
    # eigen-pseudo-inverse, which is the Moore-Penrose inverse for symmetric A.
    w, V = np.linalg.eigh(symmetrize(A))
    cutoff = max(abs(w[-1]), 1.0) * _JITTER
    inv_w = np.where(np.abs(w) > cutoff, 1.0 / np.where(w == 0, 1.0, w), 0.0)
    return V @ (inv_w[:, None] * (V.T @ B))


def spd_inv(A: np.ndarray) -> np.ndarray:
    """Return ``A^{-1}`` (pseudo-inverse if singular)."""
    n = A.shape[0]
    return spd_solve(A, np.eye(n, dtype=np.float64))
