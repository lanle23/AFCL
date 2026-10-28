"""Client side: analytic local training -- Section 3.3, Eqs. (3)-(7).

One forward pass over the local shard, two ridge solves, one Gram matrix.  No
gradients, no epochs, no optimiser state.  Everything the client uploads is
``l_e``-dimensional in its leading axis, so nothing scales with ``N_k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .class_space import ClassSplit
from .linalg import spd_solve, symmetrize

__all__ = ["ClientUpdate", "local_train"]


@dataclass
class ClientUpdate:
    """The payload client ``k`` uploads to the server.

    Per Section 3.3 this is exactly ``(W_hat_k, W_check_k, R_k)``.  None of the
    three can be used to recover the client's raw inputs or labels: ``R_k`` is a
    ``l_e x l_e`` second-moment summary and the weights are ``l_e x d``
    projections of it.

    Communication cost is ``O(l_e^2 + l_e * d_k)`` floats, independent of ``N_k``
    (Appendix E).
    """

    W_hat: np.ndarray  # (l_e, d_{k-1})   known-class local model,  Eq. (5)
    W_check: np.ndarray  # (l_e, d_k - d_{k-1}) unknown-class local model, Eq. (6)
    R: np.ndarray  # (l_e, l_e)       Regularized Gram Matrix,  Eq. (7)
    n_samples: int = 0  # bookkeeping only; not used by the aggregation

    @property
    def payload_floats(self) -> int:
        return self.W_hat.size + self.W_check.size + self.R.size


def local_train(
    features: np.ndarray,
    labels: np.ndarray,
    split: ClassSplit,
    gamma: float = 1.0,
) -> ClientUpdate:
    """Run one round of analytic local training for a single (virtual) client.

    Parameters
    ----------
    features : (N_k, l_e) float array
        ``F_k = Backbone(X_k, Theta_B)`` -- Eq. (3).  Already extracted, because
        the backbone is frozen and shared, so this is a pure forward pass that
        can be cached across the whole run.
    labels : (N_k,) int array
        Raw local labels ``Y_k``.
    split : ClassSplit
        The known/unknown encoders handed down by the server.
    gamma : float
        Ridge regularisation.  The paper reports AFCL is flat in ``gamma`` over
        ``[0, 10]`` (Table 2); ``gamma = 0`` is admissible but makes ``R_k``
        singular whenever ``N_k < l_e``, which is handled by the pseudo-inverse
        fallback in :mod:`afcl.linalg`.

    Returns
    -------
    ClientUpdate
    """
    F = np.asarray(features, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError(f"features must be (N_k, l_e), got {F.shape}")
    if F.shape[0] != len(labels):
        raise ValueError(f"features/labels length mismatch: {F.shape[0]} vs {len(labels)}")

    # Eq. (4): label encoding under the known/unknown split.
    Y_hat, Y_check = split.encode(labels)

    # Eq. (7): Regularized Gram Matrix.
    R = symmetrize(F.T @ F) + gamma * np.eye(F.shape[1], dtype=np.float64)

    # Eqs. (5)-(6), simplified per Eq. (49) to a single factorisation of R_k.
    # Solving once against the horizontally stacked right-hand side is identical
    # to solving twice and avoids a redundant Cholesky.
    FtY = F.T @ np.hstack([Y_hat, Y_check])
    W = spd_solve(R, FtY)
    W_hat = W[:, : split.d_prev]
    W_check = W[:, split.d_prev :]

    return ClientUpdate(W_hat=W_hat, W_check=W_check, R=R, n_samples=F.shape[0])
