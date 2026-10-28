"""Server side: global recursive aggregating -- Section 3.4, Eqs. (8)-(10).

The server holds three pieces of state:

  ``R_tilde_k``  = sum_{i<=k} R_i            -- Eq. (9), updated incrementally
  ``G_k``        = Global Knowledge Matrix   -- Eq. (8), grows in width as new
                                                classes arrive
  ``W_k``        = the deployable global model, read out from G by Eq. (10)

``G_k`` is *not* the model.  Lemma 2 shows ``G_k = (F_{1:k}^T F_{1:k} + k*gamma I)^-1
F_{1:k}^T Y_{1:k}`` -- i.e. it carries ``k`` copies of the ridge penalty, one per
client.  Eq. (10) strips the surplus ``(k-1)*gamma`` back off, which is what makes
Theorem 1 hold.

Three algebraically identical aggregation modes are provided; see
``AggregationMode`` for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np

from .class_space import ClassSpace, ClassSplit
from .client import ClientUpdate
from .linalg import spd_inv, spd_solve, symmetrize

__all__ = ["AFCLServer", "AggregationMode"]

AggregationMode = Literal["stable", "simplified", "paper"]
"""How to realise the recursion of Eqs. (8)-(9).

``"paper"``
    The literal expressions printed in Eq. (9)::

        A_k = I - R~_{k-1}^-1 R_k (I - R~_k^-1 R_k)
        B_k = I - R_k^-1 R~_{k-1} (I - R~_k^-1 R~_{k-1})

    Faithful, but it inverts ``R~_{k-1}`` and ``R_k`` separately and composes
    four matrix products per round, so float error compounds.  Kept for
    verification against the paper.

``"simplified"``
    The closed forms those expressions collapse to (read straight off the
    Woodbury manipulation in Eqs. (31)-(34) of Appendix C)::

        A_k = R~_k^-1 R~_{k-1}        B_k = R~_k^-1 R_k

    One inverse per round instead of three.

``"stable"`` (default)
    Skip ``A_k``/``B_k`` altogether.  Appendix C shows ``A_k G_{k-1} = R~_k^-1
    F_{1:k-1}^T Y_{1:k-1}`` and ``B_k W_hat_k = R~_k^-1 F_k^T Y_hat_k``, so it is
    enough to accumulate the cross-correlation ``C_k = F_{1:k}^T Y_{1:k}`` and
    solve once.  The server recovers each client's contribution as
    ``F_k^T Y_hat_k = R_k W_hat_k`` -- so this mode consumes *exactly* the
    protocol payload ``(W_hat_k, W_check_k, R_k)`` and changes nothing the
    clients do.  This is the numerically sound way to run AFCL.

All three produce the same ``W_k`` in exact arithmetic; ``tests/test_theorems.py``
checks they agree numerically.
"""


@dataclass
class AFCLServer:
    """Stateful AFCL server; feed it one :class:`ClientUpdate` per round.

    Parameters
    ----------
    feat_dim : ``l_e``, the backbone embedding width.
    gamma : ridge strength, shared with the clients.
    mode : see :data:`AggregationMode`.
    """

    feat_dim: int
    gamma: float = 1.0
    mode: AggregationMode = "stable"

    class_space: ClassSpace = field(default_factory=ClassSpace)
    R_tilde: np.ndarray = field(init=False)
    G: np.ndarray = field(init=False)
    _C: np.ndarray = field(init=False)  # F_{1:k}^T Y_{1:k}, "stable" mode only
    k: int = 0  # number of clients aggregated so far

    def __post_init__(self) -> None:
        le = self.feat_dim
        # Algorithm 1, line 1: W_0 = 0, R_0 = 0, G_0 = 0.
        self.R_tilde = np.zeros((le, le), dtype=np.float64)
        self.G = np.zeros((le, 0), dtype=np.float64)
        self._C = np.zeros((le, 0), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #
    def register(self, local_classes) -> ClassSplit:
        """Section 3.2: absorb a client's declared classes, return its encoders."""
        return self.class_space.register(local_classes)

    # ------------------------------------------------------------------ #
    # aggregation
    # ------------------------------------------------------------------ #
    def aggregate(self, update: ClientUpdate) -> np.ndarray:
        """Absorb one client and return the refreshed global model ``W_k``.

        Single round per client -- the server never ships a model back down, so
        there is no second leg of communication (Appendix E).
        """
        R_k = symmetrize(np.asarray(update.R, dtype=np.float64))
        W_hat = np.asarray(update.W_hat, dtype=np.float64)
        W_check = np.asarray(update.W_check, dtype=np.float64)

        if W_hat.shape[1] != self.G.shape[1]:
            raise ValueError(
                f"client's known-class block has width {W_hat.shape[1]} but the "
                f"Global Knowledge Matrix has width {self.G.shape[1]}; the client "
                f"was registered against a different class space"
            )

        R_tilde_prev = self.R_tilde
        R_tilde_new = R_tilde_prev + R_k  # Eq. (9), recursive -- no history kept

        if self.k == 0:
            # Section 3.4: "In the first round, as all the classes are previously
            # unknown, ... G_1 = W_check_1 is a special case."  (R~_0 = 0 is not
            # invertible, so A_1/B_1 are undefined.)
            G_new = W_check
            C_new = R_k @ W_check
        elif self.mode == "stable":
            C_new = np.hstack([self._C + R_k @ W_hat, R_k @ W_check])
            G_new = spd_solve(R_tilde_new, C_new)
        else:
            if self.mode == "simplified":
                A_k = spd_solve(R_tilde_new, R_tilde_prev)
                B_k = spd_solve(R_tilde_new, R_k)
            elif self.mode == "paper":
                I = np.eye(self.feat_dim, dtype=np.float64)
                inv_R_tilde_prev = spd_inv(R_tilde_prev)
                inv_R_tilde_new = spd_inv(R_tilde_new)
                inv_R_k = spd_inv(R_k)
                # Eq. (9), verbatim.
                A_k = I - inv_R_tilde_prev @ R_k @ (I - inv_R_tilde_new @ R_k)
                B_k = I - inv_R_k @ R_tilde_prev @ (I - inv_R_tilde_new @ R_tilde_prev)
            else:
                raise ValueError(f"unknown aggregation mode {self.mode!r}")

            # Eq. (8): known-class knowledge is updated in place, unknown-class
            # knowledge is concatenated as a new column block.
            G_new = np.hstack([A_k @ self.G + B_k @ W_hat, B_k @ W_check])
            C_new = np.hstack([self._C + R_k @ W_hat, R_k @ W_check])

        self.R_tilde = R_tilde_new
        self.G = G_new
        self._C = C_new
        self.k += 1
        return self.global_model()

    # ------------------------------------------------------------------ #
    # readout
    # ------------------------------------------------------------------ #
    def global_model(self) -> np.ndarray:
        """Eq. (10): ``W_k = [R~_k - (k-1) gamma I]^-1 R~_k G_k``.

        ``R~_k - (k-1) gamma I = F_{1:k}^T F_{1:k} + gamma I``, so by Theorem 1
        this equals the centralised ridge solution over ``D_{1:k}``.
        """
        if self.k == 0:
            return np.zeros((self.feat_dim, 0), dtype=np.float64)
        M = self.R_tilde - (self.k - 1) * self.gamma * np.eye(self.feat_dim)
        return spd_solve(symmetrize(M), self.R_tilde @ self.G)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Argmax over the global model's scores, decoded to original labels."""
        W = self.global_model()
        if W.shape[1] == 0:
            raise RuntimeError("no clients aggregated yet")
        scores = np.asarray(features, dtype=np.float64) @ W
        return self.class_space.decode(np.argmax(scores, axis=1))

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Top-1 accuracy in percent."""
        if len(labels) == 0:
            return float("nan")
        pred = self.predict(features)
        return 100.0 * float(np.mean(pred == np.asarray(labels)))


def joint_reference_model(
    features: np.ndarray,
    labels: np.ndarray,
    class_space: ClassSpace,
    gamma: float = 1.0,
) -> np.ndarray:
    """Centralised joint learning, Eq. (14) -- the target of Theorem 1.

    Pools every client's data and solves the single ridge problem of Eq. (1).
    Used only as a reference oracle in the tests; it is what AFCL is proven to
    reproduce without ever pooling anything.
    """
    F = np.asarray(features, dtype=np.float64)
    d = class_space.n_classes
    Y = np.zeros((F.shape[0], d), dtype=np.float64)
    for i, y in enumerate(np.asarray(labels).tolist()):
        Y[i, class_space.class_to_col[int(y)]] = 1.0
    R = symmetrize(F.T @ F) + gamma * np.eye(F.shape[1], dtype=np.float64)
    return spd_solve(R, F.T @ Y)
