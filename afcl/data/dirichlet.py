"""Dirichlet client partitioning -- spatial heterogeneity (Appendix F.1).

The paper's constraint is two-sided and worth reading carefully:

    "the data allocated to each client has an **equal number of samples** and
     follows the **Dirichlet distribution**, where the degree of spatial data
     heterogeneity among clients is controlled by the Dirichlet parameter alpha"

So this is not the usual "draw Dir(alpha) per class and let client sizes fall
where they may" recipe -- client sizes are pinned equal while the *class
composition* of each client is Dirichlet-skewed.  That is a transportation
problem with fixed row marginals (equal client sizes) and fixed column
marginals (how many samples of each class actually exist).

We solve it with iterative proportional fitting on the Dirichlet draw, then
round to integers preserving the column marginals exactly and repair the row
marginals.  The result is an exact integer allocation that honours both
constraints.

alpha -> 0 gives near-pathological label skew (a client sees one or two
classes); alpha -> infinity gives near-IID shards.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

__all__ = ["dirichlet_split"]


def _ipf(P: np.ndarray, row_sums: np.ndarray, col_sums: np.ndarray, iters: int = 500) -> np.ndarray:
    """Iterative proportional fitting of ``P`` onto the given marginals."""
    X = np.clip(P.astype(np.float64), 1e-12, None)
    # Zero out columns with no availability so they never receive mass.
    X[:, col_sums == 0] = 0.0
    for _ in range(iters):
        r = X.sum(axis=1, keepdims=True)
        X = np.divide(X, np.where(r > 0, r, 1.0)) * row_sums[:, None]
        c = X.sum(axis=0, keepdims=True)
        X = np.divide(X, np.where(c > 0, c, 1.0)) * col_sums[None, :]
    return np.nan_to_num(X)


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Allocate ``total`` integer units over ``weights`` (Hamilton's method)."""
    if total <= 0:
        return np.zeros(len(weights), dtype=np.int64)
    w = np.clip(weights.astype(np.float64), 0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    share = w / w.sum() * total
    base = np.floor(share).astype(np.int64)
    rem = total - base.sum()
    if rem > 0:
        order = np.argsort(-(share - base))
        base[order[:rem]] += 1
    return base


def _repair_rows(A: np.ndarray, row_targets: np.ndarray) -> np.ndarray:
    """Move single units between clients, within a class, to fix row sums.

    Column sums are invariant under this operation, so class availability stays
    satisfied.  Terminates because total absolute row error strictly decreases.
    """
    A = A.copy()
    for _ in range(100_000):
        diff = A.sum(axis=1) - row_targets
        if not np.any(diff):
            return A
        over = int(np.argmax(diff))
        under = int(np.argmin(diff))
        if diff[over] <= 0 or diff[under] >= 0:
            return A
        # Any class the over-client holds can be handed to the under-client.
        cand = np.where(A[over] > 0)[0]
        if len(cand) == 0:
            return A
        # Prefer a class the under-client already has, to preserve skew.
        pref = cand[np.argsort(-A[under, cand])]
        j = int(pref[0])
        A[over, j] -= 1
        A[under, j] += 1
    return A


def dirichlet_split(
    labels: Sequence[int],
    indices: Sequence[int],
    n_clients: int,
    alpha: float,
    seed: int = 0,
    equal_size: bool = True,
) -> List[np.ndarray]:
    """Split one task's samples across ``n_clients`` with Dirichlet label skew.

    Parameters
    ----------
    labels : (N,) labels of the *full* dataset.
    indices : the subset of row ids belonging to this task.
    n_clients : clients per task (5 in the paper).
    alpha : Dirichlet concentration.  Paper uses {0.1, 0.2} for CIFAR-100 and
        Tiny-ImageNet, {0.5, 1.0} for ImageNet-R.
    equal_size : enforce equal client sample counts, as the paper specifies.
        Set False for the more common size-varying Dirichlet recipe.

    Returns
    -------
    list of ``n_clients`` index arrays (subsets of ``indices``).
    """
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels)[indices]
    classes = np.unique(y)
    C, K = len(classes), n_clients
    n = len(indices)

    avail = np.array([(y == c).sum() for c in classes], dtype=np.int64)
    P = rng.dirichlet(np.full(C, alpha), size=K)  # (K, C)

    if equal_size:
        row_targets = np.full(K, n // K, dtype=np.int64)
        row_targets[: n % K] += 1
        X = _ipf(P, row_targets.astype(np.float64), avail.astype(np.float64))
        # Round column-by-column so class availability is matched exactly...
        A = np.stack([_largest_remainder(X[:, j], int(avail[j])) for j in range(C)], axis=1)
        A = _repair_rows(A, row_targets)  # ...then fix client sizes.
    else:
        A = np.stack([_largest_remainder(P[:, j], int(avail[j])) for j in range(C)], axis=1)

    # Materialise the allocation into actual sample indices.
    pools = {int(c): rng.permutation(indices[y == c]).tolist() for c in classes}
    shards: List[List[int]] = [[] for _ in range(K)]
    for j, c in enumerate(classes):
        pool = pools[int(c)]
        pos = 0
        for i in range(K):
            take = int(A[i, j])
            shards[i].extend(pool[pos : pos + take])
            pos += take

    out = [np.array(sorted(s), dtype=np.int64) for s in shards]
    assert sum(len(s) for s in out) == n, "Dirichlet split lost samples"
    return out
