"""Evaluation metrics -- Appendix F.2, Eqs. (50)-(53).

All four metrics are computed from one object: the lower-triangular accuracy
matrix ``A`` with ``A[i, j] = A^i_j`` = accuracy of the global model *after*
round ``i`` on the test set of task ``j`` (0-indexed here, 1-indexed in the
paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = ["AccuracyMatrix", "average_accuracy", "knowledge_retention", "stability", "plasticity"]


@dataclass
class AccuracyMatrix:
    """Lower-triangular ``A[i, j]``; entries with ``j > i`` are NaN (unseen task)."""

    A: np.ndarray

    @classmethod
    def empty(cls, n_tasks: int) -> "AccuracyMatrix":
        return cls(np.full((n_tasks, n_tasks), np.nan, dtype=np.float64))

    def set_row(self, i: int, accs: List[float]) -> None:
        self.A[i, : len(accs)] = accs

    def __repr__(self) -> str:  # pragma: no cover - display only
        rows = []
        for i in range(self.A.shape[0]):
            cells = " ".join(
                "  --  " if np.isnan(v) else f"{v:6.2f}" for v in self.A[i]
            )
            rows.append(f"  after task {i + 1}: {cells}")
        return "accuracy matrix A[i,j]\n" + "\n".join(rows)


def average_accuracy(A: np.ndarray, i: int) -> float:
    """Eq. (50): ``A_i = (1/i) * sum_{j<=i} A^i_j``  (0-indexed ``i``)."""
    return float(np.mean(A[i, : i + 1]))


def knowledge_retention(A: np.ndarray, i: int, paper_literal: bool = False) -> float:
    """Eq. (51): average knowledge retention after round ``i``.

    The printed formula reads ``F_i = 1/(i-1) * sum_{j<i} (A^j_j / A^i_j)``, but
    the paper then states that AFCL's retention "consistently exceeds 1.00,
    indicating its continual acquisition of new generalizable knowledge", while
    baselines sit below 1.00 "due to spatio-temporal forgetting".  Under the
    printed ratio, forgetting (``A^i_j < A^j_j``) would push the value *above*
    1.00 -- the opposite of the stated reading.

    We therefore default to ``A^i_j / A^j_j`` (current over first-learned),
    which matches the paper's interpretation and its Figure 4.  Pass
    ``paper_literal=True`` to reproduce the typeset expression instead.
    """
    if i == 0:
        return float("nan")
    ratios = []
    for j in range(i):
        first, now = A[j, j], A[i, j]
        num, den = (first, now) if paper_literal else (now, first)
        ratios.append(np.nan if den == 0 else num / den)
    return float(np.nanmean(ratios))


def stability(A: np.ndarray, i: int) -> float:
    """Eq. (52): mean, over rounds ``k=1..i``, of accuracy on tasks before ``k``."""
    if i == 0:
        return float("nan")
    per_round = [float(np.mean(A[k, :k])) for k in range(1, i + 1)]
    return float(np.mean(per_round))


def plasticity(A: np.ndarray, i: int) -> float:
    """Eq. (53): ``P_i = (1/i) * sum_{j<=i} A^j_j`` -- accuracy on each task when new."""
    return float(np.mean([A[j, j] for j in range(i + 1)]))


def summarize(A: np.ndarray, paper_literal_retention: bool = False) -> List[dict]:
    """Per-round table of all four metrics."""
    out = []
    for i in range(A.shape[0]):
        if np.isnan(A[i, 0]):
            continue
        out.append(
            {
                "task": i + 1,
                "avg_acc": average_accuracy(A, i),
                "retention": knowledge_retention(A, i, paper_literal_retention),
                "stability": stability(A, i),
                "plasticity": plasticity(A, i),
            }
        )
    return out
