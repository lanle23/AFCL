"""Known-unknown class splitting -- Section 3.2, Eq. (2).

The server never needs to know the total number of classes in advance.  Each
(virtual) client registers by declaring the set of classes present in its local
shard.  The server partitions that set into

  * ``known``   S-hat_k : classes some earlier client already registered,
  * ``unknown`` S-check_k : classes appearing for the first time at round k,

and hands back two one-hot encoders.  ``Onehot_k`` maps into the *existing*
``d_{k-1}`` columns; ``Onehot*_k`` maps into a *fresh* ``d_k - d_{k-1}`` column
block that will be concatenated onto the Global Knowledge Matrix.

Eq. (2) is exactly the statement that the next round's encoder is the current
one zero-padded on the right, with the new classes occupying the pad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

__all__ = ["ClassSplit", "ClassSpace"]


@dataclass(frozen=True)
class ClassSplit:
    """What the server sends back to client ``k`` at registration.

    Attributes
    ----------
    known_cols : class label -> column index in ``[0, d_prev)``
    unknown_cols : class label -> column index in ``[0, d_new - d_prev)``
    d_prev : ``d_{k-1}``, width of the known block.
    d_new : ``d_k``, width after this client's classes are absorbed.
    """

    known_cols: Dict[int, int]
    unknown_cols: Dict[int, int]
    d_prev: int
    d_new: int

    @property
    def n_unknown(self) -> int:
        return self.d_new - self.d_prev

    def encode(self, labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Y_hat_k, Y_check_k)`` -- Eq. (4).

        A sample of a known class contributes a one-hot row to ``Y_hat_k`` and an
        all-zero row to ``Y_check_k``; a sample of a new class does the reverse.
        Stacking the two horizontally reproduces the block structure of
        ``Y_{1:k}`` in Eq. (38).
        """
        labels = np.asarray(labels)
        n = labels.shape[0]
        Y_hat = np.zeros((n, self.d_prev), dtype=np.float64)
        Y_check = np.zeros((n, self.n_unknown), dtype=np.float64)
        for i, y in enumerate(labels.tolist()):
            if y in self.known_cols:
                Y_hat[i, self.known_cols[y]] = 1.0
            elif y in self.unknown_cols:
                Y_check[i, self.unknown_cols[y]] = 1.0
            else:
                raise KeyError(
                    f"label {y} was not declared at registration; the client must "
                    f"declare every class present in its shard"
                )
        return Y_hat, Y_check


@dataclass
class ClassSpace:
    """Server-side registry of the global class space.

    ``col_to_class[j]`` is the class owning output column ``j`` of the global
    model, so ``argmax`` over model scores maps straight back to a label.
    """

    class_to_col: Dict[int, int] = field(default_factory=dict)
    col_to_class: List[int] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.col_to_class)

    def register(self, local_classes: Iterable[int]) -> ClassSplit:
        """Split ``local_classes`` into known/unknown and extend the space.

        The extension happens immediately (the server records S-check_k on
        receipt, per Section 3.2), so the returned ``d_new`` is the output width
        the Global Knowledge Matrix will have after this round.
        """
        d_prev = self.n_classes
        # Sort for determinism: registration order of *clients* is what the
        # invariance theorem covers, but within one client we want a stable,
        # reproducible column assignment.
        declared = sorted(set(int(c) for c in local_classes))

        known_cols = {c: self.class_to_col[c] for c in declared if c in self.class_to_col}
        unknown = [c for c in declared if c not in self.class_to_col]
        unknown_cols = {c: j for j, c in enumerate(unknown)}

        for c in unknown:  # Eq. (2): new classes occupy the fresh right-hand block
            self.class_to_col[c] = len(self.col_to_class)
            self.col_to_class.append(c)

        return ClassSplit(
            known_cols=known_cols,
            unknown_cols=unknown_cols,
            d_prev=d_prev,
            d_new=self.n_classes,
        )

    def decode(self, col_indices: np.ndarray) -> np.ndarray:
        """Map model output columns back to original class labels."""
        lut = np.asarray(self.col_to_class)
        return lut[np.asarray(col_indices)]
