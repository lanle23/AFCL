"""Si-Blurry task partitioning -- temporal heterogeneity (Appendix F.1).

Following Moon et al., ICCV 2023, as the AFCL paper specifies:

  * classes are split into **disjoint** classes (all their data lives in one
    task) and **blurry** classes (their data is spread over several tasks),
    with ``r_D`` the fraction of classes that are disjoint;
  * a fraction ``r_B`` of each blurry class's samples is then randomly
    reassigned away from that class's home task.

The paper uses ``r_D = 50%`` and ``r_B = 10%``, and ``T in {5, 10}`` tasks.

Note on fidelity: the paper describes this in a single sentence ("a proportion
r_B of the data from each blurry class is randomly reassigned to other blurry
classes").  We read that as reassigning those samples to the *home tasks of
other blurry classes*, which is what makes task boundaries blurry.  The sample
keeps its own label; only the task it lands in changes.  ``blurry_mode`` lets
you switch to a uniform-over-other-tasks reading if you prefer it.

None of this affects AFCL's final model -- Theorem 2 says any repartition of
the same pool is invariant -- but it does change the *baselines*, and it
changes AFCL's intermediate per-task accuracies A^i_j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np

__all__ = ["SiBlurryConfig", "SiBlurryPlan", "si_blurry_plan", "si_blurry_split"]

BlurryMode = Literal["blurry_tasks", "uniform"]


@dataclass
class SiBlurryConfig:
    n_tasks: int = 5
    r_disjoint: float = 0.5  # r_D
    r_blurry: float = 0.1  # r_B
    blurry_mode: BlurryMode = "blurry_tasks"
    seed: int = 0


@dataclass
class SiBlurryPlan:
    """The class-level layout: which classes are disjoint, and each class's home task.

    Separated out so the *same* layout can be applied to both the train and the
    test pool, which is what makes "the test set of the j-th task" well defined.
    """

    home: dict  # class -> home task
    disjoint: set  # classes whose data never leaves its home task
    blurry: set  # classes that leak r_B of their data to other tasks
    n_tasks: int


def si_blurry_plan(classes: Sequence[int], cfg: SiBlurryConfig) -> SiBlurryPlan:
    """Decide the disjoint/blurry split and each class's home task."""
    rng = np.random.default_rng(cfg.seed)
    classes = np.unique(np.asarray(classes))
    T = cfg.n_tasks

    perm = rng.permutation(classes)
    n_disjoint = int(round(cfg.r_disjoint * len(classes)))
    disjoint_classes = perm[:n_disjoint]
    blurry_classes = perm[n_disjoint:]

    home: dict = {}
    for i, c in enumerate(disjoint_classes):
        home[int(c)] = i % T
    for i, c in enumerate(blurry_classes):
        home[int(c)] = i % T

    return SiBlurryPlan(
        home=home,
        disjoint=set(int(c) for c in disjoint_classes),
        blurry=set(int(c) for c in blurry_classes),
        n_tasks=T,
    )


def si_blurry_split(
    labels: Sequence[int],
    cfg: SiBlurryConfig,
    plan: SiBlurryPlan | None = None,
    seed_offset: int = 0,
) -> List[np.ndarray]:
    """Partition sample indices into ``cfg.n_tasks`` tasks.

    Parameters
    ----------
    labels : (N,) array of integer class labels for the pool being split.
    plan : reuse a layout from :func:`si_blurry_plan` (pass the train plan when
        splitting the test pool, so task identities line up).
    seed_offset : decorrelates the *sample-level* shuffling between pools that
        share a plan.

    Returns
    -------
    list of index arrays, one per task.  The union is the full index set and
    the parts are disjoint.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if plan is None:
        plan = si_blurry_plan(classes, cfg)
    T = plan.n_tasks
    home = plan.home
    rng = np.random.default_rng(cfg.seed + 10_000 * (seed_offset + 1))

    blurry_home_tasks = sorted({home[int(c)] for c in plan.blurry}) or list(range(T))

    task_indices: List[List[int]] = [[] for _ in range(T)]

    for c in classes:
        idx = np.where(labels == c)[0]
        idx = rng.permutation(idx)
        h = home[int(c)]

        if int(c) in plan.disjoint:
            task_indices[h].extend(idx.tolist())
            continue

        # Blurry class: hold back r_B of its samples and scatter them.
        n_move = int(round(cfg.r_blurry * len(idx)))
        stay, move = idx[n_move:], idx[:n_move]
        task_indices[h].extend(stay.tolist())

        if len(move) == 0:
            continue
        if cfg.blurry_mode == "blurry_tasks":
            targets = [t for t in blurry_home_tasks if t != h] or [h]
        else:
            targets = [t for t in range(T) if t != h] or [h]
        dest = rng.choice(targets, size=len(move), replace=True)
        for j, t in zip(move.tolist(), dest.tolist()):
            task_indices[t].append(j)

    out = [np.array(sorted(t), dtype=np.int64) for t in task_indices]

    total = sum(len(t) for t in out)
    assert total == len(labels), f"Si-Blurry lost samples: {total} != {len(labels)}"
    return out


def describe_split(task_indices: Sequence[np.ndarray], labels: Sequence[int]) -> str:
    """Human-readable summary, handy for sanity-checking a config."""
    labels = np.asarray(labels)
    lines = []
    seen: set[int] = set()
    for t, idx in enumerate(task_indices):
        cls = np.unique(labels[idx])
        new = sorted(set(cls.tolist()) - seen)
        seen |= set(cls.tolist())
        lines.append(
            f"  task {t}: {len(idx):6d} samples, {len(cls):3d} classes, {len(new):3d} new"
        )
    return "\n".join(lines)
