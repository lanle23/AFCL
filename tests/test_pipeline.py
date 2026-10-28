"""Tests for the data partitioning, metrics, and the end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest

from afcl.data import SiBlurryConfig, dirichlet_split, si_blurry_plan, si_blurry_split
from afcl.metrics import average_accuracy, knowledge_retention, plasticity, stability
from afcl.simulate import FCLConfig, run_afcl


# --------------------------------------------------------------------------- #
# Si-Blurry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [5, 10])
def test_si_blurry_is_a_partition(T):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 100, size=20_000)
    tasks = si_blurry_split(y, SiBlurryConfig(n_tasks=T, seed=0))
    assert len(tasks) == T
    allidx = np.concatenate(tasks)
    assert len(allidx) == len(y)
    assert len(np.unique(allidx)) == len(y), "tasks must be disjoint and cover everything"


def test_si_blurry_disjoint_classes_stay_put():
    """A disjoint class must appear in exactly one task; a blurry one may not."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 50, size=10_000)
    cfg = SiBlurryConfig(n_tasks=5, r_disjoint=0.5, r_blurry=0.1, seed=3)
    plan = si_blurry_plan(np.unique(y), cfg)
    tasks = si_blurry_split(y, cfg, plan=plan)

    tasks_of = {c: set() for c in np.unique(y)}
    for t, idx in enumerate(tasks):
        for c in np.unique(y[idx]):
            tasks_of[int(c)].add(t)

    for c in plan.disjoint:
        assert len(tasks_of[c]) == 1, f"disjoint class {c} leaked across tasks"
    # With r_B = 10% and thousands of samples, blurry classes should spread.
    spread = [len(tasks_of[c]) for c in plan.blurry]
    assert np.mean(spread) > 1.5, "blurry classes are not actually blurry"


def test_si_blurry_plan_is_reusable_across_pools():
    """Train and test must agree on what 'task j' means."""
    cfg = SiBlurryConfig(n_tasks=5, seed=11)
    rng = np.random.default_rng(2)
    y_tr = rng.integers(0, 40, size=8000)
    y_te = rng.integers(0, 40, size=2000)
    plan = si_blurry_plan(np.unique(y_tr), cfg)
    a = si_blurry_split(y_tr, cfg, plan=plan, seed_offset=0)
    b = si_blurry_split(y_te, cfg, plan=plan, seed_offset=1)
    assert len(a) == len(b) == 5
    # The dominant class of each task should match across pools.
    for t in range(5):
        if len(a[t]) and len(b[t]):
            top_a = np.bincount(y_tr[a[t]], minlength=40).argmax()
            assert top_a in set(y_te[b[t]].tolist())


# --------------------------------------------------------------------------- #
# Dirichlet
# --------------------------------------------------------------------------- #
def test_dirichlet_gives_equal_client_sizes():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 20, size=5000)
    idx = np.arange(len(y))
    shards = dirichlet_split(y, idx, n_clients=5, alpha=0.1, seed=0, equal_size=True)
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1, f"sizes not equal: {sizes}"
    assert sum(sizes) == len(y)
    assert len(np.unique(np.concatenate(shards))) == len(y)


def test_dirichlet_alpha_controls_label_skew():
    """Small alpha => each client concentrates on few classes."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 20, size=8000)
    idx = np.arange(len(y))

    def mean_effective_classes(alpha):
        shards = dirichlet_split(y, idx, n_clients=5, alpha=alpha, seed=1)
        ents = []
        for s in shards:
            p = np.bincount(y[s], minlength=20).astype(float)
            p = p[p > 0] / p.sum()
            ents.append(np.exp(-(p * np.log(p)).sum()))  # perplexity
        return float(np.mean(ents))

    skewed, uniform = mean_effective_classes(0.05), mean_effective_classes(100.0)
    assert skewed < uniform, f"alpha did not control skew: {skewed:.2f} vs {uniform:.2f}"


def test_dirichlet_respects_class_availability():
    """No client may be handed more samples of a class than exist."""
    rng = np.random.default_rng(6)
    y = np.concatenate([np.zeros(50), np.ones(950)]).astype(int)
    rng.shuffle(y)
    shards = dirichlet_split(y, np.arange(len(y)), n_clients=4, alpha=0.1, seed=2)
    counts = np.zeros(2, dtype=int)
    for s in shards:
        counts += np.bincount(y[s], minlength=2)
    assert counts.tolist() == [50, 950]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_metrics_match_hand_computation():
    A = np.array(
        [
            [50.0, np.nan, np.nan],
            [40.0, 60.0, np.nan],
            [30.0, 50.0, 70.0],
        ]
    )
    assert average_accuracy(A, 2) == pytest.approx((30 + 50 + 70) / 3)
    assert plasticity(A, 2) == pytest.approx((50 + 60 + 70) / 3)
    # Eq. (52): mean over k=1..2 of (mean of A[k, :k])
    assert stability(A, 2) == pytest.approx(np.mean([40.0, np.mean([30.0, 50.0])]))
    # default (non-literal) retention: current / first-learned
    assert knowledge_retention(A, 2) == pytest.approx(np.mean([30 / 50, 50 / 60]))
    assert knowledge_retention(A, 2, paper_literal=True) == pytest.approx(
        np.mean([50 / 30, 60 / 50])
    )


def test_retention_detects_forgetting():
    """A forgetting model must score below 1.0 under the default convention."""
    A = np.array([[80.0, np.nan], [20.0, 80.0]])
    assert knowledge_retention(A, 1) < 1.0


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def _synthetic(seed=0, n_cls=40, le=96, per_cls=80, noise=4.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_cls, le))

    def gen(n):
        y = np.repeat(np.arange(n_cls), n)
        F = centers[y] + rng.normal(scale=noise, size=(len(y), le))
        return F.astype(np.float32), y

    return (*gen(per_cls), *gen(20))


def test_end_to_end_runs_and_learns():
    Ftr, ytr, Fte, yte = _synthetic()
    cfg = FCLConfig(n_tasks=5, n_clients=5, alpha=0.1, gamma=1.0, seed=0, verbose=False)
    res = run_afcl(Ftr, ytr, Fte, yte, cfg)

    assert res.n_virtual_clients == 25, "T x n_clients virtual clients expected"
    assert res.final_classes == 40
    assert res.final_average_accuracy > 100.0 / 40 * 3, "should beat chance comfortably"
    assert np.isfinite(res.accuracy.A[-1]).all()


def test_end_to_end_invariance_across_heterogeneity_settings():
    """The paper's headline: accuracy barely moves as T and alpha change."""
    Ftr, ytr, Fte, yte = _synthetic()
    accs = []
    for T, alpha in [(5, 0.1), (5, 0.2), (10, 0.1), (10, 0.2)]:
        cfg = FCLConfig(n_tasks=T, n_clients=5, alpha=alpha, gamma=1.0, seed=0, verbose=False)
        accs.append(run_afcl(Ftr, ytr, Fte, yte, cfg).final_average_accuracy)
    spread = max(accs) - min(accs)
    assert spread < 1.0, f"AFCL should be near-invariant, got spread {spread:.3f}pp: {accs}"


def test_modes_agree_end_to_end():
    Ftr, ytr, Fte, yte = _synthetic(seed=3)
    out = {}
    for mode in ("stable", "simplified", "paper"):
        cfg = FCLConfig(n_tasks=5, n_clients=5, alpha=0.2, gamma=1.0, mode=mode,
                        seed=1, verbose=False)
        out[mode] = run_afcl(Ftr, ytr, Fte, yte, cfg).final_average_accuracy
    assert abs(out["stable"] - out["simplified"]) < 1e-9
    assert abs(out["stable"] - out["paper"]) < 1e-3, out
