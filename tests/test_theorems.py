"""Numerical verification of the paper's two theorems.

These are the tests that matter: AFCL's entire claim is that a recursive,
single-round, gradient-free protocol lands on exactly the centralised ridge
solution, no matter how the data is scattered across clients and tasks.
"""

from __future__ import annotations

import numpy as np
import pytest

from afcl import AFCLServer, ClassSpace, joint_reference_model, local_train


def make_shards(rng, n_clients=12, le=48, n_classes=20, n_per=60):
    """A deliberately nasty class-continual partition.

    Every client sees a different, overlapping, unevenly sized subset of the
    label space, and new classes keep appearing at arbitrary rounds.
    """
    shards = []
    for k in range(n_clients):
        # Pathological heterogeneity: each client draws 2-5 classes, biased so
        # that new classes trickle in late rather than all appearing at round 1.
        hi = min(n_classes, 3 + int(n_classes * (k + 1) / n_clients))
        pool = np.arange(hi)
        n_cls = int(rng.integers(2, min(6, len(pool) + 1)))
        classes = rng.choice(pool, size=n_cls, replace=False)
        n = int(rng.integers(n_per // 2, n_per * 2))
        labels = rng.choice(classes, size=n, replace=True)
        # Give each class a distinct mean so the problem is not degenerate.
        feats = rng.normal(size=(n, le)) + labels[:, None] * 0.05
        shards.append((feats, labels))
    return shards


def run_afcl(shards, gamma, mode, le):
    server = AFCLServer(feat_dim=le, gamma=gamma, mode=mode)
    for feats, labels in shards:
        split = server.register(np.unique(labels))
        update = local_train(feats, labels, split, gamma=gamma)
        server.aggregate(update)
    return server


@pytest.mark.parametrize("gamma", [0.0, 1e-3, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("mode", ["stable", "simplified", "paper"])
def test_theorem_1_equals_centralised_joint_learning(gamma, mode, request):
    """Theorem 1: W_k from Eq. (10) == the Eq. (14) joint ridge solution."""
    if mode == "paper" and gamma == 0.0:
        # Not an implementation bug.  B_k in Eq. (9) contains an explicit
        # R_k^-1, and at gamma = 0 we have R_k = F_k^T F_k, which is singular
        # for every client holding N_k < l_e samples.  The literal formula is
        # undefined there; the algebraically equivalent "stable"/"simplified"
        # forms never build that inverse and pass at gamma = 0.  The paper
        # reports a gamma = 0 column in Table 2, so this edge is reachable in
        # their own sweep.
        request.applymarker(
            pytest.mark.xfail(
                reason="Eq. (9) B_k requires R_k^-1, undefined at gamma=0 when N_k < l_e",
                strict=True,
            )
        )
    le = 48
    rng = np.random.default_rng(0)
    shards = make_shards(rng, le=le)

    server = run_afcl(shards, gamma, mode, le)

    all_F = np.vstack([f for f, _ in shards])
    all_y = np.concatenate([y for _, y in shards])
    W_joint = joint_reference_model(all_F, all_y, server.class_space, gamma=gamma)

    W_afcl = server.global_model()
    assert W_afcl.shape == W_joint.shape
    # Relative error, since the weight scale depends on gamma.
    err = np.linalg.norm(W_afcl - W_joint) / max(np.linalg.norm(W_joint), 1e-12)
    tol = 1e-6 if mode == "paper" else 1e-9
    assert err < tol, f"mode={mode} gamma={gamma}: relative error {err:.3e}"


@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_lemma_2_global_knowledge_matrix_closed_form(gamma):
    """Lemma 2: G_k == (F^T F + k*gamma I)^-1 F^T Y, with k copies of the penalty."""
    le = 32
    rng = np.random.default_rng(7)
    shards = make_shards(rng, n_clients=8, le=le, n_classes=12)

    server = run_afcl(shards, gamma, "stable", le)
    k = server.k

    all_F = np.vstack([f for f, _ in shards])
    all_y = np.concatenate([y for _, y in shards])
    d = server.class_space.n_classes
    Y = np.zeros((all_F.shape[0], d))
    for i, y in enumerate(all_y.tolist()):
        Y[i, server.class_space.class_to_col[int(y)]] = 1.0

    G_closed = np.linalg.solve(all_F.T @ all_F + k * gamma * np.eye(le), all_F.T @ Y)
    err = np.linalg.norm(server.G - G_closed) / max(np.linalg.norm(G_closed), 1e-12)
    assert err < 1e-9, f"gamma={gamma}: G relative error {err:.3e}"


@pytest.mark.parametrize("mode", ["stable", "simplified", "paper"])
def test_theorem_2_order_invariance(mode):
    """Theorem 2a: the final model does not depend on client registration order.

    Note this is invariance of the *model*, not of the column layout -- a
    different arrival order assigns classes to different output columns.  We
    therefore compare the class-indexed weights, which is what actually
    determines predictions.
    """
    le = 40
    gamma = 1.0
    rng = np.random.default_rng(1)
    shards = make_shards(rng, n_clients=10, le=le, n_classes=16)

    def class_indexed_weights(server):
        W = server.global_model()
        order = np.argsort(server.class_space.col_to_class)
        return W[:, order], np.asarray(server.class_space.col_to_class)[order]

    W_a, cls_a = class_indexed_weights(run_afcl(shards, gamma, mode, le))

    perm = np.random.default_rng(2).permutation(len(shards))
    W_b, cls_b = class_indexed_weights(run_afcl([shards[i] for i in perm], gamma, mode, le))

    assert np.array_equal(cls_a, cls_b)
    err = np.linalg.norm(W_a - W_b) / np.linalg.norm(W_a)
    tol = 1e-6 if mode == "paper" else 1e-9
    assert err < tol, f"mode={mode}: order-invariance error {err:.3e}"


def test_theorem_2_heterogeneity_invariance():
    """Theorem 2b: repartitioning the *same* samples across clients changes nothing.

    This is the headline property.  We take one pooled dataset and shard it two
    ways -- once IID, once pathologically non-IID (each client gets one or two
    classes) -- and check the resulting global models coincide.
    """
    le, gamma, n_classes = 40, 1.0, 10
    rng = np.random.default_rng(3)
    n = 1200
    labels = rng.integers(0, n_classes, size=n)
    feats = rng.normal(size=(n, le)) + labels[:, None] * 0.05

    # (a) IID shards.
    idx = rng.permutation(n)
    iid_shards = [(feats[c], labels[c]) for c in np.array_split(idx, 10)]

    # (b) Label-skewed shards: one class each, so no client sees the task.
    skew_shards = []
    for c in range(n_classes):
        sel = np.where(labels == c)[0]
        for part in np.array_split(sel, 2):
            if len(part):
                skew_shards.append((feats[part], labels[part]))

    def class_indexed(server):
        W = server.global_model()
        order = np.argsort(server.class_space.col_to_class)
        return W[:, order]

    W_iid = class_indexed(run_afcl(iid_shards, gamma, "stable", le))
    W_skew = class_indexed(run_afcl(skew_shards, gamma, "stable", le))

    err = np.linalg.norm(W_iid - W_skew) / np.linalg.norm(W_iid)
    assert err < 1e-9, f"spatio-temporal invariance error {err:.3e}"


def test_client_payload_is_data_independent():
    """Appendix E: upload size is O(l_e^2 + l_e*d), independent of N_k."""
    le, gamma = 32, 1.0
    rng = np.random.default_rng(5)
    space = ClassSpace()
    classes = np.arange(6)

    sizes = []
    for n in (100, 10_000):
        labels = rng.choice(classes, size=n)
        feats = rng.normal(size=(n, le))
        s = ClassSpace()
        split = s.register(np.unique(labels))
        upd = local_train(feats, labels, split, gamma=gamma)
        sizes.append(upd.payload_floats)
    assert sizes[0] == sizes[1], "payload must not grow with the local sample count"


def test_single_client_reduces_to_ridge():
    """Sanity: with K=1, AFCL is plain ridge regression (G_1 = W_check_1)."""
    le, gamma = 24, 2.0
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 5, size=300)
    feats = rng.normal(size=(300, le))

    server = AFCLServer(feat_dim=le, gamma=gamma)
    split = server.register(np.unique(labels))
    server.aggregate(local_train(feats, labels, split, gamma=gamma))

    W_ref = joint_reference_model(feats, labels, server.class_space, gamma=gamma)
    assert np.allclose(server.global_model(), W_ref, atol=1e-10)
