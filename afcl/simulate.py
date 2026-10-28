"""End-to-end FCL simulation on pre-extracted features.

Deliberately torch-free: by the time we get here the frozen backbone has
already done its one job, so the whole federated continual run is numpy.  That
is also why AFCL is fast -- the "training" below is a handful of Cholesky
solves.

Virtual clients
---------------
Section 3 folds both kinds of temporal increment (new data at an existing
client, and a brand-new client) into one abstraction: every increment is a
*virtual client*.  So a run with ``T`` tasks and ``n_clients`` clients per task
is ``K = T * n_clients`` registrations, processed in sequence.  Evaluation
happens at task boundaries, which is what the paper's ``A^i_j`` indexes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .client import local_train
from .data.dirichlet import dirichlet_split
from .data.si_blurry import SiBlurryConfig, si_blurry_plan, si_blurry_split
from .metrics import AccuracyMatrix, summarize
from .server import AFCLServer, AggregationMode

__all__ = ["FCLConfig", "FCLResult", "run_afcl"]


@dataclass
class FCLConfig:
    n_tasks: int = 5  # T
    n_clients: int = 5  # clients per task
    alpha: float = 0.1  # Dirichlet concentration (spatial heterogeneity)
    gamma: float = 1.0  # ridge regularisation
    r_disjoint: float = 0.5  # r_D
    r_blurry: float = 0.1  # r_B
    mode: AggregationMode = "stable"
    equal_client_size: bool = True
    seed: int = 0
    verbose: bool = True


@dataclass
class FCLResult:
    accuracy: AccuracyMatrix
    per_task: List[dict] = field(default_factory=list)
    cumulative_runtime: List[float] = field(default_factory=list)
    n_virtual_clients: int = 0
    upload_floats: int = 0
    final_classes: int = 0

    @property
    def final_average_accuracy(self) -> float:
        A = self.accuracy.A
        return float(np.mean(A[-1, : A.shape[0]]))


def run_afcl(
    F_train: np.ndarray,
    y_train: np.ndarray,
    F_test: np.ndarray,
    y_test: np.ndarray,
    cfg: FCLConfig,
) -> FCLResult:
    """Run the full AFCL protocol and return the paper's metric table.

    Parameters
    ----------
    F_train, F_test : (N, l_e) frozen backbone features.
    y_train, y_test : (N,) integer labels.
    """
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    le = F_train.shape[1]
    T = cfg.n_tasks

    sb = SiBlurryConfig(
        n_tasks=T, r_disjoint=cfg.r_disjoint, r_blurry=cfg.r_blurry, seed=cfg.seed
    )
    # One class-level layout shared by train and test, so task j means the same
    # thing on both sides.
    plan = si_blurry_plan(np.unique(y_train), sb)
    train_tasks = si_blurry_split(y_train, sb, plan=plan, seed_offset=0)
    test_tasks = si_blurry_split(y_test, sb, plan=plan, seed_offset=1)

    server = AFCLServer(feat_dim=le, gamma=cfg.gamma, mode=cfg.mode)
    acc = AccuracyMatrix.empty(T)
    runtimes: List[float] = []
    upload = 0
    k = 0
    t0 = time.perf_counter()

    for t in range(T):
        idx = train_tasks[t]
        if len(idx) == 0:
            if cfg.verbose:
                print(f"[task {t + 1}/{T}] empty, skipping")
            continue

        shards = dirichlet_split(
            labels=y_train,
            indices=idx,
            n_clients=cfg.n_clients,
            alpha=cfg.alpha,
            seed=cfg.seed * 1000 + t,
            equal_size=cfg.equal_client_size,
        )

        for c, shard in enumerate(shards):
            if len(shard) == 0:
                continue
            Fk, yk = F_train[shard], y_train[shard]

            # --- the three lines that are the entire protocol --------------
            split = server.register(np.unique(yk))  # Section 3.2
            update = local_train(Fk, yk, split, gamma=cfg.gamma)  # Section 3.3
            server.aggregate(update)  # Section 3.4
            # ---------------------------------------------------------------

            upload += update.payload_floats
            k += 1

        runtimes.append(time.perf_counter() - t0)

        # Evaluate on every task seen so far: A^t_j for j <= t.
        row = []
        for j in range(t + 1):
            tj = test_tasks[j]
            row.append(server.score(F_test[tj], y_test[tj]) if len(tj) else float("nan"))
        acc.set_row(t, row)

        if cfg.verbose:
            print(
                f"[task {t + 1}/{T}] clients={k:3d} classes={server.class_space.n_classes:3d} "
                f"avg_acc={np.mean(row):6.2f}%  elapsed={runtimes[-1]:6.2f}s"
            )

    return FCLResult(
        accuracy=acc,
        per_task=summarize(acc.A),
        cumulative_runtime=runtimes,
        n_virtual_clients=k,
        upload_floats=upload,
        final_classes=server.class_space.n_classes,
    )
