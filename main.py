#!/usr/bin/env python3
"""Run AFCL on one of the paper's benchmarks.

Examples
--------
    # CIFAR-100, T=5, alpha=0.1  (Table 1, row 1)
    python main.py --dataset cifar100 --tasks 5 --alpha 0.1

    # Reproduce the invariance claim: identical accuracy across all four settings
    python main.py --dataset cifar100 --sweep

    # No dataset download required -- synthetic features, exercises the full pipeline
    python main.py --dataset synthetic --tasks 5 --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from afcl.metrics import summarize
from afcl.simulate import FCLConfig, run_afcl


def load_features(args):
    """Return ``(F_train, y_train, F_test, y_test)``."""
    if args.dataset == "synthetic":
        rng = np.random.default_rng(args.seed)
        n_cls, le = args.synthetic_classes, args.synthetic_dim
        centers = rng.normal(scale=1.0, size=(n_cls, le))

        def gen(n_per):
            y = np.repeat(np.arange(n_cls), n_per)
            F = centers[y] + rng.normal(scale=args.synthetic_noise, size=(len(y), le))
            return F.astype(np.float32), y

        Ftr, ytr = gen(args.synthetic_per_class)
        Fte, yte = gen(max(20, args.synthetic_per_class // 5))
        return Ftr, ytr, Fte, yte

    # Real benchmark: run the frozen backbone once, then cache.
    from afcl.data.datasets import build_datasets, dataset_targets
    from afcl.features import BackboneConfig, FeatureExtractor

    bb = BackboneConfig(
        name=args.backbone,
        pretrained=not args.no_pretrained,
        buffer_size=args.buffer_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    train_set, test_set, _ = build_datasets(args.dataset, args.data_root, args.image_size)
    fx = FeatureExtractor(bb)

    os.makedirs(args.cache_dir, exist_ok=True)
    tag = f"{args.dataset}_{fx.signature}"
    print(f"extracting features with frozen {args.backbone} (l_e={fx.dim}) ...")
    Ftr, ytr = fx.extract(train_set, os.path.join(args.cache_dir, f"{tag}_train.npz"))
    Fte, yte = fx.extract(test_set, os.path.join(args.cache_dir, f"{tag}_test.npz"))
    return Ftr, ytr, Fte, yte


def print_table(result) -> None:
    print()
    print(result.accuracy)
    print()
    print(f"{'task':>5} {'avg acc':>9} {'retention':>10} {'stability':>10} {'plasticity':>11}")
    for r in result.per_task:
        print(
            f"{r['task']:>5} {r['avg_acc']:>8.2f}% {r['retention']:>10.3f} "
            f"{r['stability']:>9.2f}% {r['plasticity']:>10.2f}%"
        )
    print()
    print(f"final average accuracy : {result.final_average_accuracy:.2f}%")
    print(f"virtual clients        : {result.n_virtual_clients}")
    print(f"classes registered     : {result.final_classes}")
    print(f"total uploaded floats  : {result.upload_floats:,}")
    if result.cumulative_runtime:
        print(f"cumulative runtime     : {result.cumulative_runtime[-1]:.2f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="AFCL (CVPR 2026) reimplementation")
    p.add_argument("--dataset", default="synthetic",
                   choices=["cifar100", "tiny-imagenet", "imagenet-r", "synthetic"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--cache-dir", default="./cache")

    p.add_argument("--tasks", type=int, default=5, help="T")
    p.add_argument("--clients", type=int, default=5, help="clients per task")
    p.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration")
    p.add_argument("--gamma", type=float, default=1.0, help="ridge regularisation")
    p.add_argument("--r-disjoint", type=float, default=0.5)
    p.add_argument("--r-blurry", type=float, default=0.1)
    p.add_argument("--mode", default="stable", choices=["stable", "simplified", "paper"])
    p.add_argument("--unequal-clients", action="store_true",
                   help="allow varying client sizes (paper specifies equal)")

    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--buffer-size", type=int, default=None,
                   help="enable Algorithm 1's Tanh buffer layer with this width")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--synthetic-classes", type=int, default=100)
    p.add_argument("--synthetic-dim", type=int, default=512)
    p.add_argument("--synthetic-per-class", type=int, default=120)
    # Calibrated so the synthetic benchmark lands near the paper's CIFAR-100
    # operating point (~56%), rather than saturating at 100%.
    p.add_argument("--synthetic-noise", type=float, default=7.0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", action="store_true",
                   help="run the four (T, alpha) settings of Table 1 and compare")
    p.add_argument("--json", default=None, help="write results to this path")
    args = p.parse_args()

    Ftr, ytr, Fte, yte = load_features(args)
    print(f"train {Ftr.shape}  test {Fte.shape}  classes {len(np.unique(ytr))}")

    def run(tasks, alpha, verbose=True):
        cfg = FCLConfig(
            n_tasks=tasks, n_clients=args.clients, alpha=alpha, gamma=args.gamma,
            r_disjoint=args.r_disjoint, r_blurry=args.r_blurry, mode=args.mode,
            equal_client_size=not args.unequal_clients, seed=args.seed, verbose=verbose,
        )
        return run_afcl(Ftr, ytr, Fte, yte, cfg)

    if args.sweep:
        settings = [(5, 0.1), (5, 0.2), (10, 0.1), (10, 0.2)]
        print("\nSpatio-temporal invariance sweep (Table 1 layout)")
        print(f"{'T':>4} {'alpha':>7} {'final avg acc':>15}")
        rows = []
        for T, a in settings:
            r = run(T, a, verbose=False)
            rows.append({"T": T, "alpha": a, "acc": r.final_average_accuracy})
            print(f"{T:>4} {a:>7} {r.final_average_accuracy:>14.2f}%")
        spread = max(x["acc"] for x in rows) - min(x["acc"] for x in rows)
        print(f"\nspread across settings: {spread:.4f} pp")
        print("(Theorem 2 predicts the *model* is identical; residual spread comes "
              "only from how Si-Blurry assigns test samples to tasks.)")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=2)
        return

    result = run(args.tasks, args.alpha)
    print_table(result)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "accuracy_matrix": result.accuracy.A.tolist(),
                    "per_task": result.per_task,
                    "final_average_accuracy": result.final_average_accuracy,
                    "n_virtual_clients": result.n_virtual_clients,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
