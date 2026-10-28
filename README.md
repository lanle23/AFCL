# AFCL — Analytic Federated Continual Learning

Reimplementation of:

> Tang, He, Fan, He, Wang, Liu, Song, Wang, Zhu, Zhuang, Liu.
> **AFCL: Achieving Spatio-Temporal Invariance to Data Heterogeneity in Federated Continual Learning.**
> CVPR 2026, pp. 7768–7778. Preprint: [arXiv:2505.12245](https://arxiv.org/abs/2505.12245)

AFCL is a **gradient-free** federated continual learning method. A frozen pre-trained
backbone produces features; a linear classifier is then fitted by ridge regression in
closed form. Because there is no gradient descent anywhere, there is nothing for
heterogeneous data to destabilise — the aggregated model is *provably identical* to
centralised joint learning over the pooled dataset, regardless of how data is split
across clients and tasks, or in what order clients arrive.

Local training is one forward pass. Aggregation is one round per client. The server
never ships a model back down.

---

## The method in one screen

| Stage | Equation | Code |
|---|---|---|
| Known/unknown class split | (2) | `afcl/class_space.py` |
| Feature extraction `F_k = Backbone(X_k, Θ_B)` | (3) | `afcl/features.py` |
| Label encoding `Ŷ_k`, `Y̌_k` | (4) | `ClassSplit.encode` |
| Local ridge solves `Ŵ_k`, `W̌_k` | (5), (6), (49) | `afcl/client.py` |
| Regularized Gram Matrix `R_k = F_kᵀF_k + γI` | (7) | `afcl/client.py` |
| Global Knowledge Matrix recursion `G_k` | (8), (9) | `afcl/server.py` |
| Global model readout `W_k` | (10) | `AFCLServer.global_model` |
| Metrics (acc / retention / stability / plasticity) | (50)–(53) | `afcl/metrics.py` |

The protocol, verbatim from the code:

```python
split  = server.register(np.unique(y_k))          # §3.2  class splitting
update = local_train(F_k, y_k, split, gamma)      # §3.3  two ridge solves
server.aggregate(update)                          # §3.4  recursive aggregation
```

Each client uploads exactly `(Ŵ_k, W̌_k, R_k)` — `O(l_e² + l_e·d_k)` floats,
**independent of its sample count** (verified by a test).

---

## Install & run

```bash
pip install numpy scipy            # core engine
pip install torch torchvision      # only for real-image benchmarks
pip install pytest                 # only for the test suite
```

```bash
# No download needed — synthetic features, exercises the whole pipeline
python main.py --dataset synthetic --tasks 5 --alpha 0.1

# The headline claim: accuracy flat across all four heterogeneity settings
python main.py --dataset synthetic --sweep

# A real benchmark (downloads CIFAR-100, extracts once, caches)
python main.py --dataset cifar100 --tasks 5 --alpha 0.1 --gamma 1.0

pytest tests/ -q
```

Paper settings are the defaults: ResNet-18 backbone, `T ∈ {5,10}`, 5 clients per task,
`α ∈ {0.1,0.2}` (CIFAR-100 / Tiny-ImageNet) or `{0.5,1.0}` (ImageNet-R),
`r_D = 50%`, `r_B = 10%`.

---

## What the tests actually prove

`tests/test_theorems.py` is the important one — it checks the paper's claims
numerically rather than trusting them:

- **Theorem 1** — the recursively aggregated `W_K` equals the centralised joint ridge
  solution of Eq. (14) to ~1e-12 relative error, across γ ∈ {0, 1e-3, 1, 10, 100}.
- **Lemma 2** — `G_k` matches its closed form `(F ᵀF + k·γI)⁻¹FᵀY`, carrying *k* copies
  of the penalty (which is exactly what Eq. (10) strips back off).
- **Theorem 2a** — permuting client arrival order leaves the class-indexed model
  unchanged.
- **Theorem 2b** — sharding one pooled dataset IID versus one-class-per-client gives
  **the same model**. This is the spatio-temporal invariance claim, and it holds to
  1e-9.

Result: `34 passed, 1 xfailed`.

On the synthetic benchmark the invariance sweep reproduces the qualitative shape of
Table 1 — a flat column where every baseline collapses:

```
   T   alpha   final avg acc
   5     0.1          55.75%
   5     0.2          55.75%
  10     0.1          55.74%
  10     0.2          55.74%
spread: 0.0085 pp
```

The residual 0.0085 pp is *not* model drift — the model is bit-for-bit identical. It
comes only from Si-Blurry assigning test samples to task buckets differently, which
changes the per-task averaging in Eq. (50).

---

## Three places the paper is ambiguous or wrong

These are judgement calls I had to make. Each is flagged in the code and switchable.

**1. The Tanh buffer layer — contradiction.**
Algorithm 1 line 2 says the server determines `Backbone(·,Θ_B)` **and** `Tanh(·,Θ_T)`,
implying an ACIL/RanPAC-style random projection `F = tanh(F_raw W_rand)` before the
ridge solve. But Eq. (3) defines `F_k = Backbone(X_k, Θ_B)` with no such layer, and the
Conclusion lists random projections as *future* work. These cannot both be true.
→ **Default: no buffer** (matches Eq. 3, §3.3 and the Conclusion — two sources against
one). Enable the other reading with `--buffer-size 4096`. Both theorems hold either
way, since the buffer is a fixed shared map that just redefines `F`.

**2. Eq. (51) retention is inverted.**
As printed, `F_i = 1/(i-1) · Σ (A^j_j / A^i_j)`. But the paper says AFCL's retention
"consistently exceeds 1.00, indicating continual acquisition of new knowledge" while
baselines fall "below 1.00 due to forgetting". Under the printed ratio, *forgetting*
would push the value above 1.00 — backwards.
→ **Default: `A^i_j / A^j_j`**, matching the stated interpretation and Figure 4. Pass
`paper_literal=True` for the typeset version. (Our run shows retention rising 1.004 →
1.023, reproducing the described behaviour.)

**3. Eq. (9) is numerically unusable, and undefined at γ=0.**
`B_k` contains an explicit `R_k⁻¹`. At γ=0, `R_k = F_kᵀF_k` is singular for any client
with `N_k < l_e` — a *normal* situation under strong label skew. The paper reports a
γ=0 column in Table 2, so this edge is reachable in their own sweep. Even for γ>0, the
literal form inverts three matrices per round and compounds float error.
→ Three algebraically identical `--mode` options:

| mode | `A_k`, `B_k` | notes |
|---|---|---|
| `paper` | verbatim Eq. (9) | faithful; 3 inversions/round; fails at γ=0 |
| `simplified` | `A_k = R̃_k⁻¹R̃_{k-1}`, `B_k = R̃_k⁻¹R_k` | what Eq. (9) collapses to via Appendix C's Woodbury steps |
| `stable` **(default)** | none — accumulate `C_k = FᵀY` directly | Appendix C shows `B_kŴ_k = R̃_k⁻¹F_kᵀŶ_k`; the server recovers `F_kᵀŶ_k = R_kŴ_k` |

`stable` consumes **exactly** the paper's upload payload `(Ŵ_k, W̌_k, R_k)` and changes
nothing clients do — it is just the sane way to evaluate the same algebra. The test
suite asserts all three agree.

A further judgement call: Appendix F.1 says clients have "an **equal number of
samples**" *and* follow a Dirichlet distribution. That is a transport problem with
fixed row marginals (equal client sizes) and column marginals (class availability), not
the usual draw-and-let-sizes-fall recipe. `afcl/data/dirichlet.py` solves it with
iterative proportional fitting plus exact integer rounding. `--unequal-clients` gives
the conventional version.

---

## Layout

```
afcl/
  class_space.py   known/unknown splitting, dynamic one-hot (Eq. 2)
  client.py        analytic local training (Eqs. 3–7)
  server.py        recursive aggregation + readout (Eqs. 8–10)
  linalg.py        SPD solves with pseudo-inverse fallback
  features.py      frozen backbone + optional Tanh buffer, with caching
  metrics.py       Eqs. 50–53
  simulate.py      the T × n_clients virtual-client loop
  data/
    si_blurry.py   temporal heterogeneity
    dirichlet.py   spatial heterogeneity
    datasets.py    CIFAR-100 / Tiny-ImageNet / ImageNet-R
main.py            CLI
tests/             theorem verification + pipeline tests
```

**Precision note:** all AFCL state is `float64`. Eq. (8)–(9) chains explicit inverses,
and float32 loses the Theorem 1 equivalence within a few rounds. Features may be
stored as float32; they are promoted on entry.

---

## Status / caveats

- The NumPy core (`class_space`, `client`, `server`, `linalg`, `metrics`, `simulate`,
  `data/`) is **fully executed and tested** here.
- `features.py` and `data/datasets.py` need `torch`/`torchvision`, which could not be
  installed in the environment this was authored in. They are compile-checked and
  reviewed but **not executed** — expect to shake out minor issues on first real run
  (dataset paths especially). Torch is imported lazily everywhere, so the tested core
  runs without it.
- Accuracy numbers above are on **synthetic features**, calibrated to sit near the
  paper's CIFAR-100 operating point. They are not a reproduction of Table 1's absolute
  values; run `--dataset cifar100` for that.
