"""Frozen feature extraction -- Eq. (3), plus the optional Tanh buffer layer.

The backbone is never updated, so features can be extracted once for the whole
run and cached to disk.  This is why AFCL's "single-epoch local training" is
genuinely a single forward pass: there is nothing to iterate.

On the Tanh layer
-----------------
Algorithm 1 line 2 says the server determines both ``Backbone(., Theta_B)``
*and* ``Tanh(., Theta_T)`` -- i.e. an ACIL/RanPAC-style random buffer layer
``F = tanh(F_raw @ W_rand)`` that expands the embedding before the ridge solve.
But the main text's Eq. (3) defines ``F_k = Backbone(X_k, Theta_B)`` with no
such layer, and the Conclusion lists random projections as *future* work
("there is still promising potential ... by incorporating techniques such as
random projections, kernel methods, and ensemble learning").

The paper is internally inconsistent here.  We default to **no buffer**, which
matches Eq. (3), Section 3.3 and the Conclusion -- the majority reading -- and
expose ``buffer_size`` to switch the Algorithm-1 reading on.  Turning it on
typically *helps* accuracy, since it makes the classifier non-linear in the
backbone features; see the README.

Theorem 1 and Theorem 2 hold either way: the buffer is a fixed, shared,
deterministic map applied identically on every client, so it is just a
redefinition of ``F``.  Every client must use the same ``Theta_T``, which is
why the server broadcasts its seed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["BackboneConfig", "FeatureExtractor", "build_backbone"]


@dataclass
class BackboneConfig:
    name: str = "resnet18"  # the paper's unified backbone
    pretrained: bool = True  # AFCL requires a *pre-trained* frozen extractor
    buffer_size: Optional[int] = None  # Algorithm 1's Tanh(., Theta_T); None = off
    buffer_seed: int = 0
    image_size: int = 224
    batch_size: int = 256
    device: Optional[str] = None
    num_workers: int = 4


def build_backbone(cfg: BackboneConfig):
    """Return ``(module, embedding_dim)`` with the classifier head removed."""
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    name = cfg.name.lower()
    weights = "DEFAULT" if cfg.pretrained else None

    if name in {"resnet18", "resnet34", "resnet50"}:
        model = getattr(tvm, name)(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name in {"vit_b_16", "vit_b_32"}:
        model = getattr(tvm, name)(weights=weights)
        dim = model.heads.head.in_features
        model.heads = nn.Identity()
    else:
        raise ValueError(f"unsupported backbone {cfg.name!r}")

    for p in model.parameters():  # frozen: gradient-free by construction
        p.requires_grad_(False)
    model.eval()
    return model, dim


class FeatureExtractor:
    """Extract ``F = Backbone(X)``, optionally through the Tanh buffer layer."""

    def __init__(self, cfg: BackboneConfig):
        import torch

        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.raw_dim = build_backbone(cfg)
        self.model.to(self.device)

        self.buffer: Optional[np.ndarray] = None
        if cfg.buffer_size:
            # Shared random projection; identical on every client because the
            # server broadcasts Theta_T (here: its seed and shape).
            g = np.random.default_rng(cfg.buffer_seed)
            self.buffer = g.normal(
                scale=1.0 / np.sqrt(self.raw_dim), size=(self.raw_dim, cfg.buffer_size)
            )
        self.dim = cfg.buffer_size if cfg.buffer_size else self.raw_dim

    def _apply_buffer(self, F: np.ndarray) -> np.ndarray:
        if self.buffer is None:
            return F
        return np.tanh(F @ self.buffer)

    @property
    def signature(self) -> str:
        """Stable id for the cache filename."""
        c = self.cfg
        s = f"{c.name}|pre={c.pretrained}|buf={c.buffer_size}|seed={c.buffer_seed}|img={c.image_size}"
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    def extract(self, dataset, cache_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Run the frozen backbone over a torchvision-style dataset.

        Returns ``(F, y)`` as float32 / int64 numpy arrays.  Results are cached
        to ``cache_path`` because the backbone never changes, so a full
        hyper-parameter sweep pays for extraction exactly once.
        """
        import torch
        from torch.utils.data import DataLoader

        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path)
            return z["F"], z["y"]

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device == "cuda"),
        )
        feats, ys = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                out = self.model(xb)
                feats.append(out.detach().float().cpu().numpy())
                ys.append(np.asarray(yb))
        F = np.concatenate(feats, axis=0)
        y = np.concatenate(ys, axis=0).astype(np.int64)
        F = self._apply_buffer(F).astype(np.float32)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez_compressed(cache_path, F=F, y=y)
        return F, y
