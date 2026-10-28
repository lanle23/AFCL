"""Dataset loading for the paper's three benchmarks.

  * CIFAR-100      -- torchvision, downloads automatically
  * Tiny-ImageNet  -- ImageFolder; expects ``<root>/tiny-imagenet-200/{train,val}``
                      with the val split already reorganised into class folders
  * ImageNet-R     -- ImageFolder; 30k images / 200 classes, split here by a
                      fixed seed since the release ships no train/test split

All three are resized to the backbone's input size and normalised with the
ImageNet statistics the pre-trained weights expect.
"""

from __future__ import annotations

import os
from typing import Tuple

__all__ = ["build_datasets", "DATASETS"]

DATASETS = ("cifar100", "tiny-imagenet", "imagenet-r")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def build_datasets(name: str, root: str, image_size: int = 224, seed: int = 0) -> Tuple:
    """Return ``(train_set, test_set, n_classes)``.

    No augmentation anywhere: the backbone is frozen and training is a single
    deterministic forward pass, so augmentation would only add variance to the
    Gram matrices.
    """
    name = name.lower()
    tf = _transform(image_size)

    if name == "cifar100":
        from torchvision.datasets import CIFAR100

        train = CIFAR100(root=root, train=True, download=True, transform=tf)
        test = CIFAR100(root=root, train=False, download=True, transform=tf)
        return train, test, 100

    if name in {"tiny-imagenet", "tiny_imagenet", "tinyimagenet"}:
        from torchvision.datasets import ImageFolder

        base = os.path.join(root, "tiny-imagenet-200")
        train_dir, val_dir = os.path.join(base, "train"), os.path.join(base, "val")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"expected {train_dir}. Download Tiny-ImageNet-200 and, if the val "
                f"split is still flat, reorganise it into per-class folders using "
                f"val_annotations.txt."
            )
        return ImageFolder(train_dir, transform=tf), ImageFolder(val_dir, transform=tf), 200

    if name in {"imagenet-r", "imagenet_r", "imagenetr"}:
        import numpy as np
        from torch.utils.data import Subset
        from torchvision.datasets import ImageFolder

        base = os.path.join(root, "imagenet-r")
        if not os.path.isdir(base):
            raise FileNotFoundError(f"expected {base} laid out as class subfolders")
        full = ImageFolder(base, transform=tf)

        # ImageNet-R ships no split; take a stratified 80/20 with a fixed seed.
        targets = np.asarray(full.targets)
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for c in np.unique(targets):
            idx = rng.permutation(np.where(targets == c)[0])
            cut = int(round(0.8 * len(idx)))
            tr.extend(idx[:cut].tolist())
            te.extend(idx[cut:].tolist())
        return Subset(full, sorted(tr)), Subset(full, sorted(te)), len(full.classes)

    raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")


def dataset_targets(ds):
    """Labels of a dataset as a numpy array, following Subset indirection."""
    import numpy as np
    from torch.utils.data import Subset

    if isinstance(ds, Subset):
        return np.asarray(ds.dataset.targets)[np.asarray(ds.indices)]
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets)
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels)
    raise AttributeError(f"cannot read labels from {type(ds).__name__}")
