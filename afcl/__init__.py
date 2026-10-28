"""AFCL -- Analytic Federated Continual Learning.

Reimplementation of::

    Tang et al., "AFCL: Achieving Spatio-Temporal Invariance to Data
    Heterogeneity in Federated Continual Learning", CVPR 2026 (pp. 7768-7778).
    Preprint: arXiv:2505.12245.

Equation and section numbers in the docstrings refer to that paper.
"""

from .class_space import ClassSpace, ClassSplit
from .client import ClientUpdate, local_train
from .linalg import spd_inv, spd_solve
from .server import AFCLServer, joint_reference_model

__all__ = [
    "AFCLServer",
    "ClassSpace",
    "ClassSplit",
    "ClientUpdate",
    "joint_reference_model",
    "local_train",
    "spd_inv",
    "spd_solve",
]

__version__ = "0.1.0"
