from __future__ import annotations

from typing import Tuple

import numpy as np


def symmetrize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def make_spd(A: np.ndarray, eig_floor: float, jitter: float = 1e-10) -> Tuple[np.ndarray, float]:
    S = symmetrize(A)
    lam_min = float(np.min(np.linalg.eigvalsh(S)))
    shift = max(0.0, eig_floor - lam_min + jitter)
    if shift > 0.0:
        S = S + shift * np.eye(S.shape[0])
    return S, shift
