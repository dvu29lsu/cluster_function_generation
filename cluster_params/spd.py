"""Symmetric positive definite matrix helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of A."""
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def make_spd(A: np.ndarray, eig_floor: float, jitter: float = 1e-10) -> Tuple[np.ndarray, float]:
    """Return an SPD matrix by shifting the spectrum when necessary.

    Parameters
    ----------
    A:
        Square matrix to symmetrize and correct.
    eig_floor:
        Desired lower bound on the minimum eigenvalue.
    jitter:
        Small positive numerical margin added when a shift is needed.

    Returns
    -------
    Gamma:
        Symmetric positive definite matrix with lambda_min >= eig_floor.
    shift:
        The scalar shift added to the diagonal. A value of 0 means no correction
        was required.
    """
    S = symmetrize(A)
    lam_min = float(np.min(np.linalg.eigvalsh(S)))
    shift = max(0.0, eig_floor - lam_min + jitter)
    if shift > 0.0:
        S = S + shift * np.eye(S.shape[0])
    return S, shift
