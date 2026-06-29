"""Distances for c=(theta, Gamma) parameter objects."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.linalg import eigvalsh

from .config import GenerationConfig


Param = Tuple[np.ndarray, np.ndarray]



## quantify distaicne between two PSD hessian using the affine riemannian distance
def gamma_airm_distance(G1: np.ndarray, G2: np.ndarray) -> float:
    vals = eigvalsh(G2, G1)
    vals = np.maximum(vals, 1e-15)
    return float(np.linalg.norm(np.log(vals)))



# compute the scale by using the reference pools of the paramters (refer to the note)
def compute_reference_scales(
    theta_pool: Sequence[np.ndarray],
    gamma_pool: Sequence[np.ndarray],
) -> Dict[str, float]:
    """Compute fixed normalization scales for d_f.

    These scales should be computed once from the reference pool and reused for
    all separated/overlapping datasets. Do not recompute scales per dataset.
    """
    dtheta = []
    dgamma = []
    for i in range(len(theta_pool)):
        for j in range(i + 1, len(theta_pool)):
            dtheta.append(float(np.linalg.norm(theta_pool[i] - theta_pool[j])))
            dgamma.append(gamma_airm_distance(gamma_pool[i], gamma_pool[j]))
    s_theta = float(np.mean(dtheta)) if dtheta else 1.0
    s_gamma = float(np.mean(dgamma)) if dgamma else 1.0
    return {"s_theta": max(s_theta, 1e-12), "s_gamma": max(s_gamma, 1e-12)}


# function distance = theta distance + gamma distance

def function_distance(c1: Param, c2: Param, scales: Dict[str, float], cfg: GenerationConfig) -> float:
    """Weighted normalized distance between parameter objects."""
    theta1, gamma1 = c1
    theta2, gamma2 = c2
    d_theta = float(np.linalg.norm(theta1 - theta2))
    d_gamma = gamma_airm_distance(gamma1, gamma2)
    return (
        cfg.weight_theta * d_theta / scales["s_theta"]
        + cfg.weight_gamma * d_gamma / scales["s_gamma"]
    )


def pairwise_distance_matrix(
    theta_list: Sequence[np.ndarray],
    gamma_list: Sequence[np.ndarray],
    scales: Dict[str, float],
    cfg: GenerationConfig,
) -> np.ndarray:
    """Pairwise d_f matrix for a list of generated parameters."""
    n = len(theta_list)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = function_distance(
                (theta_list[i], gamma_list[i]),
                (theta_list[j], gamma_list[j]),
                scales,
                cfg,
            )
    return D
