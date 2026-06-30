from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .config import GenerationConfig
from .distances import pairwise_distance_matrix


def select_prototypes(
    theta_pool: Sequence[np.ndarray],
    gamma_pool: Sequence[np.ndarray],
    K: int,
    scales: Dict[str, float],
    cfg: GenerationConfig,
) -> List[int]:
    """Select prototype indices from a candidate pool.

    K=1: medoid.
    K=2: farthest pair.
    K=3: farthest pair plus max-min third prototype.
    """
    if K < 1 or K > 3:
        raise ValueError("Only K=1,2,3 are supported in this experiment.")

    D = pairwise_distance_matrix(theta_pool, gamma_pool, scales, cfg)

    if K == 1:
        return [int(np.argmin(D.sum(axis=1)))]

    i, j = np.unravel_index(np.argmax(D), D.shape)
    
    selected = [int(i), int(j)]
    
    if K == 2:
        return selected

    best_idx = None
    
    best_score = -np.inf

    for m in range(len(theta_pool)):
        if m in selected:
            continue
        score = min(D[m, s] for s in selected)
        if score > best_score:
            best_score = score
            best_idx = m
    
    selected.append(int(best_idx))
    
    return selected


def cluster_sizes(n_functions: int, K: int) -> List[int]:
    # balancing cluster sizes
    
    if K == 1:
        return [n_functions]
    
    if K == 2:
        return [n_functions // 2, n_functions - n_functions // 2]
    
    if K == 3:
        return [n_functions - 2 * (n_functions // 3), n_functions // 3, n_functions // 3]
    
