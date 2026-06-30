"""Cluster-quality metrics and validation."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np

from .config import GenerationConfig
from .distances import function_distance, pairwise_distance_matrix


def silhouette_score_from_distances(
    D: np.ndarray,
    labels: Sequence[int],
) -> Tuple[Optional[float], Optional[List[float]]]:
    
    #compute the Silhouette score from a pairwise distance matrix D and the labels

    labels = np.asarray(labels)
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        return None, None

    samples = []
    for i in range(len(labels)):
        own = labels == labels[i]
        own[i] = False
        a_i = float(np.mean(D[i, own])) if np.any(own) else 0.0

        b_vals = []
        for lab in unique:
            if lab == labels[i]:
                continue
            other = labels == lab
            b_vals.append(float(np.mean(D[i, other])))
        b_i = min(b_vals)

        denom = max(a_i, b_i)

        samples.append(0.0 if denom <= 1e-15 else (b_i - a_i) / denom)

    return float(np.mean(samples)), [float(x) for x in samples]


def cluster_quality_metrics(
    theta_list: Sequence[np.ndarray],
    gamma_list: Sequence[np.ndarray],
    labels: Sequence[int],
    theta_proto: Sequence[np.ndarray],
    gamma_proto: Sequence[np.ndarray],
    scales: Dict[str, float],
    cfg: GenerationConfig,
) -> Dict[str, object]:
    
    """Compute OR, pairwise OR, Silhouette, and prototype-based DB"""

    labels_arr = np.asarray(labels, dtype=int)
    K = len(theta_proto)

    radii: List[float] = []
    
    scatter: List[float] = []

    for k in range(K):
        idx = np.where(labels_arr == k)[0]
        distances = [
            function_distance((theta_list[i], gamma_list[i]), (theta_proto[k], gamma_proto[k]), scales, cfg)
            for i in idx
        ]
        radii.append(float(max(distances)))
        scatter.append(float(np.mean(distances)))

    proto_dist: Dict[str, float] = {}
    
    pairwise_or: Dict[str, float] = {}
    
    if K >= 2:
        for k in range(K):
            for ell in range(k + 1, K):
                delta = function_distance((theta_proto[k], gamma_proto[k]), (theta_proto[ell], gamma_proto[ell]), scales, cfg)
                key = f"{k}-{ell}"
                proto_dist[key] = float(delta)
                pairwise_or[key] = float((radii[k] + radii[ell]) / max(delta, 1e-15))
        overlap_ratio = float(max(pairwise_or.values()))
    else:
        overlap_ratio = None

    D = pairwise_distance_matrix(theta_list, gamma_list, scales, cfg)
    
    sil, sil_samples = silhouette_score_from_distances(D, labels_arr)

    if K >= 2:
        db_terms = []
        for k in range(K):
            worst = -np.inf
            for ell in range(K):
                if ell == k:
                    continue
                key = f"{min(k, ell)}-{max(k, ell)}"
                val = (scatter[k] + scatter[ell]) / max(proto_dist[key], 1e-15)
                worst = max(worst, val)
            db_terms.append(worst)
        db = float(np.mean(db_terms))
    else:
        db = None

    full_pairwise_overlap = None if K < 2 else all(v > 1.0 for v in pairwise_or.values())
    if K == 3:
        overlap_graph_connected = sum(v > 1.0 for v in pairwise_or.values()) >= 2
    elif K == 2:
        overlap_graph_connected = full_pairwise_overlap
    else:
        overlap_graph_connected = None

    return {
        "cluster_radii": radii,
        "cluster_scatter": scatter,
        "prototype_distances": proto_dist,
        "pairwise_overlap_ratios": pairwise_or,
        "overlap_ratio": overlap_ratio,
        "full_pairwise_overlap": full_pairwise_overlap,
        "overlap_graph_connected": overlap_graph_connected,
        "silhouette_score": sil,
        "silhouette_samples": sil_samples,
        "davies_bouldin_index": db,
    }


def is_overlap_valid(metrics: Dict[str, object], K: int, mode: str) -> bool:
    """Check whether the computed overlap ratios satisfy the requested scenario."""
    if K == 1:
        return False
    ratios = [float(v) for v in metrics["pairwise_overlap_ratios"].values()]
    if K == 2:
        return ratios[0] > 1.0
    if mode == "full":
        return all(v > 1.0 for v in ratios)
    if mode == "connected":
        return sum(v > 1.0 for v in ratios) >= 2
    raise ValueError("overlap mode must be 'full' or 'connected'")


def validate_record(record: Dict[str, object], cfg: GenerationConfig) -> Dict[str, object]:
    """Validate shape, symmetry, and positive definiteness for one dataset record."""
    gamma = np.asarray(record["gamma_list"], dtype=float)
    theta = np.asarray(record["theta_list"], dtype=float)
    labels = np.asarray(record["labels"], dtype=int)

    if theta.ndim != 2 or gamma.ndim != 3:
        raise ValueError("theta_list or gamma_list has invalid shape.")
    if theta.shape[0] != gamma.shape[0] or theta.shape[0] != labels.size:
        raise ValueError("theta_list, gamma_list, and labels have inconsistent lengths.")

    sym_errs = np.max(np.abs(gamma - np.swapaxes(gamma, 1, 2)), axis=(1, 2))
    eig_mins = np.array([np.min(np.linalg.eigvalsh(G)) for G in gamma])
    eig_maxs = np.array([np.max(np.linalg.eigvalsh(G)) for G in gamma])

    if np.max(sym_errs) > 1e-8:
        raise ValueError("At least one Gamma is not symmetric.")
    if np.min(eig_mins) <= 0.0:
        raise ValueError("At least one Gamma is not positive definite.")
    if np.min(eig_mins) < cfg.eig_floor - 1e-8:
        raise ValueError("At least one Gamma violates eig_floor.")

    return {
        "maximum_symmetry_error": float(np.max(sym_errs)),
        "minimum_gamma_eigenvalue": float(np.min(eig_mins)),
        "maximum_gamma_eigenvalue": float(np.max(eig_maxs)),
        "maximum_spd_shift": float(max(record.get("spd_shifts", [0.0]))),
    }
