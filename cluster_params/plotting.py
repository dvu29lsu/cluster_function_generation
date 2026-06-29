"""Plotting utilities for validation figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .config import GenerationConfig
from .distances import pairwise_distance_matrix


def classical_mds(D: np.ndarray) -> np.ndarray:
    """Two-dimensional classical MDS embedding from a distance matrix."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order][:2], 0.0)
    vecs = vecs[:, order]
    return vecs[:, :2] * np.sqrt(vals)


def make_plots(record: Dict[str, object], out_dir: Path, cfg: GenerationConfig) -> None:
    """Save heatmap, MDS, and silhouette plots for one dataset."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    theta = [np.asarray(x, dtype=float) for x in record["theta_list"]]
    gamma = [np.asarray(x, dtype=float) for x in record["gamma_list"]]
    labels = np.asarray(record["labels"], dtype=int)
    scales = record["distance_definition"]["scales"]
    dataset_id = record["dataset_id"]

    D = pairwise_distance_matrix(theta, gamma, scales, cfg)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(D)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{dataset_id}: d_f heatmap")
    ax.set_xlabel("agent")
    ax.set_ylabel("agent")
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset_id}_distance_heatmap.png", dpi=160)
    plt.close(fig)

    coords = classical_mds(D)
    fig, ax = plt.subplots(figsize=(5, 4))
    for lab in sorted(set(labels.tolist())):
        idx = labels == lab
        ax.scatter(coords[idx, 0], coords[idx, 1], label=f"cluster {lab}")
    ax.set_title(f"{dataset_id}: MDS of d_f")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset_id}_mds.png", dpi=160)
    plt.close(fig)

    samples = record["cluster_metrics"].get("silhouette_samples")
    if samples is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(len(labels)), samples)
        ax.axhline(0.0, linewidth=1)
        ax.set_title(f"{dataset_id}: silhouette samples")
        ax.set_xlabel("agent")
        ax.set_ylabel("s_i")
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset_id}_silhouette.png", dpi=160)
        plt.close(fig)
