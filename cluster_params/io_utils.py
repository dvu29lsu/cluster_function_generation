"""Input/output helpers for parameter datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.io import loadmat

from .config import GenerationConfig
from .spd import make_spd


def load_set1_mat(path: Path, cfg: GenerationConfig) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load theta and Gamma/Phi arrays from Set1.mat.

    Expected keys are 'theta' and 'Phi', matching the old project files.
    """
    data = loadmat(path)
    if "theta" not in data or "Phi" not in data:
        raise KeyError("Expected Set1.mat to contain keys 'theta' and 'Phi'.")

    theta_raw = data["theta"].ravel()
    gamma_raw = data["Phi"].ravel()
    theta_pool = [np.asarray(t, dtype=float).reshape(-1) for t in theta_raw]
    gamma_pool = [make_spd(np.asarray(G, dtype=float), cfg.eig_floor, cfg.jitter)[0] for G in gamma_raw]
    return theta_pool, gamma_pool


# if set 1 not available, generate a random pool of parameters for testing
def make_example_pool(cfg: GenerationConfig, n_pool: int = 12) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    theta_pool: List[np.ndarray] = []
    gamma_pool: List[np.ndarray] = []
    for _ in range(n_pool):
        theta = rng.normal(0.0, 1.0, size=cfg.dimension)
        B = rng.normal(0.0, 0.7, size=(cfg.dimension, cfg.dimension))
        gamma = B.T @ B + cfg.eig_floor * np.eye(cfg.dimension)
        theta_pool.append(theta)
        gamma_pool.append(gamma)
    return theta_pool, gamma_pool


def to_jsonable(obj):
    """Convert numpy objects and dataclasses into JSON-serializable objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_jsonl_record(path: Path, record: Dict[str, object]) -> None:
    """Save one JSON object as a single-line JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(record)) + "\n")


def load_jsonl_record(path: Path) -> Dict[str, object]:
    """Load the first record from a JSONL dataset file."""
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
    return json.loads(line)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def config_to_dict(cfg: GenerationConfig) -> Dict[str, object]:
    """Dataclass config as a JSON-safe dictionary."""
    return asdict(cfg)
