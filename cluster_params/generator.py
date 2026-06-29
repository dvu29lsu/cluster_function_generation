"""Core dataset-generation routines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import GenerationConfig
from .distances import function_distance, compute_reference_scales
from .io_utils import (
    config_to_dict,
    load_set1_mat,
    make_example_pool,
    save_jsonl_record,
    to_jsonable,
    write_csv,
)
from .metrics import cluster_quality_metrics, is_overlap_valid, validate_record
from .plotting import make_plots
from .prototypes import cluster_sizes, select_prototypes
from .spd import make_spd, symmetrize


def minimum_prototype_distance(
    theta_proto: Sequence[np.ndarray],
    gamma_proto: Sequence[np.ndarray],
    scales: Dict[str, float],
    cfg: GenerationConfig,
) -> Optional[float]:
    """Minimum pairwise prototype distance. Undefined for K=1."""
    if len(theta_proto) < 2:
        return None
    values = []
    for k in range(len(theta_proto)):
        for ell in range(k + 1, len(theta_proto)):
            values.append(function_distance((theta_proto[k], gamma_proto[k]), (theta_proto[ell], gamma_proto[ell]), scales, cfg))
    return float(min(values))


def sample_around_prototype(
    theta0: np.ndarray,
    gamma0: np.ndarray,
    rng: np.random.Generator,
    cfg: GenerationConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample one parameter object around one prototype."""
    theta = theta0 + cfg.theta_noise * rng.normal(size=theta0.shape)
    A = rng.normal(size=gamma0.shape)
    E = cfg.gamma_noise * symmetrize(A)
    gamma, shift = make_spd(gamma0 + E, cfg.eig_floor, cfg.jitter)
    return theta, gamma, shift


def generate_with_radius_rule(
    theta_proto: Sequence[np.ndarray],
    gamma_proto: Sequence[np.ndarray],
    scenario: str,
    radius_upper: Optional[float],
    radius_lower: float,
    scales: Dict[str, float],
    cfg: GenerationConfig,
    seed: int,
) -> Dict[str, object]:
    """Generate samples around prototypes with optional radius acceptance."""
    rng = np.random.default_rng(seed)
    K = len(theta_proto)
    sizes = cluster_sizes(cfg.n_functions, K)

    theta_list: List[np.ndarray] = []
    gamma_list: List[np.ndarray] = []
    labels: List[int] = []
    attempts: List[int] = []
    spd_shifts: List[float] = []

    for k, n_k in enumerate(sizes):
        for _ in range(n_k):
            for attempt in range(1, cfg.max_attempts_per_sample + 1):
                theta, gamma, shift = sample_around_prototype(theta_proto[k], gamma_proto[k], rng, cfg)
                r = function_distance((theta, gamma), (theta_proto[k], gamma_proto[k]), scales, cfg)
                upper_ok = True if radius_upper is None else r <= radius_upper + 1e-12
                lower_ok = r >= radius_lower - 1e-12
                if lower_ok and upper_ok:
                    theta_list.append(theta)
                    gamma_list.append(gamma)
                    labels.append(k)
                    attempts.append(attempt)
                    spd_shifts.append(shift)
                    break
            else:
                raise RuntimeError(
                    f"Could not sample point for cluster {k}. Try increasing radius_upper, "
                    "decreasing noise, or lowering radius_lower_fraction."
                )

    return {
        "scenario": scenario,
        "theta_list": theta_list,
        "gamma_list": gamma_list,
        "labels": labels,
        "theta_prototypes": list(theta_proto),
        "gamma_prototypes": list(gamma_proto),
        "cluster_sizes": sizes,
        "radius_upper": radius_upper,
        "radius_lower": radius_lower,
        "sampling_attempts": attempts,
        "spd_shifts": spd_shifts,
    }


def contract_prototypes(
    theta_proto: Sequence[np.ndarray],
    gamma_proto: Sequence[np.ndarray],
    contraction_factor: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Move prototypes toward their common center.

    Gamma contraction uses the arithmetic SPD mean. Since the SPD cone is convex,
    Gamma_c + rho*(Gamma_k-Gamma_c) remains SPD for rho in [0,1].
    """
    theta_c = np.mean(np.stack(theta_proto), axis=0)
    gamma_c = np.mean(np.stack(gamma_proto), axis=0)
    theta_new = [theta_c + contraction_factor * (t - theta_c) for t in theta_proto]
    gamma_new = [gamma_c + contraction_factor * (G - gamma_c) for G in gamma_proto]
    return theta_new, gamma_new


def generate_separated(
    theta_proto: Sequence[np.ndarray],
    gamma_proto: Sequence[np.ndarray],
    scales: Dict[str, float],
    cfg: GenerationConfig,
    seed: int,
) -> Dict[str, object]:
    """Generate homogeneous or separated parameter data."""
    K = len(theta_proto)
    if K == 1:
        R = cfg.one_cluster_radius
        scenario = "homogeneous"
    else:
        delta_min = minimum_prototype_distance(theta_proto, gamma_proto, scales, cfg)
        assert delta_min is not None
        R = cfg.separation_alpha * delta_min / 2.0
        scenario = "separated"

    return generate_with_radius_rule(
        theta_proto,
        gamma_proto,
        scenario,
        radius_upper=R,
        radius_lower=cfg.radius_lower_fraction * R,
        scales=scales,
        cfg=cfg,
        seed=seed,
    )


def generate_overlapping(
    theta_proto_base: Sequence[np.ndarray],
    gamma_proto_base: Sequence[np.ndarray],
    scales: Dict[str, float],
    cfg: GenerationConfig,
    seed: int,
) -> Dict[str, object]:
    """Generate overlapping data by prototype contraction and metric validation."""
    K = len(theta_proto_base)
    if K == 1:
        raise ValueError("Overlap is not defined for K=1.")

    last_candidate = None
    for rho in cfg.contraction_grid:
        theta_c, gamma_c = contract_prototypes(theta_proto_base, gamma_proto_base, rho)
        candidate = generate_with_radius_rule(
            theta_c,
            gamma_c,
            "overlapping",
            radius_upper=None,
            radius_lower=0.0,
            scales=scales,
            cfg=cfg,
            seed=seed,
        )
        candidate["contraction_factor"] = rho
        metrics = cluster_quality_metrics(
            candidate["theta_list"],
            candidate["gamma_list"],
            candidate["labels"],
            candidate["theta_prototypes"],
            candidate["gamma_prototypes"],
            scales,
            cfg,
        )
        candidate["cluster_metrics"] = metrics
        last_candidate = candidate
        if is_overlap_valid(metrics, K, cfg.overlap_mode):
            candidate["overlap_search_status"] = "accepted_largest_valid_contraction"
            return candidate

    raise RuntimeError(
        "No contraction factor in the grid produced the requested overlap. "
        f"Last metrics were: {last_candidate.get('cluster_metrics') if last_candidate else None}"
    )


def build_record(
    dataset_id: str,
    K: int,
    scenario_data: Dict[str, object],
    metrics: Dict[str, object],
    prototype_indices: List[int],
    source_description: str,
    scales: Dict[str, float],
    cfg: GenerationConfig,
) -> Dict[str, object]:
    """Build one JSONL-ready dataset record."""
    record = {
        "schema_version": "params-v2-modular",
        "dataset_id": dataset_id,
        "description": "Clustered parameter dataset c_i=(theta_i,Gamma_i). No function family is fixed here.",
        "source_description": source_description,
        "n_functions": cfg.n_functions,
        "dimension": int(len(scenario_data["theta_list"][0])),
        "n_clusters": K,
        "scenario": scenario_data["scenario"],
        "contraction_factor": scenario_data.get("contraction_factor", 1.0),
        "prototype_indices": prototype_indices,
        "labels": scenario_data["labels"],
        "cluster_sizes": scenario_data["cluster_sizes"],
        "theta_list": scenario_data["theta_list"],
        "gamma_list": scenario_data["gamma_list"],
        "theta_prototypes": scenario_data["theta_prototypes"],
        "gamma_prototypes": scenario_data["gamma_prototypes"],
        "radius_upper": scenario_data["radius_upper"],
        "radius_lower": scenario_data["radius_lower"],
        "sampling_attempts": scenario_data["sampling_attempts"],
        "spd_shifts": scenario_data["spd_shifts"],
        "distance_definition": {
            "representation": "c_i=(theta_i,Gamma_i)",
            "theta_distance": "Euclidean norm",
            "gamma_distance": "AIRM: ||log(Gamma_i^{-1/2} Gamma_j Gamma_i^{-1/2})||_F",
            "combined_distance": "w_theta*d_theta/s_theta + w_gamma*d_gamma/s_gamma",
            "weight_theta": cfg.weight_theta,
            "weight_gamma": cfg.weight_gamma,
            "scales": scales,
            "important_note": "The same distance and scales are used for all five datasets.",
        },
        "generation_config": config_to_dict(cfg),
        "cluster_metrics": metrics,
    }
    record["validation"] = validate_record(record, cfg)
    return record


def generate_all(
    output_dir: Path,
    source_mat: Optional[Path],
    make_figures: bool,
    cfg: GenerationConfig,
) -> List[Dict[str, object]]:
    """Generate all five parameter datasets and write outputs."""
    if source_mat is not None:
        theta_pool, gamma_pool = load_set1_mat(source_mat, cfg)
        source_description = f"prototypes selected from {source_mat}"
    else:
        theta_pool, gamma_pool = make_example_pool(cfg)
        source_description = "reproducible example pool generated inside the package"

    scales = compute_reference_scales(theta_pool, gamma_pool)
    dataset_dir = output_dir / "datasets"
    figure_dir = output_dir / "figures"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if make_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    prototype_indices_by_K: Dict[int, List[int]] = {}
    prototypes_by_K: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    for K in (1, 2, 3):
        idx = select_prototypes(theta_pool, gamma_pool, K, scales, cfg)
        prototype_indices_by_K[K] = idx
        prototypes_by_K[K] = ([theta_pool[i] for i in idx], [gamma_pool[i] for i in idx])
        print(f"K={K} prototype indices: {idx}")

    scenario_specs = []

    # K=1 homogeneous.
    theta_p, gamma_p = prototypes_by_K[1]
    hom = generate_separated(theta_p, gamma_p, scales, cfg, seed=cfg.seed + 1)
    hom["cluster_metrics"] = cluster_quality_metrics(
        hom["theta_list"], hom["gamma_list"], hom["labels"], hom["theta_prototypes"], hom["gamma_prototypes"], scales, cfg
    )
    scenario_specs.append(("params_K1_homogeneous", 1, hom))

    # K=2 and K=3 separated/overlapping.
    for K in (2, 3):
        theta_p, gamma_p = prototypes_by_K[K]

        sep = generate_separated(theta_p, gamma_p, scales, cfg, seed=cfg.seed + 10 * K)
        sep["cluster_metrics"] = cluster_quality_metrics(
            sep["theta_list"], sep["gamma_list"], sep["labels"], sep["theta_prototypes"], sep["gamma_prototypes"], scales, cfg
        )
        scenario_specs.append((f"params_K{K}_separated", K, sep))

        ov = generate_overlapping(theta_p, gamma_p, scales, cfg, seed=cfg.seed + 100 * K)
        scenario_specs.append((f"params_K{K}_overlapping", K, ov))

    records: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for dataset_id, K, data in scenario_specs:
        metrics = data["cluster_metrics"]
        record = build_record(dataset_id, K, data, metrics, prototype_indices_by_K[K], source_description, scales, cfg)
        save_jsonl_record(dataset_dir / f"{dataset_id}.jsonl", record)
        if make_figures:
            make_plots(record, figure_dir, cfg)
        records.append(record)

        summary_rows.append({
            "dataset_id": dataset_id,
            "scenario": record["scenario"],
            "n_clusters": K,
            "cluster_sizes": "/".join(str(x) for x in record["cluster_sizes"]),
            "contraction_factor": record["contraction_factor"],
            "overlap_ratio": metrics["overlap_ratio"],
            "pairwise_overlap_ratios": json.dumps(metrics["pairwise_overlap_ratios"]),
            "full_pairwise_overlap": metrics["full_pairwise_overlap"],
            "overlap_graph_connected": metrics["overlap_graph_connected"],
            "silhouette_score": metrics["silhouette_score"],
            "davies_bouldin_index": metrics["davies_bouldin_index"],
            "minimum_gamma_eigenvalue": record["validation"]["minimum_gamma_eigenvalue"],
            "maximum_spd_shift": record["validation"]["maximum_spd_shift"],
        })

    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    with (output_dir / "generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable({
            "source_description": source_description,
            "config": config_to_dict(cfg),
            "distance_scales": scales,
            "prototype_indices_by_K": prototype_indices_by_K,
            "datasets": [r["dataset_id"] for r in records],
        }), f, indent=2)

    print(f"Saved datasets to: {dataset_dir}")
    print(f"Saved metrics to:  {output_dir / 'metrics_summary.csv'}")
    return records
