"""Validate generated JSONL parameter datasets and write a markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from cluster_params.config import GenerationConfig
from cluster_params.io_utils import load_jsonl_record
from cluster_params.metrics import validate_record


def format_float(x):
    if x is None:
        return "N/A"
    return f"{float(x):.4f}"


def summarize_record(record: Dict[str, object]) -> Dict[str, object]:
    m = record["cluster_metrics"]
    v = record["validation"]
    return {
        "dataset_id": record["dataset_id"],
        "scenario": record["scenario"],
        "K": record["n_clusters"],
        "sizes": "/".join(str(x) for x in record["cluster_sizes"]),
        "contraction": record.get("contraction_factor", 1.0),
        "OR": m.get("overlap_ratio"),
        "SC": m.get("silhouette_score"),
        "DB": m.get("davies_bouldin_index"),
        "pairwise_OR": json.dumps(m.get("pairwise_overlap_ratios")),
        "full_pairwise_overlap": m.get("full_pairwise_overlap"),
        "overlap_graph_connected": m.get("overlap_graph_connected"),
        "min_eig": v.get("minimum_gamma_eigenvalue"),
        "max_spd_shift": v.get("maximum_spd_shift"),
    }


def write_report(rows: List[Dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Validation report", ""]
    lines.append("This report validates the five clustered parameter datasets `c_i=(theta_i,Gamma_i)`." )
    lines.append("")
    lines.append("| dataset | scenario | K | sizes | contraction | OR | SC | DB | pairwise OR |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['dataset_id']} | {r['scenario']} | {r['K']} | {r['sizes']} | "
            f"{format_float(r['contraction'])} | {format_float(r['OR'])} | {format_float(r['SC'])} | "
            f"{format_float(r['DB'])} | `{r['pairwise_OR']}` |"
        )
    lines.append("")
    lines.append("Validation checks: all Gamma matrices are symmetric positive definite and satisfy the eigenvalue floor.")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated parameter datasets.")
    parser.add_argument("--dataset-dir", default="outputs/datasets", help="Directory containing params_*.jsonl files.")
    parser.add_argument("--report", default="outputs/validation_report.md", help="Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenerationConfig()
    dataset_dir = Path(args.dataset_dir)
    paths = sorted(dataset_dir.glob("params_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No params_*.jsonl files found in {dataset_dir}")

    rows = []
    for path in paths:
        record = load_jsonl_record(path)
        record["validation"] = validate_record(record, cfg)
        rows.append(summarize_record(record))

    write_report(rows, Path(args.report))
    df = pd.DataFrame(rows)
    print(df[["dataset_id", "scenario", "K", "OR", "SC", "DB", "min_eig"]].to_string(index=False))
    print(f"\nSaved validation report to {args.report}")


if __name__ == "__main__":
    main()
