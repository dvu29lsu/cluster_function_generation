"""Command-line entry point for generating the five parameter datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from cluster_params.config import GenerationConfig
from cluster_params.generator import generate_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clustered (theta, Gamma) parameter datasets.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--source-mat", default=None, help="Optional Set1.mat path. If omitted, use example pool.")
    parser.add_argument("--overlap-mode", default="full", choices=["full", "connected"], help="K=3 overlap condition.")
    parser.add_argument("--make-figures", action="store_true", help="Save validation figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenerationConfig(overlap_mode=args.overlap_mode)
    source_mat = Path(args.source_mat) if args.source_mat else None
    generate_all(Path(args.output_dir), source_mat, args.make_figures, cfg)


if __name__ == "__main__":
    main()
