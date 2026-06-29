"""Configuration for clustered parameter generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GenerationConfig:
    """All tunable parameters used by the generator.

    The generated object is c_i=(theta_i, Gamma_i), where Gamma_i is SPD.
    Function families such as quadratic or quartic are intentionally not fixed
    in this generation layer.
    """

    # Dataset size and dimension.
    n_functions: int = 10
    dimension: int = 5

    # Distance weights in d_f.
    weight_theta: float = 0.5
    weight_gamma: float = 0.5

    # Noise used to sample local parameters around prototypes.
    theta_noise: float = 0.10
    gamma_noise: float = 0.10

    # SPD correction: every Gamma is shifted if needed so lambda_min >= eig_floor.
    eig_floor: float = 1.0
    jitter: float = 1e-10

    # Separated case: R_sep = separation_alpha * Delta_min / 2.
    # Any value below 1 guarantees separated prototype balls.
    separation_alpha: float = 0.50

    # Optional shell lower bound. It is not needed for mathematical separation;
    # it only prevents all samples from being too close to the prototype.
    radius_lower_fraction: float = 0.15

    # K=1 has no inter-prototype distance, so use a fixed radius in d_f units.
    one_cluster_radius: float = 0.25

    # Search grid for overlap by contraction. Larger values mean less contraction.
    # We choose the largest value satisfying the requested overlap condition.
    contraction_grid: Tuple[float, ...] = (
        0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60,
        0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20,
        0.15, 0.10, 0.08, 0.06, 0.04, 0.02,
    )

    # For K=3:
    # - full: every pair must satisfy O_kl > 1.
    # - connected: at least two pairwise overlaps connect all three clusters.
    overlap_mode: str = "full"

    # Randomness and rejection-sampling budget.
    seed: int = 20260629
    max_attempts_per_sample: int = 20_000
