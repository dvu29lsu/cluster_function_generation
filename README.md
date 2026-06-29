# Modular clustered parameter generation

This project generates only the five parameter datasets needed for MTGP-ADMM experiments:

```text
params_K1_homogeneous.jsonl
params_K2_separated.jsonl
params_K2_overlapping.jsonl
params_K3_separated.jsonl
params_K3_overlapping.jsonl
```

Each dataset contains

```text
(theta_i, Gamma_i, label_i),  Gamma_i positive definite.
```

No objective family is fixed here. Later, the ADMM experiment can instantiate the same parameters as quadratic, coordinate-quartic, or directional-quartic objectives.

## Why this structure

The cluster-generation layer only controls the local objective geometry:

\[
c_i=(\theta_i,\Gamma_i).
\]

The cluster quality is measured using

\[
d_f(c_i,c_j)
= w_\theta \frac{\|\theta_i-\theta_j\|_2}{s_\theta}
+ w_\Gamma \frac{d_{\mathrm{AIRM}}(\Gamma_i,\Gamma_j)}{s_\Gamma}.
\]

Therefore, generating separate files for quadratic/quartic/directional-quartic functions is unnecessary unless we later define a function-evaluation-based distance. For now, the same parameter datasets can be reused for all local objective families.

## Modular code layout

```text
cluster_params/
  config.py       # all parameters in one dataclass
  spd.py          # SPD correction utilities
  distances.py    # AIRM and d_f distances
  io_utils.py     # Set1.mat loading, JSONL, CSV helpers
  prototypes.py   # prototype selection and cluster sizes
  metrics.py      # OR, Silhouette, DB, validation
  generator.py    # separated/overlap generation logic
  plotting.py     # heatmap, MDS, silhouette figures

generate_datasets.py   # thin CLI entry point
validate_datasets.py   # validation/report script
validate_generation.ipynb
```

## Run with built-in example pool

```bash
python generate_datasets.py --output-dir outputs --make-figures
python validate_datasets.py --dataset-dir outputs/datasets --report outputs/validation_report.md
```

## Run with your `Set1.mat`

```bash
python generate_datasets.py --source-mat /path/to/Set1.mat --output-dir outputs --make-figures
python validate_datasets.py --dataset-dir outputs/datasets --report outputs/validation_report.md
```

The expected MATLAB file keys are `theta` and `Phi`.

## Main generation logic

### Separated datasets

For K=2 or K=3, let

\[
\Delta_{\min}=\min_{k<\ell}d_f(\bar c_k,\bar c_\ell).
\]

The separated radius is

\[
R_{\mathrm{sep}}=\alpha_{\mathrm{sep}}\frac{\Delta_{\min}}{2},
\qquad 0<\alpha_{\mathrm{sep}}<1.
\]

Every generated point must satisfy

\[
d_f(c_i,\bar c_k)\le R_{\mathrm{sep}}.
\]

This guarantees

\[
\frac{R_k+R_\ell}{\Delta_{k\ell}}<1
\]

for every cluster pair.

### Overlapping datasets

For overlap, the same base prototypes are moved toward their common center:

\[
\bar\theta_k(\rho_c)=\theta_c+\rho_c(\bar\theta_k-\theta_c),
\]

\[
\bar\Gamma_k(\rho_c)=\Gamma_c+\rho_c(\bar\Gamma_k-\Gamma_c).
\]

The code searches for the largest contraction factor \(\rho_c\) that satisfies the requested overlap condition. By default, `overlap_mode="full"`, meaning every pair must satisfy

\[
O_{k\ell}=\frac{R_k+R_\ell}{\Delta_{k\ell}}>1.
\]

For K=3, you can use a weaker connected-overlap condition with

```bash
python generate_datasets.py --overlap-mode connected
```

## Metrics

The validation reports:

- pairwise overlap ratios \(O_{k\ell}\);
- overall overlap ratio OR;
- Silhouette Score;
- prototype-based Davies-Bouldin index;
- SPD validation for every Gamma matrix.
