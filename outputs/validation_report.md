# Validation report

This report validates the five clustered parameter datasets `c_i=(theta_i,Gamma_i)`.

| dataset | scenario | K | sizes | contraction | OR | SC | DB | pairwise OR |
|---|---|---:|---|---:|---:|---:|---:|---|
| params_K1_homogeneous | homogeneous | 1 | 10 | 1.0000 | N/A | N/A | N/A | `{}` |
| params_K2_overlapping | overlapping | 2 | 5/5 | 0.2000 | 1.1365 | 0.4849 | 0.8742 | `{"0-1": 1.1365483388085567}` |
| params_K2_separated | separated | 2 | 5/5 | 1.0000 | 0.2250 | 0.8685 | 0.1919 | `{"0-1": 0.22502730426374504}` |
| params_K3_overlapping | overlapping | 3 | 4/3/3 | 0.2000 | 1.6276 | 0.3558 | 1.2974 | `{"0-1": 1.1300010135998766, "0-2": 1.4679157053107559, "1-2": 1.6275990466981676}` |
| params_K3_separated | separated | 3 | 4/3/3 | 1.0000 | 0.3043 | 0.8522 | 0.2393 | `{"0-1": 0.2358181560567406, "0-2": 0.23998848973754314, "1-2": 0.3043459592373447}` |

Validation checks: all Gamma matrices are symmetric positive definite and satisfy the eigenvalue floor.