# Validation report

This report validates the five clustered parameter datasets `c_i=(theta_i,Gamma_i)`.

| dataset | scenario | K | sizes | contraction | OR | SC | DB | pairwise OR |
|---|---|---:|---|---:|---:|---:|---:|---|
| params_K1_homogeneous | homogeneous | 1 | 10 | 1.0000 | N/A | N/A | N/A | `{}` |
| params_K2_overlapping | overlapping | 2 | 5/5 | 0.0800 | 1.1745 | 0.4014 | 0.9006 | `{"0-1": 1.1745003971012118}` |
| params_K2_separated | separated | 2 | 5/5 | 1.0000 | 0.1318 | 0.9315 | 0.0973 | `{"0-1": 0.13181333815194438}` |
| params_K3_overlapping | overlapping | 3 | 4/3/3 | 0.0800 | 1.8100 | 0.3758 | 1.3985 | `{"0-1": 1.231117797600119, "0-2": 1.628764822315515, "1-2": 1.8099635909116791}` |
| params_K3_separated | separated | 3 | 4/3/3 | 1.0000 | 0.1302 | 0.9232 | 0.1137 | `{"0-1": 0.10865999356676559, "0-2": 0.13019045320919828, "1-2": 0.12548076088263266}` |

Validation checks: all Gamma matrices are symmetric positive definite and satisfy the eigenvalue floor.