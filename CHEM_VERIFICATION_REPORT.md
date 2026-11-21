# CHEM v0.2.0 Verification Report

- **C Matrix-Free Adjoint Max Error (`K=4..16`, Mode A & B):** `1.20e-16` (`< 1e-14` PASS)
- **C Matrix-Free `T2` Adjoint Max Error (`Ks<=6`, Mode A & B):** `1.07e-16` (`< 1e-14` PASS)
- **Python/NumPy Independent Adjoint Max Error (DQG / T2):** `3.20e-16` / `7.98e-18` (`< 1e-14` PASS)
- **Mode A vs Mode B `H4 (Ry=2.0)` Hierarchy:** `E_LB(B) = -1.94111588 <= E_LB(A) = -1.93950079 <= E_FCI = -1.93943162` (`|A-B| = 1.615 mHa`, PASS)
- **Independent Certificate Audit (`28` `.cert.json` files):** max `|E_lb_py - E_lb_c| = 2.84e-14` Ha (PASS)
- **Process Peak RSS (up through `(16e,16o)` Mode A & B + `T2` + Observables):** `41.86 MB` (`< 50 MB` PASS)

## Acceptance Gate 2: Small & Mid-Small Active Spaces vs Exact Python/NumPy FCI
| Benchmark | $(N, K)$ | Mode | $E_{\text{LB}}^{\text{cert}}$ (Ha) | $E_{\text{FCI}}$ (Ha) | $E_{\text{HF}}$ (Ha) | $|\Delta E_{\text{FCI}}|$ | Gap $E_{\text{FCI}} - E_{\text{LB}}$ (mHa) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `h2_r1.4` | `(2e,2o)` | `A_SINGLET` | `-1.13727594` | `-1.13727594` | `-1.11671433` | `4.44e-16` | `0.0000` |
| `h2_r5.0` | `(2e,2o)` | `A_SINGLET` | `-0.93488935` | `-0.93488935` | `-0.68641593` | `2.22e-16` | `0.0000` |
| `h3_doublet` | `(3e,3o)` | `B_GENERAL` | `-1.41624901` | `-1.41624485` | `-1.33638734` | `2.22e-16` | `0.0042` |
| `h4_ry2.0` | `(4e,4o)` | `A_SINGLET` | `-1.93950079` | `-1.93943162` | `-1.77677036` | `2.44e-15` | `0.0692` |
| `h4_ry2.0_modeB` | `(4e,4o)` | `B_GENERAL` | `-1.94111588` | `-1.93943162` | `-1.77677036` | `2.44e-15` | `1.6843` |
| `h4_ry2.2` | `(4e,4o)` | `A_SINGLET` | `-1.97936713` | `-1.97770201` | `-1.86021052` | `2.22e-15` | `1.6651` |
| `h4_ry2.6` | `(4e,4o)` | `A_SINGLET` | `-2.06474585` | `-2.06395840` | `-1.96895873` | `4.44e-16` | `0.7875` |
| `h4_ry4.5` | `(4e,4o)` | `A_SINGLET` | `-2.17270654` | `-2.17233524` | `-2.09276362` | `1.33e-15` | `0.3713` |
| `h4_ry2.0_triplet` | `(4e,4o)` | `B_GENERAL` | `-1.92120223` | `-1.92120170` | `-1.87682599` | `4.44e-16` | `0.0005` |
| `h2o_eq` | `(10e,7o)` | `A_SINGLET` | `-75.01677505` | `-75.01242615` | `-74.96294062` | `0.00e+00` | `4.3489` |
| `h2o_stretched` | `(8e,6o)` | `A_SINGLET` | `-74.82259229` | `-74.81434599` | `-74.57691229` | `0.00e+00` | `8.2463` |
| `n2_eq_6e6o` | `(6e,6o)` | `A_SINGLET` | `-107.62611947` | `-107.62055050` | `-107.49523936` | `1.42e-14` | `5.5690` |
| `n2_stretched_6e6o` | `(6e,6o)` | `A_SINGLET` | `-107.44457905` | `-107.43369288` | `-106.99387943` | `4.26e-14` | `10.8862` |

## Acceptance Gate 3: Beyond-FCI & Spin-State Ordering Certification
| Benchmark | $(N, K)$ | Mode | $E_{\text{LB}}^{\text{cert}}$ (Ha) | $E_{\text{UB}}$ (Ha) | Width (mHa) | Iters | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `n2_stretched_10e8o` | `(10e,8o)` | `A_SINGLET` | `-107.46186292` | `-107.44782295` | `14.040` | `600` | `1.59` |
| `tm_spin_s0_8e8o` | `(8e,8o)` | `A_SINGLET` | `2.44323282` | `2.54914617` | `105.913` | `600` | `0.80` |
| `tm_spin_s2_8e8o` | `(8e,8o)` | `B_GENERAL` | `1.10890504` | `1.10971428` | `0.809` | `600` | `1.28` |
| `h12_chain_12e12o` | `(12e,12o)` | `A_SINGLET` | `-6.27904702` | `-6.17036243` | `108.685` | `500` | `2.41` |
| `h12_chain_triplet_12e12o` | `(12e,12o)` | `B_GENERAL` | `-6.21981104` | `-6.07595954` | `143.851` | `500` | `5.06` |
| `h16_chain_16e16o` | `(16e,16o)` | `A_SINGLET` | `-8.39855894` | `-8.14183326` | `256.726` | `400` | `6.32` |
| `h16_chain_triplet_16e16o` | `(16e,16o)` | `B_GENERAL` | `-8.33941573` | `-7.99366154` | `345.754` | `400` | `22.68` |

- **Certified Transition-Metal Spin Gap (`tm_spin` `S=2` vs `S=0`):** `E_UB(S=2) = 1.109714 < E_LB(S=0) = 2.443233` (certified separation `+1333.52 mHa`, high-spin quintet ground state rigorously proved!).

## Anytime Early-Veto Speedups (False-Low AI Candidate `E_ai = min(E_UB - 0.15, E_LB - 0.05) Ha`)
| Benchmark | Full Iters | Veto Stopped Iter | Speedup | Vetoed |
| :--- | :---: | :---: | :---: | :---: |
| `h4_ry2.0` | `450` | `15` | `30.0x` | `True` |
| `n2_stretched_6e6o` | `600` | `15` | `40.0x` | `True` |
| `h12_chain_12e12o` | `500` | `160` | `3.1x` | `True` |
| `h16_chain_16e16o` | `400` | `185` | `2.2x` | `True` |

## Acceptance Gate 5 (`v0.2.0`): `DQG+T2`, Certified Observables & Force Bounds, `.flowchem`, and Python SDK

### 5a. Matrix-Free `DQG` vs `DQG+T2` Subspace Lower-Bound Tightening
| Benchmark | $(K_s/K)$ | $E_{\text{LB}}^{\text{DQG}}$ (Ha) | $E_{\text{LB}}^{\text{DQG}+T_2}$ (Ha) | Gain (mHa) | $E_{\text{UB}}$ (Ha) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `h4_ry2.2` | `(4/4)` | `-1.97936713` | `-1.97805859` | `+1.309` | `-1.97770201` | `0.59` |
| `h2o_stretched` | `(4/6)` | `-74.82259229` | `-74.81860141` | `+3.991` | `-74.81434599` | `1.02` |
| `n2_stretched_6e6o` | `(6/6)` | `-107.44457905` | `-107.43654021` | `+8.039` | `-107.43369288` | `5.05` |
| `tm_spin_s0_8e8o` | `(4/8)` | `2.44323282` | `2.44794207` | `+4.709` | `2.54914617` | `2.18` |

### 5b. Certified Observable & Hellmann-Feynman Force Envelope Bounds (`O_LB <= <O>_FCI <= O_UB`)
| Observable | Mode | $O_{\text{LB}}^{\text{cert}}$ | $\langle O \rangle_{\text{FCI}}$ | $O_{\text{UB}}^{\text{cert}}$ | Width | $|O_{\text{var}} - O_{\text{FCI}}|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `h4_docc0` | `A_SINGLET` | `0.937941` | `0.944298` | `0.947627` | `0.00969` | `0.00e+00` |
| `h4_sdot01` | `A_SINGLET` | `-0.032307` | `-0.025240` | `-0.019703` | `0.01260` | `3.82e-17` |
| `h4_hf_force` | `A_SINGLET` | `-0.098947` | `-0.070123` | `-0.044939` | `0.05401` | `1.57e-14` |
| `h3_occ0` | `B_GENERAL` | `1.820283` | `1.831824` | `1.843423` | `0.02314` | `2.22e-16` |

### 5c. Compact Binary `.flowchem` (`FLCH0002`) vs JSON `.cert.json` Storage
| Certificate | `.cert.json` (B) | `.flowchem` (B) | Reduction |
| :--- | :---: | :---: | :---: |
| `h4_ry2.0` | `11036` | `3811` | `2.90x` |
| `n2_stretched_6e6o` | `56050` | `20412` | `2.75x` |
| `tm_spin_s0_8e8o` | `145334` | `50698` | `2.87x` |
| `h16_chain_triplet_16e16o` | `5696346` | `1975243` | `2.88x` |
| `h4_ry2.2.t2` | `108298` | `37539` | `2.88x` |
| `h2o_stretched.t2` | `147492` | `51336` | `2.87x` |
| `n2_stretched_6e6o.t2` | `1246314` | `428940` | `2.91x` |
| `tm_spin_s0_8e8o.t2` | `243946` | `84426` | `2.89x` |

### 5d. Live `ChemOracle` Python SDK (`oracles/chem_sdk.py`) Integration
- `sdk.bracket(h4_ry2.0, t2=True)`: `E_LB = -1.93957512 <= E_UB = -1.93943162` (PASS)
- `sdk.cert_verify(h4_ry2.0.flowchem)`: `build/certs/h4_ry2.0.flowchem -> BINARY CERT VERIFIED: sha=OK E_lb=-1.93950079 (diff=0.00e+00)` (PASS)
- `sdk.mlip_screen` false-low energy (`E_ai=-2.5`): `energy_vetoed=True` at iter `10` (PASS)
- `sdk.mlip_screen` bad Hellmann-Feynman force (`f_ai=0.5` vs `[-0.0989, -0.0449]`): `force_vetoed=True` (PASS)
- `sdk.mlip_screen` exact FCI Hellmann-Feynman force (`f_ai=-0.070123`): `vetoed=False` (PASS)

ALL CHEM v0.2.0 PHASE 1-5 ACCEPTANCE GATES PASSED WITH ZERO VIOLATIONS.
