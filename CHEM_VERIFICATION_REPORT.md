# CHEM v0.3.0 Verification Report

- **C Matrix-Free Adjoint Max Error (`K=4..16`, Mode A & B):** `1.20e-16` (`< 1e-14` PASS)
- **C Matrix-Free `T2` Adjoint Max Error (`Ks<=6`, Mode A & B):** `1.07e-16` (`< 1e-14` PASS)
- **Python/NumPy Independent Adjoint Max Error (DQG / T2):** `3.20e-16` / `7.98e-18` (`< 1e-14` PASS)
- **Mode A vs Mode B `H4 (Ry=2.0)` Hierarchy:** `E_LB(B) = -1.94111588 <= E_LB(A) = -1.93950090 <= E_FCI = -1.93943162` (`|A-B| = 1.615 mHa`, PASS)
- **Independent Certificate Audit (`28` `.cert.json` files):** max `|E_lb_py - E_lb_c| = 2.13e-14` Ha (PASS)
- **Process Peak RSS (up through `(16e,16o)` Mode A & B + `T2` + Observables):** `49.28 MB` (`< 64 MB` PASS)

## Acceptance Gate 2: Small & Mid-Small Active Spaces vs Exact Python/NumPy FCI
| Benchmark | $(N, K)$ | Mode | $E_{\text{LB}}^{\text{cert}}$ (Ha) | $E_{\text{FCI}}$ (Ha) | $E_{\text{HF}}$ (Ha) | $|\Delta E_{\text{FCI}}|$ | Gap $E_{\text{FCI}} - E_{\text{LB}}$ (mHa) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `h2_r1.4` | `(2e,2o)` | `A_SINGLET` | `-1.13727594` | `-1.13727594` | `-1.11671433` | `4.44e-16` | `0.0000` |
| `h2_r5.0` | `(2e,2o)` | `A_SINGLET` | `-0.93488935` | `-0.93488935` | `-0.68641593` | `2.22e-16` | `0.0000` |
| `h3_doublet` | `(3e,3o)` | `B_GENERAL` | `-1.41624487` | `-1.41624485` | `-1.33638734` | `2.22e-16` | `0.0000` |
| `h4_ry2.0` | `(4e,4o)` | `A_SINGLET` | `-1.93950090` | `-1.93943162` | `-1.77677036` | `2.44e-15` | `0.0693` |
| `h4_ry2.0_modeB` | `(4e,4o)` | `B_GENERAL` | `-1.94111588` | `-1.93943162` | `-1.77677036` | `2.44e-15` | `1.6843` |
| `h4_ry2.2` | `(4e,4o)` | `A_SINGLET` | `-1.97936701` | `-1.97770201` | `-1.86021052` | `2.22e-15` | `1.6650` |
| `h4_ry2.6` | `(4e,4o)` | `A_SINGLET` | `-2.06474575` | `-2.06395840` | `-1.96895873` | `4.44e-16` | `0.7873` |
| `h4_ry4.5` | `(4e,4o)` | `A_SINGLET` | `-2.17270654` | `-2.17233524` | `-2.09276362` | `1.33e-15` | `0.3713` |
| `h4_ry2.0_triplet` | `(4e,4o)` | `B_GENERAL` | `-1.92120575` | `-1.92120170` | `-1.87682599` | `4.44e-16` | `0.0040` |
| `h2o_eq` | `(10e,7o)` | `A_SINGLET` | `-75.01705812` | `-75.01242615` | `-74.96294062` | `0.00e+00` | `4.6320` |
| `h2o_stretched` | `(8e,6o)` | `A_SINGLET` | `-74.82259272` | `-74.81434599` | `-74.57691229` | `0.00e+00` | `8.2467` |
| `n2_eq_6e6o` | `(6e,6o)` | `A_SINGLET` | `-107.62611956` | `-107.62055050` | `-107.49523936` | `1.42e-14` | `5.5691` |
| `n2_stretched_6e6o` | `(6e,6o)` | `A_SINGLET` | `-107.44458120` | `-107.43369288` | `-106.99387943` | `4.26e-14` | `10.8883` |

## Acceptance Gate 3: Beyond-FCI & Spin-State Ordering Certification
| Benchmark | $(N, K)$ | Mode | $E_{\text{LB}}^{\text{cert}}$ (Ha) | $E_{\text{UB}}$ (Ha) | Width (mHa) | Iters | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `n2_stretched_10e8o` | `(10e,8o)` | `A_SINGLET` | `-107.46164378` | `-107.44782295` | `13.821` | `600` | `1.62` |
| `tm_spin_s0_8e8o` | `(8e,8o)` | `A_SINGLET` | `2.44359301` | `2.54914617` | `105.553` | `600` | `0.81` |
| `tm_spin_s2_8e8o` | `(8e,8o)` | `B_GENERAL` | `1.10887550` | `1.10971428` | `0.839` | `600` | `1.25` |
| `h12_chain_12e12o` | `(12e,12o)` | `A_SINGLET` | `-6.27900630` | `-6.17036243` | `108.644` | `500` | `2.44` |
| `h12_chain_triplet_12e12o` | `(12e,12o)` | `B_GENERAL` | `-6.21975751` | `-6.07595954` | `143.798` | `500` | `5.30` |
| `h16_chain_16e16o` | `(16e,16o)` | `A_SINGLET` | `-8.39822207` | `-8.14183326` | `256.389` | `400` | `6.27` |
| `h16_chain_triplet_16e16o` | `(16e,16o)` | `B_GENERAL` | `-8.33929155` | `-7.99366154` | `345.630` | `400` | `22.97` |

- **Certified Transition-Metal Spin Gap (`tm_spin` `S=2` vs `S=0`):** `E_UB(S=2) = 1.109714 < E_LB(S=0) = 2.443593` (certified separation `+1333.88 mHa`, high-spin quintet ground state rigorously proved!).

## Anytime Early-Veto Speedups (False-Low AI Candidate `E_ai = min(E_UB - 0.15, E_LB - 0.05) Ha`)
| Benchmark | Full Iters | Veto Stopped Iter | Speedup | Vetoed |
| :--- | :---: | :---: | :---: | :---: |
| `h4_ry2.0` | `450` | `15` | `30.0x` | `True` |
| `n2_stretched_6e6o` | `600` | `15` | `40.0x` | `True` |
| `h12_chain_12e12o` | `500` | `155` | `3.2x` | `True` |
| `h16_chain_16e16o` | `400` | `185` | `2.2x` | `True` |

## Acceptance Gate 5 (`v0.2.0`): `DQG+T2`, Certified Observables & Force Bounds, `.flowchem`, and Python SDK

### 5a. Matrix-Free `DQG` vs `DQG+T2` Subspace Lower-Bound Tightening
| Benchmark | $(K_s/K)$ | $E_{\text{LB}}^{\text{DQG}}$ (Ha) | $E_{\text{LB}}^{\text{DQG}+T_2}$ (Ha) | Gain (mHa) | $E_{\text{UB}}$ (Ha) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `h4_ry2.2` | `(4/4)` | `-1.97936701` | `-1.97805713` | `+1.310` | `-1.97770201` | `0.56` |
| `h2o_stretched` | `(4/6)` | `-74.82259272` | `-74.81860572` | `+3.987` | `-74.81434599` | `0.76` |
| `n2_stretched_6e6o` | `(6/6)` | `-107.44458120` | `-107.43658445` | `+7.997` | `-107.43369288` | `4.63` |
| `tm_spin_s0_8e8o` | `(4/8)` | `2.44359301` | `2.44794084` | `+4.348` | `2.54914617` | `1.70` |

### 5b. Certified Observable & Hellmann-Feynman Force Envelope Bounds (`O_LB <= <O>_FCI <= O_UB`)
| Observable | Mode | $O_{\text{LB}}^{\text{cert}}$ | $\langle O \rangle_{\text{FCI}}$ | $O_{\text{UB}}^{\text{cert}}$ | Width | $|O_{\text{var}} - O_{\text{FCI}}|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `h4_docc0` | `A_SINGLET` | `0.937941` | `0.944298` | `0.947583` | `0.00964` | `0.00e+00` |
| `h4_sdot01` | `A_SINGLET` | `-0.032307` | `-0.025240` | `-0.019620` | `0.01269` | `3.82e-17` |
| `h4_hf_force` | `A_SINGLET` | `-0.098946` | `-0.070123` | `-0.044673` | `0.05427` | `1.57e-14` |
| `h3_occ0` | `B_GENERAL` | `1.820174` | `1.831824` | `1.843449` | `0.02327` | `2.22e-16` |

### 5c. Compact Binary `.flowchem` (`FLCH0002`) vs JSON `.cert.json` Storage
| Certificate | `.cert.json` (B) | `.flowchem` (B) | Reduction |
| :--- | :---: | :---: | :---: |
| `h4_ry2.0` | `11011` | `3811` | `2.89x` |
| `n2_stretched_6e6o` | `56031` | `20412` | `2.75x` |
| `tm_spin_s0_8e8o` | `145344` | `50698` | `2.87x` |
| `h16_chain_triplet_16e16o` | `5696306` | `1975243` | `2.88x` |
| `h4_ry2.2.t2` | `108333` | `37539` | `2.89x` |
| `h2o_stretched.t2` | `147589` | `51336` | `2.87x` |
| `n2_stretched_6e6o.t2` | `1246511` | `428940` | `2.91x` |
| `tm_spin_s0_8e8o.t2` | `243941` | `84426` | `2.89x` |

### 5d. Live `ChemOracle` Python SDK (`oracles/chem_sdk.py`) Integration
- `sdk.bracket(h4_ry2.0, t2=True)`: `E_LB = -1.93957551 <= E_UB = -1.93943162` (PASS)
- `sdk.cert_verify(h4_ry2.0.flowchem)`: `build/certs/h4_ry2.0.flowchem -> BINARY CERT VERIFIED: sha=OK E_lb=-1.93950090 (diff=0.00e+00)` (PASS)
- `sdk.mlip_screen` false-low energy (`E_ai=-2.5`): `energy_vetoed=True` at iter `10` (PASS)
- `sdk.mlip_screen` bad Hellmann-Feynman force (`f_ai=0.5` vs `[-0.0989, -0.0447]`): `force_vetoed=True` (PASS)
- `sdk.mlip_screen` exact FCI Hellmann-Feynman force (`f_ai=-0.070123`): `vetoed=False` (PASS)

## Acceptance Gate 6 (`v0.3.0`): Zero-FP-Trust Lean Proofs, Hubbard Mott Phase Diagram & Generalized Pauli Constraints

### 6a. Zero-FP-Trust Per-Molecule Lean 4 + `leanchecker` Certificates (`0 sorryAx`, `0 Lean.ofReduceBool`)
| Lean Module | Molecule | Lines | Blocks | $E_{\text{LB}}^{\text{IntChol}}$ (Ha) | $|\Delta E_{\text{Weyl}}|$ (Ha) | Kernel Audit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `CertH2R14` | `h2_r1.4` | `84` | `6` | `-1.13727657` | `6.26e-07` | `leanchecker OK` |
| `CertH2R50` | `h2_r5.0` | `84` | `6` | `-0.93488997` | `6.15e-07` | `leanchecker OK` |
| `CertH4Ry20` | `h4_ry2.0` | `84` | `6` | `-1.93950297` | `2.07e-06` | `leanchecker OK` |

### 6b. Certified Hubbard Mott Phase Diagram & `DQG` vs `DQG+T2` Relaxation-Gap Atlas
| Model | $U/t$ | Gap `DQG` (mHa) | Gap `DQG+T2` (mHa) | $S_1$ | `docc0` Bracket ($\langle d_0\rangle_{\text{FCI}}$) | `sdot01` Bracket ($\langle \mathbf{S}_0\cdot\mathbf{S}_1\rangle_{\text{FCI}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `hubbard4_u0.5` | `0.5` | `1.260` | `0.029` | `0.187` | `[0.1994, 0.2404]` (`0.2200`) | `[-0.3522, -0.3229]` (`-0.3368`) |
| `hubbard4_u2.0` | `2.0` | `14.804` | `0.391` | `1.576` | `[0.0903, 0.1914]` (`0.1398`) | `[-0.4971, -0.3966]` (`-0.4462`) |
| `hubbard4_u4.0` | `4.0` | `48.654` | `0.738` | `3.283` | `[0.0267, 0.1307]` (`0.0728`) | `[-0.6163, -0.4420]` (`-0.5524`) |
| `hubbard4_u8.0` | `8.0` | `68.550` | `1.169` | `4.710` | `[0.0045, 0.0621]` (`0.0253`) | `[-0.7043, -0.4480]` (`-0.6359`) |
| `hubbard4_u12.0` | `12.0` | `59.250` | `0.613` | `5.138` | `[0.0015, 0.0338]` (`0.0121`) | `[-0.7317, -0.4427]` (`-0.6603`) |
| `hubbard4_frust_u6` | `6.0` | `0.526` | `0.765` | `4.476` | `[0.0509, 0.0624]` (`0.0580`) | `[-0.6092, -0.2488]` (`-0.4400`) |

- **Certified Mott Metal-to-Insulator Separation (`U=0.5` vs `U=12.0`):** `docc0` separated by `+0.1655`, `sdot01` separated by `+0.0905` (disjoint intervals, Theorem C13 PASS).

### 6c. Generalized Pauli Constraints (Borland–Dennis $\mathcal{B}_{3,6}$ & $\mathcal{B}_{2,4}$)
| System | Sector | Pairing Err | $\mathcal{B}_{3,6}$ Slack | Pinned | $S_1$ | Certified $\mathcal{B}_{3,6}$ Bracket |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `h3_doublet` | `(3e,3o)` | `2.22e-16` | `-4.16e-17` | `True` | `1.1540` | `[-0.000001, 1.042693]` |
| `h2_r1.4` | `(2e,2o)` | `0.00e+00` | `-` | `-` | `0.2724` | `-` |
| `h2_r5.0` | `(2e,2o)` | `0.00e+00` | `-` | `-` | `2.7300` | `-` |

### 6d. Exact Full-Space Variance, Weinstein Bound & Eckart Wavefunction Fidelity (`h4_ry2.0`, 4-det SHCI)
- Truncated 4-det energy `E_tr = -1.93943162 Ha` (`sigma_tr = 0.000000` Ha), Weinstein LB `= -1.93943162 <= E_FCI = -1.93943162` (PASS)
- Certified Eckart Ground-State Fidelity LB: `0.999519 <= fidelity_exact = 1.000000` (PASS)
- Live SDK `v0.3.0`: `sdk.exact_verify(h4_ry2.0.flowchem)` PASS (`D=100000000`), `sdk.gpc(h3_doublet, certify_bd=True)` PASS

ALL CHEM v0.3.0 PHASE 1-6 ACCEPTANCE GATES PASSED WITH ZERO VIOLATIONS.
