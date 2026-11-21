# CHEM Frontier 1 & Frontier 2 Scientific Discovery & Optimization Report

**Repository**: `/Users/kuangtalin/Documents/CHEM`  
**Verification Gate Status**: `make clean && make verify` exited `0` (0 C compiler warnings across `build/chem_cli` and `build/libchem.dylib`, Peak RSS `45.16 MB < 50.0 MB`, `selftest` PASS, 28 certs audited, `leanchecker ChemCert` + `leanchecker CertDiscoveredT1` PASS with `0 sorryAx`, `0 Lean.ofReduceBool`).

---

## 1. Solver Core & Pipeline Optimization Benchmarks

To eliminate computational bottlenecks during dense topology sweeps and multi-seed adversarial Hamiltonian searches, four architectural optimizations were implemented cleanly in [chem_rdm2_mf.c](file:///Users/kuangtalin/Documents/CHEM/src/chem_rdm2_mf.c), [chem_rdm2.h](file:///Users/kuangtalin/Documents/CHEM/include/chem_rdm2.h), [Makefile](file:///Users/kuangtalin/Documents/CHEM/Makefile), and [chem_sdk.py](file:///Users/kuangtalin/Documents/CHEM/oracles/chem_sdk.py):

1. **Normal-Equation Elimination (`U_eq` + `H_chol`)**:
   - Stores `U_eq = H^{-1} A_eq^*` (`[nD × meq]`, memory-capped at `nD × meq ≤ 262144` ≈ 2 MB) to replace the first `pcg_solve_H` call per BPSDP iteration with one exact `cblas_dgemv`.
   - For `nD ≤ 512` (covering all `K ≤ 6` Mode A and `K ≤ 5` Mode B problems within `< 2 MB`), factors `H` once via LAPACK `dpotrf_` and solves `v_D` via `dpotrs_`, eliminating **both** PCG solves per iteration (zero `apply_H_op` calls in the loop), while large active spaces (`K = 12, 16`) remain strictly matrix-free (`Peak RSS = 45.16 MB < 50.0 MB`).
2. **Upper-Triangle Symmetric Packing (`svec_pack_upper` + `reconstruct_sym_upper_from_eigs`)**:
   - Avoids mirroring the lower triangle after `cblas_dsyrk(CblasUpper)` in `split_one_block`.
3. **Persistent Worker Thread Pool (`ChemWorkerPool`)**:
   - Eliminates per-iteration `pthread_create` / `pthread_join` overhead in both `spectral_split_all` and `certify_with_ctx`, plus adjusts default `check_every` from `25` to `50` when early-veto is disabled.
4. **Persistent In-Process C API (`ChemSession` / `build/libchem.dylib`) + `ChemFastSession` (`ctypes`) + 4-Core `ProcessPoolExecutor`**:
   - Reuses factorized `H_chol`, `U_eq`, `S_eq`, worker pool, and warm-started `(X, Z, sigma)` state across optimization trajectories without subprocess startup overhead.

| Benchmark Case (800 iters, `tol=1e-6`, `/tmp/prof_k4.fcidump` & `/tmp/prof_k5.fcidump`) | Before (s) | After (s) | Speedup | Observed `E_lb_cert` (Ha) |
| :--- | :---: | :---: | :---: | :---: |
| `K=4` (`4e4o`, Mode A) `DQG` (`threads=1`) | `0.090` | `0.057` | `1.58×` | `-13.669955` |
| `K=4` (`4e4o`, Mode A) `DQG+T1` (`threads=1`) | `0.152` | `0.086` | `1.77×` | `-13.632738` |
| `K=4` (`4e4o`, Mode A) `DQG+T2` (`threads=1`) | `0.662` | `0.433` | `1.53×` | `-13.626989` |
| `K=4` (`4e4o`, Mode A) `DQG+T1+T2` (`threads=1`) | `0.863` | `0.460` | **`1.88×`** | `-13.627708` |
| `K=5` (`5e5o`, Mode B) `DQG` (`threads=1`) | `0.590` | `0.387` | `1.52×` | `-14.334153` |
| `K=5` (`5e5o`, Mode B) `DQG+T1` (`threads=1`) | `1.281` | `0.759` | `1.69×` | `-13.873790` |
| `K=5` (`5e5o`, Mode B) `DQG+T2` (`threads=1`) | `5.299` | `3.544` | `1.50×` | `-13.848765` |
| `K=5` (`5e5o`, Mode B) `DQG+T1+T2` (`threads=1`) | `6.326` | `4.040` | `1.57×` | `-13.843490` |
| `K=5` (`5e5o`, Mode B) `DQG+T1+T2` (`threads=4`) | `4.092` | `2.154` | **`1.90×`** | `-13.843490` |

- **In-Process Session Equivalence (`/tmp/prof_k5_same.fcidump`)**: `|E_lb(ChemFastSession) - E_lb(chem_cli)| = 5.684e-14 Ha` (`cold_dt = 4.204s`), and primal Hellmann–Feynman gradient identity `|E_primal - <H, G>| = 3.55e-15`.
- **End-to-End Frontier 2 Throughput**: Full 5-hunt convergence-proof search + 3500-iter precision audit + 20 `exact-verify` audits + Lean `CertDiscoveredT1` verification completed end-to-end in **`219.94 s`** (`168.09 s` search phase).

---

## 2. Frontier 1: Graph-Topology Relaxation Law & Energy–Observable–Gap Decoupling

Script: [exp1_topology_decoupling.py](file:///Users/kuangtalin/Documents/CHEM/experiments/exp1_topology_decoupling.py)  
Raw Data: [exp1_results.json](file:///Users/kuangtalin/Documents/CHEM/experiments/exp1_results.json)  
Formal Proofs: [ChemCert.lean](file:///Users/kuangtalin/Documents/CHEM/lean/ChemCert.lean) (`C15a`, `C15b`, `C16`)

### 2.1 Experiment 1A: Exact Graph-Topology Survey Across $K = 4, 5, 6$

All relaxation errors $\varepsilon_M = \max(0, E_0^{\text{FCI}} - E_{\text{lb},M}^{\text{cert}})$ are reported in Hartree (Hubbard $t = 1$). Zero lower-bound violations occurred across all 44 topology runs.

#### Table 1A-1: $K = 4$ Connected Graphs (`4e4o`) Selected Highlights Across $U \in \{2, 4, 6, 8\}$

| Graph | $2S$ | Tris | Bipartite | $U$ | $S_1^{\text{FCI}}$ | $\varepsilon_{\text{DQG}}$ | $\varepsilon_{\text{DQG+T1}}$ | $\varepsilon_{\text{DQG+T2}}$ | $\varepsilon_{\text{DQG+T1+T2}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P4` | 0 | 0 | True | 4 | `3.283` | `0.04865` | `0.00231` | `0.00050` | `0.00038` |
| `C4` | 0 | 0 | True | 6 | `4.489` | `0.00034` | `0.00040` | `0.00051` | `0.00028` |
| `K13_star_S0` | 0 | 0 | True | 4 | `4.228` | **`0.12009`** | `0.00953` | `0.00953` | `0.00953` |
| `K13_star_S1` | 2 | 0 | True | 4 | `1.437` | **`0.00000`** | `0.00000` | `0.00001` | `0.00000` |
| `Paw` | 0 | 1 | False | 6 | `4.102` | `0.05075` | `0.00150` | `0.00053` | `0.00047` |
| `Diamond` | 0 | 2 | False | 4 | `3.982` | `0.02641` | `0.00090` | `0.00045` | `0.00034` |
| `K4_tetra` | 0 | 4 | False | 4 | `4.024` | `0.00168` | `0.00194` | `0.00039` | `0.00042` |

- **Spin-Sector Exactness Collapse on Bipartite Star (`K13_star`)**: On identical $K=4$ star geometry at $U=4$, the frustrated excited/unfavored $S=0$ sector exhibits the **largest $K=4$ `DQG` error (`0.12009 Ha`)**, whereas switching to the Lieb ferrimagnetic $S=1$ ground sector collapses `DQG` error across all $U \in \{2,4,6,8\}$ to **`0.00000–0.00001 Ha`** ($>10^4\times$ reduction).

#### Table 1A-2: $K = 5$ Doublet (`5e5o`, $2S=1$) & $K = 6$ Singlet (`6e6o`, $2S=0$, $U=4$) Surveys

| System / Graph | Tris | Bipartite | $U$ | $S_1^{\text{FCI}}$ | $\varepsilon_{\text{DQG}}$ | $\varepsilon_{\text{DQG+T1}}$ | $\varepsilon_{\text{DQG+T2}}$ | $\varepsilon_{\text{DQG+T1+T2}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $K=5$ `P5` | 0 | True | 4 | `3.380` | `0.05776` | `0.00595` | `0.00364` | `0.00301` |
| $K=5$ `C5` | 0 | False | 4 | `3.318` | `0.20144` | `0.04811` | `0.03380` | `0.03098` |
| $K=5$ `K14_star` | 0 | True | 6 | `5.124` | `0.18248` | `0.06165` | `0.01960` | `0.00862` |
| $K=5$ `W5_wheel` | 4 | False | 4 | `4.027` | `0.22285` | `0.05746` | `0.06815` | `0.04909` |
| $K=5$ `W5_wheel` | 4 | False | 6 | `4.477` | `0.15639` | **`0.02041`** | **`0.04755`** | `0.03922` |
| $K=5$ `K5_complete` | 10 | False | 4 | `4.331` | `0.03405` | `0.03630` | `0.02945` | `0.03265` |
| $K=6$ `P6` | 0 | True | 4 | `4.899` | `0.14266` | `0.02533` | `0.01162` | `0.01121` |
| $K=6$ `C6` | 0 | True | 4 | `4.194` | `0.20969` | `0.05976` | `0.02379` | `0.02391` |
| $K=6$ `Ladder_2x3` | 0 | True | 4 | `5.061` | `0.18254` | `0.06435` | `0.03158` | `0.03080` |
| $K=6$ `TriLadder` | 4 | False | 4 | `4.925` | `0.43371` | `0.14105` | `0.09215` | `0.07709` |
| $K=6$ `Prism_6` | 2 | False | 4 | `5.990` | `0.56975` | `0.21086` | `0.13540` | `0.13066` |
| $K=6$ `Octahedron_6` | 8 | False | 4 | `5.341` | `0.47059` | `0.22002` | `0.16844` | **`0.08673`** |

**Empirical Graph-Topology Laws Observed**:
1. **Finite-Size vs. Bulk Bipartite Crossover**: At $K=4$, high-symmetry cliques (`K4_tetra`) have smaller mean `DQG` error than bipartite chains (`non-bip mean = 0.01811` vs `bip = 0.03637`), but by **$K=6$**, bipartite vs. frustrated separation becomes stark: non-bipartite frustrated graphs (`TriLadder`, `Prism_6`, `Octahedron_6`) average **`0.49135 Ha` (`2.76×` larger `DQG` error)** and **`0.09816 Ha` (`4.47×` larger `DQG+T1+T2` error)** compared to bipartite graphs (`0.17830 Ha` `DQG`, `0.02197 Ha` `DQG+T1+T2`). Even odd-cycle frustration without triangles (`C5` vs `P5` at $U=4$) multiplies `DQG` error by `3.49×` and `DQG+T1+T2` error by `10.29×`.
2. **Genuine Converged `T1 > T2` on Frustrated Wheel `W5_wheel U=6`**: High-precision `2000-iter` (`tol=1e-7`) verification on unit-normalized `W5_wheel U=6` (`norm = 14.00`) confirms that `DQG+T1` strictly beats `DQG+T2` by **`+0.001471 Ha/unit-Frob`** (`+20.59 mHa` at physical scale: `FCI=-0.158470`, `T1=-0.159557`, `T2=-0.161028`). By contrast, on `W5_wheel U=4` the 1000-iter `T1 > T2` lead (`0.05746` vs `0.06815`) was a transient `T2` convergence effect that reversed to `-0.000579` at `2500` iters.
3. **Super-Additive `T1 + T2` Synergy on `Octahedron_6`**: Adding both `T1` and `T2` drops residual error to `0.08673 Ha` (reducing `DQG` error by `81.6%` and halving the better of `T1`-alone `0.22002` or `T2`-alone `0.16844`).

### 2.2 Experiment 1B: Energy–Observable–Gap Decoupling & Theorem C15 Scaling

Across all **40** sweep configurations (Hubbard clean ring, Hubbard `pert=0.03` avoided crossing, STO-3G $\text{H}_4$ $R_y \in [1.60, 2.60]$ Bohr), certified observable containment held 100% (`O_lb <= O_fci <= O_ub`) with **`0` Theorem C15 violations**.

- **Hubbard 4-Site Clean Ring (`t' in [-2, +2], U=6`, observable `sdot:0:1`)**:
  At the exact symmetry crossings ($t' = \pm 1.00$, $\Delta_{\text{gap}} = 0$), energy relaxation slack remains small (`dE_relax_t12 = 0.00040 Ha`), yet certified observable width `W_t12` explodes from `0.02991` (at $t'=0$) to **`0.86550`** — an amplification ratio **`W_t12 / dE_relax_t12 = 2142.8×`**.
- **Hubbard 4-Site Avoided Crossing (`pert=0.03`, 11 points `t' in [0.60, 1.40]`, `sdot:0:1`)**:
  As the spectral gap narrows from `0.79591` ($t'=1.40$, `W_t12 = 0.04998`, `73.0×`) down to the avoided-crossing minimum `0.03974` ($t'=1.04$, `m01 = 0.4247`, `W_t12 = 0.21641`, `332.2×`), the normalized observable decoupling factor $W_{T12} / (m_{01}\sqrt{\delta E_{\text{relax}}})$ follows a clean power law against $\Delta_{\text{gap}}$:
  $$\log\left(\frac{W_{T12}}{m_{01}\sqrt{\delta E_{\text{relax}}}}\right) \propto -0.3932 \log \Delta_{\text{gap}} \quad (R^2 = 0.8664,\ n = 11)$$
  while raw $W_{T12} \propto \Delta_{\text{gap}}^{-0.3515}$ ($R^2 = 0.7008$).
- **STO-3G $\text{H}_4$ Square Avoided Crossing (`Rx=2.0`, $R_y \in [1.60, 2.60]$ Bohr, observable `occ:1`)**:
  At $R_y = 2.00$ Bohr ($D_{4h}$ square), the spectral gap reaches its minimum `0.15170 Ha` and off-diagonal coupling surges to `m01 = 0.9209` (vs. `m01 = 0.0000` at $R_y = 1.60, 2.60$). Even though symmetry tightens the energy relaxation slack by $4\times$ to `dE_t12 = 0.000115 Ha`, the certified observable width remains huge (`W_t12 = 0.31611`), producing a peak amplification ratio **`W_t12 / dE_t12 = 2738.9×`** (`W_dqg = 0.22107`, `W_t12` peaks at `0.34990` at $R_y=2.10$).

### 2.3 Formalized Lean 4 Foundations (Zero `sorryAx`, Zero `Lean.ofReduceBool`)

Verified directly via `lean -DwarningAsError=true` and `leanchecker ChemCert` in [ChemCert.lean](file:///Users/kuangtalin/Documents/CHEM/lean/ChemCert.lean):
- **Theorem C15a (`ChemCert.observable_gap_decoupling_feasibility`)**: Proves via 2×2 integer PSD density matrices $N \rho_\pm = \begin{pmatrix} f & \pm c \\ \pm c & e \end{pmatrix}$ that energy feasibility $e (E_1 - E_0) \le (f+e)(E_{\text{ub}} - E_0)$ implies exact feasibility $f E_0 + e E_1 \le (f+e) E_{\text{ub}}$ and enforces $4 c m \le (f+e)(O_{\text{ub}} - O_{\text{lb}})$. Axioms: `[propext, Classical.choice, Quot.sound]`.
- **Theorem C15b (`ChemCert.certified_observable_gap_lower_bound`)**: Proves from $c^2 \le f e$ and $e \Delta \le (f+e)\delta E$ both the $\Delta^{-1/2}$ coherence scaling $c^2 \Delta \le (f+e)^2 \delta E$ and quadratic observable width floor $16 c^2 m^2 \le ((f+e) W)^2$. Axioms: `[propext, Classical.choice, Quot.sound]`.
- **Theorem C16 (`ChemCert.subspace_t1_trace_upper_bound`)**: Proves for composite `T1 = D3 + Q3` in any projected subspace `dim_sub` that `min dim_sub (tr_d3_full + tr_q3_full) * lam_cert ≤ (tr_d3_sub + tr_q3_sub) * lam_cert` when `lam_cert ≤ 0`. Axioms: `[propext, Quot.sound]`.

---

## 3. Frontier 2: Exact Matrix-Free `T1 = D3 + Q3` & Adversarial Extremal Hamiltonians

### 3.1 Independent Fock-Space Oracle & Matrix-Free Adjoint Verification

Implemented in [fci_exact_oracle.py](file:///Users/kuangtalin/Documents/CHEM/experiments/fci_exact_oracle.py) and tested directly against `a^+_p a^+_q a^+_r a_w a_v a_u + a_p a_q a_r a^+_w a^+_v a^+_u` in full Slater-determinant Fock space:
- **Singlet `(4e, 4o)` (`seed=11`)**: `max |T1_fock - T1_map| = 2.398e-14`, `min_eig(T1) = -1.345e-15`.
- **Triplet `(4e, 4o)` (`seed=23`)**: `max |T1_fock - T1_map| = 2.753e-14`, `min_eig(T1) = -1.492e-15`.
- **Doublet `(5e, 5o)` (`seed=31`)**: `max |T1_fock - T1_map| = 5.329e-14`, `min_eig(T1) = 1.009e-15`.
- **C Matrix-Free Solver Verification (`chem_cli selftest`)**: `T1` adjoint error `3.593e-17` (`T2 = 5.930e-17`), H4 exact FCI `T1` PSD error `6.02e-16` (`triplet -5.26e-16`), and cross-check vs `fci_exact_oracle` on random `(4e,4o)` (`E_ci` `5.329e-15`, `occ:0` `2.2e-15`, `docc:0` `8.5e-16`, `sdot:0:1` `8e-17`).

### 3.2 Audited Adversarial Extremal Hamiltonians (`3500 iters`, `tol=1e-7`, Unit Frobenius Norm)

Script: [exp2_adversarial_hunt.py](file:///Users/kuangtalin/Documents/CHEM/experiments/exp2_adversarial_hunt.py)  
Raw Data: [exp2_results.json](file:///Users/kuangtalin/Documents/CHEM/experiments/exp2_results.json)

All 20 audited certificates (`chem-cert-v2` and `chem-cert-v3` with `T1`) passed **`chem_cli exact-verify`** (`all_blocks_sdd = True`, `E_lb_int_chol <= E0_fci`), confirming that every discovered separation holds unconditionally under exact integer Cholesky + SDD arithmetic.

| Discovered Hamiltonian (`||H||_F = 1`) | $E_0^{\text{FCI}}$ | `DQG` ($E_{\text{lb}}$ / IntChol) | `DQG+T1` ($E_{\text{lb}}$ / IntChol) | `DQG+T2` ($E_{\text{lb}}$ / IntChol) | `DQG+T1+T2` ($E_{\text{lb}}$ / IntChol) | Key Discovery Margin |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| [discovered_hunt1a_k5_t1_beats_t2.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_hunt1a_k5_t1_beats_t2.fcidump) (`K=5, 2S=1`) | `-0.219521` | `-0.241579` / `-0.241585` | **`-0.225722`** / `-0.225731` | `-0.226132` / `-0.226164` | `-0.223382` / `-0.223417` | `T1 > T2`: **`+0.410 mHa`** Weyl (`+0.433 mHa` IntChol) |
| [discovered_hunt1b_k4_t1_beats_t2.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_hunt1b_k4_t1_beats_t2.fcidump) (`K=4, 2S=0`) | `-2.058504` | `-2.064108` / `-2.064110` | `-2.058546` / `-2.058548` | `-2.058516` / `-2.058525` | `-2.058516` / `-2.058525` | `T1 - T2 = -0.029 mHa` ($\binom{n_\alpha}{3}=\binom{K-n_\alpha}{3}=0$ disables `T1_aaa`) |
| [discovered_hunt2_k5_strict4tier.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_hunt2_k5_strict4tier.fcidump) (`K=5, 2S=1`) | `-1.023032` | `-1.285323` / `-1.285329` | **`-1.072712`** / `-1.072721` | `-1.089006` / `-1.089044` | **`-1.062510`** / `-1.062549` | **Strict 5-Tier Total Ordering** `DQG < T2 < T1 < T1+T2 < FCI`: `T1 - T2 = +16.29 mHa`, `(T2-DQG, T12-T2, FCI-T12) = (196.32, 26.50, 39.48)` mHa |
| [discovered_hunt3a_k4_adversary.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_hunt3a_k4_adversary.fcidump) (`K=4, 2S=0`) | `-0.995302` | `-1.207147` / `-1.207149` | `-1.093763` / `-1.093766` | `-1.090568` / `-1.090577` | **`-1.090568`** / `-1.090577` | Extremal $K=4$ `FCI - T12`: **`95.265 mHa`** (`95.275 mHa` IntChol, **45.0% of DQG error survives**) |
| [discovered_hunt3b_k5_adversary.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_hunt3b_k5_adversary.fcidump) (`K=5, 2S=1`) | `-1.077124` | `-1.379283` / `-1.379289` | `-1.142965` / `-1.142974` | `-1.127212` / `-1.127248` | **`-1.119471`** / `-1.119509` | Extremal $K=5$ `FCI - T12`: **`42.347 mHa`** (`42.385 mHa` IntChol) |

### 3.3 Structural Anatomy of Extremal Adversaries & Combinatorial `T1` Insight

1. **Why `DQG+T1` Dominates on Frustrated Mott & Pair-Hopping `K=5` (`Hunt 1A` vs. `Hunt 2`)**:
   - In `Hunt 1A`, the optimizer converged to a pure on-site Hubbard frustrated cluster (`eri_onsite_frac = 98.21%`, `h1_frac = 9.71%`, `gap_fci = 0.000000`).
   - Remarkably, **`Hunt 2` (`K=5, 2S=1`) simultaneously discovered a `+16.29 mHa/unit-Frob` (`+16.32 mHa` IntChol) `T1 > T2` victory** alongside strict 4-tier gaps `(196.32, 26.50, 39.48)` mHa (`min_adjacent_gap >= 10.20 mHa` across `DQG < DQG+T2 < DQG+T1 < DQG+T1+T2 < FCI`). In `Hunt 2`, `93.72%` of $\|H\|_F^2$ sits in ERI with `79.18%` off-diagonal pair-hopping and strongly polarized natural occupations `[0.9644, 0.9581, 0.9086, 0.1265, 0.0424]` — activating pure same-spin `T1_aaa = D3_aaa + Q3_aaa` ($\binom{3}{3}=1$, omitted by `T2`).
   - By contrast, at half-filled `(4e, 4o)` singlet (`Hunt 1B`), $\binom{n_\alpha}{3} = \binom{K - n_\alpha}{3} = 0$ leaves `T1_aaa` with zero particles/holes, explaining cleanly why strong `T1 > T2` separation emerges once $K \ge 5, n_\alpha \ge 3$ or high-spin/doped sectors open same-spin 3-body correlations.
2. **Extremal `DQG+T1+T2` Adversary (`Hunt 3A`, `K=4, 4e4o`, `95.27 mHa/unit-Frob` Gap)**:
   - Near-maximal orbital entanglement $S_1 = \mathbf{5.3876}$ (close to the 4-orbital cap $8\ln 2 = 5.545$), nearly flat fractional natural occupations **`[0.6012, 0.5965, 0.4053, 0.3970]`**, `92.96%` two-body norm share, and `80.84%` off-diagonal ERI pair-hopping/3- & 4-index couplings. This geometry saturates 4-positivity ($D_4, Q_4$) constraints invisible to every 2- and 3-RDM positivity block.
3. **Standalone Lean 4 Kernel Proof of Discovered `T1` Certificate**:
   - Emitted [CertDiscoveredT1.lean](file:///Users/kuangtalin/Documents/CHEM/experiments/CertDiscoveredT1.lean) (`hunt3a_k4_adversary` `DQG+T1`, 8 PSD blocks, `scale_D = 1000000`) and verified with `lean -DwarningAsError=true` + `leanchecker CertDiscoveredT1`:
     - `CertDiscoveredT1.molecule_bracket_valid`: **0 axioms** (`Prop`).
     - `CertDiscoveredT1.block_0_psd`..`block_7_psd` and `CertDiscoveredT1.molecule_lower_bound_sound`: `[propext, Classical.choice, Quot.sound]` (`0 sorryAx`, `0 Lean.ofReduceBool`).

---

## Section 4: Direction C Practical Tooling — Entanglement `--auto-no` Concentration, Haar O(K) Invariance & `.xyz` MLIP Screener

To turn Frontier 1's natural-orbital entanglement insight into a standalone workflow tool, we implemented orthogonal Hamiltonian/observable rotation (`chem_ham_rotate_orthogonal`, `chem_obs_rotate_orthogonal`), pilot-`DQG` natural-orbital canonicalization (`chem_ham_build_dqg_no`, `--auto-no`, `--save-no-fcidump`, `chem_fcidump_write`), and the multi-core `.xyz` reaction-path and MLIP hallucination screener ([chem_screener.py](file:///Users/kuangtalin/Documents/CHEM/oracles/chem_screener.py)). Benchmark script: [exp_tool_auto_no_and_screener.py](file:///Users/kuangtalin/Documents/CHEM/experiments/exp_tool_auto_no_and_screener.py); raw JSON: [exp_tool_results.json](file:///Users/kuangtalin/Documents/CHEM/experiments/exp_tool_results.json).

### 4.1 Entanglement-Driven `--auto-no` Subspace Concentration

| System | Config | `DQG` `E_lb` (Ha) | Naive Subspace `E_lb` (Ha) | `--auto-no` `E_lb` (Ha) | `--auto-no` `ΔLB` vs Naive (mHa) | Bracket Width Naive -> `--auto-no` (mHa) | Convergence (`it / conv`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `n2_stretched_10e8o` (`K=8`) | `DQG+T2(5)` | -107.46078354 | -107.45963514 | -107.46000764 | -0.373 | 11.812 -> 12.185 | `1600 / 0` |
| `n2_stretched_10e8o` (`K=8`) | `DQG+T1+T2(5)` | -107.46078354 | -107.45999214 | -107.46032008 | -0.328 | 12.169 -> 12.497 | `1600 / 0` |
| `n2_stretched_10e8o` (`K=8`) | `DQG+T2(6)` | -107.46078354 | -107.45420433 | -107.45370879 | **+0.496** | 6.381 -> **5.886** | `1600 / 0` |
| `tm_spin_s0_8e8o` (`K=8`) | `DQG+T2(5)` | 2.44708588 | 2.44646428 | 2.44768132 | **+1.217** | 101.380 -> **59.802** (`-40.361` UB) | `1200 / 0` |
| `tm_spin_s0_8e8o` (`K=8`) | `DQG+T1+T2(5)` | 2.44708588 | 2.44560492 | 2.44743030 | **+1.825** | 102.239 -> **60.053** (`-40.361` UB) | `1200 / 0` |
| `tm_spin_s0_8e8o` (`K=8`) | `DQG+T2(6)` | 2.44708588 | 2.45102441 | 2.45202150 | **+0.997** | 96.820 -> **55.462** (`-40.361` UB) | `1200 / 0` |
| `h2o_stretched` (`K=6`) | `DQG+T2(4)` | -74.82239194 | -74.81764419 | -74.81680462 | **+0.840** | 3.298 -> **2.459** | `1200 / 0` |
| `h2o_stretched` (`K=6`) | `DQG+T1+T2(4)` | -74.82239194 | -74.81766499 | -74.81687581 | **+0.789** | 3.319 -> **2.530** | `1200 / 0` |
| `h12_chain_12e12o` (`K=12, 4t`) | `DQG+T2(5)` | -6.27503234 | -6.27635813 | -6.27638436 | -0.026 | 105.996 -> 365.901 (`+259.879` UB) | `1600 / 0` |
| `h12_chain_12e12o` (`K=12, 4t`) | `DQG+T1+T2(5)` | -6.27503234 | -6.27652628 | -6.27705641 | -0.530 | 106.164 -> 366.573 (`+259.879` UB) | `1600 / 0` |

- **Scope Boundary Observed (`tm_spin_s0_8e8o` vs. `h12_chain_12e12o`)**: On canonical/model MO systems (`tm_spin_s0_8e8o`, `h2o_stretched`, `n2_stretched_10e8o` at `Ks=6`), `--auto-no` lifts lower bounds (`+0.496` to `+1.825 mHa`) and on `tm_spin_s0_8e8o` concentrates bounded-det SHCI (`max_dets=1024`) enough to slash `E_ub` by `40.361 mHa` (`width` `101.380 -> 59.802 mHa`). Conversely, on the half-filled 1D Hubbard chain `h12_chain_12e12o` (`DQG` converged in `1105` iters), delocalizing site orbitals into Bloch-like Natural Orbitals spreads CI weight across determinants (`E_ub` `-6.17036243 -> -5.91048365`, `+259.879 mHa` wider), and `T2(5)/T1+T2(5)` remain unconverged at `1600` iterations (`-1.352 mHa` / `-2.024 mHa` vs `DQG`).

### 4.2 Haar O(K) Basis-Invariance & `.xyz` MLIP Hallucination Screener

1. **Haar O(K) Basis-Invariance Recovery (`seed=42`)**:
   - `h2o_stretched` (`DQG+T1+T2, Ks=4`): Naive subspace shifts by **`-4.7360 mHa`** under random orthogonal rotation; `--auto-no` recovers **`-0.000000 mHa`** (`max|dS| = 0.000e+00`).
   - `n2_stretched_10e8o` (`DQG+T2, Ks=5`): Naive shifts **`+0.8189 mHa`**; `--auto-no` recovers **`+0.000000 mHa`** (`max|dS| = 0.000e+00`), and exporting `--save-no-fcidump` passes `chem_cli exact-verify` (`returncode=0, sha_ok=1, all_blocks_sdd=1, E_lb_int_chol=-107.46255319`).
   - `n2_stretched_10e8o` (`DQG+T1+T2, Ks=5`): Naive shifts **`+1.4214 mHa`**; `--auto-no` cuts basis sensitivity by `>30x` to **`+0.044377 mHa`** (`max|dS| = 0.000e+00`).
2. **Parallel `.xyz` Trajectory & MLIP Hallucination Screening (`veto_tol = 5e-4 Ha`)**:
   - Note on C3 veto margin: `chem_cli veto` triggers on `E_lb_cert > E_ai_arg - veto_tol_arg` (Theorem C3). To enforce strict false-low exclusion `E_ai < E_lb_cert - veto_tol` with zero false positives even when brackets are narrower than `0.5 mHa`, `chem_screener.py` maps `e_ai -> e_ai + vtol` and `veto_tol -> 0.0`.
   - **H4 Rectangular Isomerization** ([traj_h4_isomerization.xyz](file:///Users/kuangtalin/Documents/CHEM/experiments/traj_h4_isomerization.xyz), `7` frames in `2.56s`, widths `0.10–0.72 mHa`): Frames `{0,1,2,4,5,6}` pass as `CERTIFIED_CONSISTENT`; the `-25 mHa` injected false-low hallucination at TS frame `3` (`ry=2.00` Bohr, `S1_no=1.661, MR=True`) is caught by Anytime Early-Veto as `VETOED_FALSE_LOW` at iteration `110/800`.
   - **H2O Symmetric Dissociation** ([traj_h2o_dissoc.xyz](file:///Users/kuangtalin/Documents/CHEM/experiments/traj_h2o_dissoc.xyz), active `(8e, 6o)`, `5` frames in `12.44s`): `S1_no` grows monotonically (`0.300 -> 0.748 -> 1.480 -> 2.087 -> 2.778`), triggering `MR=True` at `R >= 1.55` Å; frames `{0,1,2}` pass `CERTIFIED_CONSISTENT`, `+20 mHa` at `R=1.80` Å triggers `FLAGGED_ABOVE_UB`, and `-25 mHa` at `R=2.05` Å is vetoed early at iteration `490/800` as `VETOED_FALSE_LOW`.

---

## Section 5: Theory Capstone — Physical Coulomb PSD Manifold, 3D Geometry Manifold & Theorem C17

In [ChemCert.lean](file:///Users/kuangtalin/Documents/CHEM/lean/ChemCert.lean), Theorem **C17 (`coulomb_psd_cholesky_nonneg`)** certifies purely in Lean 4 `Init` (`leanchecker ChemCert`, 0 `sorryAx`, 0 `Lean.ofReduceBool`) that any ERI operator admitting a symmetric Cholesky factorization `V_tilde = B^T B` induces non-negative pair quadratic forms `x^T (B^T B) x >= 0`. Using the exact isometric pair parameterization (`w_pq = 1` for `p=q`, `2` for `p>q`, `||V_tilde||_F = ||ERI||_F`) and quotient Riemannian gradient verified against central finite differences (`max_err_h1 = 4.160e-10`, `max_err_B = 3.567e-10`), [exp3_coulomb_manifold.py](file:///Users/kuangtalin/Documents/CHEM/experiments/exp3_coulomb_manifold.py) ([exp3_coulomb_results.json](file:///Users/kuangtalin/Documents/CHEM/experiments/exp3_coulomb_results.json)) compares all three nested Hamiltonian manifolds under `3500`-iteration `tol=1e-7` CLI + `exact-verify` audits (`scale_D = 100000000`, `all_blocks_sdd = True` on every `T1` and `T1+T2` run):

| Manifold & Winner | `(K, ne, 2S)` | `FCI - DQG+T1+T2` (`mHa/unit-Frob`) | `%` of `Hunt 3A` | `min_eig(V_tilde)` / `rank` | `eriFrac` / `offdiag` | `S1` | Audited FCIDUMP |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| General 8-Fold `Hunt 3A` | `(4, 4, 0)` | 95.265 | 100.0% | Indefinite | 92.96% / 80.84% | 5.388 | [discovered_hunt3a_k4_adversary.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_hunt3a_k4_adversary.fcidump) |
| **M1a** Coulomb PSD (`R=1`) | `(4, 4, 0)` | **112.695** | **118.3%** | `-5.66e-17` / `1` | 96.92% / 72.35% | 1.602 | [discovered_coulomb_k4_r1.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_coulomb_k4_r1.fcidump) |
| **M1a** Coulomb PSD (`R=2`) | `(4, 4, 0)` | 71.850 | 75.4% | `-9.19e-17` / `2` | 98.01% / 56.76% | 4.087 | [discovered_coulomb_k4_r2.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_coulomb_k4_r2.fcidump) |
| **M1a** Coulomb PSD (`R=4`) | `(4, 4, 0)` | 102.392 | 107.5% | `-2.51e-17` / `4` | 97.92% / 46.52% | 5.493 | [discovered_coulomb_k4_r4.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_coulomb_k4_r4.fcidump) |
| **M1a** Coulomb PSD (`R=10`) | `(4, 4, 0)` | **120.491** | **126.5%** | `+1.40e-04` / `10` | 98.49% / 55.19% | 5.519 | [discovered_coulomb_k4_r10.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_coulomb_k4_r10.fcidump) |
| **M1b** Coulomb PSD (`R=15`) | `(5, 5, 1)` | **76.430** | 80.2% | `+3.53e-05` / `15` | 99.35% / 42.68% | 4.959 | [discovered_coulomb_k5_adv.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_coulomb_k5_adv.fcidump) |
| **M2** 3D STO-3G Worst (`H4_rhomb_s2.6_a60`) | `(4, 4, 0)` | **0.0096** | **0.01%** | `+2.28e-06` / `10` | **32.77% / 8.50%** | 3.808 | [discovered_mol3d_k4_worst.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_mol3d_k4_worst.fcidump) |

1. **Physical Coulomb Positivity Does Not Prevent Large `FCI - DQG+T1+T2` Gaps — Even at Rank 1**: Restricting ERI strictly into the PSD Cholesky cone (`V_tilde = B^T B >= 0`, `min_eig >= -9.19e-17`) still produces `112.695 mHa/unit-Frob` at `R=1` and **`120.491 mHa/unit-Frob`** at `R=10` (exceeding Frontier 2's unconstrained `Hunt 3A = 95.265 mHa`), while `K=5` reaches `76.430 mHa/unit-Frob` (`DQG=212.453 -> T1=115.941 -> T2=81.521 -> T12=76.430 mHa`, all 4 tiers cleanly separated).
2. **`T1 > T2` Inversion Vanishes in Physical Coulomb PSD Search (`coulomb_k5_t1gt2`)**: Unlike Frontier 2's indefinite 8-fold tensor adversary (`Hunt 1A`: `T1 - T2 = +16.292 mHa`), every seed inside the `K=5, R=15` Coulomb PSD cone (including `W5_wheel U=6` initialization) failed to invert `T1` above `T2`, settling at `DQG+T1` gap `0.0475 mHa` vs `DQG+T2` gap `0.0184 mHa` (`T1 - T2 = -0.0291 mHa` Weyl, [discovered_coulomb_k5_t1gt2.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_coulomb_k5_t1gt2.fcidump)).
3. **Real 3D STO-3G Geometry Manifold Collapses the Gap by `>12,500x`**: Across `120` diverse 3D `H4`, `HeH3+`, and `He2H2^2+` geometries plus stochastic coordinate refinement, the largest transient `600`-iter `tol=1e-6` gap appeared at stretched rhombus `H4_rhomb_s2.6_a60` (`gap_frob = 2.0540 mHa`, `gap_raw = 7.2677 mHa`; [discovered_mol3d_k4_worst_raw.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_mol3d_k4_worst_raw.fcidump)). However, converging tightly to `3500` iterations (`tol=1e-7`) reduces `mol3d_k4_worst` from `DQG = 4.5069 mHa` -> `T1 = 0.1721 mHa` -> `T2 = 0.0113 mHa` -> **`DQG+T1+T2 = 0.0096 mHa/unit-Frob`** (`0.01%` of `Hunt 3A`). Real nuclear attraction and Gaussian overlap lock `eriFrac` to `32.77%` and `eri_offdiag` to `8.50%` (vs. `96.92%–98.49%` ERI and `46.52%–72.35%` off-diagonal pair-hopping in adversarial PSD cones), making `DQG+T1+T2` essentially exact across all tested 3D `4e4o` molecules.

## 6. Frontier 4 Theory Trilogy: Rank-1 Integrality Theorem, Physical Coulomb `T1 > T2` Phase Diagram, and Melting Boundaries (2026-11-22)

Running [exp4_theory_trilogy.py](file:///Users/kuangtalin/Documents/CHEM/experiments/exp4_theory_trilogy.py) against the feature-frozen `v0.3.1` core (structured data in [exp4_theory_results.json](file:///Users/kuangtalin/Documents/CHEM/experiments/exp4_theory_results.json)) yields three non-depreciating theoretical results with formal Lean 4 theorems C18 & C19 ([ChemCert.lean](file:///Users/kuangtalin/Documents/CHEM/lean/ChemCert.lean)):

### 6.1 Analytical Rank-1 Coulomb PSD Integrality & Full `3-POS` Separation (Theorems C18 & C19)
Dissecting [discovered_coulomb_k4_r1.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_coulomb_k4_r1.fcidump) in the eigenbasis of its unique nonzero `V_tilde` mode (`top_Vt_eig = 0.984478`, `max|other_Vt| = 1.74e-16`) reveals asymmetric rank-1 pair weights `evals_B = [-0.388115, -0.382835, 0.575695, 0.596538]` (ratios `[-1.0, -0.9864, 1.4833, 1.5370]`) and nearly diagonal one-body couplings (`max|h1_eig_offdiag| = 5.277e-04`, `||h1||_F = 0.175507`). Distilling this into an exact zero-$h_1$ analytical model [discovered_rank1_exact_m1m122.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_rank1_exact_m1m122.fcidump) with $\beta = (-1, -1, 2, 2)$ ($\|\text{eri}_{\text{raw}}\|_F = 10.0$, normalized $\|\text{eri}\|_F = 1.0$) produces:
- **Exact Algebraic Law (Lean Theorem C18 `rank1_coulomb_integrality_gap_exact`)**: For every integer charge partition $(n_A, n_B)$ with $n_A + n_B = 4$ on the two orbital blocks $\{0,1\}$ and $\{2,3\}$, $8 H(n_A) = 9(2 n_A - 5)^2 - 33$. Because $2 n_A - 5$ is odd for all $n_A \in \mathbb{Z}$, $(2 n_A - 5)^2 \ge 1$ locks $E_{\text{FCI}} = -3/10 = -0.300000\text{ Ha}$ (at $n_A \in \{2, 3\}$), while unquantized relaxation at $m_1 = 5/2$ achieves $E_{\text{DQG}} = -33/80 = -0.412500\text{ Ha}$. Verified CLI (`3500` iters, `tol=1e-7`) + `exact-verify` confirm all four tiers coincide (`DQG = 112.5010`, `T1 = 112.5004`, `T2 = 112.5008`, `T12 = 112.5020 mHa/unit-Frob`, `all_blocks_sdd = 1`).
- **Strict 3-Tier `DQG < 3-POS < FCI` Separation (Lean Theorem C19 `rank1_coulomb_3pos_strict_separation`)**: Across a `401 x 401` fine grid (`14,111 / 160,801` feasible points) enforcing all six symmetric 3-marginal/`3-POS` conditions ($D_3, Q_3, \det F_3, \det E_3, E_3^{\text{mix}}, F_3^{\text{mix}} \ge 0$) at $m_1 = 5/2$:
  - Diagonal 3-marginal probabilities alone permit $m_2 = 6.25$ (`DQG`: witness `(m1=2.5, m2=6.25, m3=15.0)` with `(r3, r2, r1, r0) = (5/96, 25/96, 5/96, 1/96) > 0`, failing only $\det F_3 = -1.5625 < 0$).
  - Coupling $F_3 \succeq 0$ ($m_1 m_3 \ge m_2^2$) with $Q_3 \ge 0$ ($m_3 \le 9 m_2 - 41$) enforces $2 m_2^2 - 45 m_2 + 205 \le 0 \implies m_2 \ge (45 - \sqrt{385})/4 = 6.34464578$ (`m3 = 16.101812`, `grid_min_m2 = 6.345000`, `E_3POS* = -0.369909 Ha`), cutting **`42.5906 mHa` (`37.86%`)** above `DQG` while leaving **`69.9094 mHa` (`62.14%`)** as an irreducible 4-marginal integrality obstruction strictly below `FCI = -0.300000 Ha` (witnessed explicitly in C19 by `(m1=2.5, m2=6.35, m3=16.13)`).

### 6.2 Physical Coulomb PSD `T1 > T2` Multi-Sector Phase Diagram (`3500` iters, `tol=1e-7`)
Searching 6 spin/filling sectors strictly inside $\tilde{V} = B^T B \succeq 0$ overturns Frontier 3's closed-shell null result once **spin-polarized open-shell imbalance** (`K=5, na=3, nb=1, 2S=2`) enters:

| Sector `(K, na, nb, 2S)` | `DQG` Gap (`mHa`) | `DQG+T1` Gap (`mHa`) | `DQG+T2` Gap (`mHa`) | `DQG+T1+T2` Gap (`mHa`) | `T1 - T2` Weyl (`mHa`) | `T1 - T2` intChol (`mHa`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `k4_3e4o_s1` (`2,1,1`) | `0.184` | `0.013` | `0.000` | `0.000` | `-0.013` | `-0.005` |
| `k4_4e4o_s0` (`2,2,0`) | `1.173` | `0.064` | `0.071` | `0.078` | `+0.006` | `+0.012` |
| `k4_4e4o_s2` (`3,1,2`) | `0.001` | `0.000` | `0.003` | `0.003` | `+0.002` | `+0.011` |
| `k5_3e5o_s1` (`2,1,1`) | `0.064` | `0.075` | `0.017` | `0.012` | `-0.057` | `-0.045` |
| **`k5_4e5o_s2` (`3,1,2`)** | **`41.420`** | **`3.665`** | **`5.599`** | **`2.421`** | **`+1.934`** | **`+1.957`** |
| `k5_5e5o_s3` (`4,1,3`) | `0.000` | `0.000` | `0.000` | `0.000` | `-0.000` | `+0.020` |

- **Verified Physical Coulomb PSD `T1 > T2` Witness ([discovered_coulomb_t1_beats_t2.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_coulomb_t1_beats_t2.fcidump))**: In `k5_4e5o_s2` (`Mode B`, `h1_frac = 5.40%`, `eri_offdiag = 30.96%`, `min_eig_Vtilde = +9.901e-4`), pure 3-particle/3-hole `T1` cuts `37.755 mHa` off `DQG`, outperforming `DQG+T2` by **`+1.934 mHa` (Weyl)** and **`+1.957 mHa` (exact integer Cholesky)** (`exact-verify`: `T1 = -0.500898 Ha` intChol / `-0.500890 Ha` Weyl vs `T2 = -0.502855 Ha` intChol / `-0.502824 Ha` Weyl, `nblocks = 13`, `sha_ok = 1`, `all_blocks_sdd = 1`). *(Note: `chem_cli emit-lean` emitted the 13-block `K=5` Mode-B certificate, which exceeds Lean 4's default `200000` maxHeartbeats/`maxRecDepth` on `50x50` block literals `block_6, block_10, block_12`, so C `exact-verify` remains the active soundness verifier at `K=5`.)*

### 6.3 Quantitative Melting Boundaries (`112.502 mHa` Blockade $\leftrightarrow$ `0.0096 mHa` Molecule)
1. **Convex PSD Interpolation ($\lambda \in [0, 1]$, `21` pts between `discovered_rank1_exact_m1m122` and `discovered_mol3d_k4_worst`)**:
   - Everywhere PSD (`min_eig_Vtilde >= -7.70e-17`), `DQG+T1+T2` drops monotonically from `112.5020 mHa` ($\lambda = 0$) across critical thresholds **`50%` (`56.25 mHa`) at $\lambda = 0.3101$** (`h1_frac ≈ 11%`), **`10%` (`11.25 mHa`) at $\lambda = 0.5117$** (`h1_frac ≈ 33%`), and **`1 mHa` at $\lambda = 0.5955$** (`h1_frac ≈ 43%`), collapsing to `0.0096 mHa` at $\lambda = 1.00$. Meanwhile `DQG` (`0.1918 mHa` at $\lambda = 0.95$) rebounds to `4.5069 mHa` at $\lambda = 1.00$ due to `H4_rhomb_s2.6_a60`'s quasi-degenerate `E1 - E0 = 3.78 mHa` cusp (`36.84 mHa` at $\lambda = 0.95$), which `T1+T2` eliminates.
2. **Pure Kinetic Inter-Cluster Tunneling ($\tau \in [0, 1.5]$, `21` pts, $t_{02} = t_{13} = -\tau$)**:
   - Turning on one-body hopping across the cluster boundary sharply melts the integrality blockade: **`50%` at $\tau = 0.0473$ (frob) / `0.0474` (raw)** (`h1_frac < 1%`), **`10%` at $\tau = 0.1329$ (frob) / `0.1344` (raw)**, and **`1 mHa` at $\tau = 0.3723$ (frob) / `0.4137` (raw)** (`h1_frac ≈ 36%`, `Var_FCI(nA)` surging from `0.1997` at $\tau = 0$ to `0.5424` at $\tau = 0.375$ and `0.8121` at $\tau = 1.50$).
   - **Selective `T2` Mechanism**: At $\tau = 0.075$, `DQG` (`29.8218 mHa`) and `DQG+T1` (`29.8218 mHa`) give zero improvement over each other, whereas **`DQG+T2` (`23.3293 mHa`)** accounts for the entire `T12` reduction (`23.3292 mHa`), proving inter-cluster tunneling coherence is captured selectively by mixed particle-hole `T2` ($E_3 + F_3$) triples.

### 6.4 Universal $p$-POS Obstruction Scaling: Theorem C20 & $K=6$ Verification
- **General Linear-Size Obstruction Theorem (Lean Theorem C20 `rank1_coulomb_p_pos_hierarchy_obstruction`)**: Generalizing C18–C19 to $K = 2p$ spatial orbitals ($M = 2p$ spin-orbitals per cluster, $N = 2p$ electrons) with $\beta = (-(p-1),\dots,-(p-1), p,\dots,p)$ and $h_1 = 0$, Theorem C20 formally proves ($0$ `sorryAx`, `[propext, Quot.sound]`) that:
  1. Every integer partition $(n_A, n_B)$ satisfies $8 H(n_A) = (2p-1)^2 (2 n_A - (2p+1))^2 - (2p-1)(4p^2 - 2p - 1)$, locking an integer quantization gap of $(2p-1)^2$ in $8H$-units above the continuous minimum at $c = p + 1/2$.
  2. Unlike $K = 4$ (`M = 4 < 6`, where $Q_3$ boundary saturation forced `3-POS` to cut `37.86%`), for ANY fixed $p \ge 2$ the choice $M = 2p$ keeps all `2^k`-scaled hypergeometric witness marginals strictly positive for every $(m_1, m_0) \le (p, p)$ with zero charge variance and exact 1-orbital marginal consistency—rendering **$p$-POS completely blind (`0%` tightening)**—while breaking sharply at order $p+1$ where $Q_{p+1}$ (`m0 = p+1`, $M - c = p - 1/2 < p$) becomes strictly negative.
- **$K = 6, N = 6$ & Multi-Order Gram Verification ([exp4_ppos_scaling.py](file:///Users/kuangtalin/Documents/CHEM/experiments/exp4_ppos_scaling.py), [exp4_ppos_results.json](file:///Users/kuangtalin/Documents/CHEM/experiments/exp4_ppos_results.json))**:
  - Normalized $K = 6, 6\text{e}6\text{o}$ rank-1 Hamiltonian [discovered_rank1_k6_p3.fcidump](file:///Users/kuangtalin/Documents/CHEM/experiments/discovered_rank1_k6_p3.fcidump) ($\beta = (-2,-2,-2,3,3,3)$, $\|\text{eri}_{\text{raw}}\|_F = 39.0$, normalized $\|\text{eri}\|_F = 1.0$, `min_eig_Vtilde = -7.20e-17`): exact `FCI E0 = -15/39 = -0.384615 Ha` vs `DQG = 80.1288`, `T1 = 80.1291`, `T2 = 80.1302`, **`DQG+T1+T2 = 80.1290 mHa/unit-Frob`** (analytical gap `25/312 = 80.1282 mHa`, `exact-verify` `all_blocks_sdd = 1` across `T1, T2, T12`) — **`0%` tightening from `3-POS`**.
  - Full fermionic localizing Gram spectra confirm `min_eig >= -7.37e-16` across all tested $M = 2p$ blocks (`p=2, M=4`; `p=3, M=6`; `p=4, M=8`) and across the full `12`-spin-orbital $K=6$ product state $\omega_A \otimes \omega_B$ (`cA = 3.5, cB = 2.5`: `<N> = 6.000000`, `Var(N) = 7.11e-14`, `13x13` 2-body Gram `-5.22e-16`, `12x12` $E_3/F_3$ blocks `>= -3.79e-16`, `min_triple_prob = 0.015625`), while failing at `(m1=0, m0=p+1)` for `(p=3, M=4) -> -1.5625e-2`, `(p=4, M=6) -> -2.6042e-3`, and `(p=5, M=8) -> -4.8828e-4`.



