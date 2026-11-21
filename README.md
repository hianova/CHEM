# CHEM — Certified Ab Initio Quantum Chemistry & Zero-FP-Trust Discovery Engine

`CHEM` (`v0.3.1`) is a standalone first-principles quantum chemistry certification, reaction-path screening, and theoretical discovery engine built on [`../FLOW`](../FLOW/README.md) (`v8.0.0`: `core/flow.h`, `core/searu.h`, `Flow.MolecularBundle`, `Flow.Interval`).

Its mission is to deliver **mathematically certified two-sided ground-state energy and observable brackets**

$$E_{\text{LB}} \le E_{\text{exact}} \le E_{\text{UB}}, \qquad O_{\text{LB}} \le \langle \psi_0 | O | \psi_0 \rangle \le O_{\text{UB}}$$

backed by **Zero-FP-Trust per-molecule Lean 4 certificates** (`leanchecker` verified, `0 sorryAx`, `0 Lean.ofReduceBool`), **subspace partial 3-RDM `DQG+T1+T2` tightening**, **entanglement-driven Natural Orbital canonicalization (`--auto-no`)**, **Generalized Pauli Constraints (Borland–Dennis $\mathcal{B}_{3,6} / \mathcal{B}_{2,4}$)**, and **Anytime Early-Veto** filtering against AI/MLIP force-field hallucinations up through `(16e,16o)` active spaces.

For detailed benchmark verification and Frontier 1–3 / Direction C theoretical discoveries, see [CHEM_VERIFICATION_REPORT.md](file:///Users/kuangtalin/Documents/CHEM/CHEM_VERIFICATION_REPORT.md) and [CHEM_DISCOVERY_REPORT.md](file:///Users/kuangtalin/Documents/CHEM/experiments/CHEM_DISCOVERY_REPORT.md).

---

## Core Architecture (4 Layers)

1. **Layer 1 — Chemistry, Lattice & Trajectory Ingestion (`FCIDUMP` & `.xyz`)**
   - Ingests standard Molpro/PySCF `FCIDUMP` Hamiltonians $(E_{\text{core}}, h_{pq}, V_{pqrs})$ with 8-fold symmetry verification, orthogonal basis rotation (`chem_ham_rotate_orthogonal`, `chem_fcidump_write`), Cauchy–Schwarz validation, and SHA-256 commitment ([include/chem_fcidump.h](file:///Users/kuangtalin/Documents/CHEM/include/chem_fcidump.h)).
   - Supports multi-frame `.xyz` geometries (built-in STO-3G RHF + frozen-core active space + ionic `--charge`) via [oracles/chem_screener.py](file:///Users/kuangtalin/Documents/CHEM/oracles/chem_screener.py).
2. **Layer 2 — Matrix-Free `DQG+T1+T2` Dual SDP, `--auto-no` & Observable Envelopes**
   - Matrix-free boundary-point dual augmented Lagrangian solver accelerated by `FLOW` 2-Jet (`core/flow.h`), half-rank `cblas_dsyrk` spectral reconstruction, POSIX `pthreads` parallel eigendecomposition (`Accelerate` `dsyevd_`), precomputed `H^{-1} A_{\text{eq}}^*` / Schur Cholesky (`S_eq`) / subspace `H_chol` + Jacobi `H_diag_inv` normal equations, and persistent warm-startable sessions (`build/libchem.dylib` `ChemSession`).
   - Supports exact $O(K_s^5)$ subspace `T2` and $O(K_s^6)$ partial 3-RDM `T1 = D3 + Q3` constraints (Mode A singlet: up to 12 PSD blocks; Mode B spin-resolved: up to 17 PSD blocks), plus entanglement-driven pilot-`DQG` Natural Orbital rotation (`chem_ham_build_dqg_no`, `--auto-no`, `--save-no-fcidump`).
   - Computes **Anytime Valid** dual lower bounds (`Flow.Interval` Weyl rounding shifts) and **Certified Observable, Borland–Dennis, Mott Order-Parameter & Hellmann–Feynman Force Envelope Bounds** (`chem_certify_observable`).
3. **Layer 3 — Variational Witness Funnel, GPC Analyzer & Exact Full-Space Variance**
   - Internal RHF/UHF + `SEARU × 2-Jet` orbital rotation $\to$ Jordan–Wigner FCI / spin-projected Selected CI (`SHCI`), external `--witness` JSON ingestion, full $\alpha,\beta$ 1-RDM SVD Generalized Pauli analyzer (`chem_var_analyze_gpc`), and exact full-Fock-space Hamiltonian variance $\sigma^2 = \|(H - E_{\text{var}})|\psi\rangle\|_2^2$ + Eckart ground-state fidelity lower bounds (`chem_var_eval_variance`).
4. **Layer 4 — Zero-FP-Trust `ChemI128` Cholesky+SDD & Kernel-Checked Lean 4 Certificates**
   - Emits `.cert.json`, compact binary `.flowchem` (`FLCH0002`, `2.75x–2.91x` smaller), and standalone **per-molecule `.lean` formal proofs** via pure ISO C17 128-bit integer Cholesky+SDD certification (`chem_cert_verify_exact_chol`, `chem_cert_emit_lean`), compiled with `lean -DwarningAsError=true` and audited by `leanchecker` against [lean/ChemCert.lean](file:///Users/kuangtalin/Documents/CHEM/lean/ChemCert.lean) (Theorems `C1–C17`, including `T1` trace bound `C16` and Coulomb Cholesky PSD `C17`, `0 sorryAx`, `0 Lean.ofReduceBool`).

---

## Repository Layout

```text
CHEM/
├── README.md
├── CHEM_VERIFICATION_REPORT.md      # Gates 1–6 independent verification report
├── Makefile                         # C17 (-O3 -Werror -Wconversion -pthread) + Accelerate + Lean 4
├── include/
│   ├── chem_fcidump.h               # FCIDUMP loader/writer, 8-fold ERI check & SHA-256
│   ├── chem_rdm2.h                  # Matrix-free DQG+T1+T2 dual SDP, ChemSession, NO rotation, Chol+SDD & Lean emitter
│   └── chem_shci.h                  # RHF/UHF+DIIS, SEARU×2-Jet, FCI/SHCI, GPC Borland-Dennis & full-space variance
├── src/
│   ├── chem_fcidump.c               # Molpro/PySCF FCIDUMP loader/writer & SHA-256
│   ├── chem_rdm2_mf.c               # Matrix-free Mode A/B DQG+T1+T2, pthreads, NO rotation, ChemI128 & Lean emitter
│   ├── chem_shci.c                  # RHF/UHF, SEARU×2-Jet orbital rot, FCI/SHCI, 1-RDM GPC SVD & exact variance
│   └── chem_cli.c                   # CLI: selftest, bracket, veto, observable, cert-verify, exact-verify, emit-lean, gpc, atlas, bench
├── oracles/
│   ├── gen_benchmarks.py            # Deterministic STO-3G, HF-force, TM spin & Hubbard Mott FCIDUMP generator (27 files)
│   ├── chem_sdk.py                  # ChemOracle, ChemFastSession, read/write_fcidump, rotate_integrals & mlip_screen
│   ├── chem_screener.py             # Parallel .xyz trajectory & MLIP hallucination screener (--auto-no, --csv, --cert-dir)
│   ├── test_tools.py                # End-to-end CLI, SDK, ChemFastSession & chem_screener integration tests
│   └── verify_chem_oracle.py        # Independent NumPy FCI, DQG/T2 adjoints, Mott/GPC/Lean auditor (Gates 1–6)
├── benchmarks/                      # 27 FCIDUMP benchmarks ((2e,2o)–(16e,16o))
├── experiments/                     # Frontier 1–3 & Direction C discovery scripts, JSONs, .xyz trajectories & CHEM_DISCOVERY_REPORT.md
└── lean/
    └── ChemCert.lean                # Pure-Init Lean 4 Theorems C1–C17 & checkCholeskySDD_sound (0 sorryAx)
```

---

## Build, CLI, Python SDK, Screener & Verified Highlights

```bash
make all         # Build build/chem_cli, build/libchem.dylib & build/ChemCert.olean
make selftest    # Run 15 C17 self-tests (adjoints, T1/T2, --auto-no invariance, exact Chol+SDD, BD36 GPC, variance, FLCH0002)
make benchmarks  # Generate 27 FCIDUMP benchmarks into benchmarks/
make atlas       # Run Hubbard Mott phase diagram, DQG+T2 gap atlas, GPC & Eckart fidelity -> build/chem_atlas.json
make test-tools  # Run end-to-end CLI, ChemOracle, ChemFastSession & chem_screener integration suite
make verify      # Run all Gates 1–6 + leanchecker + Python oracle + test-tools
```

CLI subcommands (`./build/chem_cli`):

```text
chem_cli --help | --version
chem_cli selftest [--json <out.json>]
chem_cli bracket <fcidump> [--mode auto|a|b] [--two-s <2S>] [--no-s2] [--max-iters <n>] [--tol <f>]
                 [--max-dets <n>] [--t2] [--t2-orbs <Ks>] [--t1] [--t1-orbs <Ks>]
                 [--auto-no] [--save-no-fcidump <no.fcidump>] [--threads <n>]
                 [--witness <w.json>] [--dump-rdm-grad <grad.json>]
                 [--cert <out.json>] [--cert-bin <out.flowchem>] [--json <out.json>]
chem_cli veto <fcidump> --e-ai <Ha> [--veto-tol <Ha>] [--mode auto|a|b] [--t2] [--t2-orbs <Ks>]
              [--t1] [--t1-orbs <Ks>] [--auto-no] [--witness <w.json>] [--cert <out.json>] [--cert-bin <out.flowchem>]
chem_cli observable <fcidump> --type occ:<p>|docc:<p>|sdot:<p>:<q>|hf-force:<plus>:<minus>:<dR>
                    [--mode auto|a|b] [--t2] [--t1] [--auto-no] [--save-no-fcidump <no.fcidump>]
                    [--witness <w.json>] [--json <out.json>]
chem_cli cert-verify [<fcidump>] <cert.json|cert.flowchem>
chem_cli exact-verify [<fcidump>] <cert.json|cert.flowchem> [--json <out.json>]
chem_cli emit-lean <fcidump> <cert.json|cert.flowchem> --out <out.lean> [--name <ModName>]
chem_cli gpc <fcidump> [--mode auto|a|b] [--two-s <2S>] [--witness <w.json>] [--certify-bd] [--json <out.json>]
chem_cli atlas [--dir <benchmarks>] [--out <atlas.json>]
chem_cli bench [--dir <benchmarks>] [--out <bench.json>] [--cert-dir <certs>]
```

Reaction-Path & MLIP Trajectory Screener ([oracles/chem_screener.py](file:///Users/kuangtalin/Documents/CHEM/oracles/chem_screener.py)):

```bash
python3 oracles/chem_screener.py \
  --xyz experiments/traj_h2o_dissoc.xyz \
  --active-el 8 --active-orb 6 --sub-orbs 5 --charge 0 \
  --mlip-json mlip_energies.json --veto-tol 0.0005 --fast-veto \
  --cert-dir build/screen_certs --workers 4 --out report.json --csv report.csv
```

Python SDK ([oracles/chem_sdk.py](file:///Users/kuangtalin/Documents/CHEM/oracles/chem_sdk.py)):

```python
from chem_sdk import ChemOracle, ChemFastSession, read_fcidump, write_fcidump, rotate_integrals

oracle = ChemOracle("build/chem_cli")
ev = oracle.exact_verify("build/certs/h4_ry2.0.flowchem")        # Zero-FP-Trust integer Cholesky+SDD audit
gpc = oracle.gpc("benchmarks/h3_doublet.fcidump", certify_bd=True) # Borland-Dennis B_{3,6} pinning + SDP bounds
res = oracle.mlip_screen(
    "benchmarks/h4_ry2.0.fcidump",
    e_ai=-1.93949, f_ai=-0.070,
    f_plus="benchmarks/h4_ry2.0_fplus.fcidump",
    f_minus="benchmarks/h4_ry2.0_fminus.fcidump",
    dr=0.005, t1=True, t1_orbs=4, t2=True, t2_orbs=4,
)

# High-throughput in-process warm-started solver via build/libchem.dylib
with ChemFastSession.from_fcidump("benchmarks/h4_ry2.0.fcidump", t1=True, t1_orbs=4, t2=True, t2_orbs=4)[0] as sess:
    ham = read_fcidump("benchmarks/h4_ry2.0.fcidump")
    sol = sess.solve(ham["e_core"], ham["h1"], ham["eri"], max_iters=800, tol=1e-6)
```

**Verified Highlights** ([CHEM_VERIFICATION_REPORT.md](file:///Users/kuangtalin/Documents/CHEM/CHEM_VERIFICATION_REPORT.md) & [CHEM_DISCOVERY_REPORT.md](file:///Users/kuangtalin/Documents/CHEM/experiments/CHEM_DISCOVERY_REPORT.md)):
- **Performance & Memory**: `Peak RSS = 49.22 MB` (`< 50 MB`) across full `make verify` through `(16e,16o)` Mode B (`4.874s` SDP / `5.054s` total at 400 iters, 4 threads); precomputed `scr_t2_pairs` + `H_diag_inv` + gated `H_chol` speed up `h12_chain_12e12o` `T2(5)` (`5.702s -> 3.616s` SDP, `1.58x`), `tm_spin_s0_8e8o` `T2(6)` (`4.969s -> 4.154s` SDP), and `n2_stretched_10e8o` `DQG` (`0.577s -> 0.556s` SDP, `Peak RSS = 8.11 MB`); single-pass `chem_screener.py` cuts 5-frame `(8e,6o)` `H2O` trajectory screening from `12.44s` to `8.83s` (`6.74s` with `--fast-veto`).
- **Matrix-Free Adjoints & Orthogonal Invariance**: `selftest` adjoints `< 1e-14` (`DQG=2.174e-16`, `T2=5.930e-17`, `T1=3.593e-17`); Haar $O(K)$ orthogonal rotation (`seed=42`) invariance recovered to `0.000000 mHa` on `h2o_stretched` (`T1+T2(4)`) and `n2_stretched_10e8o` (`T2(5)`) with `--auto-no`.
- **Zero-FP-Trust & Lean 4 Formalization**: All benchmark JSON/`.flowchem` certificates pass `cert-verify` & `exact-verify` (`SDD=1`), plus [ChemCert.lean](file:///Users/kuangtalin/Documents/CHEM/lean/ChemCert.lean) Theorems `C1–C17` and [CertDiscoveredT1.lean](file:///Users/kuangtalin/Documents/CHEM/experiments/CertDiscoveredT1.lean) verified by `leanchecker` (`0 sorryAx`, `0 Lean.ofReduceBool`).
- **Coulomb PSD vs. 3D Molecular Manifold Discovery**: Within the physical Coulomb PSD cone (`V_tilde = B^T B >= 0`, Theorem `C17`), rank-1 and rank-10 Hamiltonians still sustain `112.695` and `120.491 mHa/unit-Frob` `FCI - DQG+T1+T2` gaps (`126.5%` of unconstrained `Hunt 3A`), whereas real 3D STO-3G nuclear-Gaussian geometries (`120` sampled 3D shapes) collapse at convergence to `0.0096 mHa/unit-Frob`.

---

## Explicit Scientific Boundaries

1. **Active-Space & Basis Scope**: Dual 2-RDM / subspace `T1/T2` SDP lower bounds and observable intervals are rigorous mathematical bounds *within the specified active-space Hamiltonian and one-particle basis* (`FCIDUMP`), not complete-basis-set (CBS) limits.
2. **`DQG`, `T1/T2` & `--auto-no` Tradeoffs**: Pure 2-positivity (`DQG`) runs in $O(K^6)$ per iteration and catches severe multi-reference hallucinations (`15–200+ mHa`), while sub-`4 mHa` accuracy on stretched bonds and strongly correlated open shells uses subspace `T2` ($O(K_s^5)$) and `T1` ($O(K_s^6)$). Enabling `--auto-no` provides basis-invariant concentration on canonical molecular/model orbitals, while site-local orbitals can remain tighter for bounded-det SHCI `E_ub` on strongly correlated 1D Hubbard lattices.
3. **Target Wedge**: `CHEM` is a high-assurance **Truth Oracle & Active-Learning Veto Filter** for AI force-field training sets, reaction barriers, catalysis active sites, transition-metal spin-state ordering, and biradicals—not a replacement for routine closed-shell equilibrium DFT geometry scans.
