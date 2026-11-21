# CHEM v0.2.0 — Certified Ab Initio Quantum Chemistry Anti-Hallucination Sieve

`CHEM` is a standalone first-principles quantum chemistry certification engine built on [`../FLOW`](../FLOW/README.md) (`v8.0.0`: `core/flow.h`, `core/searu.h`, `Flow.MolecularBundle`, `Flow.Interval`).

Its mission is to deliver **mathematically certified two-sided ground-state energy and observable brackets**

$$E_{\text{LB}} \le E_{\text{exact}} \le E_{\text{UB}}, \qquad O_{\text{LB}} \le \langle \psi_0 | O | \psi_0 \rangle \le O_{\text{UB}}$$

with **Anytime Early-Veto** capability—catching strong-correlation and force hallucinations in AI/MLIP force fields and single-reference electronic-structure methods (`DFT`, `RHF`, non-variational `CCSD(T)`), and certifying spin-state or reaction-barrier orderings on active spaces from `(4e,4o)` up to `(16e,16o)` where Full CI is combinatorially intractable.

---

## Core Architecture (4 Layers)

1. **Layer 1 — Standard Chemistry Ingestion (`FCIDUMP` & Active-Space Interface)**
   - Ingests standard Molpro/PySCF `FCIDUMP` Hamiltonians $(E_{\text{core}}, h_{pq}, V_{pqrs})$ with 8-fold symmetry expansion, Cauchy–Schwarz validation, and SHA-256 commitment (`include/chem_fcidump.h`).
2. **Layer 2 — Matrix-Free `DQG` & Exact Subspace `DQG+T2` Dual SDP + Observable Envelopes**
   - Matrix-free boundary-point / dual augmented Lagrangian solver accelerated by `FLOW` 2-Jet (`core/flow.h`), half-rank `cblas_dsyrk` spectral reconstruction, POSIX `pthreads` parallel block eigendecomposition (`Accelerate` `dsyevd_`), and exact $O(K_s^5)$ full/entropy-frontier `T2` partial 3-RDM constraints (Mode A: 8 PSD blocks, Mode B: 13 PSD blocks).
   - Computes **Anytime Valid** dual lower bounds (`Flow.Interval` Weyl rounding shifts) and **Certified Observable & Hellmann–Feynman Force Envelope Bounds** conditioned on a variational upper bound $E_{\text{UB}}$ (`chem_certify_observable`).
3. **Layer 3 — Tiered Variational Upper-Bound Funnel & External Witness Ingestion (`--witness`)**
   - Internal RHF/UHF + `SEARU × 2-Jet` orbital rotation $\to$ Jordan–Wigner FCI / spin-projected Selected CI (`SHCI`), plus external wavefunction witness JSON ingestion (`chem_var_load_witness_json`, `chem_eval_observable_witness`).
4. **Layer 4 — Proof-Carrying JSON & Compact Binary `.flowchem` Certificates + Lean 4 Audit**
   - Emits both audit-ready `.cert.json` (`chem-cert-v1` / `chem-cert-v2`) and compact binary `.flowchem` (`FLCH0002`, `2.75x–2.91x` smaller) certificates verified independently in C (`chem_cli cert-verify`), Python/NumPy (`oracles/verify_chem_oracle.py`), and formally in Lean 4 (`lean/ChemCert.lean`, Theorems C1–C8, `0 sorryAx`).

---

## Repository Layout

```text
CHEM/
├── README.md
├── SPEC.md                    # Mathematical & engineering specification
├── CHEM_VERIFICATION_REPORT.md# Independent audit & Gate 1–5 verification report
├── Makefile                   # C17 (-O3 -Wall -Wextra -Werror -Wpedantic -Wconversion -pthread) + Accelerate + Lean 4
├── include/
│   ├── chem_fcidump.h         # FCIDUMP parser, 8-fold ERI check & SHA-256 commitment
│   ├── chem_rdm2.h            # Matrix-free DQG/T2 dual SDP, observables, Weyl certifier & binary/JSON cert API
│   └── chem_shci.h            # RHF/UHF+DIIS, SEARU×2-Jet, FCI/SHCI, <S^2> & witness evaluator API
├── src/
│   ├── chem_fcidump.c         # Molpro/PySCF FCIDUMP loader, Cauchy-Schwarz & SHA-256
│   ├── chem_rdm2_mf.c         # Matrix-free Mode A/B DQG+T2 SDP, half-rank dsyrk, pthreads, observables, FLCH0002
│   ├── chem_shci.c            # RHF/UHF, SEARU×2-Jet orbital rot, FCI/SHCI & external witness JSON/obs eval
│   └── chem_cli.c             # CLI: selftest, bracket, veto, observable, cert-verify, bench
├── oracles/
│   ├── gen_benchmarks.py      # Deterministic STO-3G & model active-space FCIDUMP generator (21 files)
│   ├── chem_sdk.py            # ChemOracle Python SDK, PySCF CASCI export & MLIP energy/force filter
│   └── verify_chem_oracle.py  # Independent NumPy FCI, DQG/T2 adjoints, observable & cert auditor (Gates 1–5)
├── benchmarks/                # 21 FCIDUMP cases ((2e,2o)–(16e,16o), H4 Hellmann-Feynman +/-, N2, H2O, TM spin)
└── lean/
    └── ChemCert.lean          # Lean 4 Theorems C1–C8 (Flow.MolecularBundle, Flow.Interval, 0 sorryAx)
```

---

## Build, CLI, Python SDK & Verified Highlights

```bash
make all         # Build build/chem_cli (C17 + Accelerate + pthreads) & build/ChemCert.olean (Lean 4)
make selftest    # Run C17 v0.2.0 self-tests (DQG/T2 adjoints, CAR/PSD, veto, observables, FLCH0002 & tamper reject)
make benchmarks  # Generate 21 FCIDUMP benchmarks into benchmarks/
make verify      # Run selftest + bench + CLI cert-verify (.cert.json & .flowchem) + Python oracle report
```

CLI subcommands (`./build/chem_cli`):

```text
chem_cli selftest [--json <out.json>]
chem_cli bracket <fcidump> [--mode auto|a|b] [--two-s <2S>] [--t2] [--t2-orbs <k>] [--threads <n>]
                 [--witness <w.json>] [--cert <out.json>] [--cert-bin <out.flowchem>]
chem_cli veto <fcidump> --e-ai <Ha> [--veto-tol <Ha>] [--mode auto|a|b] [--t2] [--witness <w.json>]
              [--cert <out.json>] [--cert-bin <out.flowchem>]
chem_cli observable <fcidump> --type occ:<p>|docc:<p>|sdot:<p>:<q>|hf-force:<plus>:<minus>:<dR>
                    [--mode auto|a|b] [--t2] [--witness <w.json>] [--json <out.json>]
chem_cli cert-verify [<fcidump>] <cert.json|cert.flowchem>
chem_cli bench [--dir <benchmarks>] [--out <bench.json>] [--cert-dir <certs>]
```

Python SDK (`oracles/chem_sdk.py`):

```python
from chem_sdk import ChemOracle

oracle = ChemOracle("build/chem_cli")
res = oracle.mlip_screen(
    "benchmarks/h4_ry2.0.fcidump",
    e_ai=-1.90, f_ai=0.5,
    f_plus="benchmarks/h4_ry2.0_fplus.fcidump",
    f_minus="benchmarks/h4_ry2.0_fminus.fcidump",
    dr=0.005,
)
# Checks certified energy lower bound first, then Hellmann-Feynman force interval [F_lb, F_ub]
```

**Verified `v0.2.0` Results** ([CHEM_VERIFICATION_REPORT.md](file:///Users/kuangtalin/Documents/CHEM/CHEM_VERIFICATION_REPORT.md)):
- **Peak RSS**: `41.86 MB` (`< 50 MB`) through `(16e,16o)` Mode B (`22.68s`) + `DQG+T2` + observables.
- **Matrix-Free Adjoints**: C max error `1.20e-16` (`DQG`) / `1.07e-16` (`T2`); Python `3.20e-16` / `7.98e-18`.
- **Audited Certificates & Exact FCI**: `28` `.cert.json` & `8` `.flowchem` files (`max_diff = 2.84e-14`), `13` exact FCI cases & `4` observables (`0` violations).
- **`DQG+T2` Strict Tightening**: `h4_ry2.2` `+1.309 mHa` (`gap = 0.357 mHa`), `h2o_stretched` `+3.991 mHa` (`gap = 4.255 mHa`), `n2_stretched_6e6o` `+8.039 mHa` (`gap = 2.847 mHa`), `tm_spin_s0_8e8o` `+4.709 mHa`.
- **Certified TM Spin Gap & HF Force**: `E_UB(S=2) = 1.109714 < E_LB(S=0) = 2.443233` (`+1333.52 mHa`); `h4_hf_force` `[-0.098947, -0.044939]` encloses exact FCI `-0.070123`.

---

## Roadmap Status

- **v0.1.0 (Complete) — Matrix-Free `FCIDUMP` + Spin-Adapted `DQG` Dual SDP + Anytime Veto**: $O(K^4)$ operator maps, Mode A & Mode B exact adjoints, tiered RHF/UHF + SHCI upper bounds, and Lean 4 formal soundness (Theorems C1–C6).
- **v0.2.0 (Current, Complete & Verified) — Subspace `DQG+T2`, Parallel Half-Rank Acceleration, Certified Observables, Binary `.flowchem` & Python SDK**: Exact matrix-free `T2` partial 3-RDM tightening (`--t2` / `--t2-orbs`), half-rank `cblas_dsyrk` + `pthreads` eigendecomposition, certified observable & Hellmann–Feynman force intervals (Theorem C7), entropy-frontier `T2` trace bound (Theorem C8), external `--witness` ingestion, `FLCH0002` compact binary certs, and `oracles/chem_sdk.py`.
- **Next Horizons — Large-Scale AI/MLIP Anti-Hallucination & Dynamic-Correlation Envelope**: Benchmark 50–100 strongly correlated catalyst/transition-metal systems and couple active-space brackets to external NEVPT2/CASPT2/AFQMC dynamic-correlation estimators.

---

## Explicit Scientific Boundaries

1. **Active-Space & Basis Scope**: Dual 2-RDM / subspace `T2` SDP lower bounds and observable intervals are rigorous mathematical bounds *within the specified active-space Hamiltonian and one-particle basis* (`FCIDUMP`), not complete-basis-set (CBS) / full-molecule dynamic-correlation limits.
2. **`DQG` vs. `DQG+T2` Tightness–Scaling Tradeoff**: Pure 2-positivity (`DQG`) runs in $O(K^6)$ per iteration and catches severe multi-reference hallucinations (`15–200+ mHa`), while sub-`4 mHa` accuracy on stretched bonds and strongly correlated open shells uses subspace `T2` ($O(K_s^5)$ for frontier subset $K_s \le K$).
3. **Target Wedge**: `CHEM` is a high-assurance **Truth Oracle & Active-Learning Veto Filter** for AI force-field training sets, catalysis active sites, transition-metal spin-state ordering, biradicals, and bond-breaking barriers—not a replacement for routine closed-shell DFT equilibrium geometry scans.
