#!/usr/bin/env python3
"""
Theorem C20 Computational & Structural Companion:
Universal Rank-1 Coulomb PSD Obstruction of p-POS on K = 2p Orbitals (Verify K=6, p=3 & p=2..4 Blocks)
"""
import itertools
import json
import math
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "experiments")
sys.path.insert(0, os.path.join(ROOT, "oracles"))
sys.path.insert(0, EXP_DIR)

import chem_sdk
import exp4_theory_trilogy as e4


def falling(x: float, k: int) -> float:
    val = 1.0
    for j in range(k):
        val *= (x - float(j))
    return val


def omega_cluster(M: int, c: float, s1_set, s0_set) -> float:
    """Product hypergeometric witness functional on a single M-spin-orbital cluster."""
    if s1_set & s0_set:
        return 0.0
    m1 = len(s1_set)
    m0 = len(s0_set)
    return falling(c, m1) * falling(float(M) - c, m0) / falling(float(M), m1 + m0)


def check_all_ppos_blocks_single_cluster(p: int, M: int, c: float):
    """Build and eigendecompose every fermionic (m1, m0, d) localizing block with m1 + m0 + 2*d <= p."""
    blocks = []
    min_eig_all = 1e9
    worst = None
    for m1 in range(p + 1):
        for m0 in range(p + 1 - m1):
            m = m1 + m0
            d = (p - m) // 2
            s1 = set(range(m1))
            s0 = set(range(m1, m))
            rem = list(range(m, M))
            basis = []
            for deg in range(d + 1):
                for comb in itertools.combinations(rem, deg):
                    basis.append(set(comb))
            nb = len(basis)
            G = np.zeros((nb, nb), dtype=np.float64)
            for i, bi in enumerate(basis):
                for j, bj in enumerate(basis):
                    G[i, j] = omega_cluster(M, c, s1 | bi | bj, s0)
            ev = np.linalg.eigvalsh(G)
            me = float(np.min(ev))
            row = {"m1": m1, "m0": m0, "d": d, "dim": nb, "min_eig": me}
            blocks.append(row)
            if me < min_eig_all:
                min_eig_all = me
                worst = row
    return {"p": p, "M": M, "c": c, "n_blocks": len(blocks), "min_eig_all": min_eig_all, "worst_block": worst}


def omega_two_cluster(M: int, cA: float, cB: float, s1_A, s0_A, s1_B, s0_B) -> float:
    return omega_cluster(M, cA, s1_A, s0_A) * omega_cluster(M, cB, s1_B, s0_B)


def check_full_12orb_k6_3pos():
    """Verify full cross-cluster omega_AB = omega_A (x) omega_B on all 12 spin-orbitals at K=6 (cA=3.5, cB=2.5)."""
    M = 6
    cA, cB = 3.5, 2.5
    # 1. Exact mean & variance of total electron count n_A + n_B
    mean_nA = sum(omega_two_cluster(M, cA, cB, {i}, set(), set(), set()) for i in range(M))
    mean_nB = sum(omega_two_cluster(M, cA, cB, set(), set(), {i}, set()) for i in range(M))
    e_n2 = 0.0
    for c1 in ("A", "B"):
        for i in range(M):
            for c2 in ("A", "B"):
                for j in range(M):
                    s1A = ({i} if c1 == "A" else set()) | ({j} if c2 == "A" else set())
                    s1B = ({i} if c1 == "B" else set()) | ({j} if c2 == "B" else set())
                    e_n2 += omega_two_cluster(M, cA, cB, s1A, set(), s1B, set())
    mean_N = mean_nA + mean_nB
    var_N = e_n2 - mean_N * mean_N

    # 2. All cross-cluster 3-POS blocks on 12 spin-orbitals:
    # (a) 2-body (d=1, m=0): 13x13 Gram matrix on {1} union {n_k : k in A union B}
    orbs = [("A", i) for i in range(M)] + [("B", j) for j in range(M)]
    basis0 = [None] + orbs
    G0 = np.zeros((13, 13), dtype=np.float64)
    for r, o1 in enumerate(basis0):
        for c, o2 in enumerate(basis0):
            s1A, s1B = set(), set()
            for o in (o1, o2):
                if o is not None:
                    (s1A if o[0] == "A" else s1B).add(o[1])
            G0[r, c] = omega_two_cluster(M, cA, cB, s1A, set(), s1B, set())
    min_eig_G0 = float(np.min(np.linalg.eigvalsh(G0)))

    # (b) F3 (m1=1, m0=0, d=1) and E3 (m1=0, m0=1, d=1) 12x12 localizing blocks for representative pin in A and B
    loc_blocks = []
    min_eig_loc = 1e9
    for kind in ("F3", "E3"):
        for pin in (("A", 0), ("B", 0)):
            other_orbs = [o for o in orbs if o != pin]
            basis = [None] + other_orbs
            G = np.zeros((len(basis), len(basis)), dtype=np.float64)
            for r, o1 in enumerate(basis):
                for c, o2 in enumerate(basis):
                    s1A, s0A, s1B, s0B = set(), set(), set(), set()
                    if kind == "F3":
                        (s1A if pin[0] == "A" else s1B).add(pin[1])
                    else:
                        (s0A if pin[0] == "A" else s0B).add(pin[1])
                    for o in (o1, o2):
                        if o is not None:
                            (s1A if o[0] == "A" else s1B).add(o[1])
                    G[r, c] = omega_two_cluster(M, cA, cB, s1A, s0A, s1B, s0B)
            me = float(np.min(np.linalg.eigvalsh(G)))
            min_eig_loc = min(min_eig_loc, me)
            loc_blocks.append({"kind": kind, "pin": pin, "dim": len(basis), "min_eig": me})

    # (c) All m1 + m0 = 3 pure occupation/hole triple sectors across A and B
    min_triple = 1e9
    for m1A in range(4):
        for m0A in range(4 - m1A):
            for m1B in range(4 - m1A - m0A):
                m0B = 3 - m1A - m0A - m1B
                val = omega_two_cluster(
                    M, cA, cB,
                    set(range(m1A)), set(range(m1A, m1A + m0A)),
                    set(range(m1B)), set(range(m1B, m1B + m0B)),
                )
                min_triple = min(min_triple, val)

    assert abs(mean_N - 6.0) < 1e-12
    assert abs(var_N) < 1e-12
    assert min_eig_G0 >= -1e-13
    assert min_eig_loc >= -1e-13
    assert min_triple > 0.0
    return {
        "M_per_cluster": M,
        "cA": cA,
        "cB": cB,
        "mean_N": float(mean_N),
        "var_N": float(var_N),
        "min_eig_G0_13x13": min_eig_G0,
        "min_eig_loc_12x12": min_eig_loc,
        "loc_blocks": loc_blocks,
        "min_triple_prob_3orb": float(min_triple),
    }


def main():
    t0 = time.time()
    K = 6
    beta = np.array([-2.0, -2.0, -2.0, 3.0, 3.0, 3.0], dtype=np.float64)
    h1 = np.zeros((K, K), dtype=np.float64)
    eri_raw = np.zeros((K, K, K, K), dtype=np.float64)
    for p in range(K):
        for q in range(K):
            eri_raw[p, p, q, q] = beta[p] * beta[q]
    frob_raw = math.sqrt(float(np.sum(eri_raw * eri_raw)))
    assert abs(frob_raw - 39.0) < 1e-12, f"Expected Frobenius norm 39.0, got {frob_raw}"
    eri = eri_raw / frob_raw

    fcidump_path = os.path.join(EXP_DIR, "discovered_rank1_k6_p3.fcidump")
    chem_sdk.write_fcidump(fcidump_path, K, 6, 0, 0.0, h1, eri)

    cli = chem_sdk.ChemOracle(os.path.join(ROOT, "build", "chem_cli"))
    audit = e4.audit_all_tiers_cli(
        "rank1_k6_p3_m2m2m2333",
        fcidump_path,
        "cert_exp4_rank1_k6_p3",
        K, 3, 3, 0, "a", 0.0, h1, eri, cli,
    )
    moments = e4.cluster_charge_moments_fci(audit["rdm"], orbitals=(0, 1, 2))
    print(
        f"[K=6 Rank-1] FCI E0={audit['fci_E0']:.6f} (-15/39={-15.0/39.0:.6f}), "
        f"DQG={audit['tiers']['DQG']['gap_frob_mha']:.4f}, "
        f"T1={audit['tiers']['DQG+T1']['gap_frob_mha']:.4f}, "
        f"T2={audit['tiers']['DQG+T2']['gap_frob_mha']:.4f}, "
        f"T12={audit['tiers']['DQG+T1+T2']['gap_frob_mha']:.4f} mHa "
        f"(exact 25/312={25.0/312.0*1000:.4f} mHa)",
        flush=True,
    )

    ppos_pass = []
    for p, M, c in [(2, 4, 2.5), (2, 4, 1.5), (3, 6, 3.5), (3, 6, 2.5), (4, 8, 4.5), (4, 8, 3.5)]:
        res = check_all_ppos_blocks_single_cluster(p, M, c)
        assert res["min_eig_all"] >= -1e-13, f"p-POS pass check failed: {res}"
        ppos_pass.append(res)
        print(f"  [PASS {p}-POS M={M} c={c}] blocks={res['n_blocks']}, min_eig={res['min_eig_all']:+.3e}", flush=True)

    ppos_fail = []
    for p, M, c in [(3, 4, 2.5), (4, 6, 3.5), (5, 8, 4.5)]:
        res = check_all_ppos_blocks_single_cluster(p, M, c)
        assert res["min_eig_all"] < -1e-5, f"(p+1)-POS expected breakdown did not fail: {res}"
        ppos_fail.append(res)
        print(
            f"  [FAIL {p}-POS M={M} c={c}] min_eig={res['min_eig_all']:+.6e} "
            f"at (m1={res['worst_block']['m1']}, m0={res['worst_block']['m0']})",
            flush=True,
        )

    cross_12 = check_full_12orb_k6_3pos()
    print(
        f"  [Full 12-Orb K=6 3-POS] <N>={cross_12['mean_N']:.6f}, Var(N)={cross_12['var_N']:.2e}, "
        f"min_eig(G0)={cross_12['min_eig_G0_13x13']:+.3e}, "
        f"min_eig(E3/F3 12x12)={cross_12['min_eig_loc_12x12']:+.3e}, "
        f"min_triple_prob={cross_12['min_triple_prob_3orb']:.6f}",
        flush=True,
    )

    out = {
        "schema": "chem.exp4.ppos_scaling.v1",
        "elapsed_sec": float(time.time() - t0),
        "k6_rank1_model": {
            **e4.json_sanitize(audit),
            "fcidump_path": fcidump_path,
            "beta": beta.tolist(),
            "analytical_FCI_E": -15.0 / 39.0,
            "analytical_DQG_E": -18.125 / 39.0,
            "analytical_gap_mha": (25.0 / 312.0) * 1000.0,
            "fci_moments_nA": moments,
        },
        "single_cluster_ppos_pass": ppos_pass,
        "boundary_p1_breakdown_fail": ppos_fail,
        "full_12orb_k6_3pos_product": cross_12,
    }
    out_path = os.path.join(EXP_DIR, "exp4_ppos_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] Saved C20 p-POS scaling verification to {out_path} in {out['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
