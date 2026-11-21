#!/usr/bin/env python3
"""
Frontier 4 Theory Trilogy (CHEM v0.3.1, Feature-Frozen C Core):
  1. Analytical Rank-1 Coulomb PSD Integrality Theorem + Full 3-POS Dissection
  2. Physical Coulomb PSD T1 > T2 Multi-Sector Phase Diagram (6 (K, na, nb, 2S) sectors)
  3. Quantitative Phase Boundary:
       3A. Convex PSD interpolation lambda in [0, 1]: Rank-1 Monster <-> 3D Molecule
       3B. Pure kinetic inter-cluster tunneling tau in [0, 1.5]
"""
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "experiments")
sys.path.insert(0, os.path.join(ROOT, "oracles"))
sys.path.insert(0, EXP_DIR)

import chem_sdk
import fci_exact_oracle as fo
import exp3_coulomb_manifold as e3


def spectral_cone_stats(h1: np.ndarray, eri: np.ndarray):
    """Verify V_tilde >= 0, V_unw >= 0 and compute Frobenius partition."""
    K = h1.shape[0]
    pairs, _, s_vec, _ = e3.sym_pairs(K)
    M = len(pairs)
    V_unw = np.zeros((M, M), dtype=np.float64)
    for a, (p, q) in enumerate(pairs):
        for b, (r, s) in enumerate(pairs):
            V_unw[a, b] = float(eri[p, q, r, s])
    V_tilde = (V_unw * s_vec[:, None]) * s_vec[None, :]
    evals_tilde = np.linalg.eigvalsh(V_tilde)
    evals_unw = np.linalg.eigvalsh(V_unw)
    min_eig_tilde = float(np.min(evals_tilde))
    min_eig_unw = float(np.min(evals_unw))
    h1_norm2 = float(np.sum(h1 * h1))
    eri_norm2 = float(np.sum(eri * eri))
    tot_norm2 = h1_norm2 + eri_norm2
    norm = math.sqrt(max(1e-30, tot_norm2))
    coul_norm2 = float(sum(eri[i, i, j, j] ** 2 for i in range(K) for j in range(K)))
    exch_norm2 = float(sum(eri[i, j, i, j] ** 2 for i in range(K) for j in range(K) if i != j))
    offdiag_frac = max(0.0, 1.0 - (coul_norm2 + 2.0 * exch_norm2) / max(1e-15, eri_norm2))
    return {
        "V_unw": V_unw,
        "V_tilde": V_tilde,
        "evals_tilde": evals_tilde,
        "evals_unw": evals_unw,
        "min_eig_tilde": min_eig_tilde,
        "min_eig_unw": min_eig_unw,
        "num_rank": int(np.sum(evals_tilde > 1e-8)),
        "norm": norm,
        "h1_frac": h1_norm2 / max(1e-30, tot_norm2),
        "eri_frac": eri_norm2 / max(1e-30, tot_norm2),
        "eri_offdiag_frac": offdiag_frac,
    }


def cluster_charge_moments_fci(rdm: dict, orbitals=(0, 1)):
    """Compute <n_A>, <n_A^2>, Var(n_A) from FCI 1- and 2-RDMs."""
    n1 = float(sum(rdm["grad_h1"][p, p] for p in orbitals))
    pair_sum = float(sum(rdm["grad_eri"][p, p, q, q] for p in orbitals for q in orbitals))
    n2 = n1 + 2.0 * pair_sum
    var = max(0.0, n2 - n1 * n1)
    return {"mean_nA": n1, "second_moment_nA2": n2, "var_nA": var}


def audit_all_tiers_cli(
    name: str,
    fcidump_path: str,
    cert_prefix: str,
    K: int,
    na: int,
    nb: int,
    two_s: int,
    mode: str,
    e_core: float,
    h1: np.ndarray,
    eri: np.ndarray,
    cli: chem_sdk.ChemOracle,
    max_iters: int = 3500,
    tol: float = 1e-7,
):
    """Exact FCI + 4-tier CLI solve + exact-verify audit on T1, T2, T12."""
    spec = spectral_cone_stats(h1, eri)
    assert spec["min_eig_tilde"] >= -1e-11, f"{name} V_tilde < 0: {spec['min_eig_tilde']}"
    assert spec["min_eig_unw"] >= -1e-11, f"{name} V_unw < 0: {spec['min_eig_unw']}"

    fci = fo.solve_fci_sector(K, na, nb, e_core, h1, eri, two_s=two_s, n_eigs=4)
    rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)
    norm = spec["norm"]

    tiers = {}
    for tname, tslug, use_t1, use_t2, do_ev in [
        ("DQG", "dqg", False, False, False),
        ("DQG+T1", "t1", True, False, True),
        ("DQG+T2", "t2", False, True, True),
        ("DQG+T1+T2", "t12", True, True, True),
    ]:
        cert_path = os.path.join(EXP_DIR, f"{cert_prefix}_{tslug}.json") if do_ev else None
        br = cli.bracket(
            fcidump_path,
            mode=mode,
            two_s=two_s,
            t1=use_t1,
            t1_orbs=K if use_t1 else 0,
            t2=use_t2,
            t2_orbs=K if use_t2 else 0,
            max_iters=max_iters,
            tol=tol,
            max_dets=2048,
            threads=1,
            cert_path=cert_path,
        )
        ev_info = None
        if do_ev:
            ev = cli.exact_verify(cert_path, fcidump_path)
            assert ev.get("ok") and ev.get("returncode", -1) == 0, f"exact-verify failed {cert_path}: {ev}"
            assert ev.get("all_blocks_sdd"), f"SDD failed {cert_path}: {ev}"
            ev_info = {
                "E_lb_int_chol": float(ev["E_lb_int_chol"]),
                "E_lb_weyl": float(ev["E_lb_weyl"]),
                "all_blocks_sdd": bool(ev["all_blocks_sdd"]),
                "scale_D": ev.get("scale_D"),
            }
        gap_raw_mha = float(fci["E0"] - br["E_lb_cert"]) * 1000.0
        tiers[tname] = {
            "E_lb_cert": float(br["E_lb_cert"]),
            "E_primal": float(br.get("E_primal", br["E_lb_cert"])),
            "E_ub": float(br["E_ub"]),
            "gap_raw_mha": gap_raw_mha,
            "gap_frob_mha": gap_raw_mha / norm,
            "iters": int(br.get("iters", 0)),
            "cert_path": cert_path,
            "exact_verify": ev_info,
        }

    return {
        "name": name,
        "K": K,
        "na": na,
        "nb": nb,
        "nelec": na + nb,
        "two_s": two_s,
        "mode": mode,
        "norm": norm,
        "num_rank": spec["num_rank"],
        "min_eig_Vtilde": spec["min_eig_tilde"],
        "min_eig_Vunw": spec["min_eig_unw"],
        "h1_frac": spec["h1_frac"],
        "eri_frac": spec["eri_frac"],
        "eri_offdiag_frac": spec["eri_offdiag_frac"],
        "fci_evals": [float(x) for x in fci["evals"]],
        "fci_E0": float(fci["E0"]),
        "fci_E1": float(fci["E1"]),
        "fci_S1": float(rdm["S1"]),
        "tiers": tiers,
        "fci": fci,
        "rdm": rdm,
        "spec": spec,
    }


def json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items() if k not in ("fci", "rdm", "spec", "h1", "B", "eri", "V_unw", "V_tilde")}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ==============================================================================
# PART 1: Analytical Rank-1 Coulomb PSD Integrality + Full 3-POS Dissection
# ==============================================================================
def run_part1_rank1_dissection(cli: chem_sdk.ChemOracle):
    print("\n=== Part 1: Analytical Rank-1 Coulomb PSD Integrality & Full 3-POS Dissection ===", flush=True)
    K = 4
    beta = np.array([-1.0, -1.0, 2.0, 2.0], dtype=np.float64)
    h1_zero = np.zeros((K, K), dtype=np.float64)
    eri_raw = np.zeros((K, K, K, K), dtype=np.float64)
    for p in range(K):
        for q in range(K):
            eri_raw[p, p, q, q] = beta[p] * beta[q]
    frob_raw = math.sqrt(float(np.sum(eri_raw * eri_raw)))
    assert abs(frob_raw - 10.0) < 1e-12, f"Expected Frobenius norm 10.0, got {frob_raw}"
    eri_exact = eri_raw / frob_raw

    r1_exact_path = os.path.join(EXP_DIR, "discovered_rank1_exact_m1m122.fcidump")
    chem_sdk.write_fcidump(r1_exact_path, K, 4, 0, 0.0, h1_zero, eri_exact)

    exact_audit = audit_all_tiers_cli(
        "rank1_exact_m1m122",
        r1_exact_path,
        "cert_exp4_rank1_exact",
        K, 2, 2, 0, "a", 0.0, h1_zero, eri_exact, cli,
    )
    moments_exact = cluster_charge_moments_fci(exact_audit["rdm"], orbitals=(0, 1))
    print(
        f"  [Exact beta=(-1,-1,2,2)] FCI E0={exact_audit['fci_E0']:.6f}, E1={exact_audit['fci_E1']:.6f}, "
        f"DQG={exact_audit['tiers']['DQG']['gap_frob_mha']:.4f}, "
        f"T1={exact_audit['tiers']['DQG+T1']['gap_frob_mha']:.4f}, "
        f"T2={exact_audit['tiers']['DQG+T2']['gap_frob_mha']:.4f}, "
        f"T12={exact_audit['tiers']['DQG+T1+T2']['gap_frob_mha']:.4f} mHa | "
        f"FCI Var(nA)={moments_exact['var_nA']:.6f}",
        flush=True,
    )

    # Dissect discovered_coulomb_k4_r1.fcidump in its Rank-1 B eigenbasis
    r1_disc_path = os.path.join(EXP_DIR, "discovered_coulomb_k4_r1.fcidump")
    r1_data = chem_sdk.read_fcidump(r1_disc_path)
    pairs, _, s_vec, _ = e3.sym_pairs(K)
    spec_disc = spectral_cone_stats(r1_data["h1"], r1_data["eri"])
    evals_Vt = spec_disc["evals_tilde"]
    _, evecs_Vt = np.linalg.eigh(spec_disc["V_tilde"])
    lam_top = float(evals_Vt[-1])
    v_top = evecs_Vt[:, -1]
    B_mat = np.zeros((K, K), dtype=np.float64)
    scale_top = math.sqrt(max(0.0, lam_top))
    for a, (p, q) in enumerate(pairs):
        val = scale_top * float(v_top[a]) / float(s_vec[a])
        B_mat[p, q] = val
        B_mat[q, p] = val
    evals_B, U_B = np.linalg.eigh(B_mat)
    # Fix sign so evals_B matches [-, -, +, +] ordered ascending (negatives first)
    idx_sort = np.argsort(evals_B)
    evals_B = evals_B[idx_sort]
    U_B = U_B[:, idx_sort]
    if evals_B[0] > 0:
        evals_B = -evals_B
        idx_sort = np.argsort(evals_B)
        evals_B = evals_B[idx_sort]
        U_B = U_B[:, idx_sort]

    h1_eig, eri_eig = chem_sdk.rotate_integrals(r1_data["h1"], r1_data["eri"], U_B)
    max_h1_offdiag = float(np.max(np.abs(h1_eig - np.diag(np.diag(h1_eig)))))
    fci_disc_eig = fo.solve_fci_sector(K, 2, 2, 0.0, h1_eig, eri_eig, two_s=0, n_eigs=4)
    rdm_disc_eig = fo.compute_fci_12rdm_and_grad(fci_disc_eig, state_idx=0)
    mom_disc = cluster_charge_moments_fci(rdm_disc_eig, orbitals=(0, 1))
    beta_ratios = evals_B / abs(float(evals_B[0]))

    print(
        f"  [Dissect k4_r1] top Vt_eig={lam_top:.6f}, max|other_Vt|={float(np.max(np.abs(evals_Vt[:-1]))):.2e} | "
        f"evals_B={[round(float(x), 6) for x in evals_B]} (ratios {[round(float(x), 4) for x in beta_ratios]}) | "
        f"max|h1_offdiag|={max_h1_offdiag:.3e} | FCI E0={fci_disc_eig['E0']:.6f}, E1={fci_disc_eig['E1']:.6f}",
        flush=True,
    )

    # Full 3-POS 401x401 fine grid on (m2, m3) at m1 = 2.5
    m1 = 2.5
    m2_grid = np.linspace(6.25, 6.50, 401)
    m3_grid = np.linspace(15.5, 17.0, 401)
    best_3pos = None
    n_feasible = 0
    for m2 in m2_grid:
        for m3 in m3_grid:
            D3 = m3 - 3.0 * m2 + 2.0 * m1
            Q3 = 24.0 - 26.0 * m1 + 9.0 * m2 - m3
            F3_det = m1 * m3 - m2 * m2
            E3_det = (4.0 - m1) * (4.0 * m2 - m3) - (4.0 * m1 - m2) ** 2
            mixE3 = 12.0 * m1 - 7.0 * m2 + m3
            mixF3 = 5.0 * m2 - 4.0 * m1 - m3
            if min(D3, Q3, F3_det, E3_det, mixE3, mixF3) >= -1e-12:
                n_feasible += 1
                E8 = 36.0 * m2 - 180.0 * m1 + 192.0
                if best_3pos is None or E8 < best_3pos["E8"] - 1e-12:
                    best_3pos = {
                        "m1": m1,
                        "m2": float(m2),
                        "m3": float(m3),
                        "var": float(m2 - m1 * m1),
                        "E8": float(E8),
                        "E_norm_ha": float(E8 / 80.0),
                        "D3": float(D3),
                        "Q3": float(Q3),
                        "F3_det": float(F3_det),
                        "E3_det": float(E3_det),
                        "mixE3": float(mixE3),
                        "mixF3": float(mixF3),
                    }

    # Exact closed-form extremum of F3_det >= 0 & Q3 >= 0: 2*m2^2 - 45*m2 + 205 <= 0
    m2_exact_bound = (45.0 - math.sqrt(385.0)) / 4.0
    m3_exact_bound = 9.0 * m2_exact_bound - 41.0
    E8_exact_bound = 36.0 * m2_exact_bound - 180.0 * m1 + 192.0
    E_exact_3pos = E8_exact_bound / 80.0

    # Check rational witness (m1=2.5, m2=6.35, m3=16.13) and diagonal 3-marginal failure at m2=6.25
    rat_m2, rat_m3 = 6.35, 16.13
    rat_witness = {
        "m1": m1,
        "m2": rat_m2,
        "m3": rat_m3,
        "D3": rat_m3 - 3.0 * rat_m2 + 2.0 * m1,
        "Q3": 24.0 - 26.0 * m1 + 9.0 * rat_m2 - rat_m3,
        "F3_det": m1 * rat_m3 - rat_m2 * rat_m2,
        "E3_det": (4.0 - m1) * (4.0 * rat_m2 - rat_m3) - (4.0 * m1 - rat_m2) ** 2,
        "mixE3": 12.0 * m1 - 7.0 * rat_m2 + rat_m3,
        "mixF3": 5.0 * rat_m2 - 4.0 * m1 - rat_m3,
        "var": rat_m2 - m1 * m1,
        "E8": 36.0 * rat_m2 - 180.0 * m1 + 192.0,
        "E_norm_ha": (36.0 * rat_m2 - 180.0 * m1 + 192.0) / 80.0,
    }
    diag3_m2, diag3_m3 = 6.25, 15.0
    diag3_witness = {
        "m1": m1,
        "m2": diag3_m2,
        "m3": diag3_m3,
        "probs_r3_r2_r1_r0": [5.0 / 96.0, 25.0 / 96.0, 5.0 / 96.0, 1.0 / 96.0],
        "F3_det": m1 * diag3_m3 - diag3_m2 * diag3_m2,
        "var": diag3_m2 - m1 * m1,
        "E_norm_ha": (36.0 * diag3_m2 - 180.0 * m1 + 192.0) / 80.0,
    }

    print(
        f"  [Full 3-POS] feasible={n_feasible}/160801 | grid_min_m2={best_3pos['m2']:.6f} "
        f"(E_3POS={best_3pos['E_norm_ha']:.6f}) vs exact_bound={m2_exact_bound:.8f} "
        f"(E_3POS*={E_exact_3pos:.6f} Ha, FCI-3POS={(-0.3 - E_exact_3pos)*1000:.4f} mHa, "
        f"3POS-DQG={(E_exact_3pos - (-0.4125))*1000:.4f} mHa)",
        flush=True,
    )

    return {
        "exact_r1_model": {
            **json_sanitize(exact_audit),
            "fcidump_path": r1_exact_path,
            "beta": beta.tolist(),
            "analytical_DQG_E": -0.4125,
            "analytical_FCI_E": -0.3,
            "analytical_gap_mha": 112.5,
            "fci_moments_nA": moments_exact,
        },
        "discovered_k4_r1_dissection": {
            "evals_Vt": [float(x) for x in evals_Vt],
            "top_Vt_eig": lam_top,
            "evals_B": [float(x) for x in evals_B],
            "beta_ratios": [float(x) for x in beta_ratios],
            "h1_eig_diag": [float(x) for x in np.diag(h1_eig)],
            "max_h1_eig_offdiag": max_h1_offdiag,
            "h1_frob": float(np.linalg.norm(h1_eig)),
            "fci_evals_eig": [float(x) for x in fci_disc_eig["evals"]],
            "fci_moments_nA": mom_disc,
        },
        "three_pos_dissection": {
            "grid_size": "401x401",
            "n_feasible": n_feasible,
            "grid_min": best_3pos,
            "exact_analytical_min": {
                "m2_min": m2_exact_bound,
                "m3_at_min": m3_exact_bound,
                "var_min": m2_exact_bound - m1 * m1,
                "E8_min": E8_exact_bound,
                "E_norm_ha": E_exact_3pos,
                "gap_DQG_to_3POS_mha": (E_exact_3pos - (-0.4125)) * 1000.0,
                "gap_3POS_to_FCI_mha": (-0.3 - E_exact_3pos) * 1000.0,
            },
            "rational_witness_3pos": rat_witness,
            "diagonal_3marginal_witness_m2_625": diag3_witness,
        },
    }


# ==============================================================================
# PART 2: Physical Coulomb PSD T1 > T2 Multi-Sector Phase Diagram
# ==============================================================================
def run_part2_t1_gt_t2_phase_diagram(cli: chem_sdk.ChemOracle):
    print("\n=== Part 2: Physical Coulomb PSD T1 > T2 Multi-Sector Phase Diagram (6 Sectors) ===", flush=True)
    sectors = [
        {"name": "k4_3e4o_s1", "K": 4, "na": 2, "nb": 1, "two_s": 1, "mode": "b", "R": 10, "hub": False, "seed": 4100},
        {"name": "k4_4e4o_s0", "K": 4, "na": 2, "nb": 2, "two_s": 0, "mode": "a", "R": 10, "hub": False, "seed": 4200},
        {"name": "k4_4e4o_s2", "K": 4, "na": 3, "nb": 1, "two_s": 2, "mode": "b", "R": 10, "hub": False, "seed": 4300},
        {"name": "k5_3e5o_s1", "K": 5, "na": 2, "nb": 1, "two_s": 1, "mode": "b", "R": 15, "hub": True,  "seed": 4400},
        {"name": "k5_4e5o_s2", "K": 5, "na": 3, "nb": 1, "two_s": 2, "mode": "b", "R": 15, "hub": True,  "seed": 4500},
        {"name": "k5_5e5o_s3", "K": 5, "na": 4, "nb": 1, "two_s": 3, "mode": "b", "R": 15, "hub": True,  "seed": 4600},
    ]

    winners = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for sec in sectors:
            win = e3.run_psd_hunt(
                pool,
                f"T1>T2_{sec['name']}",
                "t1_gt_t2",
                sec["K"], sec["na"], sec["nb"], sec["two_s"], sec["mode"], sec["R"],
                n_seeds=4, n_steps=22,
                cold_iters=600, warm_iters=450, tol=2e-6, eta0=0.20,
                base_seed=sec["seed"], hubbard_seed0=sec["hub"],
            )
            winners[sec["name"]] = (sec, win)

    audited_sectors = []
    best_candidate = None
    tmp_fcidump = os.path.join(EXP_DIR, "_tmp_exp4_sector_audit.fcidump")

    for sec in sectors:
        s_name = sec["name"]
        _, win = winners[s_name]
        h1, eri = win["h1"], win["eri"]
        chem_sdk.write_fcidump(tmp_fcidump, sec["K"], sec["na"] + sec["nb"], sec["two_s"], 0.0, h1, eri)
        au = audit_all_tiers_cli(
            s_name, tmp_fcidump, f"cert_exp4_{s_name}",
            sec["K"], sec["na"], sec["nb"], sec["two_s"], sec["mode"],
            0.0, h1, eri, cli,
        )
        t1_tier = au["tiers"]["DQG+T1"]
        t2_tier = au["tiers"]["DQG+T2"]
        t12_tier = au["tiers"]["DQG+T1+T2"]
        dqg_tier = au["tiers"]["DQG"]

        margin_primal_mha = (t1_tier["E_lb_cert"] - max(t2_tier["E_lb_cert"], t2_tier["E_primal"])) * 1000.0
        margin_lb_mha = (t1_tier["E_lb_cert"] - t2_tier["E_lb_cert"]) * 1000.0
        margin_intchol_mha = (t1_tier["exact_verify"]["E_lb_int_chol"] - t2_tier["exact_verify"]["E_lb_int_chol"]) * 1000.0
        margin_weyl_mha = (t1_tier["exact_verify"]["E_lb_weyl"] - t2_tier["exact_verify"]["E_lb_weyl"]) * 1000.0

        row = {
            **json_sanitize(au),
            "search_seed": int(win["seed_idx"]),
            "search_step": int(win.get("step", -1)),
            "margin_T1_minus_T2_lb_mha": margin_lb_mha,
            "margin_T1_minus_T2_strict_primal_mha": margin_primal_mha,
            "margin_intchol_mha": margin_intchol_mha,
            "margin_weyl_mha": margin_weyl_mha,
            "gain_DQG_to_T12_mha": (t12_tier["E_lb_cert"] - dqg_tier["E_lb_cert"]) * 1000.0,
        }
        audited_sectors.append(row)
        print(
            f"  [Audit {s_name:10s}] DQG_gap={dqg_tier['gap_frob_mha']:6.3f} T1_gap={t1_tier['gap_frob_mha']:6.3f} "
            f"T2_gap={t2_tier['gap_frob_mha']:6.3f} T12_gap={t12_tier['gap_frob_mha']:6.3f} | "
            f"T1-T2(intChol={margin_intchol_mha:+.3f}, Weyl={margin_weyl_mha:+.3f}, LB={margin_lb_mha:+.3f}) mHa",
            flush=True,
        )

        strict_margin = min(margin_intchol_mha, margin_weyl_mha)
        if best_candidate is None or strict_margin > best_candidate["strict_margin"]:
            best_candidate = {
                "strict_margin": strict_margin,
                "row": row,
                "sec": sec,
                "h1": h1.copy(),
                "eri": eri.copy(),
                "t1_cert": t1_tier["cert_path"],
            }

    if os.path.exists(tmp_fcidump):
        os.remove(tmp_fcidump)

    witness_export = {"exported": False}
    if best_candidate is not None and best_candidate["row"]["margin_intchol_mha"] > 0.5 and best_candidate["row"]["margin_weyl_mha"] > 0.5:
        out_fcidump = os.path.join(EXP_DIR, "discovered_coulomb_t1_beats_t2.fcidump")
        sec_b = best_candidate["sec"]
        chem_sdk.write_fcidump(
            out_fcidump, sec_b["K"], sec_b["na"] + sec_b["nb"], sec_b["two_s"],
            0.0, best_candidate["h1"], best_candidate["eri"],
        )
        # Re-solve T1 cert cleanly against the exported fcidump path and attempt emit-lean
        out_cert_t1 = os.path.join(EXP_DIR, "cert_exp4_t1_beats_t2_t1.json")
        out_cert_t2 = os.path.join(EXP_DIR, "cert_exp4_t1_beats_t2_t2.json")
        cli.bracket(
            out_fcidump, mode=sec_b["mode"], two_s=sec_b["two_s"],
            t1=True, t1_orbs=sec_b["K"], t2=False,
            max_iters=3500, tol=1e-7, max_dets=2048, threads=1, cert_path=out_cert_t1,
        )
        cli.bracket(
            out_fcidump, mode=sec_b["mode"], two_s=sec_b["two_s"],
            t1=False, t2=True, t2_orbs=sec_b["K"],
            max_iters=3500, tol=1e-7, max_dets=2048, threads=1, cert_path=out_cert_t2,
        )
        ev_t1 = cli.exact_verify(out_cert_t1, out_fcidump)
        ev_t2 = cli.exact_verify(out_cert_t2, out_fcidump)

        lean_out = os.path.join(ROOT, "lean", "CertCoulombT1BeatsT2.lean")
        cli_bin = os.path.join(ROOT, "build", "chem_cli")
        p_emit = subprocess.run(
            [cli_bin, "emit-lean", out_fcidump, out_cert_t1, "--out", lean_out, "--name", "CertCoulombT1BeatsT2"],
            capture_output=True, text=True, check=False,
        )
        emit_ok = (p_emit.returncode == 0)
        leanchecker_ok = False
        lean_msg = (p_emit.stdout + p_emit.stderr).strip()
        if emit_ok:
            env = dict(os.environ)
            env["PATH"] = os.path.join(os.path.expanduser("~"), ".elan", "bin") + ":" + env.get("PATH", "")
            env["LEAN_PATH"] = "/Users/kuangtalin/Documents/FLOW/build/lean:/Users/kuangtalin/Documents/CHEM/build"
            olean_out = os.path.join(ROOT, "build", "CertCoulombT1BeatsT2.olean")
            p_lean = subprocess.run(
                ["lean", "-DwarningAsError=true", "-o", olean_out, lean_out],
                env=env, capture_output=True, text=True, check=False,
            )
            if p_lean.returncode == 0:
                p_chk = subprocess.run(
                    ["leanchecker", "CertCoulombT1BeatsT2"],
                    env=env, capture_output=True, text=True, check=False,
                )
                leanchecker_ok = (p_chk.returncode == 0)
                lean_msg = (p_chk.stdout + p_chk.stderr).strip()
            else:
                lean_msg = (p_lean.stdout + p_lean.stderr).strip()
                if os.path.exists(lean_out):
                    os.remove(lean_out)

        witness_export = {
            "exported": True,
            "sector": sec_b["name"],
            "fcidump_path": out_fcidump,
            "cert_t1_path": out_cert_t1,
            "cert_t2_path": out_cert_t2,
            "ev_t1": ev_t1,
            "ev_t2": ev_t2,
            "margin_intchol_mha": (float(ev_t1["E_lb_int_chol"]) - float(ev_t2["E_lb_int_chol"])) * 1000.0,
            "margin_weyl_mha": (float(ev_t1["E_lb_weyl"]) - float(ev_t2["E_lb_weyl"])) * 1000.0,
            "emit_lean_ok": emit_ok,
            "leanchecker_ok": leanchecker_ok,
            "lean_path": lean_out if leanchecker_ok else None,
            "lean_msg": lean_msg,
        }
        print(
            f"  [Export Winner] {sec_b['name']} -> {out_fcidump} | "
            f"intChol={witness_export['margin_intchol_mha']:+.3f} mHa, "
            f"Weyl={witness_export['margin_weyl_mha']:+.3f} mHa | "
            f"emit_lean={emit_ok}, leanchecker={leanchecker_ok}",
            flush=True,
        )

    return {"sectors": audited_sectors, "witness_export": witness_export}


# ==============================================================================
# PART 3: Quantitative Phase Boundary (Convex Interpolation & Kinetic Tunneling)
# ==============================================================================
def find_crossing(xs, ys, target):
    """Return first linearly interpolated x where ys drops from >= target to <= target."""
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 >= target >= y1:
            if abs(y0 - y1) < 1e-15:
                return float(xs[i])
            frac = (y0 - target) / (y0 - y1)
            return float(xs[i] + frac * (xs[i + 1] - xs[i]))
    return None


def _sweep_point_solve(sess_map, fci_K, fci_na, fci_nb, two_s, e_core, h1, eri, rdm_orbs=(0, 1)):
    spec = spectral_cone_stats(h1, eri)
    assert spec["min_eig_tilde"] >= -1e-11, f"Sweep PSD violated: {spec['min_eig_tilde']}"
    fci = fo.solve_fci_sector(fci_K, fci_na, fci_nb, e_core, h1, eri, two_s=two_s, n_eigs=3)
    rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)
    mom = cluster_charge_moments_fci(rdm, orbitals=rdm_orbs)

    gaps_raw = {}
    gaps_frob = {}
    elbs = {}
    norm = spec["norm"]
    for tname in ("DQG", "DQG+T1", "DQG+T2", "DQG+T1+T2"):
        res = sess_map[tname].solve(e_core, h1, eri, max_iters=3500, tol=1e-7, warm_start=False, return_grad=False)
        elbs[tname] = float(res["E_lb_cert"])
        g_raw = float(fci["E0"] - res["E_lb_cert"]) * 1000.0
        gaps_raw[tname] = g_raw
        gaps_frob[tname] = g_raw / norm

    return {
        "norm": norm,
        "min_eig_Vtilde": spec["min_eig_tilde"],
        "min_eig_Vunw": spec["min_eig_unw"],
        "h1_frac": spec["h1_frac"],
        "eri_frac": spec["eri_frac"],
        "eri_offdiag_frac": spec["eri_offdiag_frac"],
        "fci_E0": float(fci["E0"]),
        "fci_E1": float(fci["E1"]),
        "fci_S1": float(rdm["S1"]),
        "fci_var_nA": mom["var_nA"],
        "fci_mean_nA": mom["mean_nA"],
        "E_lb": elbs,
        "gaps_raw_mha": gaps_raw,
        "gaps_frob_mha": gaps_frob,
    }


def run_part3_phase_boundary(r1_exact_path: str):
    print("\n=== Part 3: Quantitative Phase Boundary (3A Convex PSD & 3B Kinetic Tunneling) ===", flush=True)
    d_r1 = chem_sdk.read_fcidump(r1_exact_path)
    d_mol = chem_sdk.read_fcidump(os.path.join(EXP_DIR, "discovered_mol3d_k4_worst.fcidump"))
    K = 4

    with chem_sdk.ChemFastSession(K, 4, ms2=0, mode="a", two_s=0, t1=False, t2=False, threads=1) as sdqg, \
         chem_sdk.ChemFastSession(K, 4, ms2=0, mode="a", two_s=0, t1=True, t1_orbs=K, t2=False, threads=1) as st1, \
         chem_sdk.ChemFastSession(K, 4, ms2=0, mode="a", two_s=0, t1=False, t2=True, t2_orbs=K, threads=1) as st2, \
         chem_sdk.ChemFastSession(K, 4, ms2=0, mode="a", two_s=0, t1=True, t1_orbs=K, t2=True, t2_orbs=K, threads=1) as st12:
        sess_map = {"DQG": sdqg, "DQG+T1": st1, "DQG+T2": st2, "DQG+T1+T2": st12}

        # 3A: Convex PSD Interpolation lambda in [0, 1] (21 points)
        lambdas = np.linspace(0.0, 1.0, 21)
        sweep_3a = []
        for lam in lambdas:
            lam_f = float(lam)
            ec = (1.0 - lam_f) * float(d_r1["e_core"]) + lam_f * float(d_mol["e_core"])
            h1 = (1.0 - lam_f) * d_r1["h1"] + lam_f * d_mol["h1"]
            eri = (1.0 - lam_f) * d_r1["eri"] + lam_f * d_mol["eri"]
            pt = _sweep_point_solve(sess_map, K, 2, 2, 0, ec, h1, eri)
            pt["lambda"] = lam_f
            sweep_3a.append(pt)
            print(
                f"  [3A lam={lam_f:4.2f}] h1_frac={pt['h1_frac']*100:5.1f}% offdiag={pt['eri_offdiag_frac']*100:5.1f}% | "
                f"DQG={pt['gaps_frob_mha']['DQG']:7.4f} T12={pt['gaps_frob_mha']['DQG+T1+T2']:7.4f} mHa | "
                f"min_Vt={pt['min_eig_Vtilde']:+.2e} | Var(nA)={pt['fci_var_nA']:.4f}",
                flush=True,
            )

        xs_3a = [p["lambda"] for p in sweep_3a]
        g0_3a = sweep_3a[0]["gaps_frob_mha"]["DQG+T1+T2"]
        ys_3a_frob = [p["gaps_frob_mha"]["DQG+T1+T2"] for p in sweep_3a]
        ys_3a_dqg = [p["gaps_frob_mha"]["DQG"] for p in sweep_3a]
        crossings_3a = {
            "t12_frob_50pct": find_crossing(xs_3a, ys_3a_frob, 0.5 * g0_3a),
            "t12_frob_10pct": find_crossing(xs_3a, ys_3a_frob, 0.1 * g0_3a),
            "t12_frob_1mha": find_crossing(xs_3a, ys_3a_frob, 1.0),
            "dqg_frob_50pct": find_crossing(xs_3a, ys_3a_dqg, 0.5 * sweep_3a[0]["gaps_frob_mha"]["DQG"]),
            "dqg_frob_10pct": find_crossing(xs_3a, ys_3a_dqg, 0.1 * sweep_3a[0]["gaps_frob_mha"]["DQG"]),
            "dqg_frob_1mha": find_crossing(xs_3a, ys_3a_dqg, 1.0),
        }
        print(f"  [3A Crossings (T12 frob)] 50%={crossings_3a['t12_frob_50pct']}, 10%={crossings_3a['t12_frob_10pct']}, 1mHa={crossings_3a['t12_frob_1mha']}", flush=True)

        # 3B: Pure Kinetic Inter-Cluster Tunneling tau in [0, 1.5] (21 points)
        T_hop = np.zeros((K, K), dtype=np.float64)
        T_hop[0, 2] = T_hop[2, 0] = -1.0
        T_hop[1, 3] = T_hop[3, 1] = -1.0
        taus = np.linspace(0.0, 1.5, 21)
        sweep_3b = []
        for tau in taus:
            tau_f = float(tau)
            h1_t = tau_f * T_hop
            pt = _sweep_point_solve(sess_map, K, 2, 2, 0, 0.0, h1_t, d_r1["eri"])
            pt["tau"] = tau_f
            sweep_3b.append(pt)
            print(
                f"  [3B tau={tau_f:5.3f}] h1_frac={pt['h1_frac']*100:5.1f}% Var(nA)={pt['fci_var_nA']:.4f} | "
                f"T12_frob={pt['gaps_frob_mha']['DQG+T1+T2']:7.4f} mHa (raw={pt['gaps_raw_mha']['DQG+T1+T2']:7.4f}) | "
                f"DQG_frob={pt['gaps_frob_mha']['DQG']:7.4f}",
                flush=True,
            )

        xs_3b = [p["tau"] for p in sweep_3b]
        g0_3b_frob = sweep_3b[0]["gaps_frob_mha"]["DQG+T1+T2"]
        g0_3b_raw = sweep_3b[0]["gaps_raw_mha"]["DQG+T1+T2"]
        ys_3b_frob = [p["gaps_frob_mha"]["DQG+T1+T2"] for p in sweep_3b]
        ys_3b_raw = [p["gaps_raw_mha"]["DQG+T1+T2"] for p in sweep_3b]
        crossings_3b = {
            "t12_frob_50pct": find_crossing(xs_3b, ys_3b_frob, 0.5 * g0_3b_frob),
            "t12_frob_10pct": find_crossing(xs_3b, ys_3b_frob, 0.1 * g0_3b_frob),
            "t12_frob_1mha": find_crossing(xs_3b, ys_3b_frob, 1.0),
            "t12_raw_50pct": find_crossing(xs_3b, ys_3b_raw, 0.5 * g0_3b_raw),
            "t12_raw_10pct": find_crossing(xs_3b, ys_3b_raw, 0.1 * g0_3b_raw),
            "t12_raw_1mha": find_crossing(xs_3b, ys_3b_raw, 1.0),
        }
        print(
            f"  [3B Crossings] frob(50%={crossings_3b['t12_frob_50pct']}, 10%={crossings_3b['t12_frob_10pct']}, 1mHa={crossings_3b['t12_frob_1mha']}) | "
            f"raw(50%={crossings_3b['t12_raw_50pct']}, 10%={crossings_3b['t12_raw_10pct']}, 1mHa={crossings_3b['t12_raw_1mha']})",
            flush=True,
        )

    return {
        "sweep_3a_convex_psd": {"points": sweep_3a, "crossings": crossings_3a},
        "sweep_3b_kinetic_tunneling": {"points": sweep_3b, "crossings": crossings_3b},
    }


def main():
    t0 = time.time()
    cli = chem_sdk.ChemOracle(os.path.join(ROOT, "build", "chem_cli"))

    part1 = run_part1_rank1_dissection(cli)
    part2 = run_part2_t1_gt_t2_phase_diagram(cli)
    part3 = run_part3_phase_boundary(part1["exact_r1_model"]["fcidump_path"])

    out = {
        "schema": "chem.exp4.theory_trilogy.v1",
        "elapsed_sec": float(time.time() - t0),
        "part1_rank1_integrality_and_3pos": part1,
        "part2_coulomb_psd_t1_gt_t2_phase_diagram": part2,
        "part3_phase_boundary": part3,
    }
    out_path = os.path.join(EXP_DIR, "exp4_theory_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[DONE] Saved Frontier 4 Theory Trilogy results to {out_path} in {out['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
