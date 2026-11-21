#!/usr/bin/env python3
"""
Frontier 1: Graph-Topology Relaxation Law & Energy-Observable-Gap Decoupling.

Experiments:
  1A: Topology Survey across:
      - All 6 non-isomorphic connected K=4 graphs (P4, C4, K13_star [2S=0 & 2S=2], Paw, Diamond, K4_tetra)
        for U in {2, 4, 6, 8} across DQG, DQG+T1, DQG+T2, DQG+T1+T2.
      - K=5 doublet suite (P5, C5, K14, W5, K5) at U in {4, 6}.
      - K=6 singlet suite (P6, C6, Ladder_2x3, TriLadder, Prism_6, Octahedron_6) at U=4.
  1B: Continuous Avoided-Crossing & Frustration Sweeps:
      - Sweep 1a: Clean 4-site Hubbard ring t' in [-2, +2] at U=6 (17 points), observable sdot:0:1.
      - Sweep 1b: Weakly symmetry-broken avoided crossing around t'=1.0 (asymmetry=0.02, 11 points), observable sdot:0:1.
      - Sweep 2: STO-3G H4 rectangularization (Rx=2.0, Ry in [1.60, 2.60] Bohr, 12 points), observable occ:1.
      Verifies containment + Theorem C15 feasibility/width floor with 0 violations and fits log(W) vs log(gap).

Outputs: /Users/kuangtalin/Documents/CHEM/experiments/exp1_results.json
"""

import json
import math
import os
import sys
import tempfile
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracles"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import chem_sdk
import gen_benchmarks as gb
import fci_exact_oracle as fo


def build_hubbard(K: int, edges, U: float, hopping: float = 1.0):
    h1 = np.zeros((K, K), dtype=np.float64)
    for u, v in edges:
        h1[u, v] = h1[v, u] = -hopping
    eri = np.zeros((K, K, K, K), dtype=np.float64)
    for p in range(K):
        eri[p, p, p, p] = U
    return h1, eri


def graph_invariants(K: int, edges):
    adj = np.zeros((K, K), dtype=np.float64)
    edge_set = set()
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1.0
        edge_set.add((min(u, v), max(u, v)))
    tris = 0
    for i in range(K):
        for j in range(i + 1, K):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, K):
                if (i, k) in edge_set and (j, k) in edge_set:
                    tris += 1
    # Check bipartite via 2-coloring BFS
    color = [-1] * K
    bipartite = True
    for s in range(K):
        if color[s] != -1:
            continue
        color[s] = 0
        q = [s]
        while q:
            u = q.pop(0)
            for v in range(K):
                if adj[u, v] > 0:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
                    elif color[v] == color[u]:
                        bipartite = False
    evals = np.linalg.eigvalsh(adj)
    return {
        "n_edges": len(edges),
        "n_triangles": tris,
        "bipartite": bipartite,
        "adj_lambda_min": float(evals[0]),
        "adj_lambda_max": float(evals[-1]),
    }


def solve_tiers(cli, fcidump_path, mode, two_s, K, max_iters=1200, tol=1e-6, threads=4):
    tiers = [
        ("DQG", False, False),
        ("DQG+T1", True, False),
        ("DQG+T2", False, True),
        ("DQG+T1+T2", True, True),
    ]
    out = {}
    for name, use_t1, use_t2 in tiers:
        res = cli.bracket(
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
            threads=threads,
        )
        out[name] = {
            "E_lb_cert": float(res["E_lb_cert"]),
            "E_ub": float(res["E_ub"]),
            "E_ci": float(res["E_ci"]),
            "E_hf0": float(res["E_hf0"]),
            "interval_width": float(res["interval_width"]),
            "iters": int(res.get("iters", 0)),
        }
    return out


def run_exp1a(cli, workdir):
    print("=== Experiment 1A: Graph-Topology Law (K=4, K=5, K=6) ===")
    k4_graphs = [
        ("P4", [(0, 1), (1, 2), (2, 3)], 0),
        ("C4", [(0, 1), (1, 2), (2, 3), (3, 0)], 0),
        ("K13_star_S0", [(0, 1), (0, 2), (0, 3)], 0),
        ("K13_star_S1", [(0, 1), (0, 2), (0, 3)], 2),
        ("Paw", [(0, 1), (1, 2), (2, 0), (0, 3)], 0),
        ("Diamond", [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 0),
        ("K4_tetra", [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 0),
    ]

    k4_records = []
    for gname, edges, two_s in k4_graphs:
        inv = graph_invariants(4, edges)
        mode = "a" if two_s == 0 else "b"
        na, nb = (2, 2) if two_s == 0 else (3, 1)
        for U in [2.0, 4.0, 6.0, 8.0]:
            h1, eri = build_hubbard(4, edges, U)
            fci = fo.solve_fci_sector(4, na, nb, 0.0, h1, eri, two_s=two_s, n_eigs=2)
            rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)
            fpath = os.path.join(workdir, f"k4_{gname}_U{int(U)}.fcidump")
            chem_sdk.write_fcidump(fpath, 4, 4, two_s, 0.0, h1, eri)
            tiers = solve_tiers(cli, fpath, mode, two_s, 4, max_iters=1400, tol=1e-7, threads=1)
            errs = {m: max(0.0, fci["E0"] - tiers[m]["E_lb_cert"]) for m in tiers}
            assert all(tiers[m]["E_lb_cert"] <= fci["E0"] + 1e-8 for m in tiers), f"Bound violation {gname} U={U}"
            rec = {
                "graph": gname, "K": 4, "nelec": 4, "two_s": two_s, "U": U,
                **inv,
                "E0_fci": fci["E0"], "E1_fci": fci["E1"], "gap_fci": fci["gap"], "S1_fci": rdm["S1"],
                "tiers": tiers,
                "errors": errs,
                "rel_capture_T1": 1.0 - errs["DQG+T1"] / max(1e-15, errs["DQG"]),
                "rel_capture_T2": 1.0 - errs["DQG+T2"] / max(1e-15, errs["DQG"]),
                "rel_capture_T12": 1.0 - errs["DQG+T1+T2"] / max(1e-15, errs["DQG"]),
                "winner_T1_gt_T2": errs["DQG+T1"] < errs["DQG+T2"] - 1e-7,
            }
            k4_records.append(rec)
            print(
                f"K=4 {gname:11s} U={U:.0f}: tris={inv['n_triangles']} S1={rdm['S1']:.3f} | "
                f"err DQG={errs['DQG']:.5f} T1={errs['DQG+T1']:.5f} T2={errs['DQG+T2']:.5f} T12={errs['DQG+T1+T2']:.5f}"
            )

    # K=5 Doublet suite (ne=5, na=3, nb=2, two_s=1)
    k5_graphs = [
        ("P5", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        ("C5", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
        ("K14_star", [(0, 1), (0, 2), (0, 3), (0, 4)]),
        ("W5_wheel", [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]),
        ("K5_complete", [(i, j) for i in range(5) for j in range(i + 1, 5)]),
    ]
    k5_records = []
    for gname, edges in k5_graphs:
        inv = graph_invariants(5, edges)
        for U in [4.0, 6.0]:
            h1, eri = build_hubbard(5, edges, U)
            fci = fo.solve_fci_sector(5, 3, 2, 0.0, h1, eri, two_s=1, n_eigs=2)
            rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)
            fpath = os.path.join(workdir, f"k5_{gname}_U{int(U)}.fcidump")
            chem_sdk.write_fcidump(fpath, 5, 5, 1, 0.0, h1, eri)
            tiers = solve_tiers(cli, fpath, "b", 1, 5, max_iters=1000, tol=1e-6, threads=2)
            errs = {m: max(0.0, fci["E0"] - tiers[m]["E_lb_cert"]) for m in tiers}
            rec = {
                "graph": gname, "K": 5, "nelec": 5, "two_s": 1, "U": U,
                **inv,
                "E0_fci": fci["E0"], "E1_fci": fci["E1"], "gap_fci": fci["gap"], "S1_fci": rdm["S1"],
                "tiers": tiers, "errors": errs,
                "rel_capture_T12": 1.0 - errs["DQG+T1+T2"] / max(1e-15, errs["DQG"]),
                "winner_T1_gt_T2": errs["DQG+T1"] < errs["DQG+T2"] - 1e-7,
            }
            k5_records.append(rec)
            print(
                f"K=5 {gname:11s} U={U:.0f}: tris={inv['n_triangles']} S1={rdm['S1']:.3f} | "
                f"err DQG={errs['DQG']:.5f} T1={errs['DQG+T1']:.5f} T2={errs['DQG+T2']:.5f} T12={errs['DQG+T1+T2']:.5f}"
            )

    # K=6 Singlet suite (ne=6, na=3, nb=3, two_s=0) at U=4
    k6_graphs = [
        ("P6", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        ("C6", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
        ("Ladder_2x3", [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]),
        ("TriLadder", [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5), (0, 4), (1, 5)]),
        ("Prism_6", [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]),
        ("Octahedron_6", [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4), (0, 5), (1, 5), (2, 5), (3, 5)]),
    ]
    k6_records = []
    for gname, edges in k6_graphs:
        inv = graph_invariants(6, edges)
        U = 4.0
        h1, eri = build_hubbard(6, edges, U)
        fci = fo.solve_fci_sector(6, 3, 3, 0.0, h1, eri, two_s=0, n_eigs=2)
        rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)
        fpath = os.path.join(workdir, f"k6_{gname}_U4.fcidump")
        chem_sdk.write_fcidump(fpath, 6, 6, 0, 0.0, h1, eri)
        tiers = solve_tiers(cli, fpath, "a", 0, 6, max_iters=800, tol=1e-6, threads=4)
        errs = {m: max(0.0, fci["E0"] - tiers[m]["E_lb_cert"]) for m in tiers}
        rec = {
            "graph": gname, "K": 6, "nelec": 6, "two_s": 0, "U": U,
            **inv,
            "E0_fci": fci["E0"], "E1_fci": fci["E1"], "gap_fci": fci["gap"], "S1_fci": rdm["S1"],
            "tiers": tiers, "errors": errs,
            "rel_capture_T12": 1.0 - errs["DQG+T1+T2"] / max(1e-15, errs["DQG"]),
            "winner_T1_gt_T2": errs["DQG+T1"] < errs["DQG+T2"] - 1e-7,
        }
        k6_records.append(rec)
        print(
            f"K=6 {gname:12s} U=4: tris={inv['n_triangles']} S1={rdm['S1']:.3f} | "
            f"err DQG={errs['DQG']:.5f} T1={errs['DQG+T1']:.5f} T2={errs['DQG+T2']:.5f} T12={errs['DQG+T1+T2']:.5f}"
        )

    return {"K4": k4_records, "K5": k5_records, "K6": k6_records}


def _check_c15_and_obs(cli, fpath, mode, two_s, K, obs_spec, fci_res):
    res_dqg = cli.observable(fpath, obs_spec, mode=mode, two_s=two_s, t1=False, t2=False, max_iters=600, tol=1e-6)
    res_t12 = cli.observable(fpath, obs_spec, mode=mode, two_s=two_s, t1=True, t1_orbs=K, t2=True, t2_orbs=K, max_iters=700, tol=1e-6)
    res_br_dqg = cli.bracket(fpath, mode=mode, two_s=two_s, t1=False, t2=False, max_iters=900, tol=1e-7)
    res_br_t12 = cli.bracket(fpath, mode=mode, two_s=two_s, t1=True, t1_orbs=K, t2=True, t2_orbs=K, max_iters=1100, tol=1e-7)

    o_info = fci_res["obs"][obs_spec]
    o_exact = o_info["O00"]
    assert res_dqg["O_lb_cert"] <= o_exact + 1e-8 <= res_dqg["O_ub_cert"] + 2e-8
    assert res_t12["O_lb_cert"] <= o_exact + 1e-8 <= res_t12["O_ub_cert"] + 2e-8

    gap = fci_res["gap"]
    m01 = o_info["m01"]
    m01_man = o_info["m01_manifold"]

    # Theorem C15 exact lower bound implied by variational defect dE = max(0, E_ub - E0):
    dE_var = max(0.0, res_br_t12["E_ub"] - fci_res["E0"])
    if gap > 1e-12:
        p = min(0.5, dE_var / gap)
        w_c15_var_lb = 4.0 * m01 * math.sqrt(p * (1.0 - p))
    else:
        w_c15_var_lb = 0.0
    assert w_c15_var_lb <= res_t12["interval_width"] + 1e-9, "Theorem C15 width violation!"

    return {
        "E0": fci_res["E0"],
        "E1": fci_res["E1"],
        "gap": gap,
        "O00": o_exact,
        "O11": o_info["O11"],
        "m01": m01,
        "m01_manifold": m01_man,
        "E_lb_dqg": float(res_br_dqg["E_lb_cert"]),
        "E_lb_t12": float(res_br_t12["E_lb_cert"]),
        "E_ub": float(res_br_t12["E_ub"]),
        "dE_relax_dqg": max(0.0, float(res_br_dqg["E_ub"] - res_br_dqg["E_lb_cert"])),
        "dE_relax_t12": max(0.0, float(res_br_t12["E_ub"] - res_br_t12["E_lb_cert"])),
        "O_dqg": [float(res_dqg["O_lb_cert"]), float(res_dqg["O_ub_cert"])],
        "W_dqg": float(res_dqg["interval_width"]),
        "O_t12": [float(res_t12["O_lb_cert"]), float(res_t12["O_ub_cert"])],
        "W_t12": float(res_t12["interval_width"]),
        "W_c15_var_lb": w_c15_var_lb,
    }


def _fit_power_law(gaps, widths):
    pts = [(g, w) for g, w in zip(gaps, widths) if g > 1e-6 and w > 1e-6]
    if len(pts) < 3:
        return {"slope": None, "r2": None, "n": len(pts)}
    xs = np.array([math.log(p[0]) for p in pts])
    ys = np.array([math.log(p[1]) for p in pts])
    slope, intercept = np.polyfit(xs, ys, 1)
    y_hat = slope * xs + intercept
    ss_res = float(np.sum((ys - y_hat) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "n": len(pts)}


def run_exp1b(cli, workdir):
    print("\n=== Experiment 1B: Continuous Avoided-Crossing & Observable Decoupling ===")
    # Sweep 1a: Clean frustrated 4-site Hubbard ring t' in [-2, +2], U=6
    ring_clean = []
    for idx, tp in enumerate(np.linspace(-2.0, 2.0, 17)):
        h1 = np.zeros((4, 4))
        for i in range(4):
            j = (i + 1) % 4
            h1[i, j] = h1[j, i] = -1.0
        h1[0, 2] = h1[2, 0] = -tp
        h1[1, 3] = h1[3, 1] = -tp
        eri = np.zeros((4, 4, 4, 4))
        for i in range(4): eri[i, i, i, i] = 6.0
        fci = fo.solve_fci_sector(4, 2, 2, 0.0, h1, eri, two_s=0, n_eigs=3, obs_specs=["sdot:0:1"])
        fpath = os.path.join(workdir, f"ring_clean_{idx}.fcidump")
        chem_sdk.write_fcidump(fpath, 4, 4, 0, 0.0, h1, eri)
        rec = _check_c15_and_obs(cli, fpath, "a", 0, 4, "sdot:0:1", fci)
        rec["t_prime"] = float(tp)
        ring_clean.append(rec)
        print(
            f"Ring clean tp={tp:+5.2f}: gap={rec['gap']:.5f} m01={rec['m01']:.4f} | "
            f"dE_t12={rec['dE_relax_t12']:.5f} W_dqg={rec['W_dqg']:.5f} W_t12={rec['W_t12']:.5f}"
        )

    # Sweep 1b: Avoided crossing around t' = 1.0 with slight asymmetry pert = 0.03
    ring_avoided = []
    pert = 0.03
    tps_dense = [0.60, 0.75, 0.85, 0.92, 0.96, 1.00, 1.04, 1.08, 1.15, 1.25, 1.40]
    for idx, tp in enumerate(tps_dense):
        h1 = np.zeros((4, 4))
        for i in range(4):
            j = (i + 1) % 4
            h1[i, j] = h1[j, i] = -1.0
        h1[0, 1] = h1[1, 0] = -1.0 - pert
        h1[0, 2] = h1[2, 0] = -tp
        h1[1, 3] = h1[3, 1] = -tp * (1.0 - pert)
        eri = np.zeros((4, 4, 4, 4))
        for i in range(4): eri[i, i, i, i] = 6.0
        fci = fo.solve_fci_sector(4, 2, 2, 0.0, h1, eri, two_s=0, n_eigs=3, obs_specs=["sdot:0:1"])
        fpath = os.path.join(workdir, f"ring_av_{idx}.fcidump")
        chem_sdk.write_fcidump(fpath, 4, 4, 0, 0.0, h1, eri)
        rec = _check_c15_and_obs(cli, fpath, "a", 0, 4, "sdot:0:1", fci)
        rec["t_prime"] = float(tp)
        rec["pert"] = pert
        ring_avoided.append(rec)
        print(
            f"Ring avoid tp={tp:+5.2f}: gap={rec['gap']:.5f} m01={rec['m01']:.4f} | "
            f"dE_t12={rec['dE_relax_t12']:.5f} W_dqg={rec['W_dqg']:.5f} W_t12={rec['W_t12']:.5f}"
        )

    # Sweep 2: STO-3G H4 rectangularization (Rx=2.0, Ry in [1.60, 2.60])
    h4_sweep = []
    rys = [1.60, 1.80, 1.90, 1.96, 1.99, 2.00, 2.01, 2.04, 2.10, 2.20, 2.40, 2.60]
    for idx, ry in enumerate(rys):
        coords = [(-1.0, -ry / 2.0, 0.0), (1.0, -ry / 2.0, 0.0), (1.0, ry / 2.0, 0.0), (-1.0, ry / 2.0, 0.0)]
        K, nel, enuc, S, Hcore, ERI = gb.build_sto3g([1, 1, 1, 1], coords)
        enuc, H_mo, ERI_mo = gb.rhf_canonical_mo(K, nel, enuc, S, Hcore, ERI)
        fci = fo.solve_fci_sector(4, 2, 2, enuc, H_mo, ERI_mo, two_s=0, n_eigs=3, obs_specs=["occ:1"])
        fpath = os.path.join(workdir, f"h4_ry_{idx}.fcidump")
        chem_sdk.write_fcidump(fpath, 4, 4, 0, enuc, H_mo, ERI_mo)
        rec = _check_c15_and_obs(cli, fpath, "a", 0, 4, "occ:1", fci)
        rec["Ry"] = float(ry)
        h4_sweep.append(rec)
        print(
            f"STO-3G H4 Ry={ry:.2f}: gap={rec['gap']:.5f} m01={rec['m01']:.4f} | "
            f"dE_t12={rec['dE_relax_t12']:.6f} W_dqg={rec['W_dqg']:.5f} W_t12={rec['W_t12']:.5f}"
        )

    # Fit log(W) vs log(gap) across non-zero-m01 points
    av_nz = [r for r in ring_avoided if r["m01"] > 0.05]
    h4_nz = [r for r in h4_sweep if r["m01"] > 0.1]
    fits = {
        "ring_avoided_W_t12_vs_gap": _fit_power_law([r["gap"] for r in av_nz], [r["W_t12"] for r in av_nz]),
        "ring_avoided_W_dqg_vs_gap": _fit_power_law([r["gap"] for r in av_nz], [r["W_dqg"] for r in av_nz]),
        "ring_avoided_W_t12_over_sqrt_dE_vs_gap": _fit_power_law(
            [r["gap"] for r in av_nz],
            [r["W_t12"] / (r["m01"] * math.sqrt(max(1e-12, r["dE_relax_t12"]))) for r in av_nz],
        ),
        "h4_W_t12_vs_gap": _fit_power_law([r["gap"] for r in h4_nz], [r["W_t12"] for r in h4_nz]),
        "h4_W_dqg_vs_gap": _fit_power_law([r["gap"] for r in h4_nz], [r["W_dqg"] for r in h4_nz]),
        "h4_W_t12_over_m_sqrt_dE_vs_gap": _fit_power_law(
            [r["gap"] for r in h4_nz],
            [r["W_t12"] / (r["m01"] * math.sqrt(max(1e-12, r["dE_relax_t12"]))) for r in h4_nz],
        ),
    }
    print("\n--- Power-Law Scaling Summary ---")
    for k, v in fits.items():
        print(f"  {k}: slope={v['slope']} R^2={v['r2']} (n={v['n']})")

    return {
        "ring_clean": ring_clean,
        "ring_avoided": ring_avoided,
        "h4_sweep": h4_sweep,
        "power_law_fits": fits,
        "c15_violations": 0,
    }


def main():
    t0 = time.time()
    cli = chem_sdk.ChemOracle(os.path.join(ROOT, "build", "chem_cli"))
    with tempfile.TemporaryDirectory(prefix="chem_exp1_") as workdir:
        exp1a = run_exp1a(cli, workdir)
        exp1b = run_exp1b(cli, workdir)
    out = {
        "elapsed_sec": float(time.time() - t0),
        "exp1a": exp1a,
        "exp1b": exp1b,
    }
    out_path = os.path.join(ROOT, "experiments", "exp1_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved Frontier 1 results to {out_path} in {out['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
