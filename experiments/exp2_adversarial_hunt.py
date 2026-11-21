#!/usr/bin/env python3
"""
Frontier 2: Convergence-Proof Parallel ChemFastSession Adversarial Extremal Hunter.

Eliminates transient T1-vs-T2 iteration-speed artifacts via conservative duality-gap-bounded
objectives: E_lb(A) - max(E_lb(B), E_primal(B)), where A is evaluated tightly and B is bounded
above by its primal energy. Uses ChemFastSession (build/libchem.dylib) + ProcessPoolExecutor(4).

Hunts on 8-fold symmetric unit-Frobenius sphere (||h1||_F^2 + ||eri||_F^2 = 1):
  Hunt 1A (K=5, 5e5o, 2S=1, Mode B): maximize E_lb(T1) - max(E_lb(T2), E_primal(T2))
  Hunt 1B (K=4, 4e4o, 2S=0, Mode A): test minimal K=4 T1 > T2 margin
  Hunt 2  (K=5, 5e5o, 2S=1, Mode B): maximize min(T2-DQG, T12-T2, FCI-T12)
  Hunt 3A (K=4, 4e4o, 2S=0, Mode A): maximize E_CI - max(E_lb(T12), E_primal(T12))
  Hunt 3B (K=5, 5e5o, 2S=1, Mode B): maximize E_CI - max(E_lb(T12), E_primal(T12))

Outputs: discovered_*.fcidump, cert_*.json, CertDiscoveredT1.lean, exp2_results.json
"""

from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import subprocess
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracles"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import chem_sdk
import fci_exact_oracle as fo


def _sym_raw(h1: np.ndarray, eri: np.ndarray):
    h = 0.5 * (h1 + h1.T)
    e = 0.125 * (
        eri
        + eri.transpose(1, 0, 2, 3)
        + eri.transpose(0, 1, 3, 2)
        + eri.transpose(1, 0, 3, 2)
        + eri.transpose(2, 3, 0, 1)
        + eri.transpose(3, 2, 0, 1)
        + eri.transpose(2, 3, 1, 0)
        + eri.transpose(3, 2, 1, 0)
    )
    return h, e


def sym_ham(h1: np.ndarray, eri: np.ndarray):
    h, e = _sym_raw(h1, eri)
    norm = math.sqrt(float(np.sum(h * h) + np.sum(e * e)))
    if norm < 1e-15:
        raise ValueError("Zero Hamiltonian norm")
    return h / norm, e / norm


def read_fcidump_arrays(path: str, K: int):
    h1 = np.zeros((K, K), dtype=np.float64)
    eri = np.zeros((K, K, K, K), dtype=np.float64)
    in_body = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not in_body:
                if "&END" in s or "/" in s:
                    in_body = True
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            val = float(parts[0])
            p, q, r, t = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            if p == 0 and q == 0 and r == 0 and t == 0:
                continue
            if r == 0 and t == 0:
                h1[p - 1, q - 1] = h1[q - 1, p - 1] = val
            else:
                a, b, c, d = p - 1, q - 1, r - 1, t - 1
                for x, y in ((a, b), (b, a)):
                    for u, v in ((c, d), (d, c)):
                        eri[x, y, u, v] = val
                        eri[u, v, x, y] = val
    return sym_ham(h1, eri)


def tangent_step(h1, eri, g_h1, g_eri, eta):
    gs_h, gs_e = _sym_raw(g_h1, g_eri)
    dot = float(np.sum(gs_h * h1) + np.sum(gs_e * eri))
    t_h = gs_h - dot * h1
    t_e = gs_e - dot * eri
    tnorm = math.sqrt(float(np.sum(t_h * t_h) + np.sum(t_e * t_e)))
    if tnorm < 1e-14:
        return h1, eri, 0.0
    t_h /= tnorm
    t_e /= tnorm
    new_h = math.cos(eta) * h1 + math.sin(eta) * t_h
    new_e = math.cos(eta) * eri + math.sin(eta) * t_e
    nh, ne = sym_ham(new_h, new_e)
    return nh, ne, tnorm


def random_searu_perturbation(h1, eri, rng, strength=0.16):
    rh, re = _sym_raw(rng.standard_normal(h1.shape), rng.standard_normal(eri.shape))
    dot = float(np.sum(rh * h1) + np.sum(re * eri))
    rh -= dot * h1
    re -= dot * eri
    rn = math.sqrt(float(np.sum(rh * rh) + np.sum(re * re)))
    if rn < 1e-15:
        return h1, eri
    rh /= rn
    re /= rn
    return sym_ham(
        math.cos(strength) * h1 + math.sin(strength) * rh,
        math.cos(strength) * eri + math.sin(strength) * re,
    )


def _hubbard_ham(K, edges, U):
    h1 = np.zeros((K, K), dtype=np.float64)
    for u, v in edges:
        h1[u, v] = h1[v, u] = -1.0
    eri = np.zeros((K, K, K, K), dtype=np.float64)
    for p in range(K):
        eri[p, p, p, p] = U
    return sym_ham(h1, eri)


def make_seeds(kind: str, K: int, n_seeds: int = 4, base_seed: int = 2026):
    rng = np.random.default_rng(base_seed)
    seeds = []
    w5_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
    p3_k4 = os.path.join(ROOT, "experiments", "discovered_hunt3a_k4_adversary.fcidump")
    p3_k5 = os.path.join(ROOT, "experiments", "discovered_hunt3b_k5_adversary.fcidump")

    if kind == "hunt1a":
        seeds.append(_hubbard_ham(5, w5_edges, 6.0))
        seeds.append(_hubbard_ham(5, w5_edges, 8.0))
    elif kind == "hunt1b":
        seeds.append(_hubbard_ham(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 6.0))
        seeds.append(_hubbard_ham(4, [(0, 1), (1, 2), (2, 3), (3, 0)], 6.0))
    elif kind == "hunt2":
        if os.path.exists(p3_k5):
            seeds.append(read_fcidump_arrays(p3_k5, 5))
        seeds.append(_hubbard_ham(5, w5_edges, 6.0))
    elif kind == "hunt3a" and os.path.exists(p3_k4):
        seeds.append(read_fcidump_arrays(p3_k4, 4))
    elif kind == "hunt3b" and os.path.exists(p3_k5):
        seeds.append(read_fcidump_arrays(p3_k5, 5))

    while len(seeds) < n_seeds:
        seeds.append(sym_ham(rng.standard_normal((K, K)), rng.standard_normal((K, K, K, K))))
    return seeds[:n_seeds]


def _worker_run_seed(task):
    hunt_kind = task["hunt_kind"]
    K = task["K"]
    na = task["na"]
    nb = task["nb"]
    two_s = task["two_s"]
    mode = task["mode"]
    seed_idx = task["seed_idx"]
    h1 = task["h1"].copy()
    eri = task["eri"].copy()
    n_steps = task["n_steps"]
    cold_iters = task["cold_iters"]
    warm_iters = task["warm_iters"]
    tol = task["tol"]
    eta0 = task["eta0"]
    rng = np.random.default_rng(task["rng_seed"])

    nelec = na + nb
    ms2 = two_s
    best = {"obj": -1e9, "seed_idx": seed_idx}

    if hunt_kind in ("hunt1a", "hunt1b"):
        with chem_sdk.ChemFastSession(
            K, nelec, ms2=ms2, mode=mode, two_s=two_s, t1=True, t1_orbs=K, t2=False, threads=1
        ) as st1, chem_sdk.ChemFastSession(
            K, nelec, ms2=ms2, mode=mode, two_s=two_s, t1=False, t2=True, t2_orbs=K, threads=1
        ) as st2:
            eta = eta0
            stale = 0
            warm = False
            for step in range(n_steps):
                mi = warm_iters if warm else cold_iters
                r1 = st1.solve(0.0, h1, eri, max_iters=mi, tol=tol, warm_start=warm)
                r2 = st2.solve(0.0, h1, eri, max_iters=mi, tol=tol, warm_start=warm)
                # Conservative objective blocks under-converged T2 artifacts
                obj = r1["E_lb_cert"] - max(r2["E_lb_cert"], r2["E_primal"])
                raw_obj = r1["E_lb_cert"] - r2["E_lb_cert"]
                if obj > best["obj"]:
                    best = {
                        "obj": obj, "raw_obj": raw_obj,
                        "h1": h1.copy(), "eri": eri.copy(), "seed_idx": seed_idx, "step": step,
                    }
                    stale = 0
                else:
                    stale += 1
                    eta *= 0.78
                if stale >= 4 and "h1" in best:
                    h1, eri = random_searu_perturbation(best["h1"], best["eri"], rng, strength=0.14)
                    eta = eta0
                    stale = 0
                    warm = False
                    continue
                h1, eri, _ = tangent_step(h1, eri, r1["grad_h1"] - r2["grad_h1"], r1["grad_eri"] - r2["grad_eri"], eta)
                warm = True
        return best

    if hunt_kind == "hunt2":
        with chem_sdk.ChemFastSession(
            K, nelec, ms2=ms2, mode=mode, two_s=two_s, t1=False, t2=False, threads=1
        ) as sdqg, chem_sdk.ChemFastSession(
            K, nelec, ms2=ms2, mode=mode, two_s=two_s, t1=False, t2=True, t2_orbs=K, threads=1
        ) as st2, chem_sdk.ChemFastSession(
            K, nelec, ms2=ms2, mode=mode, two_s=two_s, t1=True, t1_orbs=K, t2=True, t2_orbs=K, threads=1
        ) as st12:
            eta = eta0
            stale = 0
            warm = False
            for step in range(n_steps):
                mi = warm_iters if warm else cold_iters
                r0 = sdqg.solve(0.0, h1, eri, max_iters=mi, tol=tol, warm_start=warm)
                r2 = st2.solve(0.0, h1, eri, max_iters=mi, tol=tol, warm_start=warm)
                r12 = st12.solve(0.0, h1, eri, max_iters=mi, tol=tol, warm_start=warm)
                fci = fo.solve_fci_sector(K, na, nb, 0.0, h1, eri, two_s=two_s, n_eigs=2)
                rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)
                g1 = r2["E_lb_cert"] - max(r0["E_lb_cert"], r0["E_primal"])
                g2 = r12["E_lb_cert"] - max(r2["E_lb_cert"], r2["E_primal"])
                g3 = fci["E0"] - max(r12["E_lb_cert"], r12["E_primal"])
                obj = min(g1, g2, g3)
                if obj > best["obj"]:
                    best = {
                        "obj": obj, "gaps": (g1, g2, g3),
                        "h1": h1.copy(), "eri": eri.copy(), "seed_idx": seed_idx, "step": step,
                    }
                    stale = 0
                else:
                    stale += 1
                    eta *= 0.80
                if stale >= 4 and "h1" in best:
                    h1, eri = random_searu_perturbation(best["h1"], best["eri"], rng, strength=0.14)
                    eta = eta0
                    stale = 0
                    warm = False
                    continue
                vals = np.array([g1, g2, g3])
                exps = np.exp(-150.0 * (vals - np.min(vals)))
                w = exps / np.sum(exps)
                comb_h = w[0] * (r2["grad_h1"] - r0["grad_h1"]) + w[1] * (r12["grad_h1"] - r2["grad_h1"]) + w[2] * (rdm["grad_h1"] - r12["grad_h1"])
                comb_e = w[0] * (r2["grad_eri"] - r0["grad_eri"]) + w[1] * (r12["grad_eri"] - r2["grad_eri"]) + w[2] * (rdm["grad_eri"] - r12["grad_eri"])
                h1, eri, _ = tangent_step(h1, eri, comb_h, comb_e, eta)
                warm = True
        return best

    # hunt3a / hunt3b
    with chem_sdk.ChemFastSession(
        K, nelec, ms2=ms2, mode=mode, two_s=two_s, t1=True, t1_orbs=K, t2=True, t2_orbs=K, threads=1
    ) as st12:
        eta = eta0
        stale = 0
        warm = False
        for step in range(n_steps):
            mi = warm_iters if warm else cold_iters
            r12 = st12.solve(0.0, h1, eri, max_iters=mi, tol=tol, warm_start=warm)
            fci = fo.solve_fci_sector(K, na, nb, 0.0, h1, eri, two_s=two_s, n_eigs=2)
            rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)
            obj = fci["E0"] - max(r12["E_lb_cert"], r12["E_primal"])
            if obj > best["obj"]:
                best = {"obj": obj, "h1": h1.copy(), "eri": eri.copy(), "seed_idx": seed_idx, "step": step}
                stale = 0
            else:
                stale += 1
                eta *= 0.85
            if stale >= 4 and "h1" in best:
                h1, eri = random_searu_perturbation(best["h1"], best["eri"], rng, strength=0.18)
                eta = eta0
                stale = 0
                warm = False
                continue
            h1, eri, _ = tangent_step(h1, eri, rdm["grad_h1"] - r12["grad_h1"], rdm["grad_eri"] - r12["grad_eri"], eta)
            warm = True
    return best


def run_parallel_hunt(pool, hunt_kind, label, K, na, nb, two_s, mode, n_seeds, n_steps, cold_iters, warm_iters, tol, eta0, base_seed):
    t0 = time.time()
    seeds = make_seeds(hunt_kind, K, n_seeds=n_seeds, base_seed=base_seed)
    tasks = [
        {
            "hunt_kind": hunt_kind,
            "K": K, "na": na, "nb": nb, "two_s": two_s, "mode": mode,
            "seed_idx": i,
            "h1": seeds[i][0], "eri": seeds[i][1],
            "n_steps": n_steps, "cold_iters": cold_iters, "warm_iters": warm_iters,
            "tol": tol, "eta0": eta0, "rng_seed": base_seed * 10 + i,
        }
        for i in range(n_seeds)
    ]
    results = list(pool.map(_worker_run_seed, tasks))
    winner = max(results, key=lambda r: r["obj"])
    print(f"  {label}: best_obj={winner['obj']:+.6f} (seed {winner['seed_idx']} step {winner.get('step')}) in {time.time()-t0:.2f}s", flush=True)
    return winner


def _audit_worker(task):
    name = task["name"]
    tag = task["tag"]
    h1 = task["h1"]
    eri = task["eri"]
    K = task["K"]
    na = task["na"]
    nb = task["nb"]
    two_s = task["two_s"]
    mode = task["mode"]
    threads = task["threads"]

    cli = chem_sdk.ChemOracle(os.path.join(ROOT, "build", "chem_cli"))
    fcidump_path = os.path.join(ROOT, "experiments", f"discovered_{tag}.fcidump")
    chem_sdk.write_fcidump(fcidump_path, K, na + nb, two_s, 0.0, h1, eri)

    fci = fo.solve_fci_sector(K, na, nb, 0.0, h1, eri, two_s=two_s, n_eigs=3)
    rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)

    tiers_info = {}
    for tname, t_slug, use_t1, use_t2 in [
        ("DQG", "dqg", False, False),
        ("DQG+T1", "t1", True, False),
        ("DQG+T2", "t2", False, True),
        ("DQG+T1+T2", "t12", True, True),
    ]:
        cert_path = os.path.join(ROOT, "experiments", f"cert_{tag}_{t_slug}.json")
        br = cli.bracket(
            fcidump_path,
            mode=mode,
            two_s=two_s,
            t1=use_t1,
            t1_orbs=K if use_t1 else 0,
            t2=use_t2,
            t2_orbs=K if use_t2 else 0,
            max_iters=3500,
            tol=1e-7,
            max_dets=2048,
            threads=threads,
            cert_path=cert_path,
        )
        ev = cli.exact_verify(cert_path, fcidump_path)
        assert ev["ok"], f"exact-verify failed on {cert_path}: {ev}"
        assert ev["all_blocks_sdd"], f"SDD failed on {cert_path}: {ev}"
        assert ev["E_lb_int_chol"] <= fci["E0"] + 1e-12, f"Int-Chol bound invalid: {ev['E_lb_int_chol']} > {fci['E0']}"
        tiers_info[tname] = {
            "E_lb_cert": float(br["E_lb_cert"]),
            "E_lb_int_chol": float(ev["E_lb_int_chol"]),
            "E_lb_weyl": float(ev["E_lb_weyl"]),
            "E_primal": float(br.get("E_primal", br["E_lb_cert"])),
            "E_ub": float(br["E_ub"]),
            "E_ci_cli": float(br["E_ci"]),
            "gap_to_FCI": float(fci["E0"] - br["E_lb_cert"]),
            "gap_int_chol_to_FCI": float(fci["E0"] - ev["E_lb_int_chol"]),
            "all_blocks_sdd": bool(ev["all_blocks_sdd"]),
            "scale_D": ev.get("scale_D"),
            "cert_path": os.path.relpath(cert_path, ROOT),
            "iters": int(br.get("iters", 0)),
        }

    h1_norm2 = float(np.sum(h1 * h1))
    eri_norm2 = float(np.sum(eri * eri))
    h1_diag2 = float(np.sum(np.diag(h1) ** 2))
    eri_hubbard_diag2 = float(sum(eri[i, i, i, i] ** 2 for i in range(K)))
    coulomb_norm2 = float(sum(eri[i, i, j, j] ** 2 for i in range(K) for j in range(K)))
    exch_norm2 = float(sum(eri[i, j, i, j] ** 2 for i in range(K) for j in range(K) if i != j))
    ga_evals = sorted([float(x) for x in np.linalg.eigvalsh(rdm["ga"])], reverse=True)

    return {
        "name": name,
        "tag": tag,
        "K": K,
        "nelec": na + nb,
        "two_s": two_s,
        "mode": mode,
        "fcidump_path": os.path.relpath(fcidump_path, ROOT),
        "E0_fci": fci["E0"],
        "E1_fci": fci["E1"],
        "gap_fci": fci["gap"],
        "S1_fci": rdm["S1"],
        "nat_occ_alpha": ga_evals,
        "anatomy": {
            "h1_frac_norm2": h1_norm2,
            "eri_frac_norm2": eri_norm2,
            "h1_diag_frac_of_h1": h1_diag2 / max(1e-15, h1_norm2),
            "eri_onsite_frac_of_eri": eri_hubbard_diag2 / max(1e-15, eri_norm2),
            "eri_coulomb_frac_of_eri": coulomb_norm2 / max(1e-15, eri_norm2),
            "eri_exchange_frac_of_eri": exch_norm2 / max(1e-15, eri_norm2),
            "eri_pairhop_offdiag_frac": max(0.0, 1.0 - (coulomb_norm2 + 2.0 * exch_norm2) / max(1e-15, eri_norm2)),
            "h1_eigvals": [float(x) for x in np.linalg.eigvalsh(h1)],
            "max_abs_h1": float(np.max(np.abs(h1))),
            "max_abs_eri": float(np.max(np.abs(eri))),
        },
        "tiers": tiers_info,
    }


def main():
    t0 = time.time()
    print("=== Frontier 2: Convergence-Proof Parallel ChemFastSession Hunts ===", flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        w1a = run_parallel_hunt(pool, "hunt1a", "Hunt 1A (K=5, 5e5o, 2S=1, T1 > T2)", 5, 3, 2, 1, "b", 4, 15, 750, 450, 2e-5, 0.12, 101)
        w1b = run_parallel_hunt(pool, "hunt1b", "Hunt 1B (K=4, 4e4o, 2S=0, T1 > T2)", 4, 2, 2, 0, "a", 4, 20, 500, 350, 2e-5, 0.12, 151)
        w2 = run_parallel_hunt(pool, "hunt2", "Hunt 2  (K=5, 5e5o, 2S=1, Strict 4-Tier)", 5, 3, 2, 1, "b", 4, 14, 700, 400, 2e-5, 0.12, 202)
        w3a = run_parallel_hunt(pool, "hunt3a", "Hunt 3A (K=4, 4e4o, 2S=0, T12 Adversary)", 4, 2, 2, 0, "a", 4, 25, 350, 250, 5e-5, 0.20, 303)
        w3b = run_parallel_hunt(pool, "hunt3b", "Hunt 3B (K=5, 5e5o, 2S=1, T12 Adversary)", 5, 3, 2, 1, "b", 4, 16, 350, 220, 5e-5, 0.20, 404)

        hunt_dt = time.time() - t0
        print(f"\n=== Parallel High-Precision 3500-iter Audit & exact-verify (hunt phase={hunt_dt:.1f}s) ===", flush=True)
        audit_tasks = [
            {"name": "Hunt 1A (K=5): DQG+T1 > DQG+T2", "tag": "hunt1a_k5_t1_beats_t2", "h1": w1a["h1"], "eri": w1a["eri"], "K": 5, "na": 3, "nb": 2, "two_s": 1, "mode": "b", "threads": 2},
            {"name": "Hunt 1B (K=4 4e4o): DQG+T1 > DQG+T2", "tag": "hunt1b_k4_t1_beats_t2", "h1": w1b["h1"], "eri": w1b["eri"], "K": 4, "na": 2, "nb": 2, "two_s": 0, "mode": "a", "threads": 1},
            {"name": "Hunt 2 (K=5): Strict 4-Tier", "tag": "hunt2_k5_strict4tier", "h1": w2["h1"], "eri": w2["eri"], "K": 5, "na": 3, "nb": 2, "two_s": 1, "mode": "b", "threads": 2},
            {"name": "Hunt 3A (K=4 4e4o): T12 Adversary", "tag": "hunt3a_k4_adversary", "h1": w3a["h1"], "eri": w3a["eri"], "K": 4, "na": 2, "nb": 2, "two_s": 0, "mode": "a", "threads": 1},
            {"name": "Hunt 3B (K=5 5e5o): T12 Adversary", "tag": "hunt3b_k5_adversary", "h1": w3b["h1"], "eri": w3b["eri"], "K": 5, "na": 3, "nb": 2, "two_s": 1, "mode": "b", "threads": 2},
        ]
        audited = list(pool.map(_audit_worker, audit_tasks))

    winners = {
        "hunt1a_k5_t1_beats_t2": audited[0],
        "hunt1b_k4_t1_beats_t2": audited[1],
        "hunt2_k5_strict4tier": audited[2],
        "hunt3a_k4_adversary": audited[3],
        "hunt3b_k5_adversary": audited[4],
    }

    t_1a = winners["hunt1a_k5_t1_beats_t2"]["tiers"]
    h1a_w = t_1a["DQG+T1"]["E_lb_cert"] - t_1a["DQG+T2"]["E_lb_cert"]
    h1a_c = t_1a["DQG+T1"]["E_lb_int_chol"] - t_1a["DQG+T2"]["E_lb_int_chol"]
    print(f"Hunt 1A (K=5)    Verified Margin: Weyl T1-T2={h1a_w:+.6f} | IntChol T1-T2={h1a_c:+.6f}")
    assert h1a_w > 0.0 and h1a_c > 0.0, f"Hunt 1A failed strict T1 > T2: weyl={h1a_w}, chol={h1a_c}"

    t_1b = winners["hunt1b_k4_t1_beats_t2"]["tiers"]
    h1b_w = t_1b["DQG+T1"]["E_lb_cert"] - t_1b["DQG+T2"]["E_lb_cert"]
    h1b_c = t_1b["DQG+T1"]["E_lb_int_chol"] - t_1b["DQG+T2"]["E_lb_int_chol"]
    print(f"Hunt 1B (K=4 4e) Verified Margin: Weyl T1-T2={h1b_w:+.6f} | IntChol T1-T2={h1b_c:+.6f}")

    t_2 = winners["hunt2_k5_strict4tier"]["tiers"]
    e0_2 = winners["hunt2_k5_strict4tier"]["E0_fci"]
    h2_w = (t_2["DQG+T2"]["E_lb_cert"] - t_2["DQG"]["E_lb_cert"], t_2["DQG+T1+T2"]["E_lb_cert"] - t_2["DQG+T2"]["E_lb_cert"], e0_2 - t_2["DQG+T1+T2"]["E_lb_cert"])
    h2_c = (t_2["DQG+T2"]["E_lb_int_chol"] - t_2["DQG"]["E_lb_int_chol"], t_2["DQG+T1+T2"]["E_lb_int_chol"] - t_2["DQG+T2"]["E_lb_int_chol"], e0_2 - t_2["DQG+T1+T2"]["E_lb_int_chol"])
    print(f"Hunt 2  (K=5)    Verified Gaps:   Weyl={tuple(round(x,6) for x in h2_w)} | IntChol={tuple(round(x,6) for x in h2_c)}")
    assert min(h2_w) > 1e-3 and min(h2_c) > 1e-3, f"Hunt 2 failed strict 4-tier: weyl={h2_w}, chol={h2_c}"

    for key in ("hunt3a_k4_adversary", "hunt3b_k5_adversary"):
        w = winners[key]
        g_w = w["E0_fci"] - w["tiers"]["DQG+T1+T2"]["E_lb_cert"]
        g_c = w["E0_fci"] - w["tiers"]["DQG+T1+T2"]["E_lb_int_chol"]
        print(f"{key:22s}: FCI-T12 Weyl={g_w:.6f} IntChol={g_c:.6f}")

    # Emit & leanchecker-verify CertDiscoveredT1.lean on K=4 Mode A (max_dim=24) DQG+T1
    cli = chem_sdk.ChemOracle(os.path.join(ROOT, "build", "chem_cli"))
    lean_out = os.path.join(ROOT, "experiments", "CertDiscoveredT1.lean")
    olean_out = os.path.join(ROOT, "build", "CertDiscoveredT1.olean")
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.elan/bin") + ":" + env.get("PATH", "")
    env["LEAN_PATH"] = os.path.join(ROOT, "..", "FLOW", "build", "lean") + ":" + os.path.join(ROOT, "build")

    lean_target = "hunt1b_k4_t1_beats_t2" if h1b_c > 1e-4 else "hunt3a_k4_adversary"
    cand = winners[lean_target]
    fci_p = os.path.join(ROOT, cand["fcidump_path"])
    cert_p = os.path.join(ROOT, cand["tiers"]["DQG+T1"]["cert_path"])
    emit_res = cli.emit_lean(fci_p, cert_p, lean_out, module_name="CertDiscoveredT1")
    c_proc = subprocess.run(
        ["lean", "-DwarningAsError=true", "-o", olean_out, lean_out],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=60, check=True,
    )
    lc_proc = subprocess.run(
        ["leanchecker", "CertDiscoveredT1"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=30, check=True,
    )
    print(f"CertDiscoveredT1.lean ({lean_target} DQG+T1) compiled & verified by leanchecker!", flush=True)

    out = {
        "hunt_elapsed_sec": float(hunt_dt),
        "total_elapsed_sec": float(time.time() - t0),
        "winners": winners,
        "hunt1a_margin_weyl": h1a_w,
        "hunt1a_margin_int_chol": h1a_c,
        "hunt1b_margin_weyl": h1b_w,
        "hunt1b_margin_int_chol": h1b_c,
        "hunt2_gaps_weyl": list(h2_w),
        "hunt2_gaps_int_chol": list(h2_c),
        "cert_lean_source": lean_target,
        "cert_lean_path": os.path.relpath(lean_out, ROOT),
        "emit_lean": emit_res,
        "lean_axioms_stdout": c_proc.stdout.strip(),
        "leanchecker_stdout": lc_proc.stdout.strip(),
    }
    out_path = os.path.join(ROOT, "experiments", "exp2_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved Frontier 2 results to {out_path} in {out['total_elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
