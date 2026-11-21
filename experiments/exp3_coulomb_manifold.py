#!/usr/bin/env python3
"""
Direction C Step 3 — Theory Capstone:
  Physical Coulomb PSD Cone (V_tilde = B^T B >= 0) & 3D STO-3G Geometry Manifold
  vs. General 8-Fold Tensor Adversaries (Hunt 3A = 95.265 mHa/unit-Frob).

Experiments:
  0. Mandatory central finite-difference gradient check (< 1e-7) for quotient PSD map.
  1. M1a (K=4, 4e4o, 2S=0, Mode A): max(FCI - DQG+T1+T2) over rank R in {1, 2, 4, 10}.
  2. M1b (K=5, 5e5o, 2S=1, Mode B, R=15):
       (i)  max(FCI - DQG+T1+T2)
       (ii) Conservative T1 > T2 search: E_lb(T1) - max(E_lb(T2), E_primal(T2)).
  3. M2  (K=4, 4e4o, 2S=0, STO-3G RHF): 120 diverse 3D H4 / HeH3+ / He2H2^2+ geometries
       + stochastic coordinate-perturbation refinement.
  4. High-precision 3500-iter CLI audit + exact-verify on all winners & JSON report.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracles"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import chem_sdk
import gen_benchmarks as gb
import fci_exact_oracle as fo


def sym_pairs(K: int):
    pairs = [(p, q) for p in range(K) for q in range(p + 1)]
    weights = np.array([1.0 if p == q else 2.0 for (p, q) in pairs], dtype=np.float64)
    s_vec = np.sqrt(weights)
    pair_map = {(p, q): idx for idx, (p, q) in enumerate(pairs)}
    return pairs, weights, s_vec, pair_map


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


def unpack_psd_eri(K: int, B: np.ndarray, pairs, s_vec: np.ndarray):
    """Build 8-fold symmetric ERI with exact V_tilde = B^T B >= 0 isometric norm."""
    V_tilde = B.T @ B
    inv_s = 1.0 / s_vec
    V_unw = (V_tilde * inv_s[:, None]) * inv_s[None, :]
    eri = np.zeros((K, K, K, K), dtype=np.float64)
    M = len(pairs)
    for a in range(M):
        p, q = pairs[a]
        for b in range(a, M):
            r, s = pairs[b]
            val = float(V_unw[a, b])
            for x, y in ((p, q), (q, p)):
                for u, v in ((r, s), (s, r)):
                    eri[x, y, u, v] = val
                    eri[u, v, x, y] = val
    return eri, V_tilde, V_unw


def sym_normalize_psd(h1: np.ndarray, B: np.ndarray, pairs, s_vec: np.ndarray):
    """Project (h1, B) onto unit Frobenius sphere ||h1||_F^2 + ||V_tilde||_F^2 = 1."""
    h = 0.5 * (h1 + h1.T)
    V_tilde = B.T @ B
    nu2 = float(np.sum(h * h) + np.sum(V_tilde * V_tilde))
    nu = math.sqrt(max(1e-30, nu2))
    h_hat = h / nu
    B_hat = B / math.sqrt(nu)
    eri_hat, V_hat, _ = unpack_psd_eri(h.shape[0], B_hat, pairs, s_vec)
    return h_hat, B_hat, eri_hat, V_hat


def pack_grad_to_quotient(h1: np.ndarray, B: np.ndarray, eri: np.ndarray,
                          V_tilde: np.ndarray, g_h1: np.ndarray, g_eri: np.ndarray,
                          pairs, s_vec: np.ndarray):
    """Exact quotient Riemannian gradient on unit-Frobenius PSD cone at nu = 1."""
    gh_sym, ge_sym = _sym_raw(g_h1, g_eri)
    M = len(pairs)
    G_raw = np.zeros((M, M), dtype=np.float64)
    for a, (p, q) in enumerate(pairs):
        for b in range(a, M):
            r, s = pairs[b]
            v = float(ge_sym[p, q, r, s])
            G_raw[a, b] = v
            G_raw[b, a] = v
    G_tilde = (G_raw * s_vec[:, None]) * s_vec[None, :]
    J_lin = float(np.sum(h1 * gh_sym) + np.sum(eri * ge_sym))
    grad_h1 = gh_sym - J_lin * h1
    grad_B = 2.0 * (B @ (G_tilde - J_lin * V_tilde))
    return grad_h1, grad_B, J_lin


def verify_analytic_gradient():
    """Verify quotient PSD gradient against central finite differences (< 1e-7)."""
    K, R = 4, 4
    pairs, _, s_vec, _ = sym_pairs(K)
    M = len(pairs)
    rng = np.random.default_rng(77)
    h1_raw = rng.standard_normal((K, K))
    B_raw = rng.standard_normal((R, M))
    h1, B, eri, V_tilde = sym_normalize_psd(h1_raw, B_raw, pairs, s_vec)

    # Check isometric Frobenius norm identity
    norm2 = float(np.sum(h1 * h1) + np.sum(eri * eri))
    v_norm2 = float(np.sum(h1 * h1) + np.sum(V_tilde * V_tilde))
    assert abs(norm2 - 1.0) < 1e-12, f"Unit norm violated: {norm2}"
    assert abs(v_norm2 - norm2) < 1e-12, f"Isometry violated: {v_norm2} vs {norm2}"

    fci = fo.solve_fci_sector(K, 2, 2, 0.0, h1, eri, two_s=0, n_eigs=1)
    rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)
    gh_a, gB_a, J_lin = pack_grad_to_quotient(
        h1, B, eri, V_tilde, rdm["grad_h1"], rdm["grad_eri"], pairs, s_vec
    )
    assert abs(J_lin - fci["E0"]) < 1e-11, f"Euler identity mismatch: {J_lin} vs {fci['E0']}"

    eps = 1e-6
    max_err_h = 0.0
    max_err_B = 0.0
    for (i, j) in [(0, 1), (2, 2), (1, 3)]:
        hp = h1.copy(); hp[i, j] += eps; hp[j, i] += (eps if i != j else 0.0)
        hm = h1.copy(); hm[i, j] -= eps; hm[j, i] -= (eps if i != j else 0.0)
        hnp, _, erip, _ = sym_normalize_psd(hp, B, pairs, s_vec)
        hnm, _, erim, _ = sym_normalize_psd(hm, B, pairs, s_vec)
        ep = fo.solve_fci_sector(K, 2, 2, 0.0, hnp, erip, two_s=0, n_eigs=1)["E0"]
        em = fo.solve_fci_sector(K, 2, 2, 0.0, hnm, erim, two_s=0, n_eigs=1)["E0"]
        fd = (ep - em) / (2.0 * eps)
        an = gh_a[i, j] if i == j else 2.0 * gh_a[i, j]
        max_err_h = max(max_err_h, abs(fd - an))

    for (r_idx, c_idx) in [(0, 0), (1, 4), (2, 7), (3, 9)]:
        Bp = B.copy(); Bp[r_idx, c_idx] += eps
        Bm = B.copy(); Bm[r_idx, c_idx] -= eps
        hnp, _, erip, _ = sym_normalize_psd(h1, Bp, pairs, s_vec)
        hnm, _, erim, _ = sym_normalize_psd(h1, Bm, pairs, s_vec)
        ep = fo.solve_fci_sector(K, 2, 2, 0.0, hnp, erip, two_s=0, n_eigs=1)["E0"]
        em = fo.solve_fci_sector(K, 2, 2, 0.0, hnm, erim, two_s=0, n_eigs=1)["E0"]
        fd = (ep - em) / (2.0 * eps)
        an = gB_a[r_idx, c_idx]
        max_err_B = max(max_err_B, abs(fd - an))

    max_err = max(max_err_h, max_err_B)
    print(f"[Grad Check] max_fd_err_h1={max_err_h:.3e}, max_fd_err_B={max_err_B:.3e} (< 1e-7 required)", flush=True)
    assert max_err < 1e-7, f"Quotient PSD gradient FD check failed: {max_err}"
    return {"max_err_h1": max_err_h, "max_err_B": max_err_B}


def _hubbard_wheel_psd_seed(R: int, rng: np.random.Generator, perturb: float = 0.05):
    K = 5
    pairs, _, s_vec, pair_map = sym_pairs(K)
    M = len(pairs)
    h1 = np.zeros((K, K), dtype=np.float64)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]:
        h1[u, v] = h1[v, u] = -1.0
    h1 += perturb * 0.5 * (rng.standard_normal((K, K)) + rng.standard_normal((K, K)).T)
    B = perturb * rng.standard_normal((R, M))
    for p in range(K):
        idx = pair_map[(p, p)]
        B[p, idx] += math.sqrt(6.0)
    return h1, B


def _seed_worker(task):
    kind = task["kind"]
    K = task["K"]
    na = task["na"]
    nb = task["nb"]
    two_s = task["two_s"]
    mode = task["mode"]
    R = task["R"]
    n_steps = task["n_steps"]
    cold_iters = task["cold_iters"]
    warm_iters = task["warm_iters"]
    tol = task["tol"]
    eta0 = task["eta0"]
    seed_idx = task["seed_idx"]
    rng = np.random.default_rng(task["rng_seed"])

    pairs, _, s_vec, _ = sym_pairs(K)
    M = len(pairs)
    if task.get("hubbard_seed"):
        h1_0, B_0 = _hubbard_wheel_psd_seed(R, rng, perturb=0.05)
    else:
        h1_0 = rng.standard_normal((K, K))
        B_0 = rng.standard_normal((R, M))

    h1, B, eri, V_tilde = sym_normalize_psd(h1_0, B_0, pairs, s_vec)
    nelec = na + nb
    ms2 = two_s
    best = {"obj": -1e9, "seed_idx": seed_idx}

    if kind == "fci_minus_t12":
        with chem_sdk.ChemFastSession(
            K, nelec, ms2=ms2, mode=mode, two_s=two_s,
            t1=True, t1_orbs=K, t2=True, t2_orbs=K, threads=1,
        ) as st12:
            eta = eta0
            stale = 0
            warm = False
            for step in range(n_steps):
                mi = warm_iters if warm else cold_iters
                r12 = st12.solve(0.0, h1, eri, max_iters=mi, tol=tol, warm_start=warm)
                fci = fo.solve_fci_sector(K, na, nb, 0.0, h1, eri, two_s=two_s, n_eigs=1)
                rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)

                obj = float(fci["E0"] - max(r12["E_lb_cert"], r12["E_primal"]))
                raw_gap = float(fci["E0"] - r12["E_lb_cert"])
                if obj > best["obj"]:
                    best = {
                        "obj": obj,
                        "raw_gap": raw_gap,
                        "h1": h1.copy(),
                        "B": B.copy(),
                        "eri": eri.copy(),
                        "seed_idx": seed_idx,
                        "step": step,
                    }
                    stale = 0
                else:
                    stale += 1
                    eta *= 0.78

                if stale >= 4 and "h1" in best:
                    rh = rng.standard_normal((K, K))
                    rB = rng.standard_normal((R, M))
                    h1, B, eri, V_tilde = sym_normalize_psd(
                        best["h1"] + 0.14 * rh, best["B"] + 0.14 * rB, pairs, s_vec
                    )
                    eta = eta0
                    stale = 0
                    warm = False
                    continue

                gh1 = rdm["grad_h1"] - r12["grad_h1"]
                geri = rdm["grad_eri"] - r12["grad_eri"]
                g_h, g_B, _ = pack_grad_to_quotient(
                    h1, B, eri, V_tilde, gh1, geri, pairs, s_vec
                )
                gnorm = math.sqrt(float(np.sum(g_h * g_h) + np.sum(g_B * g_B)))
                if gnorm < 1e-13:
                    break
                h1, B, eri, V_tilde = sym_normalize_psd(
                    h1 + eta * g_h / gnorm, B + eta * g_B / gnorm, pairs, s_vec
                )
                warm = True
        return best

    if kind == "t1_gt_t2":
        with chem_sdk.ChemFastSession(
            K, nelec, ms2=ms2, mode=mode, two_s=two_s,
            t1=True, t1_orbs=K, t2=False, threads=1,
        ) as st1, chem_sdk.ChemFastSession(
            K, nelec, ms2=ms2, mode=mode, two_s=two_s,
            t1=False, t2=True, t2_orbs=K, threads=1,
        ) as st2:
            eta = eta0
            stale = 0
            warm = False
            for step in range(n_steps):
                mi = warm_iters if warm else cold_iters
                r1 = st1.solve(0.0, h1, eri, max_iters=mi, tol=tol, warm_start=warm)
                r2 = st2.solve(0.0, h1, eri, max_iters=mi, tol=tol, warm_start=warm)
                obj = float(r1["E_lb_cert"] - max(r2["E_lb_cert"], r2["E_primal"]))
                raw_obj = float(r1["E_lb_cert"] - r2["E_lb_cert"])
                if obj > best["obj"]:
                    best = {
                        "obj": obj,
                        "raw_gap": raw_obj,
                        "h1": h1.copy(),
                        "B": B.copy(),
                        "eri": eri.copy(),
                        "seed_idx": seed_idx,
                        "step": step,
                    }
                    stale = 0
                else:
                    stale += 1
                    eta *= 0.78

                if stale >= 4 and "h1" in best:
                    rh = rng.standard_normal((K, K))
                    rB = rng.standard_normal((R, M))
                    h1, B, eri, V_tilde = sym_normalize_psd(
                        best["h1"] + 0.14 * rh, best["B"] + 0.14 * rB, pairs, s_vec
                    )
                    eta = eta0
                    stale = 0
                    warm = False
                    continue

                gh1 = r1["grad_h1"] - r2["grad_h1"]
                geri = r1["grad_eri"] - r2["grad_eri"]
                g_h, g_B, _ = pack_grad_to_quotient(
                    h1, B, eri, V_tilde, gh1, geri, pairs, s_vec
                )
                gnorm = math.sqrt(float(np.sum(g_h * g_h) + np.sum(g_B * g_B)))
                if gnorm < 1e-13:
                    break
                h1, B, eri, V_tilde = sym_normalize_psd(
                    h1 + eta * g_h / gnorm, B + eta * g_B / gnorm, pairs, s_vec
                )
                warm = True
        return best

    raise ValueError(f"Unknown kind {kind}")


def run_psd_hunt(pool, label, kind, K, na, nb, two_s, mode, R, n_seeds, n_steps,
                 cold_iters, warm_iters, tol, eta0, base_seed, hubbard_seed0=False):
    t0 = time.time()
    tasks = [
        {
            "kind": kind, "K": K, "na": na, "nb": nb, "two_s": two_s, "mode": mode,
            "R": R, "n_steps": n_steps, "cold_iters": cold_iters, "warm_iters": warm_iters,
            "tol": tol, "eta0": eta0, "seed_idx": i,
            "rng_seed": base_seed + i,
            "hubbard_seed": (hubbard_seed0 and i == 0),
        }
        for i in range(n_seeds)
    ]
    res = list(pool.map(_seed_worker, tasks))
    win = max(res, key=lambda r: r["obj"])
    print(
        f"  {label:32s} | best_obj={win['obj']*1000:+.3f}mHa (seed {win['seed_idx']} step {win.get('step')}) "
        f"in {time.time()-t0:.1f}s",
        flush=True,
    )
    return win


def _generate_120_geometries():
    geoms = []

    # 1. Squares & Rectangles H4
    for rx, ry in [
        (1.0, 1.0), (1.4, 1.4), (1.8, 1.8), (2.0, 2.0), (2.4, 2.4), (3.0, 3.0), (3.6, 3.6),
        (1.4, 2.2), (1.6, 2.6), (1.8, 3.0), (2.0, 2.6), (2.0, 3.4), (1.5, 3.8), (2.4, 3.8),
        (1.2, 2.8), (2.8, 3.6),
    ]:
        geoms.append((
            f"H4_rect_{rx:.2f}x{ry:.2f}",
            [1, 1, 1, 1],
            [(0.0, 0.0, 0.0), (rx, 0.0, 0.0), (0.0, ry, 0.0), (rx, ry, 0.0)],
        ))

    # 2. Rhombuses & Parallelograms H4
    for side in [1.4, 1.8, 2.2, 2.6, 3.2]:
        for deg in [40, 60, 75]:
            ang = math.radians(deg)
            c, s = side * math.cos(ang), side * math.sin(ang)
            geoms.append((
                f"H4_rhomb_s{side:.1f}_a{deg}",
                [1, 1, 1, 1],
                [(0.0, 0.0, 0.0), (side, 0.0, 0.0), (c, s, 0.0), (side + c, s, 0.0)],
            ))

    # 3. Linear & Alternating Linear H4
    for r in [1.1, 1.4, 1.8, 2.2, 2.6, 3.2, 3.8]:
        geoms.append((
            f"H4_lin_r{r:.1f}",
            [1, 1, 1, 1],
            [(0.0, 0.0, 0.0), (r, 0.0, 0.0), (2.0 * r, 0.0, 0.0), (3.0 * r, 0.0, 0.0)],
        ))
    for r1, r2 in [(1.2, 2.2), (1.4, 2.8), (1.6, 3.4), (1.3, 1.9), (1.8, 2.6)]:
        geoms.append((
            f"H4_dimer_{r1:.1f}_{r2:.1f}",
            [1, 1, 1, 1],
            [(0.0, 0.0, 0.0), (r1, 0.0, 0.0), (r1 + r2, 0.0, 0.0), (2.0 * r1 + r2, 0.0, 0.0)],
        ))

    # 4. Trapezoids & Out-of-plane Butterfly H4
    for base, top, h, dz in [
        (2.4, 1.4, 1.6, 0.0), (3.0, 1.6, 2.0, 0.0), (2.2, 2.2, 1.8, 0.8),
        (2.0, 2.0, 1.8, 1.4), (2.6, 2.0, 2.2, 1.2), (1.8, 1.8, 1.5, 1.0),
    ]:
        dx = 0.5 * (base - top)
        geoms.append((
            f"H4_trap_{base:.1f}_{top:.1f}_dz{dz:.1f}",
            [1, 1, 1, 1],
            [(0.0, 0.0, 0.0), (base, 0.0, 0.0), (dx, h, dz), (dx + top, h, -dz)],
        ))

    # 5. Regular & Distorted 3D Tetrahedra H4
    tet_base = np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ], dtype=np.float64) / math.sqrt(8.0)  # edge = 1.0
    for edge in [1.2, 1.6, 2.0, 2.4, 2.8, 3.4]:
        pts = tet_base * edge
        geoms.append((f"H4_tet_e{edge:.1f}", [1, 1, 1, 1], [tuple(p) for p in pts]))
    for scale_z in [0.55, 0.75, 1.35, 1.75]:
        for edge in [1.8, 2.4]:
            pts = tet_base.copy() * edge
            pts[:, 2] *= scale_z
            geoms.append((f"H4_dtet_e{edge:.1f}_sz{scale_z:.2f}", [1, 1, 1, 1], [tuple(p) for p in pts]))

    # 6. Y-shapes, T-shapes, Pyramids H4
    for r in [1.4, 1.9, 2.5, 3.2]:
        geoms.append((
            f"H4_Y_r{r:.1f}",
            [1, 1, 1, 1],
            [(0.0, 0.0, 0.0), (r, 0.0, 0.0), (-0.5 * r, 0.8660254 * r, 0.0), (-0.5 * r, -0.8660254 * r, 0.0)],
        ))
        geoms.append((
            f"H4_pyr_r{r:.1f}",
            [1, 1, 1, 1],
            [(r, 0.0, 0.0), (-0.5 * r, 0.8660254 * r, 0.0), (-0.5 * r, -0.8660254 * r, 0.0), (0.0, 0.0, 0.85 * r)],
        ))

    # 7. Cationic HeH3+ and He2H2^2+ (4e, 4o)
    for edge in [1.3, 1.7, 2.1, 2.6, 3.2]:
        pts = tet_base * edge
        geoms.append((f"HeH3+_tet_{edge:.1f}", [2, 1, 1, 1], [tuple(p) for p in pts]))
        geoms.append((f"He2H2_tet_{edge:.1f}", [2, 2, 1, 1], [tuple(p) for p in pts]))
    for rx, ry in [(1.4, 1.8), (1.8, 2.4), (2.2, 3.0)]:
        geoms.append((
            f"HeH3+_rect_{rx:.1f}x{ry:.1f}",
            [2, 1, 1, 1],
            [(0.0, 0.0, 0.0), (rx, 0.0, 0.0), (0.0, ry, 0.0), (rx, ry, 0.0)],
        ))
        geoms.append((
            f"He2H2_rect_{rx:.1f}x{ry:.1f}",
            [2, 2, 1, 1],
            [(0.0, 0.0, 0.0), (rx, 0.0, 0.0), (0.0, ry, 0.0), (rx, ry, 0.0)],
        ))
    for r in [1.3, 1.8, 2.4]:
        geoms.append((
            f"HeH3+_lin_{r:.1f}",
            [2, 1, 1, 1],
            [(0.0, 0.0, 0.0), (r, 0.0, 0.0), (2.0 * r, 0.0, 0.0), (3.0 * r, 0.0, 0.0)],
        ))
        geoms.append((
            f"He2H2_lin_{r:.1f}",
            [2, 2, 1, 1],
            [(0.0, 0.0, 0.0), (r, 0.0, 0.0), (2.0 * r, 0.0, 0.0), (3.0 * r, 0.0, 0.0)],
        ))

    # 8. Deterministic Random 3D Clusters (min_dist >= 0.90 Bohr) until >= 120
    rng = np.random.default_rng(202611)
    idx_rand = 0
    while len(geoms) < 120:
        pts = rng.uniform(-1.9, 1.9, size=(4, 3))
        dmin = min(
            float(np.linalg.norm(pts[i] - pts[j]))
            for i in range(4) for j in range(i + 1, 4)
        )
        if dmin < 0.90:
            continue
        if idx_rand % 4 == 0:
            Z = [2, 1, 1, 1]
            prefix = "HeH3+_rand"
        else:
            Z = [1, 1, 1, 1]
            prefix = "H4_rand"
        geoms.append((f"{prefix}_{idx_rand:02d}", Z, [tuple(float(v) for v in p) for p in pts]))
        idx_rand += 1

    return geoms[:120]


def _eval_3d_geom(Z_list, coords_bohr, sess):
    coords = np.asarray(coords_bohr, dtype=np.float64)
    dmin = min(
        float(np.linalg.norm(coords[i] - coords[j]))
        for i in range(len(Z_list)) for j in range(i + 1, len(Z_list))
    )
    if dmin < 0.80:
        return None
    K, _, enuc, S, Hcore, ERI = gb.build_sto3g(Z_list, [tuple(p) for p in coords])
    min_eig_s = float(np.min(np.linalg.eigvalsh(S)))
    if min_eig_s < 1e-4:
        return None
    enuc, h_mo, eri_mo = gb.rhf_canonical_mo(K, 4, enuc, S, Hcore, ERI)
    norm = math.sqrt(float(np.sum(h_mo * h_mo) + np.sum(eri_mo * eri_mo)))
    res = sess.solve(float(enuc), h_mo, eri_mo, max_iters=600, tol=1e-6, warm_start=False, return_grad=False)
    fci = fo.solve_fci_sector(4, 2, 2, float(enuc), h_mo, eri_mo, two_s=0, n_eigs=2)
    gap_raw_mha = float(fci["E0"] - res["E_lb_cert"]) * 1000.0
    gap_frob_mha = gap_raw_mha / norm
    return {
        "enuc": float(enuc),
        "h1": h_mo,
        "eri": eri_mo,
        "norm": norm,
        "E0_fci": float(fci["E0"]),
        "E_lb_t12": float(res["E_lb_cert"]),
        "gap_raw_mha": gap_raw_mha,
        "gap_frob_mha": gap_frob_mha,
        "min_dist_bohr": dmin,
        "min_eig_S": min_eig_s,
    }


def run_m2_3d_manifold():
    print("\n=== M2: Real 3D STO-3G Geometry Manifold (K=4, 4e4o, 2S=0, 120 Geometries + Perturb) ===", flush=True)
    t0 = time.time()
    geoms = _generate_120_geometries()
    evaluated = []
    with chem_sdk.ChemFastSession(
        4, 4, ms2=0, mode="a", two_s=0, t1=True, t1_orbs=4, t2=True, t2_orbs=4, threads=1
    ) as sess:
        for label, Z_list, coords in geoms:
            ev = _eval_3d_geom(Z_list, coords, sess)
            if ev is not None:
                evaluated.append({
                    "label": label,
                    "Z_list": Z_list,
                    "coords_bohr": [list(c) for c in coords],
                    **ev,
                })

        evaluated.sort(key=lambda r: r["gap_frob_mha"], reverse=True)
        print(f"  Surveyed {len(evaluated)} valid 3D geometries in {time.time()-t0:.1f}s. Top 5:")
        for r in evaluated[:5]:
            print(
                f"    {r['label']:24s} | gap_frob={r['gap_frob_mha']:.4f} mHa | "
                f"gap_raw={r['gap_raw_mha']:.4f} mHa | norm={r['norm']:.3f}"
            )

        # Stochastic coordinate-perturbation refinement on top 3 geometries (40 steps each)
        rng = np.random.default_rng(909)
        refined_candidates = []
        for idx, top in enumerate(evaluated[:3]):
            Z = top["Z_list"]
            cur_coords = np.array(top["coords_bohr"], dtype=np.float64)
            cur = top
            sigma = 0.14
            for step in range(40):
                cand_coords = cur_coords + rng.normal(0.0, sigma, size=cur_coords.shape)
                cand = _eval_3d_geom(Z, cand_coords, sess)
                if cand is not None and cand["gap_frob_mha"] > cur["gap_frob_mha"]:
                    cur_coords = cand_coords
                    cur = {
                        "label": f"{top['label']}_ref{step}",
                        "Z_list": Z,
                        "coords_bohr": cand_coords.tolist(),
                        **cand,
                    }
                    sigma = min(0.18, sigma * 1.05)
                else:
                    sigma = max(0.02, sigma * 0.92)
            refined_candidates.append(cur)

    best_3d = max(refined_candidates, key=lambda r: r["gap_frob_mha"])
    print(
        f"  Refined M2 Winner: {best_3d['label']} | gap_frob={best_3d['gap_frob_mha']:.4f} mHa | "
        f"gap_raw={best_3d['gap_raw_mha']:.4f} mHa (Z={best_3d['Z_list']})",
        flush=True,
    )
    return {
        "n_surveyed": len(evaluated),
        "top5_survey": [
            {k: v for k, v in r.items() if k not in ("h1", "eri")} for r in evaluated[:5]
        ],
        "winner_raw": best_3d,
        "elapsed_sec": float(time.time() - t0),
    }


def _audit_winner(task):
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

    pairs, _, s_vec, _ = sym_pairs(K)
    M = len(pairs)
    cli = chem_sdk.ChemOracle(os.path.join(ROOT, "build", "chem_cli"))
    fcidump_path = os.path.join(ROOT, "experiments", f"discovered_{tag}.fcidump")
    chem_sdk.write_fcidump(fcidump_path, K, na + nb, two_s, 0.0, h1, eri)

    # Coulomb PSD cone spectral verification
    V_unw = np.zeros((M, M), dtype=np.float64)
    for a, (p, q) in enumerate(pairs):
        for b, (r, s) in enumerate(pairs):
            V_unw[a, b] = float(eri[p, q, r, s])
    V_tilde = (V_unw * s_vec[:, None]) * s_vec[None, :]
    evals_tilde = np.linalg.eigvalsh(V_tilde)
    evals_unw = np.linalg.eigvalsh(V_unw)
    min_eig_tilde = float(np.min(evals_tilde))
    min_eig_unw = float(np.min(evals_unw))
    assert min_eig_tilde >= -1e-11, f"{tag} violated V_tilde >= 0: {min_eig_tilde}"
    assert min_eig_unw >= -1e-11, f"{tag} violated V_unw >= 0: {min_eig_unw}"
    num_rank = int(np.sum(evals_tilde > 1e-8))

    fci = fo.solve_fci_sector(K, na, nb, 0.0, h1, eri, two_s=two_s, n_eigs=3)
    rdm = fo.compute_fci_12rdm_and_grad(fci, state_idx=0)

    tiers = {}
    for tname, tslug, use_t1, use_t2, do_ev in [
        ("DQG", "dqg", False, False, False),
        ("DQG+T1", "t1", True, False, True),
        ("DQG+T2", "t2", False, True, False),
        ("DQG+T1+T2", "t12", True, True, True),
    ]:
        cert_path = os.path.join(ROOT, "experiments", f"cert_{tag}_{tslug}.json") if do_ev else None
        br = cli.bracket(
            fcidump_path, mode=mode, two_s=two_s,
            t1=use_t1, t1_orbs=K if use_t1 else 0,
            t2=use_t2, t2_orbs=K if use_t2 else 0,
            max_iters=3500, tol=1e-7, max_dets=2048,
            threads=threads, cert_path=cert_path,
        )
        ev_info = None
        if do_ev:
            ev = cli.exact_verify(cert_path, fcidump_path)
            assert ev.get("ok") and ev.get("returncode", -1) == 0, f"exact-verify failed on {cert_path}: {ev}"
            assert ev.get("all_blocks_sdd"), f"SDD failed on {cert_path}: {ev}"
            ev_info = {
                "E_lb_int_chol": float(ev["E_lb_int_chol"]),
                "E_lb_weyl": float(ev["E_lb_weyl"]),
                "all_blocks_sdd": bool(ev["all_blocks_sdd"]),
                "scale_D": ev.get("scale_D"),
            }
        tiers[tname] = {
            "E_lb_cert": float(br["E_lb_cert"]),
            "E_primal": float(br.get("E_primal", br["E_lb_cert"])),
            "E_ub": float(br["E_ub"]),
            "gap_to_FCI_mha": float(fci["E0"] - br["E_lb_cert"]) * 1000.0,
            "iters": int(br.get("iters", 0)),
            "exact_verify": ev_info,
        }

    h1_norm2 = float(np.sum(h1 * h1))
    eri_norm2 = float(np.sum(eri * eri))
    coulomb_norm2 = float(sum(eri[i, i, j, j] ** 2 for i in range(K) for j in range(K)))
    exch_norm2 = float(sum(eri[i, j, i, j] ** 2 for i in range(K) for j in range(K) if i != j))
    offdiag_frac = max(0.0, 1.0 - (coulomb_norm2 + 2.0 * exch_norm2) / max(1e-15, eri_norm2))
    gap_fci_t12_mha = tiers["DQG+T1+T2"]["gap_to_FCI_mha"]
    ratio_to_hunt3a = gap_fci_t12_mha / 95.2647

    return {
        "name": name,
        "tag": tag,
        "K": K,
        "nelec": na + nb,
        "two_s": two_s,
        "mode": mode,
        "fcidump_path": os.path.relpath(fcidump_path, ROOT),
        "E0_fci": float(fci["E0"]),
        "E1_fci": float(fci["E1"]),
        "gap_fci": float(fci["gap"]),
        "S1_fci": float(rdm["S1"]),
        "min_eig_V_tilde": min_eig_tilde,
        "min_eig_V_unweighted": min_eig_unw,
        "V_tilde_rank": num_rank,
        "eri_norm_frac": eri_norm2 / max(1e-15, h1_norm2 + eri_norm2),
        "eri_offdiag_frac": offdiag_frac,
        "gap_fci_t12_mha": gap_fci_t12_mha,
        "ratio_to_hunt3a": ratio_to_hunt3a,
        "tiers": tiers,
    }


def main():
    t0 = time.time()
    print("=== Direction C Step 3: Coulomb PSD Cone & 3D Geometry Manifold ===", flush=True)
    grad_check = verify_analytic_gradient()

    with ProcessPoolExecutor(max_workers=4) as pool:
        print("\n--- M1a: K=4 (4e4o) 2S=0 Coulomb PSD Cone across Rank R={1,2,4,10} ---", flush=True)
        w_r1 = run_psd_hunt(pool, "M1a K=4 R=1",  "fci_minus_t12", 4, 2, 2, 0, "a", 1,  4, 30, 500, 250, 1e-5, 0.16, 1100)
        w_r2 = run_psd_hunt(pool, "M1a K=4 R=2",  "fci_minus_t12", 4, 2, 2, 0, "a", 2,  4, 30, 500, 250, 1e-5, 0.16, 1200)
        w_r4 = run_psd_hunt(pool, "M1a K=4 R=4",  "fci_minus_t12", 4, 2, 2, 0, "a", 4,  4, 30, 500, 250, 1e-5, 0.16, 1300)
        w_r10 = run_psd_hunt(pool, "M1a K=4 R=10", "fci_minus_t12", 4, 2, 2, 0, "a", 10, 4, 30, 500, 250, 1e-5, 0.16, 1400)

        print("\n--- M1b: K=5 (5e5o) 2S=1 Coulomb PSD Cone (R=15) ---", flush=True)
        w_k5_adv = run_psd_hunt(pool, "M1b K=5 R=15 FCI-T12", "fci_minus_t12", 5, 3, 2, 1, "b", 15, 4, 20, 600, 300, 2e-5, 0.16, 1500)
        w_k5_t1 = run_psd_hunt(pool, "M1b K=5 R=15 T1>T2", "t1_gt_t2", 5, 3, 2, 1, "b", 15, 4, 18, 750, 450, 2e-5, 0.12, 1600, hubbard_seed0=True)

        m2 = run_m2_3d_manifold()
        m2_w = m2["winner_raw"]
        m2_norm = m2_w["norm"]
        m2_h1_n = 0.5 * (m2_w["h1"] + m2_w["h1"].T) / m2_norm
        m2_eri_n = m2_w["eri"] / m2_norm
        m2_raw_fcidump = os.path.join(ROOT, "experiments", "discovered_mol3d_k4_worst_raw.fcidump")
        chem_sdk.write_fcidump(m2_raw_fcidump, 4, 4, 0, m2_w["enuc"], m2_w["h1"], m2_w["eri"])

        print("\n--- High-Precision 3500-iter CLI Audit & Exact-Verify ---", flush=True)
        audit_tasks = [
            {"name": "M1a K=4 R=1 Coulomb PSD",  "tag": "coulomb_k4_r1",    "h1": w_r1["h1"],     "eri": w_r1["eri"],     "K": 4, "na": 2, "nb": 2, "two_s": 0, "mode": "a", "threads": 1},
            {"name": "M1a K=4 R=2 Coulomb PSD",  "tag": "coulomb_k4_r2",    "h1": w_r2["h1"],     "eri": w_r2["eri"],     "K": 4, "na": 2, "nb": 2, "two_s": 0, "mode": "a", "threads": 1},
            {"name": "M1a K=4 R=4 Coulomb PSD",  "tag": "coulomb_k4_r4",    "h1": w_r4["h1"],     "eri": w_r4["eri"],     "K": 4, "na": 2, "nb": 2, "two_s": 0, "mode": "a", "threads": 1},
            {"name": "M1a K=4 R=10 Coulomb PSD", "tag": "coulomb_k4_r10",   "h1": w_r10["h1"],    "eri": w_r10["eri"],    "K": 4, "na": 2, "nb": 2, "two_s": 0, "mode": "a", "threads": 1},
            {"name": "M1b K=5 R=15 Coulomb PSD", "tag": "coulomb_k5_adv",   "h1": w_k5_adv["h1"], "eri": w_k5_adv["eri"], "K": 5, "na": 3, "nb": 2, "two_s": 1, "mode": "b", "threads": 2},
            {"name": "M1b K=5 T1>T2 Coulomb PSD","tag": "coulomb_k5_t1gt2", "h1": w_k5_t1["h1"],  "eri": w_k5_t1["eri"],  "K": 5, "na": 3, "nb": 2, "two_s": 1, "mode": "b", "threads": 2},
            {"name": "M2 K=4 3D STO-3G Worst",   "tag": "mol3d_k4_worst",   "h1": m2_h1_n,        "eri": m2_eri_n,        "K": 4, "na": 2, "nb": 2, "two_s": 0, "mode": "a", "threads": 1},
        ]
        audited = list(pool.map(_audit_winner, audit_tasks))

    winners = {a["tag"]: a for a in audited}
    t1_t2_k5 = (
        winners["coulomb_k5_t1gt2"]["tiers"]["DQG+T1"]["E_lb_cert"]
        - winners["coulomb_k5_t1gt2"]["tiers"]["DQG+T2"]["E_lb_cert"]
    ) * 1000.0
    t1_t2_k5_chol = (
        winners["coulomb_k5_t1gt2"]["tiers"]["DQG+T1"]["exact_verify"]["E_lb_int_chol"]
        - winners["coulomb_k5_t1gt2"]["tiers"]["DQG+T2"]["E_lb_cert"]
    ) * 1000.0

    for a in audited:
        print(
            f"  {a['tag']:18s} | FCI-T12={a['gap_fci_t12_mha']:6.3f}mHa "
            f"({a['ratio_to_hunt3a']*100:5.1f}% of Hunt3A) | "
            f"minEig={a['min_eig_V_tilde']:+.2e} rank={a['V_tilde_rank']:2d} | "
            f"S1={a['S1_fci']:.3f} offdiag={a['eri_offdiag_frac']*100:5.1f}%",
            flush=True,
        )
    print(f"  K=5 Coulomb PSD (T1 - T2) Weyl={t1_t2_k5:+.4f}mHa | IntChol(T1)-Weyl(T2)={t1_t2_k5_chol:+.4f}mHa", flush=True)

    payload = {
        "total_elapsed_sec": float(time.time() - t0),
        "grad_check": grad_check,
        "baseline_frontier2": {
            "hunt3a_k4_fci_t12_mha": 95.2647,
            "hunt3b_k5_fci_t12_mha": 69.048,
            "hunt1a_k5_t1_beats_t2_mha": 16.292,
        },
        "m2_geometry_manifold": {
            "n_surveyed": m2["n_surveyed"],
            "top5_survey": m2["top5_survey"],
            "winner_metadata": {k: v for k, v in m2_w.items() if k not in ("h1", "eri")},
            "winner_raw_fcidump": os.path.relpath(m2_raw_fcidump, ROOT),
            "elapsed_sec": m2["elapsed_sec"],
        },
        "k5_coulomb_t1_minus_t2_weyl_mha": t1_t2_k5,
        "winners": winners,
    }
    out_path = os.path.join(ROOT, "experiments", "exp3_coulomb_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path} in {payload['total_elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
