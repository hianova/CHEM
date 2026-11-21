#!/usr/bin/env python3
"""
Direction C Step 2:
  Part A: Entanglement-driven --auto-no subspace concentration & Haar O(K) basis-invariance benchmark.
  Part B: Practical .xyz reaction-path & MLIP hallucination screening demo via oracles/chem_screener.py.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracles"))

import chem_sdk
import chem_screener

BOHR_TO_ANGSTROM = 0.529177210903


def load_fcidump_full(path: str):
    """Load an FCIDUMP file preserving exact unnormalized integrals and e_core."""
    header_lines = []
    with open(path) as f:
        lines = f.readlines()

    body_start = 0
    for idx, line in enumerate(lines):
        header_lines.append(line)
        if "&END" in line or "/" in line.strip():
            body_start = idx + 1
            break

    header_text = " ".join(header_lines)
    m_k = re.search(r"NORB\s*=\s*(\d+)", header_text, re.IGNORECASE)
    m_ne = re.search(r"NELEC\s*=\s*(\d+)", header_text, re.IGNORECASE)
    m_ms2 = re.search(r"MS2\s*=\s*(-?\d+)", header_text, re.IGNORECASE)
    if not (m_k and m_ne):
        raise ValueError(f"Missing NORB/NELEC in FCIDUMP header: {path}")

    K = int(m_k.group(1))
    nelec = int(m_ne.group(1))
    ms2 = int(m_ms2.group(1)) if m_ms2 else 0

    e_core = 0.0
    h1 = np.zeros((K, K), dtype=np.float64)
    eri = np.zeros((K, K, K, K), dtype=np.float64)

    for line in lines[body_start:]:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        val = float(parts[0])
        p, q, r, t = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        if p == 0 and q == 0 and r == 0 and t == 0:
            e_core = val
        elif r == 0 and t == 0:
            a, b = p - 1, q - 1
            h1[a, b] = h1[b, a] = val
        else:
            a, b, c, d = p - 1, q - 1, r - 1, t - 1
            for x, y in ((a, b), (b, a)):
                for u, v in ((c, d), (d, c)):
                    eri[x, y, u, v] = val
                    eri[u, v, x, y] = val

    return K, nelec, ms2, e_core, h1, eri


def rotate_fcidump_haar(src_path: str, dst_path: str, seed: int = 42):
    """Apply a random Haar O(K) orthogonal rotation to an FCIDUMP Hamiltonian."""
    K, nelec, ms2, e_core, h1, eri = load_fcidump_full(src_path)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((K, K))
    Q, R = np.linalg.qr(A)
    signs = np.where(np.diag(R) >= 0.0, 1.0, -1.0)
    Q = Q * signs[None, :]
    ortho_err = float(np.linalg.norm(Q.T @ Q - np.eye(K)))
    if ortho_err > 1e-11:
        raise RuntimeError(f"Haar Q orthogonality error {ortho_err}")

    h1_rot = Q.T @ h1 @ Q
    h1_rot = 0.5 * (h1_rot + h1_rot.T)

    t1 = np.tensordot(eri, Q, axes=([0], [0]))
    t2 = np.tensordot(t1, Q, axes=([0], [0]))
    t3 = np.tensordot(t2, Q, axes=([0], [0]))
    eri_rot = np.tensordot(t3, Q, axes=([0], [0]))
    eri_sym = 0.125 * (
        eri_rot
        + eri_rot.transpose(1, 0, 2, 3)
        + eri_rot.transpose(0, 1, 3, 2)
        + eri_rot.transpose(1, 0, 3, 2)
        + eri_rot.transpose(2, 3, 0, 1)
        + eri_rot.transpose(3, 2, 0, 1)
        + eri_rot.transpose(2, 3, 1, 0)
        + eri_rot.transpose(3, 2, 1, 0)
    )

    chem_sdk.write_fcidump(dst_path, K, nelec, ms2, e_core, h1_rot, eri_sym)
    return {
        "K": K,
        "nelec": nelec,
        "ms2": ms2,
        "ortho_err": ortho_err,
        "det_Q": float(np.linalg.det(Q)),
    }


def _pack_bracket_entry(res):
    return {
        "E_lb_cert": float(res["E_lb_cert"]),
        "E_ub": float(res["E_ub"]),
        "width_mha": float(res["interval_width"]) * 1000.0,
        "iters": int(res["iters"]),
        "converged": bool(res.get("converged", 1)),
        "auto_no": bool(res.get("auto_no", False)),
        "no_occs": [float(x) for x in res["no_occs"]] if res.get("no_occs") else None,
        "no_entropies": [float(x) for x in res["no_entropies"]] if res.get("no_entropies") else None,
    }


def run_part_a(cli: chem_sdk.ChemOracle):
    print("=== Part A: --auto-no Subspace Concentration & Haar O(K) Basis Invariance ===")
    benchmarks = [
        {
            "name": "n2_stretched_10e8o",
            "path": os.path.join(ROOT, "benchmarks", "n2_stretched_10e8o.fcidump"),
            "max_iters": 1600,
            "tol": 1e-6,
            "threads": 0,
            "configs": [
                {"label": "DQG", "t1": False, "t2": False, "sub_orbs": 0},
                {"label": "DQG+T2(5)", "t1": False, "t2": True, "sub_orbs": 5},
                {"label": "DQG+T1+T2(5)", "t1": True, "t2": True, "sub_orbs": 5},
                {"label": "DQG+T2(6)", "t1": False, "t2": True, "sub_orbs": 6},
            ],
        },
        {
            "name": "tm_spin_s0_8e8o",
            "path": os.path.join(ROOT, "benchmarks", "tm_spin_s0_8e8o.fcidump"),
            "max_iters": 1200,
            "tol": 1e-6,
            "threads": 0,
            "configs": [
                {"label": "DQG", "t1": False, "t2": False, "sub_orbs": 0},
                {"label": "DQG+T2(5)", "t1": False, "t2": True, "sub_orbs": 5},
                {"label": "DQG+T1+T2(5)", "t1": True, "t2": True, "sub_orbs": 5},
                {"label": "DQG+T2(6)", "t1": False, "t2": True, "sub_orbs": 6},
            ],
        },
        {
            "name": "h2o_stretched",
            "path": os.path.join(ROOT, "benchmarks", "h2o_stretched.fcidump"),
            "max_iters": 1200,
            "tol": 1e-6,
            "threads": 0,
            "configs": [
                {"label": "DQG", "t1": False, "t2": False, "sub_orbs": 0},
                {"label": "DQG+T2(4)", "t1": False, "t2": True, "sub_orbs": 4},
                {"label": "DQG+T1+T2(4)", "t1": True, "t2": True, "sub_orbs": 4},
            ],
        },
        {
            "name": "h12_chain_12e12o",
            "path": os.path.join(ROOT, "benchmarks", "h12_chain_12e12o.fcidump"),
            "max_iters": 1600,
            "tol": 1e-6,
            "threads": 4,
            "configs": [
                {"label": "DQG", "t1": False, "t2": False, "sub_orbs": 0},
                {"label": "DQG+T2(5)", "t1": False, "t2": True, "sub_orbs": 5},
                {"label": "DQG+T1+T2(5)", "t1": True, "t2": True, "sub_orbs": 5},
            ],
        },
    ]

    concentration_results = {}
    for bm in benchmarks:
        name = bm["name"]
        fpath = bm["path"]
        rows = {}
        t0 = time.time()
        dqg_lb = None
        for cfg in bm["configs"]:
            lbl = cfg["label"]
            if not cfg["t1"] and not cfg["t2"]:
                t_s = time.time()
                res = cli.bracket(
                    fpath, mode="auto", two_s=0, constrain_s2=True,
                    max_iters=bm["max_iters"], tol=bm["tol"],
                    threads=bm["threads"],
                )
                entry = _pack_bracket_entry(res)
                entry["wall_sec"] = float(time.time() - t_s)
                dqg_lb = entry["E_lb_cert"]
                rows[lbl] = entry
                print(
                    f"  {name:20s} | {lbl:13s} | LB={entry['E_lb_cert']:.8f} UB={entry['E_ub']:.8f} "
                    f"w={entry['width_mha']:.3f}mHa it={entry['iters']}/c={int(entry['converged'])} ({entry['wall_sec']:.2f}s)"
                )
            else:
                k_sub = cfg["sub_orbs"]
                t_n = time.time()
                r_naive = cli.bracket(
                    fpath, mode="auto", two_s=0, constrain_s2=True,
                    max_iters=bm["max_iters"], tol=bm["tol"],
                    t1=cfg["t1"], t1_orbs=k_sub if cfg["t1"] else 0,
                    t2=cfg["t2"], t2_orbs=k_sub if cfg["t2"] else 0,
                    auto_no=False, threads=bm["threads"],
                )
                w_n = float(time.time() - t_n)

                t_a = time.time()
                r_auto = cli.bracket(
                    fpath, mode="auto", two_s=0, constrain_s2=True,
                    max_iters=bm["max_iters"], tol=bm["tol"],
                    t1=cfg["t1"], t1_orbs=k_sub if cfg["t1"] else 0,
                    t2=cfg["t2"], t2_orbs=k_sub if cfg["t2"] else 0,
                    auto_no=True, threads=bm["threads"],
                )
                w_a = float(time.time() - t_a)

                en = _pack_bracket_entry(r_naive)
                en["wall_sec"] = w_n
                ea = _pack_bracket_entry(r_auto)
                ea["wall_sec"] = w_a

                gain_vs_naive_mha = (ea["E_lb_cert"] - en["E_lb_cert"]) * 1000.0
                gain_vs_dqg_naive_mha = (en["E_lb_cert"] - dqg_lb) * 1000.0
                gain_vs_dqg_auto_mha = (ea["E_lb_cert"] - dqg_lb) * 1000.0
                rows[lbl] = {
                    "naive": en,
                    "auto_no": ea,
                    "gain_auto_vs_naive_mha": gain_vs_naive_mha,
                    "gain_naive_vs_dqg_mha": gain_vs_dqg_naive_mha,
                    "gain_auto_vs_dqg_mha": gain_vs_dqg_auto_mha,
                }
                print(
                    f"  {name:20s} | {lbl:13s} | naive LB={en['E_lb_cert']:.8f} it={en['iters']}/c={int(en['converged'])} ({w_n:.2f}s) | "
                    f"auto-no LB={ea['E_lb_cert']:.8f} it={ea['iters']}/c={int(ea['converged'])} ({w_a:.2f}s) | gain={gain_vs_naive_mha:+.3f}mHa "
                    f"(vs DQG: {gain_vs_dqg_naive_mha:+.3f} -> {gain_vs_dqg_auto_mha:+.3f}mHa)"
                )
        concentration_results[name] = {
            "elapsed_sec": float(time.time() - t0),
            "configs": rows,
        }

    # Haar O(K) rotation invariance benchmark
    print("\n--- Haar O(K) Basis-Invariance Test (seed=42) ---")
    invariance_results = {}
    with tempfile.TemporaryDirectory(prefix="chem_haar_") as tmpdir:
        h2o_rot = os.path.join(tmpdir, "h2o_haar42.fcidump")
        n2_rot = os.path.join(tmpdir, "n2_haar42.fcidump")
        h2o_meta = rotate_fcidump_haar(os.path.join(ROOT, "benchmarks", "h2o_stretched.fcidump"), h2o_rot, seed=42)
        n2_meta = rotate_fcidump_haar(os.path.join(ROOT, "benchmarks", "n2_stretched_10e8o.fcidump"), n2_rot, seed=42)

        haar_cases = [
            {
                "key": "h2o_stretched_t12_k4",
                "canon_path": os.path.join(ROOT, "benchmarks", "h2o_stretched.fcidump"),
                "rot_path": h2o_rot,
                "rot_meta": h2o_meta,
                "t1": True, "t2": True, "sub_orbs": 4,
                "verify_exact": False,
            },
            {
                "key": "n2_stretched_t2_k5",
                "canon_path": os.path.join(ROOT, "benchmarks", "n2_stretched_10e8o.fcidump"),
                "rot_path": n2_rot,
                "rot_meta": n2_meta,
                "t1": False, "t2": True, "sub_orbs": 5,
                "verify_exact": True,
            },
            {
                "key": "n2_stretched_t12_k5",
                "canon_path": os.path.join(ROOT, "benchmarks", "n2_stretched_10e8o.fcidump"),
                "rot_path": n2_rot,
                "rot_meta": n2_meta,
                "t1": True, "t2": True, "sub_orbs": 5,
                "verify_exact": False,
            },
        ]

        for hc in haar_cases:
            k_sub = hc["sub_orbs"]
            c_naive = cli.bracket(
                hc["canon_path"], mode="auto", two_s=0, constrain_s2=True,
                max_iters=1200, tol=1e-6,
                t1=hc["t1"], t1_orbs=k_sub if hc["t1"] else 0,
                t2=hc["t2"], t2_orbs=k_sub if hc["t2"] else 0,
                auto_no=False,
            )
            r_naive = cli.bracket(
                hc["rot_path"], mode="auto", two_s=0, constrain_s2=True,
                max_iters=1200, tol=1e-6,
                t1=hc["t1"], t1_orbs=k_sub if hc["t1"] else 0,
                t2=hc["t2"], t2_orbs=k_sub if hc["t2"] else 0,
                auto_no=False,
            )
            c_auto = cli.bracket(
                hc["canon_path"], mode="auto", two_s=0, constrain_s2=True,
                max_iters=1200, tol=1e-6,
                t1=hc["t1"], t1_orbs=k_sub if hc["t1"] else 0,
                t2=hc["t2"], t2_orbs=k_sub if hc["t2"] else 0,
                auto_no=True,
            )

            save_no_path = None
            cert_path = None
            exact_verify_res = None
            if hc["verify_exact"]:
                save_no_path = os.path.join(tmpdir, "n2_rot_no.fcidump")
                cert_path = os.path.join(tmpdir, "n2_rot_no.cert.json")

            r_auto = cli.bracket(
                hc["rot_path"], mode="auto", two_s=0, constrain_s2=True,
                max_iters=1200, tol=1e-6,
                t1=hc["t1"], t1_orbs=k_sub if hc["t1"] else 0,
                t2=hc["t2"], t2_orbs=k_sub if hc["t2"] else 0,
                auto_no=True,
                save_no_fcidump=save_no_path,
                cert_path=cert_path,
            )

            if hc["verify_exact"]:
                exact_verify_res = cli.exact_verify(cert_path, fcidump_path=save_no_path)
                if exact_verify_res.get("returncode", -1) != 0:
                    raise RuntimeError(f"exact-verify failed on rotated+NO FCIDUMP: {exact_verify_res}")

            delta_naive_mha = (float(r_naive["E_lb_cert"]) - float(c_naive["E_lb_cert"])) * 1000.0
            delta_auto_no_mha = (float(r_auto["E_lb_cert"]) - float(c_auto["E_lb_cert"])) * 1000.0
            ent_c = np.array(c_auto["no_entropies"], dtype=np.float64)
            ent_r = np.array(r_auto["no_entropies"], dtype=np.float64)
            max_ent_diff = float(np.max(np.abs(ent_c - ent_r)))

            item = {
                "rot_meta": hc["rot_meta"],
                "canon_naive": _pack_bracket_entry(c_naive),
                "rot_naive": _pack_bracket_entry(r_naive),
                "canon_auto_no": _pack_bracket_entry(c_auto),
                "rot_auto_no": _pack_bracket_entry(r_auto),
                "delta_naive_mha": delta_naive_mha,
                "delta_auto_no_mha": delta_auto_no_mha,
                "max_abs_no_entropy_diff": max_ent_diff,
                "exact_verify": exact_verify_res,
            }
            invariance_results[hc["key"]] = item
            print(
                f"  {hc['key']:22s} | delta_naive={delta_naive_mha:+.4f}mHa | "
                f"delta_auto_no={delta_auto_no_mha:+.6f}mHa | max|dS|={max_ent_diff:.3e}"
                + (f" | exact-verify rc={exact_verify_res['returncode']}" if exact_verify_res else "")
            )

    return {
        "concentration": concentration_results,
        "haar_invariance": invariance_results,
    }


def _write_h4_xyz(path: str):
    rx_bohr = 2.0
    ry_list = [1.6, 1.8, 1.95, 2.0, 2.05, 2.2, 2.5]
    with open(path, "w") as f:
        for i, ry in enumerate(ry_list):
            rx_a = rx_bohr * BOHR_TO_ANGSTROM
            ry_a = ry * BOHR_TO_ANGSTROM
            f.write("4\n")
            f.write(f"H4_rect_rx2.0_ry{ry:.2f}Bohr_frame{i}\n")
            f.write(f"H  0.000000000000   0.000000000000   0.000000000000\n")
            f.write(f"H  {rx_a:.12f}   0.000000000000   0.000000000000\n")
            f.write(f"H  0.000000000000   {ry_a:.12f}   0.000000000000\n")
            f.write(f"H  {rx_a:.12f}   {ry_a:.12f}   0.000000000000\n")
    return ry_list


def _write_h2o_xyz(path: str):
    r_list = [0.96, 1.25, 1.55, 1.80, 2.05]
    half_theta = math.radians(104.5 / 2.0)
    sin_ht = math.sin(half_theta)
    cos_ht = math.cos(half_theta)
    with open(path, "w") as f:
        for i, r_ang in enumerate(r_list):
            hx = r_ang * sin_ht
            hy = r_ang * cos_ht
            f.write("3\n")
            f.write(f"H2O_sym_R{r_ang:.2f}Ang_frame{i}\n")
            f.write("O  0.000000000000   0.000000000000   0.000000000000\n")
            f.write(f"H  {hx:.12f}   {hy:.12f}   0.000000000000\n")
            f.write(f"H  {-hx:.12f}  {hy:.12f}   0.000000000000\n")
    return r_list


def _run_screener_cli(xyz_path: str, mlip_energies, out_json_path: str,
                      active_el=None, active_orb=None):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        mlip_json_path = tf.name
        json.dump({"energies": [float(e) for e in mlip_energies]}, tf)

    try:
        cmd = [
            sys.executable,
            os.path.join(ROOT, "oracles", "chem_screener.py"),
            "--xyz", xyz_path,
            "--mlip-json", mlip_json_path,
            "--two-s", "0",
            "--sub-orbs", "5",
            "--veto-tol", "0.0005",
            "--workers", "4",
            "--out", out_json_path,
        ]
        if active_el is not None:
            cmd += ["--active-el", str(int(active_el))]
        if active_orb is not None:
            cmd += ["--active-orb", str(int(active_orb))]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"chem_screener.py failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        print(proc.stdout.strip())
        with open(out_json_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(mlip_json_path):
            os.remove(mlip_json_path)


def run_part_b():
    print("\n=== Part B: .xyz Reaction-Path & MLIP Hallucination Screener Demo ===")
    h4_xyz = os.path.join(ROOT, "experiments", "traj_h4_isomerization.xyz")
    h2o_xyz = os.path.join(ROOT, "experiments", "traj_h2o_dissoc.xyz")
    h4_ry = _write_h4_xyz(h4_xyz)
    h2o_r = _write_h2o_xyz(h2o_xyz)

    # 1. Reference screening via Python API without MLIP candidate
    ref_h4 = chem_screener.screen_trajectory(
        xyz_path=h4_xyz, two_s=0, t1=True, t2=True, sub_orbs=5, auto_no=True, workers=4
    )
    ref_h2o = chem_screener.screen_trajectory(
        xyz_path=h2o_xyz, two_s=0, active_el=8, active_orb=6,
        t1=True, t2=True, sub_orbs=5, auto_no=True, workers=4
    )

    # 2. Construct MLIP candidates with injected hallucinations
    mlip_h4 = []
    for idx, row in enumerate(ref_h4["frames"]):
        mid = 0.5 * (row["E_lb_cert"] + row["E_ub"])
        if idx == 3:
            mlip_h4.append(row["E_lb_cert"] - 0.025)
        else:
            mlip_h4.append(mid)

    mlip_h2o = []
    for idx, row in enumerate(ref_h2o["frames"]):
        mid = 0.5 * (row["E_lb_cert"] + row["E_ub"])
        if idx == 3:
            mlip_h2o.append(row["E_ub"] + 0.020)
        elif idx == 4:
            mlip_h2o.append(row["E_lb_cert"] - 0.025)
        else:
            mlip_h2o.append(mid)

    with tempfile.TemporaryDirectory(prefix="chem_screen_out_") as tmpdir:
        h4_out = os.path.join(tmpdir, "h4_screen.json")
        h2o_out = os.path.join(tmpdir, "h2o_screen.json")
        cli_h4 = _run_screener_cli(h4_xyz, mlip_h4, h4_out)
        cli_h2o = _run_screener_cli(h2o_xyz, mlip_h2o, h2o_out, active_el=8, active_orb=6)

    # Assertions on H4 isomerization trajectory
    f_h4 = cli_h4["frames"]
    assert cli_h4["n_frames"] == 7
    assert cli_h4["n_flagged"] == 1, f"Expected H4 n_flagged=1, got {cli_h4['n_flagged']}"
    for idx, row in enumerate(f_h4):
        st = row["mlip_check"]["status"]
        if idx == 3:
            assert st == "VETOED_FALSE_LOW", f"H4 frame 3 status={st}"
            assert row["mlip_check"]["early_vetoed"] is True, "H4 frame 3 early_vetoed must be True"
            assert row["mr_alert"] is True, "H4 TS frame 3 must trigger MR alert"
        else:
            assert st == "CERTIFIED_CONSISTENT", f"H4 frame {idx} status={st}"

    # Assertions on H2O symmetric dissociation trajectory
    f_h2o = cli_h2o["frames"]
    assert cli_h2o["n_frames"] == 5
    assert cli_h2o["n_flagged"] == 2, f"Expected H2O n_flagged=2, got {cli_h2o['n_flagged']}"
    for idx, row in enumerate(f_h2o):
        st = row["mlip_check"]["status"]
        if idx == 3:
            assert st == "FLAGGED_ABOVE_UB", f"H2O frame 3 status={st}"
        elif idx == 4:
            assert st == "VETOED_FALSE_LOW", f"H2O frame 4 status={st}"
            assert row["mlip_check"]["early_vetoed"] is True, "H2O frame 4 early_vetoed must be True"
        else:
            assert st == "CERTIFIED_CONSISTENT", f"H2O frame {idx} status={st}"

    s1_h2o = [float(r["S1_no"]) for r in f_h2o]
    for i in range(len(s1_h2o) - 1):
        assert s1_h2o[i + 1] > s1_h2o[i], f"H2O S1_no not strictly increasing: {s1_h2o}"

    print("  [PASS] All Part B screener & hallucination veto assertions passed!")
    return {
        "h4_isomerization": {
            "xyz_path": h4_xyz,
            "ry_bohr": h4_ry,
            "report": cli_h4,
        },
        "h2o_dissociation": {
            "xyz_path": h2o_xyz,
            "r_angstrom": h2o_r,
            "s1_no_series": s1_h2o,
            "report": cli_h2o,
        },
    }


def main():
    t0 = time.time()
    cli = chem_sdk.ChemOracle(os.path.join(ROOT, "build", "chem_cli"))
    part_a = run_part_a(cli)
    part_b = run_part_b()

    payload = {
        "total_elapsed_sec": float(time.time() - t0),
        "part_a_auto_no": part_a,
        "part_b_screener": part_b,
    }
    out_path = os.path.join(ROOT, "experiments", "exp_tool_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path} in {payload['total_elapsed_sec']:.2f}s")


if __name__ == "__main__":
    main()
