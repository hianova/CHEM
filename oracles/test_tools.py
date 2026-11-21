#!/usr/bin/env python3
"""
End-to-End Integration Tests for CHEM CLI, Python SDK (ChemOracle / ChemFastSession),
and Reaction-Path & MLIP Hallucination Screener (oracles/chem_screener.py).
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracles"))

import chem_sdk
import chem_screener


def main():
    t0 = time.time()
    cli_path = os.path.join(ROOT, "build", "chem_cli")
    oracle = chem_sdk.ChemOracle(cli_path)

    # 1. CLI --help and --version
    p_help = subprocess.run([cli_path, "--help"], capture_output=True, text=True, check=False)
    assert p_help.returncode == 0 and "Usage:" in (p_help.stdout + p_help.stderr), f"--help failed: {p_help.returncode}"
    p_ver = subprocess.run([cli_path, "--version"], capture_output=True, text=True, check=False)
    assert p_ver.returncode == 0 and "CHEM v0.3.1" in p_ver.stdout, f"--version failed: {p_ver.stdout}"
    print("  [PASS] 1. CLI --help and --version")

    # 2. read_fcidump / write_fcidump round-trip & rotate_integrals orthogonal invariance
    h4_path = os.path.join(ROOT, "benchmarks", "h4_ry2.0.fcidump")
    ham = chem_sdk.read_fcidump(h4_path)
    assert ham["K"] == 4 and ham["nelec"] == 4
    rng = np.random.default_rng(123)
    Q, R = np.linalg.qr(rng.standard_normal((4, 4)))
    Q = Q * np.where(np.diag(R) >= 0.0, 1.0, -1.0)[None, :]
    h1_rot, eri_rot = chem_sdk.rotate_integrals(ham["h1"], ham["eri"], Q)

    with tempfile.TemporaryDirectory(prefix="chem_tools_test_") as tmpdir:
        rot_fcidump = os.path.join(tmpdir, "h4_rot.fcidump")
        chem_sdk.write_fcidump(rot_fcidump, ham["K"], ham["nelec"], ham["ms2"], ham["e_core"], h1_rot, eri_rot)
        b_orig = oracle.bracket(h4_path, t1=True, t1_orbs=4, t2=True, t2_orbs=4)
        b_rot = oracle.bracket(rot_fcidump, t1=True, t1_orbs=4, t2=True, t2_orbs=4)
        diff_rot = abs(float(b_rot["E_lb_cert"]) - float(b_orig["E_lb_cert"]))
        assert diff_rot < 1e-6, f"Rotated bracket mismatch: {diff_rot}"
        print(f"  [PASS] 2. read/write FCIDUMP & rotate_integrals invariance (diff={diff_rot:.2e})")

        # 3. ChemFastSession.from_fcidump warm-start & accuracy
        with chem_sdk.ChemFastSession.from_fcidump(
            h4_path, t1=True, t1_orbs=4, t2=True, t2_orbs=4
        )[0] as sess:
            r_cold = sess.solve(ham["e_core"], ham["h1"], ham["eri"], max_iters=800, tol=1e-6, warm_start=False)
            r_warm = sess.solve(ham["e_core"], ham["h1"], ham["eri"], max_iters=800, tol=1e-6, warm_start=True)
            r_rot = sess.solve(ham["e_core"], h1_rot, eri_rot, max_iters=800, tol=1e-6, warm_start=False)
            diff_fast = abs(r_cold["E_lb_cert"] - float(b_orig["E_lb_cert"]))
            assert diff_fast < 1e-10, f"ChemFastSession vs CLI mismatch: {diff_fast}"
            assert r_warm["E_lb_cert"] >= r_cold["E_lb_cert"] - 1e-7
            assert abs(r_rot["E_lb_cert"] - r_cold["E_lb_cert"]) < 1e-6
        print(f"  [PASS] 3. ChemFastSession.from_fcidump warm-start & accuracy (diff={diff_fast:.2e})")

        # 4. ChemOracle.mlip_screen narrow bracket (certified, -25 mHa false-low, +20 mHa above-UB, HF force)
        e_mid = 0.5 * (float(b_orig["E_lb_cert"]) + float(b_orig["E_ub"]))
        fplus = os.path.join(ROOT, "benchmarks", "h4_ry2.0_fplus.fcidump")
        fminus = os.path.join(ROOT, "benchmarks", "h4_ry2.0_fminus.fcidump")

        s_ok = oracle.mlip_screen(
            h4_path, e_mid, f_ai=-0.070, f_plus=fplus, f_minus=fminus, dr=0.005,
            veto_tol=5e-4, force_tol=1e-3, t1=True, t1_orbs=4, t2=True, t2_orbs=4,
        )
        assert s_ok["status"] == "CERTIFIED_CONSISTENT" and not s_ok["vetoed"], f"s_ok={s_ok}"

        s_low = oracle.mlip_screen(
            h4_path, float(b_orig["E_lb_cert"]) - 0.025,
            veto_tol=5e-4, t1=True, t1_orbs=4, t2=True, t2_orbs=4,
        )
        assert s_low["status"] == "VETOED_FALSE_LOW" and s_low["energy_vetoed"] and s_low["vetoed"], f"s_low={s_low}"

        s_high = oracle.mlip_screen(
            h4_path, float(b_orig["E_ub"]) + 0.020,
            veto_tol=5e-4, t1=True, t1_orbs=4, t2=True, t2_orbs=4,
        )
        assert s_high["status"] == "FLAGGED_ABOVE_UB" and s_high["above_ub"], f"s_high={s_high}"

        s_bad_f = oracle.mlip_screen(
            h4_path, e_mid, f_ai=0.500, f_plus=fplus, f_minus=fminus, dr=0.005,
            veto_tol=5e-4, force_tol=1e-3, t1=True, t1_orbs=4, t2=True, t2_orbs=4,
        )
        assert s_bad_f["force_vetoed"] and s_bad_f["vetoed"], f"s_bad_f={s_bad_f}"
        print("  [PASS] 4. ChemOracle.mlip_screen (mid consistent, -25mHa veto, +20mHa above-UB, bad force)")

        # 5. End-to-end chem_screener.py CLI: H4 trajectory (--csv, --cert-dir, --fast-veto, exact-verify)
        traj_h4 = os.path.join(ROOT, "experiments", "traj_h4_isomerization.xyz")
        ref_h4 = chem_screener.screen_trajectory(
            xyz_path=traj_h4, two_s=0, t1=True, t2=True, sub_orbs=4, auto_no=True, workers=4
        )
        mlip_h4 = []
        for idx, row in enumerate(ref_h4["frames"]):
            mlip_h4.append(row["E_lb_cert"] - 0.025 if idx == 3 else 0.5 * (row["E_lb_cert"] + row["E_ub"]))
        mlip_json = os.path.join(tmpdir, "mlip_h4.json")
        out_json = os.path.join(tmpdir, "screen_h4.json")
        out_csv = os.path.join(tmpdir, "screen_h4.csv")
        cert_dir = os.path.join(tmpdir, "screen_certs")
        with open(mlip_json, "w") as f:
            json.dump({"energies": mlip_h4}, f)

        p_scr = subprocess.run([
            sys.executable, os.path.join(ROOT, "oracles", "chem_screener.py"),
            "--xyz", traj_h4, "--mlip-json", mlip_json,
            "--sub-orbs", "4", "--veto-tol", "0.0005", "--fast-veto",
            "--cert-dir", cert_dir, "--workers", "4",
            "--out", out_json, "--csv", out_csv,
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        assert p_scr.returncode == 0, f"chem_screener failed: {p_scr.stdout}\n{p_scr.stderr}"

        with open(out_json) as f:
            rep = json.load(f)
        assert rep["n_frames"] == 7 and rep["n_flagged"] == 1
        for idx, r in enumerate(rep["frames"]):
            st = r["mlip_check"]["status"]
            if idx == 3:
                assert st == "VETOED_FALSE_LOW" and r["mlip_check"]["early_vetoed"]
            else:
                assert st == "CERTIFIED_CONSISTENT"

        with open(out_csv) as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == 7 and csv_rows[3]["mlip_status"] == "VETOED_FALSE_LOW"

        c3 = rep["frames"][3]["cert_path"]
        f3 = rep["frames"][3]["cert_fcidump_path"]
        ev3 = oracle.exact_verify(c3, f3)
        assert ev3.get("returncode", -1) == 0 and ev3.get("all_blocks_sdd") == 1, f"exact_verify frame3 failed: {ev3}"

        # Cationic HeH3+ trajectory with --charge 1 --two-s 0
        heh3_xyz = os.path.join(tmpdir, "heh3_cation.xyz")
        with open(heh3_xyz, "w") as f:
            for d in (1.1, 1.3):
                f.write("4\nHeH3_cation\n")
                f.write("He 0.0 0.0 0.0\n")
                f.write(f"H  {d:.6f} 0.0 0.0\n")
                f.write(f"H  0.0 {d:.6f} 0.0\n")
                f.write(f"H  0.0 0.0 {d:.6f}\n")
        out_cat = os.path.join(tmpdir, "cat.json")
        p_cat = subprocess.run([
            sys.executable, os.path.join(ROOT, "oracles", "chem_screener.py"),
            "--xyz", heh3_xyz, "--charge", "1", "--two-s", "0",
            "--sub-orbs", "4", "--workers", "2", "--out", out_cat,
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        assert p_cat.returncode == 0, f"cation screener failed: {p_cat.stdout}\n{p_cat.stderr}"
        with open(out_cat) as f:
            cat_rep = json.load(f)
        assert cat_rep["n_frames"] == 2
        for r in cat_rep["frames"]:
            assert r["nelec"] == 4 and r["K"] == 4 and r["E_lb_cert"] <= r["E_ub"] + 1e-10
        print("  [PASS] 5. chem_screener CLI (--csv, --cert-dir + exact-verify, --fast-veto, --charge 1)")

    print(f"ALL CHEM TOOLS INTEGRATION TESTS PASSED in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
