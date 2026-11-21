#!/usr/bin/env python3
"""
CHEM Practical Reaction-Path & AI/MLIP Hallucination Screening Tool.

Accepts either:
  - A multi-frame .xyz trajectory file (Angstrom coordinates; H, He, Li, C, N, O in built-in STO-3G RHF+active space), OR
  - One or more pre-built .fcidump files.

Supports parallel screening across CPU cores via ProcessPoolExecutor, optional entanglement-driven
--auto-no (DQG Natural Orbitals) for T1/T2 subspace concentration, certified relative barrier envelopes
[dE_lb, dE_ub] relative to frame 0, static-correlation/diradical entropy alerts (S1_no), and
Anytime Early-Veto / variational-bracket MLIP hallucination detection.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oracles"))

import chem_sdk
import gen_benchmarks as gb

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
HA_TO_KCAL = 627.5094740631
ELEM_Z = {"H": 1, "HE": 2, "LI": 3, "C": 6, "N": 7, "O": 8}


def parse_xyz_trajectory(xyz_path: str):
    """Parse standard multi-frame .xyz trajectory into [(comment, Z_list, coords_bohr), ...]."""
    with open(xyz_path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    frames = []
    idx = 0
    n_lines = len(lines)
    while idx < n_lines:
        s = lines[idx].strip()
        if not s:
            idx += 1
            continue
        nat = int(s)
        comment = lines[idx + 1].strip() if idx + 1 < n_lines else ""
        Z_list = []
        coords_bohr = []
        for a in range(nat):
            parts = lines[idx + 2 + a].split()
            sym = parts[0].strip().upper()
            if sym not in ELEM_Z:
                raise ValueError(f"Unsupported element '{sym}' in built-in STO-3G (supported: {sorted(ELEM_Z)})")
            Z_list.append(ELEM_Z[sym])
            coords_bohr.append((
                float(parts[1]) * ANGSTROM_TO_BOHR,
                float(parts[2]) * ANGSTROM_TO_BOHR,
                float(parts[3]) * ANGSTROM_TO_BOHR,
            ))
        frames.append((comment, Z_list, coords_bohr))
        idx += 2 + nat
    if not frames:
        raise ValueError(f"No frames parsed from {xyz_path}")
    return frames


def build_frame_fcidump(Z_list, coords_bohr, out_path: str, two_s: int = 0,
                        charge: int = 0, active_el: int = None, active_orb: int = None):
    """Run STO-3G RHF MO transformation + optional frozen-core active space projection and write FCIDUMP."""
    K_full, nel_neutral, enuc, S, Hcore, ERI = gb.build_sto3g(Z_list, coords_bohr)
    nel_full = int(nel_neutral) - int(charge)
    if nel_full < 1 or nel_full - int(two_s) < 0 or ((nel_full - int(two_s)) % 2) != 0:
        raise ValueError(
            f"Invalid electron count nel_full={nel_full} (charge={charge}) for two_s={two_s}"
        )
    enuc, H_mo, ERI_mo = gb.rhf_canonical_mo(K_full, nel_full, enuc, S, Hcore, ERI)

    if active_el is None and active_orb is None:
        K_act, nel_act, e_core, h_act, eri_act = K_full, nel_full, float(enuc), H_mo, ERI_mo
    else:
        nel_target = nel_full if active_el is None else int(active_el)
        K_target = K_full if active_orb is None else int(active_orb)
        diff_el = nel_full - nel_target
        if diff_el < 0 or (diff_el % 2) != 0:
            raise ValueError(f"Incompatible active electrons {nel_target} vs full {nel_full}")
        n_core = diff_el // 2
        if n_core + K_target > K_full:
            raise ValueError(f"Requested {K_target} active orbs + {n_core} core > full K={K_full}")
        frozen_idx = list(range(n_core))
        active_idx = list(range(n_core, n_core + K_target))
        K_act, nel_act, e_core, h_act, eri_act = gb.project_active_space(
            enuc, H_mo, ERI_mo, frozen_idx, active_idx, nel_target
        )

    chem_sdk.write_fcidump(out_path, K_act, nel_act, two_s, e_core, h_act, eri_act)
    return {"K": K_act, "nelec": nel_act, "K_full": K_full, "nel_full": nel_full, "e_core": e_core}


def _screen_one_frame(task):
    cli = chem_sdk.ChemOracle(task["cli_path"])
    fpath = task["fcidump_path"]
    t0 = time.time()

    cert_dir = task.get("cert_dir")
    cert_path = None
    save_no_path = None
    cert_fcidump_path = None
    fpath_solve = fpath
    if cert_dir:
        os.makedirs(cert_dir, exist_ok=True)
        cert_path = os.path.join(cert_dir, f"frame_{task['idx']:03d}.cert.json")
        if task["auto_no"]:
            save_no_path = os.path.join(cert_dir, f"frame_{task['idx']:03d}_no.fcidump")
            cert_fcidump_path = save_no_path
        else:
            cert_fcidump_path = os.path.join(cert_dir, f"frame_{task['idx']:03d}.fcidump")
            if os.path.abspath(fpath) != os.path.abspath(cert_fcidump_path):
                shutil.copyfile(fpath, cert_fcidump_path)
            fpath_solve = cert_fcidump_path

    e_ai = task.get("e_ai")
    vtol = float(task["veto_tol"])
    t1_orbs = task["sub_orbs"] if task["t1"] else 0
    t2_orbs = task["sub_orbs"] if task["t2"] else 0

    veto_res = None
    if e_ai is not None:
        e_ai = float(e_ai)
        # Single-pass screening: run veto first with C3 margin mapped via e_ai + vtol, veto_tol=0.0
        # so E_lb > (e_ai + vtol) <=> e_ai < E_lb - vtol (zero false positives on narrow brackets).
        vt = cli.veto(
            fpath_solve,
            e_ai=e_ai + vtol,
            veto_tol=0.0,
            mode=task["mode"],
            two_s=task["two_s"],
            max_iters=task["max_iters"],
            tol=task["tol"],
            max_dets=task["max_dets"],
            t1=task["t1"],
            t1_orbs=t1_orbs,
            t2=task["t2"],
            t2_orbs=t2_orbs,
            auto_no=task["auto_no"],
            save_no_fcidump=save_no_path,
            cert_path=cert_path,
            threads=task["threads"],
        )
        vinfo = vt.get("veto_test") or {}
        early_vetoed = bool(vinfo.get("vetoed", 0))
        if (not early_vetoed) or task.get("fast_veto"):
            br = vt
        else:
            br = cli.bracket(
                fpath_solve,
                mode=task["mode"],
                two_s=task["two_s"],
                constrain_s2=True,
                max_iters=task["max_iters"],
                tol=task["tol"],
                max_dets=task["max_dets"],
                t1=task["t1"],
                t1_orbs=t1_orbs,
                t2=task["t2"],
                t2_orbs=t2_orbs,
                auto_no=task["auto_no"],
                save_no_fcidump=save_no_path,
                cert_path=cert_path,
                threads=task["threads"],
            )
    else:
        br = cli.bracket(
            fpath_solve,
            mode=task["mode"],
            two_s=task["two_s"],
            constrain_s2=True,
            max_iters=task["max_iters"],
            tol=task["tol"],
            max_dets=task["max_dets"],
            t1=task["t1"],
            t1_orbs=t1_orbs,
            t2=task["t2"],
            t2_orbs=t2_orbs,
            auto_no=task["auto_no"],
            save_no_fcidump=save_no_path,
            cert_path=cert_path,
            threads=task["threads"],
        )
        early_vetoed = False
        vinfo = {}

    no_ent = br.get("no_entropies")
    no_occs = br.get("no_occs")
    s1_no = float(sum(no_ent)) if no_ent else None
    max_ent = float(max(no_ent)) if no_ent else None
    mr_alert = bool(no_ent and (s1_no >= 1.0 or max_ent >= 0.35))

    e_lb = float(br["E_lb_cert"])
    e_ub = float(br["E_ub"])
    if e_ai is not None:
        if early_vetoed or e_ai < e_lb - vtol:
            hallucination_flag = "VETOED_FALSE_LOW"
        elif e_ai > e_ub + vtol:
            hallucination_flag = "FLAGGED_ABOVE_UB"
        else:
            hallucination_flag = "CERTIFIED_CONSISTENT"
        veto_res = {
            "E_ai": e_ai,
            "veto_tol": vtol,
            "early_vetoed": early_vetoed,
            "iter_stopped": int(vinfo.get("iter_stopped", br.get("iters", 0))),
            "veto_sec": float(vinfo.get("veto_sec", 0.0)),
            "status": hallucination_flag,
        }

    return {
        "frame": task["idx"],
        "label": task["label"],
        "K": int(br["K"]),
        "nelec": int(br["nelec"]),
        "two_s": task["two_s"],
        "E_lb_cert": e_lb,
        "E_ub": e_ub,
        "E_ci": float(br["E_ci"]),
        "E_hf0": float(br["E_hf0"]),
        "width_mha": float(br["interval_width"]) * 1000.0,
        "iters": int(br["iters"]),
        "auto_no": bool(task["auto_no"]),
        "no_occs": no_occs,
        "no_entropies": no_ent,
        "S1_no": s1_no,
        "mr_alert": mr_alert,
        "mlip_check": veto_res,
        "cert_path": cert_path,
        "cert_fcidump_path": cert_fcidump_path,
        "wall_sec": float(time.time() - t0),
        **(task.get("meta") or {}),
    }


def write_csv_report(report: dict, csv_path: str):
    """Export screening report frames to a flat CSV table."""
    fieldnames = [
        "frame", "label", "K", "nelec", "E_lb_cert", "E_ub", "width_mha",
        "dE_lb_kcal", "dE_ub_kcal", "S1_no", "mr_alert", "E_ai",
        "mlip_status", "early_vetoed", "iter_stopped", "wall_sec",
    ]
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report["frames"]:
            mc = r.get("mlip_check") or {}
            writer.writerow({
                "frame": r["frame"],
                "label": r["label"],
                "K": r["K"],
                "nelec": r["nelec"],
                "E_lb_cert": f"{r['E_lb_cert']:.10f}",
                "E_ub": f"{r['E_ub']:.10f}",
                "width_mha": f"{r['width_mha']:.6f}",
                "dE_lb_kcal": f"{r['dE_lb_kcal']:.6f}",
                "dE_ub_kcal": f"{r['dE_ub_kcal']:.6f}",
                "S1_no": "" if r["S1_no"] is None else f"{r['S1_no']:.6f}",
                "mr_alert": r["mr_alert"],
                "E_ai": "" if not mc else f"{mc['E_ai']:.10f}",
                "mlip_status": mc.get("status", "NO_CANDIDATE"),
                "early_vetoed": "" if not mc else mc.get("early_vetoed", False),
                "iter_stopped": "" if not mc else mc.get("iter_stopped", ""),
                "wall_sec": f"{r['wall_sec']:.6f}",
            })


def screen_trajectory(
    xyz_path=None,
    fcidump_paths=None,
    mlip_energies=None,
    two_s: int = 0,
    charge: int = 0,
    mode: str = "auto",
    active_el: int = None,
    active_orb: int = None,
    t1: bool = True,
    t2: bool = True,
    sub_orbs: int = 5,
    auto_no: bool = True,
    max_iters: int = 800,
    tol: float = 1e-6,
    max_dets: int = 1024,
    veto_tol: float = 5e-4,
    fast_veto: bool = False,
    cert_dir: str = None,
    workers: int = 4,
    cli_path: str = None,
):
    if cli_path is None:
        cli_path = os.path.join(ROOT, "build", "chem_cli")
    if cert_dir:
        os.makedirs(cert_dir, exist_ok=True)
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="chem_screen_") as tmpdir:
        tasks = []
        if xyz_path is not None:
            frames = parse_xyz_trajectory(xyz_path)
            for idx, (comment, Z_list, coords) in enumerate(frames):
                fpath = os.path.join(tmpdir, f"frame_{idx:03d}.fcidump")
                meta = build_frame_fcidump(
                    Z_list, coords, fpath, two_s=two_s, charge=charge,
                    active_el=active_el, active_orb=active_orb,
                )
                e_ai = mlip_energies[idx] if (mlip_energies is not None and idx < len(mlip_energies)) else None
                tasks.append({
                    "idx": idx,
                    "label": comment or f"frame_{idx}",
                    "fcidump_path": fpath,
                    "cli_path": cli_path,
                    "mode": mode, "two_s": two_s,
                    "max_iters": max_iters, "tol": tol, "max_dets": max_dets,
                    "t1": t1, "t2": t2, "sub_orbs": sub_orbs, "auto_no": auto_no,
                    "fast_veto": fast_veto, "cert_dir": cert_dir,
                    "threads": 1, "e_ai": e_ai, "veto_tol": veto_tol, "meta": meta,
                })
        elif fcidump_paths:
            for idx, fp in enumerate(fcidump_paths):
                e_ai = mlip_energies[idx] if (mlip_energies is not None and idx < len(mlip_energies)) else None
                tasks.append({
                    "idx": idx,
                    "label": os.path.basename(fp),
                    "fcidump_path": fp,
                    "cli_path": cli_path,
                    "mode": mode, "two_s": two_s,
                    "max_iters": max_iters, "tol": tol, "max_dets": max_dets,
                    "t1": t1, "t2": t2, "sub_orbs": sub_orbs, "auto_no": auto_no,
                    "fast_veto": fast_veto, "cert_dir": cert_dir,
                    "threads": 1, "e_ai": e_ai, "veto_tol": veto_tol, "meta": {},
                })
        else:
            raise ValueError("Specify xyz_path or fcidump_paths")

        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            results = list(pool.map(_screen_one_frame, tasks))

    # Relative barrier envelopes referenced to frame 0
    ref_lb = results[0]["E_lb_cert"]
    ref_ub = results[0]["E_ub"]
    for row in results:
        de_lb_ha = row["E_lb_cert"] - ref_ub
        de_ub_ha = row["E_ub"] - ref_lb
        row["dE_lb_mha"] = de_lb_ha * 1000.0
        row["dE_ub_mha"] = de_ub_ha * 1000.0
        row["dE_lb_kcal"] = de_lb_ha * HA_TO_KCAL
        row["dE_ub_kcal"] = de_ub_ha * HA_TO_KCAL

    n_vetoed = sum(
        1 for r in results if r["mlip_check"] and r["mlip_check"]["status"] != "CERTIFIED_CONSISTENT"
    )
    return {
        "n_frames": len(results),
        "n_flagged": n_vetoed,
        "elapsed_sec": float(time.time() - t0),
        "frames": results,
    }


def main():
    ap = argparse.ArgumentParser(description="CHEM Reaction-Path & MLIP Screening Tool")
    ap.add_argument("--xyz", help="Multi-frame .xyz trajectory path")
    ap.add_argument("--fcidumps", nargs="*", help="FCIDUMP file list")
    ap.add_argument("--mlip-json", help="JSON file containing list of E_ai values (Ha)")
    ap.add_argument("--two-s", type=int, default=0)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--active-el", type=int, default=None)
    ap.add_argument("--active-orb", type=int, default=None)
    ap.add_argument("--sub-orbs", type=int, default=5)
    ap.add_argument("--no-auto-no", action="store_true")
    ap.add_argument("--no-t1", action="store_true")
    ap.add_argument("--no-t2", action="store_true")
    ap.add_argument("--max-iters", type=int, default=800)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--veto-tol", type=float, default=5e-4)
    ap.add_argument("--fast-veto", action="store_true")
    ap.add_argument("--cert-dir", help="Optional directory to save .cert.json and verified .fcidump files")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", help="Output JSON summary path")
    ap.add_argument("--csv", help="Output CSV summary path")
    args = ap.parse_args()

    mlip_e = None
    if args.mlip_json:
        with open(args.mlip_json) as f:
            raw = json.load(f)
        mlip_e = raw["energies"] if isinstance(raw, dict) else raw

    report = screen_trajectory(
        xyz_path=args.xyz,
        fcidump_paths=args.fcidumps,
        mlip_energies=mlip_e,
        two_s=args.two_s,
        charge=args.charge,
        active_el=args.active_el,
        active_orb=args.active_orb,
        t1=not args.no_t1,
        t2=not args.no_t2,
        sub_orbs=args.sub_orbs,
        auto_no=not args.no_auto_no,
        max_iters=args.max_iters,
        tol=args.tol,
        veto_tol=args.veto_tol,
        fast_veto=args.fast_veto,
        cert_dir=args.cert_dir,
        workers=args.workers,
    )
    print(f"=== CHEM Screener: {report['n_frames']} frames in {report['elapsed_sec']:.2f}s (flagged={report['n_flagged']}) ===")
    for r in report["frames"]:
        st = r["mlip_check"]["status"] if r["mlip_check"] else "NO_MLIP"
        print(
            f"  [{r['frame']:2d}] {r['label']:24s} | [LB,UB]=[{r['E_lb_cert']:.6f}, {r['E_ub']:.6f}] "
            f"(w={r['width_mha']:.2f}mHa) dE=[{r['dE_lb_kcal']:+6.2f},{r['dE_ub_kcal']:+6.2f}]kcal "
            f"| S1={r['S1_no'] if r['S1_no'] is not None else 0:.2f} MR={r['mr_alert']} | {st}"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    if args.csv:
        write_csv_report(report, args.csv)


if __name__ == "__main__":
    main()
