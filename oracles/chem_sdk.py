"""
CHEM v0.2.0 Python SDK & AI/MLIP Active-Learning Veto Wrapper
=============================================================
Clean, zero-dependency (NumPy optional) Python interface around `build/chem_cli`
plus a PySCF CASCI/CASSCF FCIDUMP export bridge (`from_pyscf_casci`).
"""

import json
import os
import subprocess
import tempfile


def write_fcidump(path, K, nelec, ms2, e_core, h1, eri):
    """Write a canonical Molpro/PySCF FCIDUMP file."""
    with open(path, "w") as f:
        f.write(f" &FCI NORB={K},NELEC={nelec},MS2={ms2},\n")
        f.write("  ORBSYM=" + ",".join(["1"] * K) + ",\n")
        f.write("  ISYM=1,\n &END\n")
        for p in range(K):
            for q in range(p + 1):
                pq = p * (p + 1) // 2 + q
                for r in range(K):
                    for s in range(r + 1):
                        rs = r * (r + 1) // 2 + s
                        if rs > pq:
                            continue
                        val = float(eri[p, q, r, s])
                        if abs(val) > 1e-16:
                            f.write(f"{val:25.16e} {p+1:4d} {q+1:4d} {r+1:4d} {s+1:4d}\n")
        for p in range(K):
            for q in range(p + 1):
                val = float(h1[p, q])
                if abs(val) > 1e-16:
                    f.write(f"{val:25.16e} {p+1:4d} {q+1:4d}    0    0\n")
        f.write(f"{float(e_core):25.16e}    0    0    0    0\n")
    return path


def from_pyscf_casci(mc, out_fcidump_path, ms2=0):
    """
    Export an active-space Hamiltonian from a PySCF CASCI/CASSCF object.
    Imports pyscf lazily so CHEM SDK has zero mandatory runtime dependencies.
    """
    from pyscf import ao2mo  # type: ignore

    ncas = int(mc.ncas)
    ne_a, ne_b = mc.nelecas
    nelec = int(ne_a + ne_b)
    h1e, ecore = mc.get_h1eff()
    eri4 = ao2mo.restore(1, mc.get_h2eff(), ncas)
    return write_fcidump(out_fcidump_path, ncas, nelec, ms2, float(ecore), h1e, eri4)


class ChemOracle:
    """High-assurance certified lower/upper bracket & veto client wrapping chem_cli."""

    def __init__(self, cli_path="build/chem_cli"):
        self.cli_path = cli_path

    def _run(self, args):
        proc = subprocess.run([self.cli_path] + args, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"chem_cli failed (rc={proc.returncode}): {' '.join(args)}\n{proc.stderr}\n{proc.stdout}"
            )
        return proc.stdout

    def _build_common_flags(self, mode="auto", two_s=None, constrain_s2=True, max_iters=800,
                            tol=1e-6, max_dets=1024, t2=False, t2_orbs=0, threads=0,
                            witness_path=None, cert_path=None, cert_bin_path=None):
        flags = ["--mode", str(mode), "--max-iters", str(int(max_iters)),
                 "--tol", str(float(tol)), "--max-dets", str(int(max_dets))]
        if two_s is not None:
            flags += ["--two-s", str(int(two_s))]
        if not constrain_s2:
            flags.append("--no-s2")
        if t2:
            flags.append("--t2")
        if t2_orbs and int(t2_orbs) > 0:
            flags += ["--t2-orbs", str(int(t2_orbs))]
        if threads and int(threads) > 0:
            flags += ["--threads", str(int(threads))]
        if witness_path:
            flags += ["--witness", str(witness_path)]
        if cert_path:
            flags += ["--cert", str(cert_path)]
        if cert_bin_path:
            flags += ["--cert-bin", str(cert_bin_path)]
        return flags

    def bracket(self, fcidump_path, mode="auto", two_s=None, constrain_s2=True, max_iters=800,
                tol=1e-6, max_dets=1024, t2=False, t2_orbs=0, threads=0,
                witness_path=None, cert_path=None, cert_bin_path=None):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            args = ["bracket", fcidump_path] + self._build_common_flags(
                mode=mode, two_s=two_s, constrain_s2=constrain_s2, max_iters=max_iters,
                tol=tol, max_dets=max_dets, t2=t2, t2_orbs=t2_orbs, threads=threads,
                witness_path=witness_path, cert_path=cert_path, cert_bin_path=cert_bin_path
            ) + ["--json", tmp_json]
            self._run(args)
            with open(tmp_json) as f:
                payload = json.load(f)
            return payload["results"][0]
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)

    def veto(self, fcidump_path, e_ai, veto_tol=1e-4, mode="auto", two_s=None,
             constrain_s2=True, max_iters=800, tol=1e-6, max_dets=1024, t2=False,
             t2_orbs=0, threads=0, witness_path=None, cert_path=None, cert_bin_path=None):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            args = ["veto", fcidump_path, "--e-ai", str(float(e_ai)),
                    "--veto-tol", str(float(veto_tol))] + self._build_common_flags(
                mode=mode, two_s=two_s, constrain_s2=constrain_s2, max_iters=max_iters,
                tol=tol, max_dets=max_dets, t2=t2, t2_orbs=t2_orbs, threads=threads,
                witness_path=witness_path, cert_path=cert_path, cert_bin_path=cert_bin_path
            ) + ["--json", tmp_json]
            self._run(args)
            with open(tmp_json) as f:
                payload = json.load(f)
            return payload["results"][0]
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)

    def observable(self, fcidump_path, obs_type, mode="auto", two_s=None, constrain_s2=True,
                   max_iters=350, tol=1e-6, max_dets=1024, t2=False, t2_orbs=0,
                   threads=0, witness_path=None):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            args = ["observable", fcidump_path, "--type", str(obs_type)] + self._build_common_flags(
                mode=mode, two_s=two_s, constrain_s2=constrain_s2, max_iters=max_iters,
                tol=tol, max_dets=max_dets, t2=t2, t2_orbs=t2_orbs, threads=threads,
                witness_path=witness_path
            ) + ["--json", tmp_json]
            self._run(args)
            with open(tmp_json) as f:
                return json.load(f)
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)

    def cert_verify(self, cert_path, fcidump_path=None):
        args = ["cert-verify"]
        if fcidump_path:
            args.append(fcidump_path)
        args.append(cert_path)
        proc = subprocess.run([self.cli_path] + args, capture_output=True, text=True, check=False)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "message": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }

    def mlip_screen(self, fcidump_path, e_ai, f_ai=None, f_plus=None, f_minus=None,
                    dr=0.005, veto_tol=1e-4, force_tol=1e-4, mode="auto",
                    two_s=None, t2=False, max_iters=800):
        """
        Active-learning gate for AI/MLIP candidates: checks energy lower bound first via
        Anytime Early-Veto, and if force finite-difference Hamiltonians are supplied,
        also certifies Hellmann-Feynman nuclear force consistency [F_lb, F_ub].
        """
        ev = self.veto(fcidump_path, e_ai, veto_tol=veto_tol, mode=mode, two_s=two_s,
                       t2=t2, max_iters=max_iters)
        vt = ev.get("veto_test") or {}
        energy_vetoed = bool(vt.get("vetoed", 0))
        out = {
            "energy_vetoed": energy_vetoed,
            "E_lb_cert": ev["E_lb_cert"],
            "E_ub": ev["E_ub"],
            "iter_stopped": vt.get("iter_stopped"),
            "force_vetoed": False,
            "force_bracket": None,
        }
        if (not energy_vetoed) and (f_ai is not None) and f_plus and f_minus:
            fb = self.observable(fcidump_path, f"hf-force:{f_plus}:{f_minus}:{dr}",
                                 mode=mode, two_s=two_s, t2=t2)
            out["force_bracket"] = fb
            out["force_vetoed"] = (f_ai < fb["O_lb_cert"] - force_tol) or (
                f_ai > fb["O_ub_cert"] + force_tol
            )
        out["vetoed"] = out["energy_vetoed"] or out["force_vetoed"]
        return out
