"""
CHEM v0.2.0 Python SDK & AI/MLIP Active-Learning Veto Wrapper
=============================================================
Clean, zero-dependency (NumPy optional) Python interface around `build/chem_cli`
plus a PySCF CASCI/CASSCF FCIDUMP export bridge (`from_pyscf_casci`).
"""

import ctypes
import json
import os
import re
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


def read_fcidump(path):
    """Read a Molpro/PySCF FCIDUMP file into full unnormalized NumPy arrays."""
    import numpy as np

    with open(path) as f:
        lines = f.readlines()

    header_lines = []
    body_start = 0
    for idx, line in enumerate(lines):
        header_lines.append(line)
        if "&END" in line or "/" in line.strip():
            body_start = idx + 1
            break

    header = " ".join(header_lines)
    m_k = re.search(r"NORB\s*=\s*(\d+)", header, re.IGNORECASE)
    m_ne = re.search(r"NELEC\s*=\s*(\d+)", header, re.IGNORECASE)
    m_ms2 = re.search(r"MS2\s*=\s*(-?\d+)", header, re.IGNORECASE)
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

    return {"K": K, "nelec": nelec, "ms2": ms2, "e_core": e_core, "h1": h1, "eri": eri}


def rotate_integrals(h1, eri, U):
    """Apply an orthogonal basis rotation U to (h1, eri) with 8-fold symmetrization."""
    import numpy as np

    U = np.asarray(U, dtype=np.float64)
    h1_rot = U.T @ np.asarray(h1, dtype=np.float64) @ U
    h1_rot = 0.5 * (h1_rot + h1_rot.T)
    t1 = np.tensordot(np.asarray(eri, dtype=np.float64), U, axes=([0], [0]))
    t2 = np.tensordot(t1, U, axes=([0], [0]))
    t3 = np.tensordot(t2, U, axes=([0], [0]))
    e4 = np.tensordot(t3, U, axes=([0], [0]))
    eri_rot = 0.125 * (
        e4
        + e4.transpose(1, 0, 2, 3)
        + e4.transpose(0, 1, 3, 2)
        + e4.transpose(1, 0, 3, 2)
        + e4.transpose(2, 3, 0, 1)
        + e4.transpose(3, 2, 0, 1)
        + e4.transpose(2, 3, 1, 0)
        + e4.transpose(3, 2, 1, 0)
    )
    return h1_rot, eri_rot


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
                            tol=1e-6, max_dets=1024, t2=False, t2_orbs=0, t1=False, t1_orbs=0,
                            auto_no=False, save_no_fcidump=None, threads=0, witness_path=None,
                            dump_rdm_grad=None, cert_path=None, cert_bin_path=None):
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
        if t1:
            flags.append("--t1")
        if t1_orbs and int(t1_orbs) > 0:
            flags += ["--t1-orbs", str(int(t1_orbs))]
        if auto_no:
            flags.append("--auto-no")
        if save_no_fcidump:
            flags += ["--save-no-fcidump", str(save_no_fcidump)]
        if threads and int(threads) > 0:
            flags += ["--threads", str(int(threads))]
        if witness_path:
            flags += ["--witness", str(witness_path)]
        if dump_rdm_grad:
            flags += ["--dump-rdm-grad", str(dump_rdm_grad)]
        if cert_path:
            flags += ["--cert", str(cert_path)]
        if cert_bin_path:
            flags += ["--cert-bin", str(cert_bin_path)]
        return flags

    def bracket(self, fcidump_path, mode="auto", two_s=None, constrain_s2=True, max_iters=800,
                tol=1e-6, max_dets=1024, t2=False, t2_orbs=0, t1=False, t1_orbs=0,
                auto_no=False, save_no_fcidump=None, threads=0,
                witness_path=None, dump_rdm_grad=None, cert_path=None, cert_bin_path=None):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            args = ["bracket", fcidump_path] + self._build_common_flags(
                mode=mode, two_s=two_s, constrain_s2=constrain_s2, max_iters=max_iters,
                tol=tol, max_dets=max_dets, t2=t2, t2_orbs=t2_orbs, t1=t1, t1_orbs=t1_orbs,
                auto_no=auto_no, save_no_fcidump=save_no_fcidump,
                threads=threads, witness_path=witness_path, dump_rdm_grad=dump_rdm_grad,
                cert_path=cert_path, cert_bin_path=cert_bin_path
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
             t2_orbs=0, t1=False, t1_orbs=0, auto_no=False, save_no_fcidump=None,
             threads=0, witness_path=None, cert_path=None, cert_bin_path=None):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            args = ["veto", fcidump_path, "--e-ai", str(float(e_ai)),
                    "--veto-tol", str(float(veto_tol))] + self._build_common_flags(
                mode=mode, two_s=two_s, constrain_s2=constrain_s2, max_iters=max_iters,
                tol=tol, max_dets=max_dets, t2=t2, t2_orbs=t2_orbs, t1=t1, t1_orbs=t1_orbs,
                auto_no=auto_no, save_no_fcidump=save_no_fcidump,
                threads=threads, witness_path=witness_path, cert_path=cert_path,
                cert_bin_path=cert_bin_path
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
                   t1=False, t1_orbs=0, auto_no=False, save_no_fcidump=None,
                   threads=0, witness_path=None):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            args = ["observable", fcidump_path, "--type", str(obs_type)] + self._build_common_flags(
                mode=mode, two_s=two_s, constrain_s2=constrain_s2, max_iters=max_iters,
                tol=tol, max_dets=max_dets, t2=t2, t2_orbs=t2_orbs, t1=t1, t1_orbs=t1_orbs,
                auto_no=auto_no, save_no_fcidump=save_no_fcidump,
                threads=threads, witness_path=witness_path
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

    def exact_verify(self, cert_path, fcidump_path=None):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            args = ["exact-verify"]
            if fcidump_path:
                args.append(fcidump_path)
            args += [cert_path, "--json", tmp_json]
            proc = subprocess.run([self.cli_path] + args, capture_output=True, text=True, check=False)
            if os.path.exists(tmp_json) and os.path.getsize(tmp_json) > 0:
                with open(tmp_json) as f:
                    res = json.load(f)
                res["returncode"] = proc.returncode
                return res
            return {
                "ok": False,
                "returncode": proc.returncode,
                "message": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)

    def emit_lean(self, fcidump_path, cert_path, out_lean_path, module_name=None):
        args = ["emit-lean", fcidump_path, cert_path, "--out", out_lean_path]
        if module_name:
            args += ["--name", str(module_name)]
        out = self._run(args)
        return {"ok": True, "out_lean": out_lean_path, "message": out.strip()}

    def gpc(self, fcidump_path, mode="auto", two_s=None, witness_path=None, certify_bd=False):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name
        try:
            args = ["gpc", fcidump_path, "--mode", str(mode)]
            if two_s is not None:
                args += ["--two-s", str(int(two_s))]
            if witness_path:
                args += ["--witness", str(witness_path)]
            if certify_bd:
                args.append("--certify-bd")
            args += ["--json", tmp_json]
            self._run(args)
            with open(tmp_json) as f:
                return json.load(f)
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)

    def mlip_screen(self, fcidump_path, e_ai, f_ai=None, f_plus=None, f_minus=None,
                    dr=0.005, veto_tol=1e-4, force_tol=1e-4, mode="auto",
                    two_s=None, t2=False, t2_orbs=0, t1=False, t1_orbs=0,
                    auto_no=False, threads=0, max_iters=800):
        """
        Active-learning gate for AI/MLIP candidates: checks energy lower/upper bounds
        first via Anytime Early-Veto (using e_ai + veto_tol so strict exclusion margin
        is e_ai < E_lb_cert - veto_tol with zero false positives on narrow brackets),
        and if force finite-difference Hamiltonians are supplied, also certifies
        Hellmann-Feynman nuclear force consistency [F_lb, F_ub].
        """
        e_ai_f = float(e_ai)
        vtol = float(veto_tol)
        ev = self.veto(
            fcidump_path, e_ai_f + vtol, veto_tol=0.0, mode=mode, two_s=two_s,
            t2=t2, t2_orbs=t2_orbs, t1=t1, t1_orbs=t1_orbs, auto_no=auto_no,
            threads=threads, max_iters=max_iters,
        )
        vt = ev.get("veto_test") or {}
        e_lb = float(ev["E_lb_cert"])
        e_ub = float(ev["E_ub"])
        energy_vetoed = bool(vt.get("vetoed", 0)) or (e_ai_f < e_lb - vtol)
        above_ub = e_ai_f > e_ub + vtol
        if energy_vetoed:
            status = "VETOED_FALSE_LOW"
        elif above_ub:
            status = "FLAGGED_ABOVE_UB"
        else:
            status = "CERTIFIED_CONSISTENT"
        out = {
            "energy_vetoed": energy_vetoed,
            "above_ub": above_ub,
            "status": status,
            "E_lb_cert": e_lb,
            "E_ub": e_ub,
            "iter_stopped": vt.get("iter_stopped"),
            "force_vetoed": False,
            "force_bracket": None,
        }
        if (not energy_vetoed) and (f_ai is not None) and f_plus and f_minus:
            fb = self.observable(
                fcidump_path, f"hf-force:{f_plus}:{f_minus}:{dr}",
                mode=mode, two_s=two_s, t2=t2, t2_orbs=t2_orbs, t1=t1, t1_orbs=t1_orbs,
                auto_no=auto_no, threads=threads,
            )
            out["force_bracket"] = fb
            out["force_vetoed"] = (f_ai < fb["O_lb_cert"] - force_tol) or (
                f_ai > fb["O_ub_cert"] + force_tol
            )
        out["vetoed"] = out["energy_vetoed"] or out["force_vetoed"]
        return out


_LIBCHEM_CACHE = {}


def _load_libchem(lib_path=None):
    if lib_path is None:
        lib_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "build",
            "libchem.dylib",
        )
    abspath = os.path.abspath(lib_path)
    if abspath in _LIBCHEM_CACHE:
        return _LIBCHEM_CACHE[abspath]
    lib = ctypes.CDLL(abspath)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_int_p = ctypes.POINTER(ctypes.c_int)

    lib.chem_session_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
    ]
    lib.chem_session_create.restype = ctypes.c_void_p

    lib.chem_session_free.argtypes = [ctypes.c_void_p]
    lib.chem_session_free.restype = None

    lib.chem_session_solve.argtypes = [
        ctypes.c_void_p, ctypes.c_double, c_double_p, c_double_p,
        ctypes.c_int, ctypes.c_double, ctypes.c_int,
        c_double_p, c_double_p, c_int_p, c_int_p,
        c_double_p, c_double_p, ctypes.c_char_p, ctypes.c_size_t,
    ]
    lib.chem_session_solve.restype = ctypes.c_int

    _LIBCHEM_CACHE[abspath] = lib
    return lib


class ChemFastSession:
    """
    Persistent in-process BPSDP solver session backed by build/libchem.dylib.
    Reuses factorized H^{-1}, Schur complement S_eq, worker threads, and primal/dual iterates
    across warm-started Hamiltonian optimization trajectories.
    """

    MODE_MAP = {"auto": 0, "a": 1, "b": 2}

    def __init__(self, K, nelec, ms2=0, mode="auto", two_s=-1, constrain_s2=True,
                 t1=False, t1_orbs=0, t2=False, t2_orbs=0, threads=1, lib_path=None):
        self._lib = _load_libchem(lib_path)
        self.K = int(K)
        mode_i = self.MODE_MAP.get(str(mode).lower(), 0)
        ts_i = abs(int(ms2)) if two_s is None else int(two_s)
        err = ctypes.create_string_buffer(512)
        self._ptr = self._lib.chem_session_create(
            self.K, int(nelec), int(ms2), mode_i, ts_i, 1 if constrain_s2 else 0,
            1 if t1 else 0, int(t1_orbs), 1 if t2 else 0, int(t2_orbs),
            int(threads), err, ctypes.sizeof(err),
        )
        if not self._ptr:
            raise RuntimeError(f"chem_session_create failed: {err.value.decode()}")

    @classmethod
    def from_fcidump(cls, path, mode="auto", two_s=None, constrain_s2=True,
                     t1=False, t1_orbs=0, t2=False, t2_orbs=0, threads=1, lib_path=None):
        """Instantiate a ChemFastSession directly from an FCIDUMP file and return (session, ham)."""
        ham = read_fcidump(path)
        ts = ham["ms2"] if two_s is None else int(two_s)
        sess = cls(
            ham["K"], ham["nelec"], ms2=ham["ms2"], mode=mode, two_s=ts,
            constrain_s2=constrain_s2, t1=t1, t1_orbs=t1_orbs,
            t2=t2, t2_orbs=t2_orbs, threads=threads, lib_path=lib_path,
        )
        return sess, ham

    def solve(self, e_core, h1, eri, max_iters=800, tol=1e-6, warm_start=True, return_grad=True):
        import numpy as np

        if not self._ptr:
            raise RuntimeError("ChemFastSession is closed")
        K = self.K
        h1_arr = np.ascontiguousarray(h1, dtype=np.float64).reshape(K, K)
        eri_arr = np.ascontiguousarray(eri, dtype=np.float64).reshape(K, K, K, K)

        e_lb = ctypes.c_double(0.0)
        e_pr = ctypes.c_double(0.0)
        iters = ctypes.c_int(0)
        conv = ctypes.c_int(0)

        if return_grad:
            gh1 = np.empty((K, K), dtype=np.float64)
            geri = np.empty((K, K, K, K), dtype=np.float64)
            gh1_p = gh1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            geri_p = geri.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            gh1, geri = None, None
            gh1_p = ctypes.POINTER(ctypes.c_double)()
            geri_p = ctypes.POINTER(ctypes.c_double)()

        err = ctypes.create_string_buffer(512)
        rc = self._lib.chem_session_solve(
            self._ptr,
            float(e_core),
            h1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            eri_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(max_iters),
            float(tol),
            1 if warm_start else 0,
            ctypes.byref(e_lb),
            ctypes.byref(e_pr),
            ctypes.byref(iters),
            ctypes.byref(conv),
            gh1_p,
            geri_p,
            err,
            ctypes.sizeof(err),
        )
        if rc != 0:
            raise RuntimeError(f"chem_session_solve failed: {err.value.decode()}")
        return {
            "E_lb_cert": float(e_lb.value),
            "E_primal": float(e_pr.value),
            "iters": int(iters.value),
            "converged": bool(conv.value),
            "grad_h1": gh1,
            "grad_eri": geri,
        }

    def close(self):
        if getattr(self, "_ptr", None):
            self._lib.chem_session_free(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
