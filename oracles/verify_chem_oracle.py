#!/usr/bin/env python3
"""
Independent Python/NumPy Verification Oracle & Certificate Auditor for CHEM.
Zero shared C code: independently parses FCIDUMPs, verifies SHA-256 commitments,
verifies Mode A & Mode B matrix-free adjoints (< 1e-14), solves exact FCI,
audits every emitted JSON certificate (both dual Weyl lower bound & variational
wavefunction witness), verifies Gate 1-4 criteria, and writes CHEM_VERIFICATION_REPORT.md.
"""
import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import struct
import sys
import numpy as np


SQRT2 = math.sqrt(2.0)
ISQRT2 = 1.0 / SQRT2
DBL_EPS = float(np.finfo(np.float64).eps)


def parse_fcidump(path):
    with open(path, "r") as f:
        lines = f.readlines()

    hdr_lines = []
    idx = 0
    while idx < len(lines):
        ln = lines[idx]
        hdr_lines.append(ln)
        idx += 1
        if "/" in ln or "&END" in ln.upper():
            break
    hdr = " ".join(hdr_lines).upper()

    def get_int(key, default=None):
        import re
        m = re.search(rf"\b{key}\s*=\s*(-?\d+)", hdr)
        return int(m.group(1)) if m else default

    K = get_int("NORB")
    nelec = get_int("NELEC")
    ms2 = get_int("MS2", 0)
    assert K is not None and nelec is not None, f"Missing NORB/NELEC in {path}"
    na = (nelec + ms2) // 2
    nb = (nelec - ms2) // 2

    h1 = np.zeros((K, K), dtype=np.float64)
    eri = np.zeros((K, K, K, K), dtype=np.float64)
    e_core = 0.0

    for ln in lines[idx:]:
        s = ln.strip()
        if not s or s[0] in "!#*":
            continue
        toks = s.split()
        val = float(toks[0].replace("D", "E").replace("d", "e"))
        p, q, r, s_i = int(toks[1]), int(toks[2]), int(toks[3]), int(toks[4])
        if p == 0 and q == 0 and r == 0 and s_i == 0:
            e_core = val
        elif p > 0 and q > 0 and r == 0 and s_i == 0:
            h1[p - 1, q - 1] = h1[q - 1, p - 1] = val
        elif p > 0 and q > 0 and r > 0 and s_i > 0:
            p0, q0, r0, s0 = p - 1, q - 1, r - 1, s_i - 1
            for a, b in ((p0, q0), (q0, p0)):
                for c, d in ((r0, s0), (s0, r0)):
                    eri[a, b, c, d] = val
                    eri[c, d, a, b] = val

    # Canonical SHA-256 matching C chem_sha256_hex
    h1_n = np.where(h1 == 0.0, 0.0, h1).astype("<f8")
    eri_n = np.where(eri == 0.0, 0.0, eri).astype("<f8")
    ec_n = 0.0 if e_core == 0.0 else float(e_core)
    sha = hashlib.sha256()
    sha.update(struct.pack("<iii", K, nelec, ms2))
    sha.update(struct.pack("<d", ec_n))
    sha.update(h1_n.tobytes(order="C"))
    sha.update(eri_n.tobytes(order="C"))
    return {
        "K": K, "nelec": nelec, "ms2": ms2, "na": na, "nb": nb,
        "E_core": e_core, "h": h1, "eri": eri, "sha256": sha.hexdigest()
    }


def svec_unpack(n, v):
    M = np.zeros((n, n), dtype=np.float64)
    p = 0
    for i in range(n):
        M[i, i] = v[p]
        p += 1
        for j in range(i + 1, n):
            val = ISQRT2 * v[p]
            M[i, j] = val
            M[j, i] = val
            p += 1
    return M


def svec_pack_general(M):
    n = M.shape[0]
    v = np.zeros(n * (n + 1) // 2, dtype=np.float64)
    p = 0
    for i in range(n):
        v[p] = M[i, i]
        p += 1
        for j in range(i + 1, n):
            v[p] = ISQRT2 * (M[i, j] + M[j, i])
            p += 1
    return v


def idx_sym_pair(K, p, q):
    if p > q:
        p, q = q, p
    return p * K - p * (p - 1) // 2 + (q - p)


def idx_asym_pair(K, p, q):
    if p == q:
        return -1, 0
    if p > q:
        p, q = q, p
        sgn = -1
    else:
        sgn = 1
    return p * (K - 1) - p * (p - 1) // 2 + (q - p - 1), sgn


class T2HelperPy:
    """Independent vectorized/tensor partial 3-RDM T2 core forward & adjoint operators."""

    def __init__(self, t2_orbs):
        self.orbs = sorted(t2_orbs)
        self.Ks = Ks = len(self.orbs)
        self.naa = Ks * (Ks - 1) // 2
        self.nab = Ks * Ks
        self.naab = self.naa * Ks
        self.nabb = Ks ** 3
        self.nmix = self.naab + self.nabb
        self.pairs = [(p, q) for p in range(Ks) for q in range(p + 1, Ks)]
        self.P1 = np.full((Ks, Ks), -1, dtype=int)
        self.sgn1 = np.zeros((Ks, Ks))
        for p in range(Ks):
            for q in range(Ks):
                idx1, sg = idx_asym_pair(Ks, p, q)
                self.P1[p, q] = idx1
                self.sgn1[p, q] = float(sg)
        self.sz_aab = self.naab * (self.naab + 1) // 2
        self.sz_mix = self.nmix * (self.nmix + 1) // 2

    def _expand_daa4(self, Daa):
        Ks = self.Ks
        if self.naa == 0:
            return np.zeros((Ks, Ks, Ks, Ks))
        P1f = np.maximum(self.P1.ravel(), 0)
        sg = self.sgn1[:, :, None, None] * self.sgn1[None, None, :, :]
        return sg * Daa[np.ix_(P1f, P1f)].reshape(Ks, Ks, Ks, Ks)

    def _contract_daa4(self, W):
        Daa_h = np.zeros((self.naa, self.naa))
        if self.naa == 0:
            return Daa_h
        mask = (self.P1 >= 0).ravel()
        P1v = self.P1.ravel()[mask]
        Wsg = self.sgn1[:, :, None, None] * self.sgn1[None, None, :, :] * W
        np.add.at(Daa_h, (P1v[:, None], P1v[None, :]),
                  Wsg.reshape(self.Ks ** 2, self.Ks ** 2)[np.ix_(mask, mask)])
        return Daa_h

    def aab_fwd(self, gb, Daa, Dab):
        Ks, naa, naab = self.Ks, self.naa, self.naab
        D4 = Dab.reshape(Ks, Ks, Ks, Ks)
        Daa4 = self._expand_daa4(Daa)
        eye = np.eye(Ks)
        T = (
            np.einsum("pu,qv,wr->pqruvw", eye, eye, gb)
            + np.einsum("rw,pquv->pqruvw", eye, Daa4)
            - np.einsum("pu,qwvr->pqruvw", eye, D4)
            + np.einsum("qu,pwvr->pqruvw", eye, D4)
            + np.einsum("pv,qwur->pqruvw", eye, D4)
            - np.einsum("qv,pwur->pqruvw", eye, D4)
        )
        pf = np.array([p * Ks + q for p, q in self.pairs], dtype=int)
        rk = np.arange(Ks)
        return T.reshape(Ks * Ks, Ks, Ks * Ks, Ks)[np.ix_(pf, rk, pf, rk)].reshape(naab, naab)

    def aab_adj(self, Y):
        Ks, naa = self.Ks, self.naa
        pf = np.array([p * Ks + q for p, q in self.pairs], dtype=int)
        rk = np.arange(Ks)
        Y4 = np.zeros((Ks * Ks, Ks, Ks * Ks, Ks))
        Y4[np.ix_(pf, rk, pf, rk)] = Y.reshape(naa, Ks, naa, Ks)
        Yf = Y4.reshape(Ks, Ks, Ks, Ks, Ks, Ks)
        eye = np.eye(Ks)
        gb_h = np.einsum("pqruvw,pu,qv->wr", Yf, eye, eye)
        Daa_h = np.einsum("irjr->ij", Y.reshape(naa, Ks, naa, Ks))
        D4_h = (
            -np.einsum("pqruvw,pu->qwvr", Yf, eye)
            + np.einsum("pqruvw,qu->pwvr", Yf, eye)
            + np.einsum("pqruvw,pv->qwur", Yf, eye)
            - np.einsum("pqruvw,qv->pwur", Yf, eye)
        )
        return gb_h, Daa_h, D4_h.reshape(self.nab, self.nab)

    def mix_fwd(self, ga, gb, Daa, Dbb, Dab):
        Ks, naa, naab, nmix = self.Ks, self.naa, self.naab, self.nmix
        D4 = Dab.reshape(Ks, Ks, Ks, Ks)
        Daa4 = self._expand_daa4(Daa)
        Dbb4 = self._expand_daa4(Dbb)
        eye = np.eye(Ks)
        pf = np.array([p * Ks + q for p, q in self.pairs], dtype=int)
        rk = np.arange(Ks)

        Taaa = (
            np.einsum("pu,qv,wr->pqruvw", eye, eye, ga)
            + np.einsum("rw,pquv->pqruvw", eye, Daa4)
            - np.einsum("pu,qwvr->pqruvw", eye, Daa4)
            + np.einsum("qu,pwvr->pqruvw", eye, Daa4)
            + np.einsum("pv,qwur->pqruvw", eye, Daa4)
            - np.einsum("qv,pwur->pqruvw", eye, Daa4)
        )
        Maaa = Taaa.reshape(Ks * Ks, Ks, Ks * Ks, Ks)[np.ix_(pf, rk, pf, rk)].reshape(naab, naab)

        Tabb = (
            np.einsum("pu,qv,wr->pqruvw", eye, eye, gb)
            + np.einsum("rw,pquv->pqruvw", eye, D4)
            - np.einsum("pu,wqrv->pqruvw", eye, Dbb4)
            - np.einsum("qv,pwur->pqruvw", eye, D4)
        ).reshape(self.nabb, self.nabb)

        Tcr = (
            np.einsum("pu,qwrv->pqruvw", eye, D4)
            - np.einsum("qu,pwrv->pqruvw", eye, D4)
        )
        Mcr = Tcr.reshape(Ks * Ks, Ks, Ks * Ks * Ks)[np.ix_(pf, rk, range(self.nabb))].reshape(
            naab, self.nabb
        )

        M = np.zeros((nmix, nmix))
        M[:naab, :naab] = Maaa
        M[naab:, naab:] = Tabb
        M[:naab, naab:] = Mcr
        M[naab:, :naab] = Mcr.T
        return M

    def mix_adj(self, Y):
        Ks, naa, naab = self.Ks, self.naa, self.naab
        eye = np.eye(Ks)
        pf = np.array([p * Ks + q for p, q in self.pairs], dtype=int)
        rk = np.arange(Ks)

        Yaaa_blk = Y[:naab, :naab]
        Y4a = np.zeros((Ks * Ks, Ks, Ks * Ks, Ks))
        Y4a[np.ix_(pf, rk, pf, rk)] = Yaaa_blk.reshape(naa, Ks, naa, Ks)
        Yfa = Y4a.reshape(Ks, Ks, Ks, Ks, Ks, Ks)
        ga_h = np.einsum("pqruvw,pu,qv->wr", Yfa, eye, eye)
        Waa = (
            -np.einsum("pqruvw,pu->qwvr", Yfa, eye)
            + np.einsum("pqruvw,qu->pwvr", Yfa, eye)
            + np.einsum("pqruvw,pv->qwur", Yfa, eye)
            - np.einsum("pqruvw,qv->pwur", Yfa, eye)
        )
        Daa_h = np.einsum("irjr->ij", Yaaa_blk.reshape(naa, Ks, naa, Ks)) + self._contract_daa4(Waa)

        Yabb = Y[naab:, naab:].reshape(Ks, Ks, Ks, Ks, Ks, Ks)
        gb_h = np.einsum("pqruvw,pu,qv->wr", Yabb, eye, eye)
        Wbb = -np.einsum("pqruvw,pu->wqrv", Yabb, eye)
        Dbb_h = self._contract_daa4(Wbb)
        D4_h = np.einsum("pqruvw,rw->pquv", Yabb, eye) - np.einsum("pqruvw,qv->pwur", Yabb, eye)

        Ycr_blk = Y[:naab, naab:] + Y[naab:, :naab].T
        Y4c = np.zeros((Ks * Ks, Ks, self.nabb))
        Y4c[np.ix_(pf, rk, range(self.nabb))] = Ycr_blk.reshape(naa, Ks, self.nabb)
        Yfc = Y4c.reshape(Ks, Ks, Ks, Ks, Ks, Ks)
        D4_h = (
            D4_h
            + np.einsum("pqruvw,pu->qwrv", Yfc, eye)
            - np.einsum("pqruvw,qu->pwrv", Yfc, eye)
        )
        return ga_h, gb_h, Daa_h, Dbb_h, D4_h.reshape(self.nab, self.nab)


class ModeAPy:
    def __init__(self, K, ne, t2_orbs=None):
        self.K = K
        self.ne = ne
        self.m0 = K * (K + 1) // 2
        self.m1 = K * (K - 1) // 2
        self.k2 = K * K
        self.szD0 = self.m0 * (self.m0 + 1) // 2
        self.szD1 = self.m1 * (self.m1 + 1) // 2
        self.szG = self.k2 * (self.k2 + 1) // 2
        self.nD = self.szD0 + self.szD1
        self.nQ = self.nD
        self.nG = 2 * self.szG
        self.meq = 2 if ne == 1 else (1 + self.m0)
        self.block_dims = [self.m0, self.m1, self.m0, self.m1, self.k2, self.k2]
        self.block_traces = [
            0.5 * ne * (ne + 1),
            0.5 * ne * (ne - 1),
            0.5 * (K - ne) * (K - ne + 1),
            0.5 * (K - ne) * (K - ne - 1),
            float(ne * (K - ne + 2)),
            float(ne * (K - ne)),
        ]
        self.t2 = T2HelperPy(t2_orbs) if t2_orbs else None
        if self.t2 is not None:
            tr_aab = 0.5 * K * (K - 1) * ne + K * 0.5 * ne * (ne - 1) - (K - 1) * ne * ne
            tr_mix = 0.5 * K * (K - 1) * ne - 0.5 * (K - 2) * ne * (ne - 1) + K * ne * (K - ne + 1)
            self.block_dims += [self.t2.naab, self.t2.nmix]
            tr_a = tr_aab if self.t2.Ks == K else min(float(self.t2.naab), tr_aab)
            tr_m = tr_mix if self.t2.Ks == K else min(float(self.t2.nmix), tr_mix)
            self.block_traces += [max(0.0, tr_a), max(0.0, tr_m)]
        self.nT2 = (self.t2.sz_aab + self.t2.sz_mix) if self.t2 is not None else 0
        self.block_szs = [d * (d + 1) // 2 for d in self.block_dims]
        self.b_eq = np.zeros(self.meq)
        self.b_eq[0] = self.block_traces[0]
        self.q0 = np.concatenate([
            svec_pack_general(np.eye(self.m0)),
            svec_pack_general(np.eye(self.m1)) if self.m1 > 0 else np.zeros(0),
        ])

        # Precompute lookup tables for fast Mode A vectorized evaluation
        self.P0 = np.zeros((K, K), dtype=int)
        self.eta = np.zeros((K, K))
        self.beta = np.zeros((K, K))
        self.P1 = np.full((K, K), -1, dtype=int)
        self.zeta = np.zeros((K, K))
        self.sgn1 = np.zeros((K, K))
        for p in range(K):
            for q in range(K):
                self.P0[p, q] = idx_sym_pair(K, p, q)
                self.eta[p, q] = 1.0 if p == q else ISQRT2
                self.beta[p, q] = SQRT2 if p == q else 1.0
                idx1, sg = idx_asym_pair(K, p, q)
                self.P1[p, q] = idx1
                self.sgn1[p, q] = float(sg)
                self.zeta[p, q] = float(sg) * ISQRT2 if idx1 >= 0 else 0.0

    def expand_st_tt(self, d):
        D0 = svec_unpack(self.m0, d[:self.szD0])
        # Clean direct indexing via flattened P0
        P0f = self.P0.ravel()
        St = (self.eta[:, :, None, None] * self.eta[None, None, :, :] *
              D0[np.ix_(P0f, P0f)].reshape(self.K, self.K, self.K, self.K))
        if self.m1 > 0:
            D1 = svec_unpack(self.m1, d[self.szD0:])
            P1f = np.maximum(self.P1.ravel(), 0)
            Tt = (self.zeta[:, :, None, None] * self.zeta[None, None, :, :] *
                  D1[np.ix_(P1f, P1f)].reshape(self.K, self.K, self.K, self.K))
        else:
            Tt = np.zeros_like(St)
        return St, Tt

    def contract_sh_th(self, Sh, Th=None):
        K = self.K
        W0 = self.eta[:, :, None, None] * self.eta[None, None, :, :] * Sh
        D0h = np.zeros((self.m0, self.m0))
        P0f = self.P0.ravel()
        np.add.at(D0h, (P0f[:, None], P0f[None, :]), W0.reshape(K * K, K * K))
        out0 = svec_pack_general(D0h)
        if self.m1 == 0:
            return out0
        D1h = np.zeros((self.m1, self.m1))
        if Th is not None:
            W1 = self.zeta[:, :, None, None] * self.zeta[None, None, :, :] * Th
            mask = (self.P1 >= 0).ravel()
            P1v = self.P1.ravel()[mask]
            np.add.at(D1h, (P1v[:, None], P1v[None, :]),
                      W1.reshape(K * K, K * K)[np.ix_(mask, mask)])
        return np.concatenate([out0, svec_pack_general(D1h)])

    def gamma_from_st(self, St):
        return (2.0 / (self.ne + 1)) * np.einsum("pqsq->ps", St)

    def gamma_adj_sh(self, ga_h):
        K = self.K
        Sh = np.zeros((K, K, K, K))
        fac = (2.0 / (self.ne + 1)) * ga_h
        for q in range(K):
            Sh[:, q, :, q] += fac
        return Sh

    def LQ_fwd(self, d):
        K = self.K
        St, _ = self.expand_st_tt(d)
        ga = self.gamma_from_st(St)
        Q0 = svec_unpack(self.m0, d[:self.szD0])
        Q1 = svec_unpack(self.m1, d[self.szD0:]) if self.m1 > 0 else np.zeros((0, 0))
        for a in range(K):
            for b in range(K):
                gba = ga[b, a]
                for c in range(K):
                    Q0[self.P0[a, c], self.P0[b, c]] -= self.beta[a, c] * self.beta[b, c] * gba
                    if self.m1 > 0 and a != c and b != c:
                        Q1[self.P1[a, c], self.P1[b, c]] -= self.sgn1[a, c] * self.sgn1[b, c] * gba
        parts = [svec_pack_general(Q0)]
        if self.m1 > 0:
            parts.append(svec_pack_general(Q1))
        return np.concatenate(parts)

    def LQ_adj(self, yq):
        K = self.K
        Y0 = svec_unpack(self.m0, yq[:self.szD0])
        Y1 = svec_unpack(self.m1, yq[self.szD0:]) if self.m1 > 0 else np.zeros((0, 0))
        ga_h = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                acc = 0.0
                for c in range(K):
                    acc -= self.beta[a, c] * self.beta[b, c] * Y0[self.P0[a, c], self.P0[b, c]]
                    if self.m1 > 0 and a != c and b != c:
                        acc -= self.sgn1[a, c] * self.sgn1[b, c] * Y1[self.P1[a, c], self.P1[b, c]]
                ga_h[b, a] = acc
        Sh = self.gamma_adj_sh(ga_h)
        return yq + self.contract_sh_th(Sh, None)

    def LG_fwd(self, d):
        K = self.K
        St, Tt = self.expand_st_tt(d)
        ga = self.gamma_from_st(St)
        eye = np.eye(K)
        dg = np.einsum("pr,qs->pqrs", ga, eye)
        G0 = (dg + St.transpose(0, 2, 3, 1) + 3.0 * Tt.transpose(0, 2, 3, 1)).reshape(self.k2, self.k2)
        G1 = (dg - St.transpose(0, 3, 2, 1) - Tt.transpose(0, 3, 2, 1)).reshape(self.k2, self.k2)
        return np.concatenate([svec_pack_general(G0), svec_pack_general(G1)])

    def LG_adj(self, yg):
        K = self.K
        Y0 = svec_unpack(self.k2, yg[:self.szG]).reshape(K, K, K, K)
        Y1 = svec_unpack(self.k2, yg[self.szG:]).reshape(K, K, K, K)
        ga_h = np.einsum("pqrs,qs->pr", Y0 + Y1, np.eye(K))
        Sh = Y0.transpose(0, 3, 1, 2) - Y1.transpose(0, 3, 2, 1) + self.gamma_adj_sh(ga_h)
        Th = 3.0 * Y0.transpose(0, 3, 1, 2) - Y1.transpose(0, 3, 2, 1)
        return self.contract_sh_th(Sh, Th)

    def Aeq_fwd(self, d):
        D0 = svec_unpack(self.m0, d[:self.szD0])
        out = np.zeros(self.meq)
        out[0] = np.trace(D0)
        if self.ne == 1:
            if self.m1 > 0:
                D1 = svec_unpack(self.m1, d[self.szD0:])
                out[1] = np.trace(D1)
            return out
        St, Tt = self.expand_st_tt(d)
        cs, ct = float(self.ne - 1), float(self.ne + 1)
        S2 = np.einsum("pqrq->pr", St)
        T2 = np.einsum("pqrq->pr", Tt)
        for p in range(self.K):
            for r in range(p, self.K):
                mu = 1.0 if p == r else SQRT2
                out[1 + idx_sym_pair(self.K, p, r)] = 0.5 * mu * (
                    cs * (S2[p, r] + S2[r, p]) - ct * (T2[p, r] + T2[r, p])
                )
        return out

    def Aeq_adj(self, lam):
        K = self.K
        if self.ne == 1:
            parts = [svec_pack_general(lam[0] * np.eye(self.m0))]
            if self.m1 > 0:
                parts.append(svec_pack_general(lam[1] * np.eye(self.m1)))
            return np.concatenate(parts)
        Sh = np.zeros((K, K, K, K))
        Th = np.zeros((K, K, K, K))
        cs, ct = float(self.ne - 1), float(self.ne + 1)
        for p in range(K):
            for r in range(p, K):
                mu = 1.0 if p == r else SQRT2
                w = 0.5 * mu * lam[1 + idx_sym_pair(K, p, r)]
                for q in range(K):
                    Sh[p, q, r, q] += w * cs
                    Sh[r, q, p, q] += w * cs
                    Th[p, q, r, q] -= w * ct
                    Th[r, q, p, q] -= w * ct
        res = self.contract_sh_th(Sh, Th)
        res[:self.szD0] += svec_pack_general(lam[0] * np.eye(self.m0))
        return res

    def build_cost(self, ham):
        eri = ham["eri"]
        h1 = ham["h"]
        Sh = eri.transpose(0, 2, 1, 3).copy() + self.gamma_adj_sh(2.0 * h1)
        Th = 3.0 * eri.transpose(0, 2, 1, 3)
        return self.contract_sh_th(Sh, Th)

    def LT2_fwd(self, d):
        if self.t2 is None:
            return np.zeros(0)
        orbs = self.t2.orbs
        St, Tt = self.expand_st_tt(d)
        ga = self.gamma_from_st(St)[np.ix_(orbs, orbs)]
        Dab_s = (St + Tt)[np.ix_(orbs, orbs, orbs, orbs)].reshape(self.t2.nab, self.t2.nab)
        D1 = svec_unpack(self.m1, d[self.szD0:]) if self.m1 > 0 else np.zeros((0, 0))
        pf = np.array([self.P1[o1, o2] for o1 in orbs for o2 in orbs if o1 < o2], dtype=int)
        Daa_s = D1[np.ix_(pf, pf)] if self.t2.naa > 0 and self.m1 > 0 else np.zeros((self.t2.naa, self.t2.naa))
        Maab = self.t2.aab_fwd(ga, Daa_s, Dab_s)
        Mmix = self.t2.mix_fwd(ga, ga, Daa_s, Daa_s, Dab_s)
        return np.concatenate([svec_pack_general(Maab), svec_pack_general(Mmix)])

    def LT2_adj(self, yt2):
        if self.t2 is None:
            return np.zeros(self.nD)
        K, orbs = self.K, self.t2.orbs
        Yaab = svec_unpack(self.t2.naab, yt2[:self.t2.sz_aab])
        Ymix = svec_unpack(self.t2.nmix, yt2[self.t2.sz_aab:])
        gb1, daa1, dab1 = self.t2.aab_adj(Yaab)
        ga2, gb2, daa2, dbb2, dab2 = self.t2.mix_adj(Ymix)
        ga_sh = np.zeros((K, K))
        ga_sh[np.ix_(orbs, orbs)] = gb1 + ga2 + gb2
        Sh = self.gamma_adj_sh(ga_sh)
        Dab_sum = (dab1 + dab2).reshape(self.t2.Ks, self.t2.Ks, self.t2.Ks, self.t2.Ks)
        Th = np.zeros((K, K, K, K))
        Sh[np.ix_(orbs, orbs, orbs, orbs)] += Dab_sum
        Th[np.ix_(orbs, orbs, orbs, orbs)] += Dab_sum
        out = self.contract_sh_th(Sh, Th)
        if self.m1 > 0 and self.t2.naa > 0:
            D1h = svec_unpack(self.m1, out[self.szD0:])
            pf = np.array([self.P1[o1, o2] for o1 in orbs for o2 in orbs if o1 < o2], dtype=int)
            np.add.at(D1h, (pf[:, None], pf[None, :]), daa1 + daa2 + dbb2)
            out[self.szD0:] = svec_pack_general(D1h)
        return out


# ============================================================================
# Independent Python/NumPy Mode B Operators & Adjoints
# ============================================================================

class ModeBPy:
    def __init__(self, K, na, nb, two_s, constrain_s2=1, t2_orbs=None):
        self.K = K
        self.na = na
        self.nb = nb
        self.two_s = two_s
        self.constrain_s2 = constrain_s2
        self.naa = K * (K - 1) // 2
        self.nab = K * K
        self.ngd = 2 * K * K
        self.szDaa = self.naa * (self.naa + 1) // 2
        self.szDab = self.nab * (self.nab + 1) // 2
        self.szGd = self.ngd * (self.ngd + 1) // 2
        self.nD = 2 * self.szDaa + self.szDab
        self.nQ = self.nD
        self.nG = self.szGd + 2 * self.szDab

        self.block_dims = [self.naa, self.naa, self.nab, self.naa, self.naa, self.nab, self.ngd, self.nab, self.nab]
        self.block_traces = [
            0.5 * na * (na - 1),
            0.5 * nb * (nb - 1),
            float(na * nb),
            0.5 * (K - na) * (K - na - 1),
            0.5 * (K - nb) * (K - nb - 1),
            float((K - na) * (K - nb)),
            float(na * (K - na + 1) + nb * (K - nb + 1)),
            float(na * (K - nb)),
            float(nb * (K - na)),
        ]
        self.t2 = T2HelperPy(t2_orbs) if t2_orbs else None
        if self.t2 is not None:
            tr_aab = 0.5 * K * (K - 1) * nb + K * 0.5 * na * (na - 1) - (K - 1) * na * nb
            tr_mix_a = 0.5 * K * (K - 1) * na - 0.5 * (K - 2) * na * (na - 1) + K * nb * (K - nb + 1)
            tr_bba = 0.5 * K * (K - 1) * na + K * 0.5 * nb * (nb - 1) - (K - 1) * na * nb
            tr_mix_b = 0.5 * K * (K - 1) * nb - 0.5 * (K - 2) * nb * (nb - 1) + K * na * (K - na + 1)
            self.block_dims += [self.t2.naab, self.t2.nmix, self.t2.naab, self.t2.nmix]
            for dim_b, tr_f in [(self.t2.naab, tr_aab), (self.t2.nmix, tr_mix_a),
                                (self.t2.naab, tr_bba), (self.t2.nmix, tr_mix_b)]:
                tr_v = tr_f if self.t2.Ks == K else min(float(dim_b), tr_f)
                self.block_traces.append(max(0.0, tr_v))
        self.nT2 = 2 * (self.t2.sz_aab + self.t2.sz_mix) if self.t2 is not None else 0
        self.block_szs = [d * (d + 1) // 2 for d in self.block_dims]

        b_list = [0.5 * na * (na - 1), 0.5 * nb * (nb - 1), float(na * nb)]
        if constrain_s2 and na >= 1 and nb >= 1:
            b_list.append(0.5 * (na + nb) + 0.25 * (na - nb) ** 2 - 0.25 * two_s * (two_s + 2))
        if na >= 1 and nb >= 1:
            nsym_m1 = K * (K + 1) // 2 - 1
            if na >= 2:
                b_list.extend([0.0] * nsym_m1)
            if nb >= 2:
                b_list.extend([0.0] * nsym_m1)
        self.b_eq = np.array(b_list, dtype=np.float64)
        self.meq = len(b_list)

        q0_parts = []
        if self.naa > 0:
            q0_parts.extend([svec_pack_general(np.eye(self.naa)), svec_pack_general(np.eye(self.naa))])
        q0_parts.append(svec_pack_general(np.eye(self.nab)))
        self.q0 = np.concatenate(q0_parts)

        self.P1 = np.full((K, K), -1, dtype=int)
        self.sgn1 = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                idx1, sg = idx_asym_pair(K, i, j)
                self.P1[i, j] = idx1
                self.sgn1[i, j] = float(sg)

    def unpack_D(self, d):
        o1 = self.szDaa
        o2 = 2 * self.szDaa
        Daa = svec_unpack(self.naa, d[:o1]) if self.naa > 0 else np.zeros((0, 0))
        Dbb = svec_unpack(self.naa, d[o1:o2]) if self.naa > 0 else np.zeros((0, 0))
        Dab = svec_unpack(self.nab, d[o2:])
        return Daa, Dbb, Dab

    def pack_D(self, Daa, Dbb, Dab):
        parts = []
        if self.naa > 0:
            parts.append(svec_pack_general(Daa))
            parts.append(svec_pack_general(Dbb))
        parts.append(svec_pack_general(Dab))
        return np.concatenate(parts)

    def gamma_fwd(self, Daa, Dbb, Dab):
        K, na, nb = self.K, self.na, self.nb
        if na >= 1 and nb >= 1:
            D4 = Dab.reshape(K, K, K, K)
            ga = np.einsum("imkm->ik", D4) / float(nb)
            gb = np.einsum("mimk->ik", D4) / float(na)
            return ga, gb
        ga = np.zeros((K, K))
        gb = np.zeros((K, K))
        if nb == 0 and na >= 2:
            for i in range(K):
                for k in range(K):
                    s = 0.0
                    for m in range(K):
                        if i != m and k != m:
                            s += self.sgn1[i, m] * self.sgn1[k, m] * Daa[self.P1[i, m], self.P1[k, m]]
                    ga[i, k] = s / float(na - 1)
        elif na == 0 and nb >= 2:
            for i in range(K):
                for k in range(K):
                    s = 0.0
                    for m in range(K):
                        if i != m and k != m:
                            s += self.sgn1[i, m] * self.sgn1[k, m] * Dbb[self.P1[i, m], self.P1[k, m]]
                    gb[i, k] = s / float(nb - 1)
        return ga, gb

    def gamma_adj(self, ga_h, gb_h, Daa_h, Dbb_h, Dab_h):
        K, na, nb = self.K, self.na, self.nb
        if na >= 1 and nb >= 1:
            D4 = Dab_h.reshape(K, K, K, K)
            for m in range(K):
                D4[:, m, :, m] += ga_h / float(nb)
                D4[m, :, m, :] += gb_h / float(na)
            return
        if nb == 0 and na >= 2:
            for i in range(K):
                for k in range(K):
                    v = ga_h[i, k] / float(na - 1)
                    for m in range(K):
                        if i != m and k != m:
                            Daa_h[self.P1[i, m], self.P1[k, m]] += self.sgn1[i, m] * self.sgn1[k, m] * v
        elif na == 0 and nb >= 2:
            for i in range(K):
                for k in range(K):
                    v = gb_h[i, k] / float(nb - 1)
                    for m in range(K):
                        if i != m and k != m:
                            Dbb_h[self.P1[i, m], self.P1[k, m]] += self.sgn1[i, m] * self.sgn1[k, m] * v

    def LQ_fwd(self, d):
        K = self.K
        Daa, Dbb, Dab = self.unpack_D(d)
        ga, gb = self.gamma_fwd(Daa, Dbb, Dab)
        Qaa, Qbb, Qab = Daa.copy(), Dbb.copy(), Dab.copy()
        Q4 = Qab.reshape(K, K, K, K)
        for c in range(K):
            Q4[:, c, :, c] -= ga.T
            Q4[c, :, c, :] -= gb.T
        if self.naa > 0:
            for a in range(K):
                for b in range(K):
                    for c in range(K):
                        if a != c and b != c:
                            sg = self.sgn1[a, c] * self.sgn1[b, c]
                            Qaa[self.P1[a, c], self.P1[b, c]] -= sg * ga[b, a]
                            Qbb[self.P1[a, c], self.P1[b, c]] -= sg * gb[b, a]
        return self.pack_D(Qaa, Qbb, Qab)

    def LQ_adj(self, yq):
        K = self.K
        Yaa, Ybb, Yab = self.unpack_D(yq)
        Y4 = Yab.reshape(K, K, K, K)
        ga_h = np.zeros((K, K))
        gb_h = np.zeros((K, K))
        for c in range(K):
            ga_h -= Y4[:, c, :, c].T
            gb_h -= Y4[c, :, c, :].T
        if self.naa > 0:
            for a in range(K):
                for b in range(K):
                    sa = sb = 0.0
                    for c in range(K):
                        if a != c and b != c:
                            sg = self.sgn1[a, c] * self.sgn1[b, c]
                            sa -= sg * Yaa[self.P1[a, c], self.P1[b, c]]
                            sb -= sg * Ybb[self.P1[a, c], self.P1[b, c]]
                    ga_h[b, a] += sa
                    gb_h[b, a] += sb
        Daa_h, Dbb_h, Dab_h = Yaa.copy(), Ybb.copy(), Yab.copy()
        self.gamma_adj(ga_h, gb_h, Daa_h, Dbb_h, Dab_h)
        return self.pack_D(Daa_h, Dbb_h, Dab_h)

    def LG_fwd(self, d):
        K = self.K
        Daa, Dbb, Dab = self.unpack_D(d)
        ga, gb = self.gamma_fwd(Daa, Dbb, Dab)
        D4 = Dab.reshape(K, K, K, K)
        eye = np.eye(K)

        # Expand Daa/Dbb into full K^4 tensor
        if self.naa > 0:
            P1f = np.maximum(self.P1.ravel(), 0)
            sg = self.sgn1[:, :, None, None] * self.sgn1[None, None, :, :]
            Daa4 = sg * Daa[np.ix_(P1f, P1f)].reshape(K, K, K, K)
            Dbb4 = sg * Dbb[np.ix_(P1f, P1f)].reshape(K, K, K, K)
        else:
            Daa4 = np.zeros((K, K, K, K))
            Dbb4 = np.zeros((K, K, K, K))

        Gd_aa = np.einsum("ik,jl->ijkl", ga, eye) - np.einsum("ilkj->ijkl", Daa4)
        Gd_bb = np.einsum("ik,jl->ijkl", gb, eye) - np.einsum("ilkj->ijkl", Dbb4)
        Gd_ab = np.einsum("iljk->ijkl", D4)
        Gd_ba = np.einsum("likj->ijkl", D4)

        Gd = np.zeros((self.ngd, self.ngd))
        Gd[:self.nab, :self.nab] = Gd_aa.reshape(self.nab, self.nab)
        Gd[self.nab:, self.nab:] = Gd_bb.reshape(self.nab, self.nab)
        Gd[:self.nab, self.nab:] = Gd_ab.reshape(self.nab, self.nab)
        Gd[self.nab:, :self.nab] = Gd_ba.reshape(self.nab, self.nab)

        Gab = (np.einsum("ik,jl->ijkl", ga, eye) - np.einsum("ilkj->ijkl", D4)).reshape(self.nab, self.nab)
        Gba = (np.einsum("ik,jl->ijkl", gb, eye) - np.einsum("lijk->ijkl", D4)).reshape(self.nab, self.nab)
        return np.concatenate([svec_pack_general(Gd), svec_pack_general(Gab), svec_pack_general(Gba)])

    def LG_adj(self, yg):
        K = self.K
        Yd = svec_unpack(self.ngd, yg[:self.szGd])
        Yab = svec_unpack(self.nab, yg[self.szGd:self.szGd + self.szDab]).reshape(K, K, K, K)
        Yba = svec_unpack(self.nab, yg[self.szGd + self.szDab:]).reshape(K, K, K, K)

        Yaa = Yd[:self.nab, :self.nab].reshape(K, K, K, K)
        Ybb = Yd[self.nab:, self.nab:].reshape(K, K, K, K)
        Yd_ab = Yd[:self.nab, self.nab:].reshape(K, K, K, K)
        Yd_ba = Yd[self.nab:, :self.nab].reshape(K, K, K, K)

        eye = np.eye(K)
        ga_h = np.einsum("ijkl,jl->ik", Yaa + Yab, eye)
        gb_h = np.einsum("ijkl,jl->ik", Ybb + Yba, eye)

        Daa_h = np.zeros((self.naa, self.naa))
        Dbb_h = np.zeros((self.naa, self.naa))
        if self.naa > 0:
            mask = (self.P1 >= 0).ravel()
            P1v = self.P1.ravel()[mask]
            Waa = -self.sgn1[:, :, None, None] * self.sgn1[None, None, :, :] * np.einsum("ijkl->ilkj", Yaa)
            Wbb = -self.sgn1[:, :, None, None] * self.sgn1[None, None, :, :] * np.einsum("ijkl->ilkj", Ybb)
            np.add.at(Daa_h, (P1v[:, None], P1v[None, :]),
                      Waa.reshape(K * K, K * K)[np.ix_(mask, mask)])
            np.add.at(Dbb_h, (P1v[:, None], P1v[None, :]),
                      Wbb.reshape(K * K, K * K)[np.ix_(mask, mask)])

        Dab_h = (np.einsum("ijkl->iljk", Yd_ab) + np.einsum("ijkl->likj", Yd_ba)
                 - np.einsum("ijkl->ilkj", Yab) - np.einsum("ijkl->lijk", Yba)).reshape(self.nab, self.nab)
        self.gamma_adj(ga_h, gb_h, Daa_h, Dbb_h, Dab_h)
        return self.pack_D(Daa_h, Dbb_h, Dab_h)

    def Aeq_fwd(self, d):
        K, na, nb = self.K, self.na, self.nb
        Daa, Dbb, Dab = self.unpack_D(d)
        out = [np.trace(Daa), np.trace(Dbb), np.trace(Dab)]
        if self.constrain_s2 and na >= 1 and nb >= 1:
            D4 = Dab.reshape(K, K, K, K)
            out.append(float(np.einsum("pqqp->", D4)))
        if na >= 1 and nb >= 1:
            ga, gb = self.gamma_fwd(Daa, Dbb, Dab)
            if na >= 2:
                for i in range(K):
                    for k in range(i, K):
                        if i == K - 1 and k == K - 1:
                            continue
                        mu = 1.0 if i == k else SQRT2
                        da = sum(self.sgn1[i, m] * self.sgn1[k, m] * Daa[self.P1[i, m], self.P1[k, m]]
                                 for m in range(K) if i != m and k != m)
                        out.append(mu * (da - (na - 1) * 0.5 * (ga[i, k] + ga[k, i])))
            if nb >= 2:
                for i in range(K):
                    for k in range(i, K):
                        if i == K - 1 and k == K - 1:
                            continue
                        mu = 1.0 if i == k else SQRT2
                        db = sum(self.sgn1[i, m] * self.sgn1[k, m] * Dbb[self.P1[i, m], self.P1[k, m]]
                                 for m in range(K) if i != m and k != m)
                        out.append(mu * (db - (nb - 1) * 0.5 * (gb[i, k] + gb[k, i])))
        return np.array(out, dtype=np.float64)

    def Aeq_adj(self, lam):
        K, na, nb = self.K, self.na, self.nb
        Daa_h = lam[0] * np.eye(self.naa) if self.naa > 0 else np.zeros((0, 0))
        Dbb_h = lam[1] * np.eye(self.naa) if self.naa > 0 else np.zeros((0, 0))
        Dab_h = lam[2] * np.eye(self.nab)
        ga_h = np.zeros((K, K))
        gb_h = np.zeros((K, K))
        r = 3
        if self.constrain_s2 and na >= 1 and nb >= 1:
            D4 = Dab_h.reshape(K, K, K, K)
            for p in range(K):
                for q in range(K):
                    D4[p, q, q, p] += lam[r]
            r += 1
        if na >= 1 and nb >= 1:
            if na >= 2:
                for i in range(K):
                    for k in range(i, K):
                        if i == K - 1 and k == K - 1:
                            continue
                        mu = 1.0 if i == k else SQRT2
                        w = mu * lam[r]
                        r += 1
                        for m in range(K):
                            if i != m and k != m:
                                Daa_h[self.P1[i, m], self.P1[k, m]] += self.sgn1[i, m] * self.sgn1[k, m] * w
                        ga_h[i, k] -= 0.5 * (na - 1) * w
                        ga_h[k, i] -= 0.5 * (na - 1) * w
            if nb >= 2:
                for i in range(K):
                    for k in range(i, K):
                        if i == K - 1 and k == K - 1:
                            continue
                        mu = 1.0 if i == k else SQRT2
                        w = mu * lam[r]
                        r += 1
                        for m in range(K):
                            if i != m and k != m:
                                Dbb_h[self.P1[i, m], self.P1[k, m]] += self.sgn1[i, m] * self.sgn1[k, m] * w
                        gb_h[i, k] -= 0.5 * (nb - 1) * w
                        gb_h[k, i] -= 0.5 * (nb - 1) * w
        self.gamma_adj(ga_h, gb_h, Daa_h, Dbb_h, Dab_h)
        return self.pack_D(Daa_h, Dbb_h, Dab_h)

    def build_cost(self, ham):
        K = self.K
        eri = ham["eri"]
        h1 = ham["h"]
        Daa_h = np.zeros((self.naa, self.naa))
        Dbb_h = np.zeros((self.naa, self.naa))
        if self.naa > 0:
            pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
            for p, (i, j) in enumerate(pairs):
                for q, (k, l) in enumerate(pairs):
                    v = eri[i, k, j, l] - eri[i, l, j, k]
                    Daa_h[p, q] = v
                    Dbb_h[p, q] = v
        Dab_h = eri.transpose(0, 2, 1, 3).reshape(self.nab, self.nab).copy()
        self.gamma_adj(h1, h1, Daa_h, Dbb_h, Dab_h)
        return self.pack_D(Daa_h, Dbb_h, Dab_h)

    def LT2_fwd(self, d):
        if self.t2 is None:
            return np.zeros(0)
        orbs, K = self.t2.orbs, self.K
        Daa, Dbb, Dab = self.unpack_D(d)
        ga, gb = self.gamma_fwd(Daa, Dbb, Dab)
        ga_s = ga[np.ix_(orbs, orbs)]
        gb_s = gb[np.ix_(orbs, orbs)]
        pf = np.array([self.P1[o1, o2] for o1 in orbs for o2 in orbs if o1 < o2], dtype=int)
        if self.t2.naa > 0 and self.naa > 0:
            Daa_s = Daa[np.ix_(pf, pf)]
            Dbb_s = Dbb[np.ix_(pf, pf)]
        else:
            Daa_s = np.zeros((self.t2.naa, self.t2.naa))
            Dbb_s = np.zeros((self.t2.naa, self.t2.naa))
        D4_s = Dab.reshape(K, K, K, K)[np.ix_(orbs, orbs, orbs, orbs)]
        Dab_s = D4_s.reshape(self.t2.nab, self.t2.nab)
        Dab_sw = D4_s.transpose(1, 0, 3, 2).reshape(self.t2.nab, self.t2.nab)
        return np.concatenate([
            svec_pack_general(self.t2.aab_fwd(gb_s, Daa_s, Dab_s)),
            svec_pack_general(self.t2.mix_fwd(ga_s, gb_s, Daa_s, Dbb_s, Dab_s)),
            svec_pack_general(self.t2.aab_fwd(ga_s, Dbb_s, Dab_sw)),
            svec_pack_general(self.t2.mix_fwd(gb_s, ga_s, Dbb_s, Daa_s, Dab_sw)),
        ])

    def LT2_adj(self, yt2):
        if self.t2 is None:
            return np.zeros(self.nD)
        orbs, K, Ks = self.t2.orbs, self.K, self.t2.Ks
        s_a, s_m = self.t2.sz_aab, self.t2.sz_mix
        o1 = s_a
        o2 = s_a + s_m
        o3 = o2 + s_a
        gb1, daa1, dab1 = self.t2.aab_adj(svec_unpack(self.t2.naab, yt2[:o1]))
        ga2, gb2, daa2, dbb2, dab2 = self.t2.mix_adj(svec_unpack(self.t2.nmix, yt2[o1:o2]))
        ga3, dbb3, dabsw3 = self.t2.aab_adj(svec_unpack(self.t2.naab, yt2[o2:o3]))
        gb4, ga4, dbb4, daa4, dabsw4 = self.t2.mix_adj(svec_unpack(self.t2.nmix, yt2[o3:]))

        ga_h = np.zeros((K, K))
        gb_h = np.zeros((K, K))
        ga_h[np.ix_(orbs, orbs)] = ga2 + ga3 + ga4
        gb_h[np.ix_(orbs, orbs)] = gb1 + gb2 + gb4
        Daa_h = np.zeros((self.naa, self.naa))
        Dbb_h = np.zeros((self.naa, self.naa))
        if self.t2.naa > 0 and self.naa > 0:
            pf = np.array([self.P1[o1i, o2i] for o1i in orbs for o2i in orbs if o1i < o2i], dtype=int)
            np.add.at(Daa_h, (pf[:, None], pf[None, :]), daa1 + daa2 + daa4)
            np.add.at(Dbb_h, (pf[:, None], pf[None, :]), dbb2 + dbb3 + dbb4)
        Dab_sub = (dab1 + dab2).reshape(Ks, Ks, Ks, Ks) + (dabsw3 + dabsw4).reshape(Ks, Ks, Ks, Ks).transpose(1, 0, 3, 2)
        D4_h = np.zeros((K, K, K, K))
        D4_h[np.ix_(orbs, orbs, orbs, orbs)] = Dab_sub
        Dab_h = D4_h.reshape(self.nab, self.nab)
        self.gamma_adj(ga_h, gb_h, Daa_h, Dbb_h, Dab_h)
        return self.pack_D(Daa_h, Dbb_h, Dab_h)


# ============================================================================
# Independent Exact FCI & Variational Witness Evaluator
# ============================================================================

def popcnt(x):
    return bin(x).count("1")


def slater_condon_mel(ham, dI, dJ):
    K = ham["K"]
    h1 = ham["h"]
    eri = ham["eri"]

    def jw_par(det, bit):
        return -1.0 if (popcnt(det & ((1 << bit) - 1)) & 1) else 1.0

    def h1so(p, q):
        return 0.0 if ((p & 1) != (q & 1)) else h1[p >> 1, q >> 1]

    def e2so(p, q, r, s):
        if ((p & 1) != (r & 1)) or ((q & 1) != (s & 1)):
            return 0.0
        return eri[p >> 1, r >> 1, q >> 1, s >> 1]

    if dI == dJ:
        occ = [p for p in range(2 * K) if (dI >> p) & 1]
        e = ham["E_core"]
        for ii, p in enumerate(occ):
            e += h1so(p, p)
            for q in occ[ii + 1:]:
                e += e2so(p, q, p, q) - e2so(p, q, q, p)
        return e
    diff = dI ^ dJ
    nd = popcnt(diff)
    if nd == 2:
        r = (dI & diff).bit_length() - 1
        c = (dJ & diff).bit_length() - 1
        if (r & 1) != (c & 1):
            return 0.0
        sgn = jw_par(dI, r) * jw_par(dI ^ (1 << r), c)
        mat = h1so(r, c)
        common = dI & dJ
        while common:
            lb = common & -common
            p = lb.bit_length() - 1
            mat += e2so(r, p, c, p) - e2so(r, p, p, c)
            common ^= lb
        return sgn * mat
    if nd == 4:
        rem = dI & diff
        r1 = (rem & -rem).bit_length() - 1
        r2 = (rem ^ (1 << r1)).bit_length() - 1
        add = dJ & diff
        c1 = (add & -add).bit_length() - 1
        c2 = (add ^ (1 << c1)).bit_length() - 1
        t1 = dI ^ (1 << r1)
        t2 = t1 ^ (1 << r2)
        t3 = t2 ^ (1 << c2)
        sgn = jw_par(dI, r1) * jw_par(t1, r2) * jw_par(t2, c2) * jw_par(t3, c1)
        return sgn * (e2so(r1, r2, c1, c2) - e2so(r1, r2, c2, c1))
    return 0.0


def solve_fci_state_py(ham):
    K, na, nb = ham["K"], ham["na"], ham["nb"]
    acomb = list(itertools.combinations(range(K), na))
    bcomb = list(itertools.combinations(range(K), nb))
    dets = []
    for ac in acomb:
        am = sum(1 << (2 * i) for i in ac)
        for bc in bcomb:
            dets.append(am | sum(1 << (2 * i + 1) for i in bc))
    ndet = len(dets)
    Hmat = np.zeros((ndet, ndet), dtype=np.float64)
    for I in range(ndet):
        Hmat[I, I] = slater_condon_mel(ham, dets[I], dets[I])
        for J in range(I):
            diff = dets[I] ^ dets[J]
            if popcnt(diff) in (2, 4):
                v = slater_condon_mel(ham, dets[I], dets[J])
                Hmat[I, J] = Hmat[J, I] = v
    evals, evecs = np.linalg.eigh(Hmat)
    return float(evals[0]), dets, evecs[:, 0]


def solve_fci_py(ham):
    return solve_fci_state_py(ham)[0]


def eval_ham_state_py(ham, dets, psi0):
    c = np.asarray(psi0, dtype=np.float64)
    n2 = float(np.dot(c, c))
    num = 0.0
    for i, dI in enumerate(dets):
        ci = float(c[i])
        if abs(ci) < 1e-16:
            continue
        num += ci * ci * slater_condon_mel(ham, dI, dI)
        for j in range(i):
            cj = float(c[j])
            if abs(cj) < 1e-16:
                continue
            if popcnt(dI ^ dets[j]) in (2, 4):
                num += 2.0 * ci * cj * slater_condon_mel(ham, dI, dets[j])
    return num / n2


def eval_obs_fci_py(ham, dets, psi0, spec, ham_cache):
    c = np.asarray(psi0, dtype=np.float64)
    n2 = float(np.dot(c, c))
    c2 = (c * c) / n2
    if spec.startswith("occ:"):
        p = int(spec[4:])
        val = 0.0
        for i, d in enumerate(dets):
            val += float(c2[i]) * float(((d >> (2 * p)) & 1) + ((d >> (2 * p + 1)) & 1))
        return val
    if spec.startswith("docc:"):
        p = int(spec[5:])
        val = 0.0
        for i, d in enumerate(dets):
            if ((d >> (2 * p)) & 1) and ((d >> (2 * p + 1)) & 1):
                val += float(c2[i])
        return val
    if spec.startswith("sdot:"):
        _, ps, qs = spec.split(":")
        p, q = int(ps), int(qs)
        if p == q:
            val = 0.0
            for i, d in enumerate(dets):
                na_p = (d >> (2 * p)) & 1
                nb_p = (d >> (2 * p + 1)) & 1
                val += float(c2[i]) * 0.75 * float(na_p + nb_p - 2 * (na_p & nb_p))
            return val
        val = 0.0
        for i, d in enumerate(dets):
            sz_p = 0.5 * float(((d >> (2 * p)) & 1) - ((d >> (2 * p + 1)) & 1))
            sz_q = 0.5 * float(((d >> (2 * q)) & 1) - ((d >> (2 * q + 1)) & 1))
            val += float(c2[i]) * sz_p * sz_q
        mask = (1 << (2 * p)) | (1 << (2 * p + 1)) | (1 << (2 * q)) | (1 << (2 * q + 1))
        det_to_idx = {d: idx for idx, d in enumerate(dets)}
        for i, d in enumerate(dets):
            ci = float(c[i])
            if abs(ci) < 1e-16:
                continue
            na_p = (d >> (2 * p)) & 1
            nb_p = (d >> (2 * p + 1)) & 1
            na_q = (d >> (2 * q)) & 1
            nb_q = (d >> (2 * q + 1)) & 1
            if na_p != nb_p and na_q != nb_q and na_p != na_q:
                j = det_to_idx.get(d ^ mask)
                if j is not None and j < i:
                    # JW parity for swapping spins at adjacent (2p,2p+1) and (2q,2q+1) is +1
                    val += (ci * float(c[j])) / n2
        return val
    if spec.startswith("hf-force:"):
        rest = spec[len("hf-force:"):]
        p_plus, p_minus, dr_str = rest.rsplit(":", 2)
        dr = float(dr_str)
        if p_plus not in ham_cache:
            ham_cache[p_plus] = parse_fcidump(p_plus)
        if p_minus not in ham_cache:
            ham_cache[p_minus] = parse_fcidump(p_minus)
        ep = eval_ham_state_py(ham_cache[p_plus], dets, c)
        em = eval_ham_state_py(ham_cache[p_minus], dets, c)
        return (ep - em) / (2.0 * dr)
    raise ValueError(f"Unsupported observable spec: {spec}")


def eval_witness_py(ham, dets_hex, coeffs):
    dets = [int(x, 16) for x in dets_hex]
    c = np.array(coeffs, dtype=np.float64)
    n2 = float(np.dot(c, c))
    num = 0.0
    for i, dI in enumerate(dets):
        ci = c[i]
        if abs(ci) < 1e-16:
            continue
        num += ci * ci * slater_condon_mel(ham, dI, dI)
        for j in range(i):
            cj = c[j]
            if abs(cj) < 1e-16:
                continue
            if popcnt(dI ^ dets[j]) in (2, 4):
                num += 2.0 * ci * cj * slater_condon_mel(ham, dI, dets[j])
    return num / n2


def eval_spin_sq_py(na, nb, dets_hex, coeffs):
    EVEN = 0x5555555555555555
    dets = [int(x, 16) for x in dets_hex]
    c = np.array(coeffs, dtype=np.float64)
    n2 = float(np.dot(c, c))
    sz = 0.5 * (na - nb)
    num = 0.0
    for i, dI in enumerate(dets):
        ci = c[i]
        if abs(ci) < 1e-16:
            continue
        ndocc = popcnt((dI & EVEN) & ((dI >> 1) & EVEN))
        num += ci * ci * (sz * (sz + 1.0) + nb - ndocc)
        aI = dI & EVEN
        bI = (dI >> 1) & EVEN
        for j in range(i):
            cj = c[j]
            if abs(cj) < 1e-16:
                continue
            aJ = dets[j] & EVEN
            bJ = (dets[j] >> 1) & EVEN
            a_in = aI & ~aJ
            a_out = aJ & ~aI
            b_in = bI & ~bJ
            b_out = bJ & ~bI
            if a_in == b_out and a_out == b_in and popcnt(a_in) == 1 and popcnt(a_out) == 1:
                num += 2.0 * ci * cj
    return num / n2


def audit_certificate_py(cert_path, ham_cache):
    with open(cert_path) as f:
        cert = json.load(f)
    fpath = cert["fcidump_path"]
    if fpath not in ham_cache:
        ham_cache[fpath] = parse_fcidump(fpath)
    ham = ham_cache[fpath]
    assert cert["hamiltonian_sha256"] == ham["sha256"], f"SHA mismatch on {cert_path}"

    dw = cert["dual_witness"]
    mode = cert["mode"]
    t2_enabled = bool(cert.get("t2_enabled", 0))
    t2_orbs = cert.get("t2_orbs", []) if t2_enabled else None
    op = ModeAPy(ham["K"], ham["na"], t2_orbs=t2_orbs) if mode == "A_SINGLET" else ModeBPy(
        ham["K"], ham["na"], ham["nb"], cert["two_s"], cert["constrain_s2"], t2_orbs=t2_orbs
    )
    y_eq = np.array(dw["y_eq"], dtype=np.float64)
    Z_Q = np.array(dw["Z_Q"], dtype=np.float64)
    Z_G = np.array(dw["Z_G"], dtype=np.float64)
    C_D = op.build_cost(ham)
    Z_D = C_D - op.Aeq_adj(y_eq) - op.LQ_adj(Z_Q) - op.LG_adj(Z_G)
    Z_T2 = np.zeros(0, dtype=np.float64)
    if t2_enabled:
        Z_T2 = np.array(dw.get("Z_T2", []), dtype=np.float64)
        Z_D = Z_D - op.LT2_adj(Z_T2)

    e_dual = ham["E_core"] + float(np.dot(y_eq, op.b_eq)) - float(np.dot(Z_Q, op.q0))

    nD_blk = 2 if mode == "A_SINGLET" else 3
    off = 0
    floor_corr = 0.0
    sum_frob_tr = 0.0
    for b, (dim, sz, tr) in enumerate(zip(op.block_dims, op.block_szs, op.block_traces)):
        if b < nD_blk:
            blk = Z_D[off:off + sz]
            off = 0 if b == nD_blk - 1 else off + sz
        elif b < 2 * nD_blk:
            blk = Z_Q[off:off + sz]
            off = 0 if b == 2 * nD_blk - 1 else off + sz
        elif b < 3 * nD_blk:
            blk = Z_G[off:off + sz]
            off = 0 if b == 3 * nD_blk - 1 else off + sz
        else:
            blk = Z_T2[off:off + sz]
            off += sz
        if dim <= 0:
            continue
        M = svec_unpack(dim, blk)
        lam_fp = float(np.linalg.eigvalsh(M)[0])
        frob = float(np.linalg.norm(blk))
        lam_cert = lam_fp - 64.0 * DBL_EPS * math.sqrt(dim) * frob
        if lam_cert < 0.0:
            floor_corr += lam_cert * tr
        sum_frob_tr += frob * tr

    tr_D_sum = sum(op.block_traces[:nD_blk])
    delta_round = 64.0 * DBL_EPS * (
        abs(ham["E_core"])
        + float(np.linalg.norm(C_D)) * tr_D_sum
        + float(np.linalg.norm(y_eq)) * float(np.linalg.norm(op.b_eq))
        + sum_frob_tr
    )
    e_lb_py = e_dual + floor_corr - delta_round
    diff_lb = abs(e_lb_py - cert["solver"]["E_lb_cert"])
    assert diff_lb < 1e-8, f"Cert LB mismatch on {cert_path}: diff={diff_lb}"

    veto = cert["veto"]
    if veto["vetoed"]:
        assert e_lb_py > veto["E_ai"] - veto["veto_tol"], f"Veto unsound on {cert_path}"

    var = cert["variational"]
    if var.get("has_witness"):
        e_var_py = eval_witness_py(ham, var["dets"], var["psi0"])
        assert abs(e_var_py - var["E_ci"]) < 1e-7, f"Witness E_ci mismatch on {cert_path}"
        assert e_lb_py <= e_var_py + 1e-9, f"LB > Witness on {cert_path}"
        if os.path.basename(cert_path).startswith("tm_spin_s0_8e8o"):
            s2_py = eval_spin_sq_py(ham["na"], ham["nb"], var["dets"], var["psi0"])
            assert abs(s2_py) < 1e-4, f"tm_spin_s0_8e8o witness not spin-pure S=0: <S^2>={s2_py}"

    return diff_lb


def verify_py_adjoints():
    rng = np.random.default_rng(42)
    max_err = 0.0
    for mode, args in [("A", (6, 3)), ("B", (5, 3, 1, 1, 1)), ("B", (4, 3, 0, 3, 0))]:
        op = ModeAPy(*args) if mode == "A" else ModeBPy(*args)
        X = rng.standard_normal(op.nD)
        YQ = rng.standard_normal(op.nQ)
        YG = rng.standard_normal(op.nG)
        Yeq = rng.standard_normal(op.meq)
        for fwd, adj, Y in [(op.LQ_fwd, op.LQ_adj, YQ), (op.LG_fwd, op.LG_adj, YG), (op.Aeq_fwd, op.Aeq_adj, Yeq)]:
            LX = fwd(X)
            LsY = adj(Y)
            err = abs(float(np.dot(LX, Y) - np.dot(X, LsY))) / max(1.0, float(np.linalg.norm(LX) * np.linalg.norm(Y)))
            assert err < 1e-14, f"Python adjoint error {err}"
            if err > max_err:
                max_err = err

    max_t2_err = 0.0
    for mode, args in [("A", (6, 3, [0, 1, 2, 3, 4, 5])), ("B", (5, 3, 1, 1, 1, [0, 1, 2, 3]))]:
        op = ModeAPy(*args) if mode == "A" else ModeBPy(*args)
        X = rng.standard_normal(op.nD)
        YT2 = rng.standard_normal(op.nT2)
        LX = op.LT2_fwd(X)
        LsY = op.LT2_adj(YT2)
        err = abs(float(np.dot(LX, YT2) - np.dot(X, LsY))) / max(1.0, float(np.linalg.norm(LX) * np.linalg.norm(YT2)))
        assert err < 1e-14, f"Python T2 adjoint error {err}"
        if err > max_t2_err:
            max_t2_err = err
    return max(max_err, max_t2_err), max_t2_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--cert-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    py_adj_err, py_t2_adj_err = verify_py_adjoints()
    with open(args.bench) as f:
        data = json.load(f)

    # Gate 1: Adjoint checks (< 1e-14)
    c_adj_max = max(a["max_err"] for a in data["adjoints"])
    assert c_adj_max < 1e-14, f"C adjoint max_err={c_adj_max}"

    # Gate 2: Exact FCI verification on small & mid-small active spaces (<= 400 dets)
    small_fci_names = {
        "h2_r1.4", "h2_r5.0", "h3_doublet",
        "h4_ry2.0", "h4_ry2.0_modeB", "h4_ry2.2", "h4_ry2.6", "h4_ry4.5", "h4_ry2.0_triplet",
        "h2o_eq", "h2o_stretched", "n2_eq_6e6o", "n2_stretched_6e6o",
    }
    by_name = {r["name"]: r for r in data["benchmarks"]}
    ham_cache = {}
    fci_checks = []

    for r in data["benchmarks"]:
        name = r["name"]
        fname = "h4_ry2.0.fcidump" if name == "h4_ry2.0_modeB" else f"{name}.fcidump"
        fpath = os.path.join(args.bench_dir, fname)
        if fpath not in ham_cache:
            ham_cache[fpath] = parse_fcidump(fpath)
        ham = ham_cache[fpath]
        assert r["sha256"] == ham["sha256"], f"SHA mismatch for {name}"
        assert r["E_lb_cert"] <= r["E_ub"] + 1e-11, f"Bracket inverted for {name}"

        if name in small_fci_names:
            efci_py = solve_fci_py(ham)
            dfci = abs(r["E_ci"] - efci_py)
            assert dfci < 1e-10, f"FCI mismatch on {name}: |{r['E_ci']} - {efci_py}| = {dfci}"
            assert r["E_lb_cert"] <= efci_py + 1e-11, f"LB violation on {name}: {r['E_lb_cert']} > {efci_py}"
            assert efci_py <= r["E_ub"] + 1e-11, f"UB violation on {name}: {efci_py} > {r['E_ub']}"
            fci_checks.append((name, efci_py, dfci))

    # Mode A vs Mode B hierarchy on H4 Ry=2.0: Mode B <= Mode A <= FCI and Mode B within 3 mHa
    lb_a_h4 = by_name["h4_ry2.0"]["E_lb_cert"]
    lb_b_h4 = by_name["h4_ry2.0_modeB"]["E_lb_cert"]
    fci_map = {n: (e, df) for n, e, df in fci_checks}
    efci_h4 = fci_map["h4_ry2.0"][0]
    assert lb_b_h4 <= lb_a_h4 + 1e-4, f"Mode B > Mode A on H4: {lb_b_h4} > {lb_a_h4}"
    assert (efci_h4 - lb_b_h4) < 0.003, f"Mode B gap too large on H4: {efci_h4 - lb_b_h4}"
    ab_diff = abs(lb_a_h4 - lb_b_h4)

    # Gate 3: Beyond-FCI memory, bracket validity, spin-state separation, and veto speedups
    assert data["peak_rss_mb"] < 50.0, f"Peak RSS exceeded 50 MB: {data['peak_rss_mb']}"

    # Spin-state ordering separation: TM S=2 quintet strictly below S=0 singlet!
    tm_s2 = by_name["tm_spin_s2_8e8o"]
    tm_s0 = by_name["tm_spin_s0_8e8o"]
    tm_sep = tm_s0["E_lb_cert"] - tm_s2["E_ub"]
    assert tm_sep > 0.05, f"TM spin gap failed to separate: LB(S=0)={tm_s0['E_lb_cert']} vs UB(S=2)={tm_s2['E_ub']}"

    # Verify Early-Veto speedups
    veto_cases = [r for r in data["benchmarks"] if r.get("veto_test")]
    assert len(veto_cases) >= 3
    for r in veto_cases:
        vt = r["veto_test"]
        assert vt["vetoed"] == 1
        assert vt["iter_stopped"] < vt["full_iters"], f"Veto not early on {r['name']}"

    # Gate 4: Independent audit of EVERY emitted certificate (JSON DQG + DQG+T2)
    cert_files = sorted(glob.glob(os.path.join(args.cert_dir, "*.cert.json")))
    assert len(cert_files) >= len(data["benchmarks"])
    max_cert_diff = 0.0
    for cp in cert_files:
        d = audit_certificate_py(cp, ham_cache)
        if d > max_cert_diff:
            max_cert_diff = d

    # Gate 5 (v0.2.0): T2 adjoints, DQG+T2 tightening, certified observables & forces, binary certs, SDK
    c_t2_adj_max = max(a["max_err"] for a in data["t2_adjoints"])
    assert c_t2_adj_max < 1e-14, f"C T2 adjoint max_err={c_t2_adj_max}"

    t2_cases = data["v02_t2_cases"]
    assert len(t2_cases) >= 4
    strict_gains = 0
    for r in t2_cases:
        name = r["name"]
        e_lb_dqg = by_name[name]["E_lb_cert"]
        assert r["E_lb_t2"] <= r["E_ub"] + 1e-11, f"T2 bracket inverted on {name}"
        if name in fci_map:
            assert r["E_lb_t2"] <= fci_map[name][0] + 1e-11, f"T2 LB > FCI on {name}"
        assert r["E_lb_t2"] >= e_lb_dqg - 1e-5, f"T2 weakened LB on {name}: {r['E_lb_t2']} < {e_lb_dqg}"
        if r["E_lb_t2"] > e_lb_dqg + 1e-5:
            strict_gains += 1
    assert strict_gains >= 3, f"Expected >= 3 strict T2 gains, got {strict_gains}"

    obs_checks = []
    o_hf_fci = None
    for o in data["v02_observables"]:
        fpath = os.path.join(args.bench_dir, o["fcidump"])
        if fpath not in ham_cache:
            ham_cache[fpath] = parse_fcidump(fpath)
        ham = ham_cache[fpath]
        _, dets_o, psi0_o = solve_fci_state_py(ham)
        o_fci = eval_obs_fci_py(ham, dets_o, psi0_o, o["spec"], ham_cache)
        do_var = abs(o["O_var"] - o_fci)
        assert do_var < 1e-10, f"Obs witness mismatch on {o['name']}: |{o['O_var']} - {o_fci}| = {do_var}"
        assert o["O_lb_cert"] <= o_fci + 1e-10, f"Obs LB > FCI on {o['name']}: {o['O_lb_cert']} > {o_fci}"
        assert o_fci <= o["O_ub_cert"] + 1e-10, f"Obs FCI > UB on {o['name']}: {o_fci} > {o['O_ub_cert']}"
        obs_checks.append((o, o_fci, do_var))
        if o["name"] == "h4_hf_force":
            o_hf_fci = o_fci

    bin_required = ["h4_ry2.0", "n2_stretched_6e6o", "tm_spin_s0_8e8o", "h16_chain_triplet_16e16o"]
    bin_rows = []
    for base in bin_required + [f"{r['name']}.t2" for r in t2_cases]:
        jpath = os.path.join(args.cert_dir, f"{base}.cert.json")
        bpath = os.path.join(args.cert_dir, f"{base}.flowchem")
        if base in bin_required:
            assert os.path.exists(bpath), f"Missing binary cert {bpath}"
            assert os.path.exists(jpath), f"Missing json cert {jpath}"
        if os.path.exists(bpath) and os.path.exists(jpath):
            jb = os.path.getsize(jpath)
            bb = os.path.getsize(bpath)
            if base in bin_required:
                assert bb < jb, f"Binary cert not smaller on {base}: {bb} >= {jb}"
            bin_rows.append((base, jb, bb, float(jb) / float(bb)))

    # Live Python SDK (ChemOracle) integration test
    sdk_dir = os.path.dirname(os.path.abspath(__file__))
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    from chem_sdk import ChemOracle
    cli_path = os.path.join(os.path.dirname(sdk_dir), "build", "chem_cli")
    sdk = ChemOracle(cli_path=cli_path)
    h4_path = os.path.join(args.bench_dir, "h4_ry2.0.fcidump")
    hp_path = os.path.join(args.bench_dir, "h4_ry2.0_fplus.fcidump")
    hm_path = os.path.join(args.bench_dir, "h4_ry2.0_fminus.fcidump")

    r_sdk_t2 = sdk.bracket(h4_path, t2=True, max_iters=300)
    assert r_sdk_t2["E_lb_cert"] <= r_sdk_t2["E_ub"] + 1e-11
    v_sdk_bin = sdk.cert_verify(os.path.join(args.cert_dir, "h4_ry2.0.flowchem"))
    assert v_sdk_bin["ok"] is True, f"SDK binary cert_verify failed: {v_sdk_bin}"
    s_ev = sdk.mlip_screen(h4_path, e_ai=-2.5)
    assert s_ev["energy_vetoed"] is True and s_ev["vetoed"] is True
    s_fv = sdk.mlip_screen(h4_path, e_ai=-1.90, f_ai=0.5, f_plus=hp_path, f_minus=hm_path, dr=0.005)
    assert (not s_fv["energy_vetoed"]) and s_fv["force_vetoed"] and s_fv["vetoed"]
    assert o_hf_fci is not None
    s_ok = sdk.mlip_screen(h4_path, e_ai=-1.90, f_ai=o_hf_fci, f_plus=hp_path, f_minus=hm_path, dr=0.005)
    assert not s_ok["vetoed"], f"SDK valid HF force falsely vetoed: {s_ok}"

    # Emit Markdown verification report
    lines = [
        "# CHEM v0.2.0 Verification Report\n",
        f"- **C Matrix-Free Adjoint Max Error (`K=4..16`, Mode A & B):** `{c_adj_max:.2e}` (`< 1e-14` PASS)",
        f"- **C Matrix-Free `T2` Adjoint Max Error (`Ks<=6`, Mode A & B):** `{c_t2_adj_max:.2e}` (`< 1e-14` PASS)",
        f"- **Python/NumPy Independent Adjoint Max Error (DQG / T2):** `{py_adj_err:.2e}` / `{py_t2_adj_err:.2e}` (`< 1e-14` PASS)",
        f"- **Mode A vs Mode B `H4 (Ry=2.0)` Hierarchy:** `E_LB(B) = {lb_b_h4:.8f} <= E_LB(A) = {lb_a_h4:.8f} <= E_FCI = {efci_h4:.8f}` (`|A-B| = {ab_diff*1000:.3f} mHa`, PASS)",
        f"- **Independent Certificate Audit (`{len(cert_files)}` `.cert.json` files):** max `|E_lb_py - E_lb_c| = {max_cert_diff:.2e}` Ha (PASS)",
        f"- **Process Peak RSS (up through `(16e,16o)` Mode A & B + `T2` + Observables):** `{data['peak_rss_mb']:.2f} MB` (`< 50 MB` PASS)\n",
        "## Acceptance Gate 2: Small & Mid-Small Active Spaces vs Exact Python/NumPy FCI",
        "| Benchmark | $(N, K)$ | Mode | $E_{\\text{LB}}^{\\text{cert}}$ (Ha) | $E_{\\text{FCI}}$ (Ha) | $E_{\\text{HF}}$ (Ha) | $|\\Delta E_{\\text{FCI}}|$ | Gap $E_{\\text{FCI}} - E_{\\text{LB}}$ (mHa) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in data["benchmarks"]:
        if r["name"] in fci_map:
            efci, df = fci_map[r["name"]]
            gap = (efci - r["E_lb_cert"]) * 1000.0
            lines.append(
                f"| `{r['name']}` | `({r['nelec']}e,{r['K']}o)` | `{r['mode']}` | "
                f"`{r['E_lb_cert']:.8f}` | `{efci:.8f}` | `{r['E_hf0']:.8f}` | "
                f"`{df:.2e}` | `{gap:.4f}` |"
            )

    lines += [
        "\n## Acceptance Gate 3: Beyond-FCI & Spin-State Ordering Certification",
        "| Benchmark | $(N, K)$ | Mode | $E_{\\text{LB}}^{\\text{cert}}$ (Ha) | $E_{\\text{UB}}$ (Ha) | Width (mHa) | Iters | Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in data["benchmarks"]:
        if r["name"] not in fci_map:
            lines.append(
                f"| `{r['name']}` | `({r['nelec']}e,{r['K']}o)` | `{r['mode']}` | "
                f"`{r['E_lb_cert']:.8f}` | `{r['E_ub']:.8f}` | `{r['interval_width']*1000.0:.3f}` | "
                f"`{r['iters']}` | `{r['sdp_sec'] + r['var_sec']:.2f}` |"
            )

    lines += [
        f"\n- **Certified Transition-Metal Spin Gap (`tm_spin` `S=2` vs `S=0`):** "
        f"`E_UB(S=2) = {tm_s2['E_ub']:.6f} < E_LB(S=0) = {tm_s0['E_lb_cert']:.6f}` "
        f"(certified separation `+{tm_sep*1000.0:.2f} mHa`, high-spin quintet ground state rigorously proved!).",
        "\n## Anytime Early-Veto Speedups (False-Low AI Candidate `E_ai = min(E_UB - 0.15, E_LB - 0.05) Ha`)",
        "| Benchmark | Full Iters | Veto Stopped Iter | Speedup | Vetoed |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in veto_cases:
        vt = r["veto_test"]
        sp = float(vt["full_iters"]) / max(1, vt["iter_stopped"])
        lines.append(
            f"| `{r['name']}` | `{vt['full_iters']}` | `{vt['iter_stopped']}` | `{sp:.1f}x` | `{bool(vt['vetoed'])}` |"
        )

    lines += [
        "\n## Acceptance Gate 5 (`v0.2.0`): `DQG+T2`, Certified Observables & Force Bounds, `.flowchem`, and Python SDK",
        "\n### 5a. Matrix-Free `DQG` vs `DQG+T2` Subspace Lower-Bound Tightening",
        "| Benchmark | $(K_s/K)$ | $E_{\\text{LB}}^{\\text{DQG}}$ (Ha) | $E_{\\text{LB}}^{\\text{DQG}+T_2}$ (Ha) | Gain (mHa) | $E_{\\text{UB}}$ (Ha) | Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in t2_cases:
        e_dqg = by_name[r["name"]]["E_lb_cert"]
        gain_mha = (r["E_lb_t2"] - e_dqg) * 1000.0
        lines.append(
            f"| `{r['name']}` | `({r['Ks']}/{r['K']})` | `{e_dqg:.8f}` | "
            f"`{r['E_lb_t2']:.8f}` | `+{gain_mha:.3f}` | `{r['E_ub']:.8f}` | `{r['sdp_sec']:.2f}` |"
        )

    lines += [
        "\n### 5b. Certified Observable & Hellmann-Feynman Force Envelope Bounds (`O_LB <= <O>_FCI <= O_UB`)",
        "| Observable | Mode | $O_{\\text{LB}}^{\\text{cert}}$ | $\\langle O \\rangle_{\\text{FCI}}$ | $O_{\\text{UB}}^{\\text{cert}}$ | Width | $|O_{\\text{var}} - O_{\\text{FCI}}|$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for o, o_fci, do_var in obs_checks:
        lines.append(
            f"| `{o['name']}` | `{o['mode']}` | `{o['O_lb_cert']:.6f}` | `{o_fci:.6f}` | "
            f"`{o['O_ub_cert']:.6f}` | `{o['width']:.5f}` | `{do_var:.2e}` |"
        )

    lines += [
        "\n### 5c. Compact Binary `.flowchem` (`FLCH0002`) vs JSON `.cert.json` Storage",
        "| Certificate | `.cert.json` (B) | `.flowchem` (B) | Reduction |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for base, jb, bb, ratio in bin_rows:
        lines.append(f"| `{base}` | `{jb}` | `{bb}` | `{ratio:.2f}x` |")

    lines += [
        "\n### 5d. Live `ChemOracle` Python SDK (`oracles/chem_sdk.py`) Integration",
        f"- `sdk.bracket(h4_ry2.0, t2=True)`: `E_LB = {r_sdk_t2['E_lb_cert']:.8f} <= E_UB = {r_sdk_t2['E_ub']:.8f}` (PASS)",
        f"- `sdk.cert_verify(h4_ry2.0.flowchem)`: `{v_sdk_bin['message']}` (PASS)",
        f"- `sdk.mlip_screen` false-low energy (`E_ai=-2.5`): `energy_vetoed={s_ev['energy_vetoed']}` at iter `{s_ev['iter_stopped']}` (PASS)",
        f"- `sdk.mlip_screen` bad Hellmann-Feynman force (`f_ai=0.5` vs `[{s_fv['force_bracket']['O_lb_cert']:.4f}, {s_fv['force_bracket']['O_ub_cert']:.4f}]`): `force_vetoed={s_fv['force_vetoed']}` (PASS)",
        f"- `sdk.mlip_screen` exact FCI Hellmann-Feynman force (`f_ai={o_hf_fci:.6f}`): `vetoed={s_ok['vetoed']}` (PASS)",
    ]

    lines.append("\nALL CHEM v0.2.0 PHASE 1-5 ACCEPTANCE GATES PASSED WITH ZERO VIOLATIONS.\n")
    with open(args.report, "w") as f:
        f.write("\n".join(lines))

    print(f"[ORACLE PASS] Adjoints max={c_adj_max:.2e} (T2={c_t2_adj_max:.2e}) | Exact FCI checked={len(fci_checks)} (0 violations) | "
          f"Certs audited={len(cert_files)} (max_diff={max_cert_diff:.2e}) | Peak RSS={data['peak_rss_mb']:.2f} MB")
    print(f"[ORACLE PASS] TM Spin Gap = +{tm_sep*1000.0:.2f} mHa | Obs checked={len(obs_checks)} (0 violations) | "
          f"T2 strict gains={strict_gains}/{len(t2_cases)} | SDK OK | Wrote {args.report}")


if __name__ == "__main__":
    main()
