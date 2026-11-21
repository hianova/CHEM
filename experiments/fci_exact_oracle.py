#!/usr/bin/env python3
"""
Exact Spin-Sector FCI Oracle, 1/2/3-RDM Evaluator, and Independent T1 = D3 + Q3 Fock Verifier.

Strictly uses exact bitmask Slater determinant Fock-space operators (spin-up + spin-down)
for small correlated clusters (K <= 6), enabling:
  1. Ground & excited eigenstates (E0, E1, gap = E1 - E0) per exact S^2 sector.
  2. Exact observable matrix elements O00, O11, and off-diagonal transition m01 = |<psi0|O|psi1>|.
  3. Exact Hellmann-Feynman Hamiltonian gradients (grad_h1, grad_eri) matching chem_cli indexing.
  4. Independent Fock-space brute-force verification of T1 = D3 + Q3 (T1_aaa, T1_aab, T1_bbb, T1_bba)
     against closed-form 1-RDM + 2-RDM map formulas (< 1e-13 max error).
"""

import itertools
import math
import os
import sys
import numpy as np


def _popcount_below(mask: int, orb: int) -> int:
    return bin(mask & ((1 << orb) - 1)).count("1")


def apply_ann(mask: int, orb: int):
    if not ((mask >> orb) & 1):
        return None
    sgn = -1.0 if (_popcount_below(mask, orb) & 1) else 1.0
    return (mask ^ (1 << orb), sgn)


def apply_cre(mask: int, orb: int):
    if (mask >> orb) & 1:
        return None
    sgn = -1.0 if (_popcount_below(mask, orb) & 1) else 1.0
    return (mask | (1 << orb), sgn)


def apply_seq_single_spin(mask: int, ops):
    """
    Apply a list of ('c' | 'a', orb) operators LEFT-TO-RIGHT mathematically,
    i.e. ops = [op_0, op_1, ..., op_{m-1}] computes op_0 op_1 ... op_{m-1} |mask>,
    so op_{m-1} acts first.
    """
    m = mask
    sgn = 1.0
    for kind, orb in reversed(ops):
        res = apply_cre(m, orb) if kind == "c" else apply_ann(m, orb)
        if res is None:
            return None
        m, s = res
        sgn *= s
    return m, sgn


class SpinDeterminantSpace:
    """Exact (na, nb) Slater determinant basis over K spatial orbitals."""

    def __init__(self, K: int, na: int, nb: int):
        self.K = int(K)
        self.na = int(na)
        self.nb = int(nb)
        self.a_strings = [sum(1 << i for i in c) for c in itertools.combinations(range(K), na)]
        self.b_strings = [sum(1 << i for i in c) for c in itertools.combinations(range(K), nb)]
        self.a_idx = {s: idx for idx, s in enumerate(self.a_strings)}
        self.b_idx = {s: idx for idx, s in enumerate(self.b_strings)}
        self.na_det = len(self.a_strings)
        self.nb_det = len(self.b_strings)
        self.dim = self.na_det * self.nb_det

    def _single_spin_excitations(self, strings, idx_map):
        """Precompute <J| a^+_p a_q |I> tuples: (I, p, q, J, sign)."""
        K = self.K
        ex = []
        for I, m in enumerate(strings):
            for q in range(K):
                if not ((m >> q) & 1):
                    continue
                m1, s1 = apply_ann(m, q)
                for p in range(K):
                    res2 = apply_cre(m1, p)
                    if res2 is not None:
                        m2, s2 = res2
                        J = idx_map[m2]
                        ex.append((I, p, q, J, s1 * s2))
        return ex

    def build_hamiltonian_and_s2(self, e_core: float, h1: np.ndarray, eri: np.ndarray):
        """
        Build dense FCI Hamiltonian and exact S^2 matrix in chemist (pq|rs) convention:
        H = e_core + sum_{p,q,sigma} h1[p,q] a^+_{p,s} a_{q,s}
          + 0.5 * sum_{p,q,r,s,sigma,tau} eri[p,q,r,s] a^+_{p,s} a^+_{r,t} a_{s,t} a_{q,s}
        S^2 = Sz*(Sz+1) + nb - sum_{p,q} a^+_{p,a} a_{q,a} a^+_{q,b} a_{p,b}
        """
        K = self.K
        h1 = np.asarray(h1, dtype=np.float64)
        eri = np.asarray(eri, dtype=np.float64)
        na_det = self.na_det
        nb_det = self.nb_det
        dim = self.dim

        # Single-spin E_{pq} matrices: shape (K, K, n_det, n_det) where E[p, q, J, I] = <J|a^+_p a_q|I>
        Ea = np.zeros((K, K, na_det, na_det), dtype=np.float64)
        for I, m in enumerate(self.a_strings):
            for q in range(K):
                r1 = apply_ann(m, q)
                if r1 is None:
                    continue
                m1, s1 = r1
                for p in range(K):
                    r2 = apply_cre(m1, p)
                    if r2 is not None:
                        J = self.a_idx[r2[0]]
                        Ea[p, q, J, I] += s1 * r2[1]

        Eb = np.zeros((K, K, nb_det, nb_det), dtype=np.float64)
        for I, m in enumerate(self.b_strings):
            for q in range(K):
                r1 = apply_ann(m, q)
                if r1 is None:
                    continue
                m1, s1 = r1
                for p in range(K):
                    r2 = apply_cre(m1, p)
                    if r2 is not None:
                        J = self.b_idx[r2[0]]
                        Eb[p, q, J, I] += s1 * r2[1]

        # H = e_core * I + H_a (x) I_b + I_a (x) H_b + H_ab
        # First compute 1-spin 1e+2e matrices directly in alpha and beta determinant spaces:
        # 2-body same-spin operator:
        # 0.5 * sum_{p,q,r,s} eri[p,q,r,s] a^+_p a^+_r a_s a_q
        # Using a^+_p a^+_r a_s a_q = a^+_p a_q a^+_r a_s - delta_{qr} a^+_p a_s:
        Ea_flat = Ea.reshape(K * K, na_det, na_det)
        Eb_flat = Eb.reshape(K * K, nb_det, nb_det)

        h1_eff = h1 - 0.5 * np.einsum("pqqs->ps", eri, optimize=True)
        Ha_1 = np.einsum("pq,pqJI->JI", h1_eff, Ea, optimize=True)
        Hb_1 = np.einsum("pq,pqJI->JI", h1_eff, Eb, optimize=True)

        eri_mat = eri.reshape(K * K, K * K)
        # 0.5 * sum_{pq, rs} eri[pq, rs] E[pq] @ E[rs]
        Ha_2 = 0.5 * np.einsum("uv,uJK,vKI->JI", eri_mat, Ea_flat, Ea_flat, optimize=True)
        Hb_2 = 0.5 * np.einsum("uv,uJK,vKI->JI", eri_mat, Eb_flat, Eb_flat, optimize=True)

        Ha = Ha_1 + Ha_2
        Hb = Hb_1 + Hb_2

        # Assemble (na_det, nb_det, na_det, nb_det) then reshape to (dim, dim)
        H4 = (
            np.einsum("JI,LK->JLIK", Ha, np.eye(nb_det), optimize=True)
            + np.einsum("JI,LK->JLIK", np.eye(na_det), Hb, optimize=True)
            + np.einsum("uv,uJI,vLK->JLIK", eri_mat, Ea_flat, Eb_flat, optimize=True)
        )
        H = H4.reshape(dim, dim) + e_core * np.eye(dim)
        H = 0.5 * (H + H.T)

        # Build S^2 matrix: Sz*(Sz+1) + nb - sum_{p,q} Ea[p,q] (x) Eb[q,p]
        sz = 0.5 * (self.na - self.nb)
        diag_s2 = sz * (sz + 1.0) + float(self.nb)
        S2_4 = diag_s2 * np.einsum("JI,LK->JLIK", np.eye(na_det), np.eye(nb_det)) - np.einsum(
            "pqJI,qpLK->JLIK", Ea, Eb, optimize=True
        )
        S2 = S2_4.reshape(dim, dim)
        S2 = 0.5 * (S2 + S2.T)

        return H, S2, Ea, Eb

    def build_observable_matrix(self, obs_spec: str, H_parts=None):
        """
        Build FCI matrix in determinant basis for spin-free CHEM observables:
          occ:<p>
          docc:<p>
          sdot:<p>:<q>
        """
        if H_parts is None:
            raise ValueError("Pass (Ea, Eb) from build_hamiltonian_and_s2")
        Ea, Eb = H_parts
        na_det, nb_det, dim = self.na_det, self.nb_det, self.dim
        Ia = np.eye(na_det)
        Ib = np.eye(nb_det)

        if obs_spec.startswith("occ:"):
            p = int(obs_spec.split(":")[1])
            O4 = np.einsum("JI,LK->JLIK", Ea[p, p], Ib) + np.einsum("JI,LK->JLIK", Ia, Eb[p, p])
            return 0.5 * (O4.reshape(dim, dim) + O4.reshape(dim, dim).T)

        if obs_spec.startswith("docc:"):
            p = int(obs_spec.split(":")[1])
            O4 = np.einsum("JI,LK->JLIK", Ea[p, p], Eb[p, p])
            return 0.5 * (O4.reshape(dim, dim) + O4.reshape(dim, dim).T)

        if obs_spec.startswith("sdot:"):
            _, sp, sq = obs_spec.split(":")
            p, q = int(sp), int(sq)
            if p == q:
                # S_p^2 = 0.75 * (n_p - 2 * n_{p,a} n_{p,b})
                O4 = 0.75 * (
                    np.einsum("JI,LK->JLIK", Ea[p, p], Ib)
                    + np.einsum("JI,LK->JLIK", Ia, Eb[p, p])
                    - 2.0 * np.einsum("JI,LK->JLIK", Ea[p, p], Eb[p, p])
                )
                return 0.5 * (O4.reshape(dim, dim) + O4.reshape(dim, dim).T)
            # S_p . S_q = 0.25*(n_{pa}-n_{pb})(n_{qa}-n_{qb})
            #           + 0.5*(a^+_{pa} a_{pb} a^+_{qb} a_{qa} + a^+_{pb} a_{pa} a^+_{qa} a_{qb})
            # Note: a^+_{pa} a_{pb} a^+_{qb} a_{qa} = - (a^+_{pa} a_{qa}) (a^+_{qb} a_{pb})?
            # Careful! Compare: a^+_{pa} a_{qa} a^+_{qb} a_{pb}:
            # Move a_{pb} past a_{qa} a^+_{qb}:
            # Actually a^+_{pa} a_{pb} a^+_{qb} a_{qa}: a_{pb} a^+_{qb} = delta_{pq} - a^+_{qb} a_{pb}, p!=q -> -a^+_{qb} a_{pb},
            # and then moving a^+_{qb} a_{pb} past a_{qa} gives (-1)^2 = +1! So net sign is NEGATIVE?
            # Wait: let's match chem_obs_spin_corr EXACTLY!
            # Spin-free formula in chem_rdm2_mf.c:
            # S_p . S_q = -(1/4) n_p n_q - (1/2) sum_{s,t} a^+_{ps} a^+_{qt} a_{pt} a_{qs}
            # Wait: check sign explicitly via Pauli Sz*Sz + 0.5*(S+S- + S-S+)!
            # 1) Sz_p Sz_q = 0.25 * (n_{pa} - n_{pb}) * (n_{qa} - n_{qb})
            # 2) 0.5 * S_p^+ S_q^- = 0.5 * a^+_{pa} a_{pb} a^+_{qb} a_{qa}:
            #    Move a_{qa} left past a^+_{qb}: sign -1 (p!=q). Now a^+_{pa} a_{pb} a_{qa} a^+_{qb} -> a_{qa} past a_{pb}: sign -1!
            #    Wait: (-1)*(-1) = +1! -> +0.5 * (a^+_{pa} a_{qa}) * (a^+_{qb} a_{pb})? NO! Move (a^+_{qb} a_{pb})...
            # Let's track indices step by step!
            # S_p^+ S_q^- = a^+_{pa} a_{pb} a^+_{qb} a_{qa} (4 operators)
            # Swap a_{pb} and a^+_{qb} (p!=q -> anticommutator 0): = - a^+_{pa} a^+_{qb} a_{pb} a_{qa}
            # Swap a_{pb} and a_{qa}: = + a^+_{pa} a^+_{qb} a_{qa} a_{pb}
            # Swap a^+_{qb} and a_{qa}: = - a^+_{pa} a_{qa} a^+_{qb} a_{pb} !!
            # Wait: a^+_{pa} a_{qa} acts on alpha space (Ea[p, q]), and a^+_{qb} a_{pb} acts on beta space (Eb[q, p])!
            # Wait: why is there a MINUS sign when bringing alpha operators together? Let's test on singlet |up_p, down_q> - |down_p, up_q> / sqrt(2)!
            # In ordered determinant |p_a, q_b> = a^+_{pa} a^+_{qb} |0>:
            # Using our convention apply_cre / apply_ann on |0> (alpha string, beta string):
            # Determinant (1<<p, 1<<q) means a^+_{pa} |0> (x) a^+_{qb} |0>, i.e. alpha operators sit LEFT of beta operators: a^+_{pa} a^+_{qb} |0>!
            # Apply S_p^- S_q^+ = a^+_{pb} a_{pa} a^+_{qa} a_{qb} to a^+_{pa} a^+_{qb} |0>:
            # = - a^+_{pb} a_{pa} a^+_{qa} a^+_{pa} |0> ... Wait:
            # First a_{qb} hits a^+_{pa} a^+_{qb} |0> -> - a^+_{pa} |0>!
            # Next a^+_{qa} hits - a^+_{pa} |0> -> - a^+_{qa} a^+_{pa} |0>
            # Next a_{pa} hits - a^+_{qa} a^+_{pa} |0> -> + a^+_{qa} |0>
            # Next a^+_{pb} hits + a^+_{qa} |0> -> - a^+_{qa} a^+_{pb} |0> ! (since moving a^+_{pb} past a^+_{qa} flips sign!)
            # So S_p^- S_q^+ |p_a, q_b> = - |q_a, p_b> in full anti-symmetrized alpha-left-beta-right convention, WAIT:
            # AND what does (a^+_{qa} a_{pa}) (a^+_{pb} a_{qb}) do on |p_a> (x) |q_b>? It sends |p_a> -> |q_a> (+1) and |q_b> -> |p_b> (+1), giving + |q_a, p_b>!!
            # That's why Ea[q,p] (x) Eb[p,q] has a MINUS sign in front (-Ea[p,q](x)Eb[q,p] - Ea[q,p](x)Eb[p,q])!
            # Which matches EXACTLY our S^2 formula `Sz(Sz+1) + nb - sum Ea[p,q] Eb[q,p]`! Let's verify on 2-site Hubbard singlet: S1.S2 MUST be -0.75 on singlet and +0.25 on triplet!
            npa = Ea[p, p]
            npb = Eb[p, p]
            nqa = Ea[q, q]
            nqb = Eb[q, q]
            # (npa - npb)(nqa - nqb) = npa*nqa (x) Ib + Ia (x) npb*nqb - npa (x) nqb - nqa (x) npb
            O4 = 0.25 * (
                np.einsum("JI,LK->JLIK", npa @ nqa, Ib)
                + np.einsum("JI,LK->JLIK", Ia, npb @ nqb)
                - np.einsum("JI,LK->JLIK", npa, nqb)
                - np.einsum("JI,LK->JLIK", nqa, npb)
            ) - 0.5 * (
                np.einsum("JI,LK->JLIK", Ea[p, q], Eb[q, p])
                + np.einsum("JI,LK->JLIK", Ea[q, p], Eb[p, q])
            )
            return 0.5 * (O4.reshape(dim, dim) + O4.reshape(dim, dim).T)

        raise ValueError(f"Unsupported observable spec: {obs_spec}")


def solve_fci_sector(
    K: int,
    na: int,
    nb: int,
    e_core: float,
    h1: np.ndarray,
    eri: np.ndarray,
    two_s: int = None,
    n_eigs: int = 3,
    obs_specs=None,
):
    """Solve exact FCI in (na, nb) and optionally project to exact S = two_s / 2 sector."""
    space = SpinDeterminantSpace(K, na, nb)
    H, S2, Ea, Eb = space.build_hamiltonian_and_s2(e_core, h1, eri)

    if two_s is not None:
        s_val = 0.5 * float(two_s)
        target_s2 = s_val * (s_val + 1.0)
        s2_evals, s2_vecs = np.linalg.eigh(S2)
        mask = np.abs(s2_evals - target_s2) < 1e-6
        if not np.any(mask):
            raise ValueError(f"No states with 2S={two_s} in (K={K}, na={na}, nb={nb})")
        Q = s2_vecs[:, mask]
        H_sub = Q.T @ H @ Q
        H_sub = 0.5 * (H_sub + H_sub.T)
        evals, sub_vecs = np.linalg.eigh(H_sub)
        evecs = Q @ sub_vecs
        s2_out = np.full(len(evals), target_s2)
    else:
        evals, evecs = np.linalg.eigh(H)
        s2_out = np.diag(evecs.T @ S2 @ evecs)

    k_keep = min(n_eigs, len(evals))
    psi0 = evecs[:, 0]
    gap = float(evals[1] - evals[0]) if len(evals) >= 2 else float("inf")

    obs_info = {}
    if obs_specs:
        for spec in obs_specs:
            Omat = space.build_observable_matrix(spec, (Ea, Eb))
            o00 = float(psi0 @ Omat @ psi0)
            if len(evals) >= 2:
                psi1 = evecs[:, 1]
                o11 = float(psi1 @ Omat @ psi1)
                m01 = float(abs(psi0 @ Omat @ psi1))
                # If level 1..m are nearly degenerate, take norm of transition coupling to E1 manifold:
                deg_idx = np.where(np.abs(evals - evals[1]) < 1e-8)[0]
                m01_eff = float(np.linalg.norm([psi0 @ Omat @ evecs[:, j] for j in deg_idx]))
            else:
                o11 = o00
                m01 = 0.0
                m01_eff = 0.0
            obs_info[spec] = {
                "O00": o00,
                "O11": o11,
                "m01": m01,
                "m01_manifold": m01_eff,
            }

    return {
        "space": space,
        "Ea": Ea,
        "Eb": Eb,
        "evals": evals[:k_keep],
        "E0": float(evals[0]),
        "E1": float(evals[1]) if len(evals) >= 2 else float("nan"),
        "gap": gap,
        "psi0": psi0,
        "evecs": evecs[:, :k_keep],
        "s2_evals": s2_out[:k_keep],
        "obs": obs_info,
    }


def compute_fci_12rdm_and_grad(fci_res, state_idx: int = 0):
    """
    Return exact ga, gb, Daa (K,K,K,K), Dbb, Dab, S1 (spin-orbital vN entropy),
    and Hellmann-Feynman Hamiltonian gradients (grad_h1, grad_eri) matching chem_cli.
    """
    space = fci_res["space"]
    K = space.K
    na_det, nb_det = space.na_det, space.nb_det
    Ea, Eb = fci_res["Ea"], fci_res["Eb"]
    psi = fci_res["evecs"][:, state_idx].reshape(na_det, nb_det)

    # 1-RDMs: ga[p, q] = <psi| a^+_{p,a} a_{q,a} |psi>
    ga = np.einsum("IL,pqJI,JL->pq", psi, Ea, psi, optimize=True)
    gb = np.einsum("IL,pqLK,IK->pq", psi, Eb, psi, optimize=True)
    ga = 0.5 * (ga + ga.T)
    gb = 0.5 * (gb + gb.T)

    # Full 2-RDMs in tensor form:
    # Daa[p, q, r, s] = <a^+_p a^+_q a_s a_r> = sum_t (Ea[p,r] Ea[q,s]) - delta_{qr} Ea[p,s]
    rho_a = psi @ psi.T  # (na_det, na_det): <I| rho_a |J> = sum_L psi[I,L] psi[J,L]
    rho_b = psi.T @ psi  # (nb_det, nb_det)
    t_aa = np.einsum("JI,prJK,qsKI->pqrs", rho_a, Ea, Ea, optimize=True)
    for q in range(K):
        t_aa[:, q, q, :] -= ga  # delta_{q,r} * <a^+_p a_s> = delta_{q,r} * ga[p, s]
    t_bb = np.einsum("LK,prLM,qsMK->pqrs", rho_b, Eb, Eb, optimize=True)
    for q in range(K):
        t_bb[:, q, q, :] -= gb

    # Dab[p, q, r, s] = <a^+_{p,a} a^+_{q,b} a_{s,b} a_{r,a}> = <E^a_{pr} E^b_{qs}>
    Dab = np.einsum("IL,prJI,qsKL,JK->pqrs", psi, Ea, Eb, psi, optimize=True)

    # Spin-orbital von Neumann entropy matching chem_shci.c
    ev_a = np.clip(np.linalg.eigvalsh(ga), 1e-15, 1.0 - 1e-15)
    ev_b = np.clip(np.linalg.eigvalsh(gb), 1e-15, 1.0 - 1e-15)
    S1 = 0.0
    for lam in np.concatenate([ev_a, ev_b]):
        S1 -= float(lam * math.log(lam) + (1.0 - lam) * math.log(1.0 - lam))

    # CHEM grad_h1 and grad_eri indexing:
    # E_2body = 0.5 * sum_{p,r,q,s} eri[p,r,q,s] (Daa[p,q,r,s] + Dbb[p,q,r,s] + 2*Dab[p,q,r,s])
    grad_h1 = ga + gb
    grad_eri = (
        Dab.transpose(0, 2, 1, 3)
        + 0.5 * (t_aa.transpose(0, 2, 1, 3) + t_bb.transpose(0, 2, 1, 3))
    )

    return {
        "ga": ga,
        "gb": gb,
        "Daa": t_aa,
        "Dbb": t_bb,
        "Dab": Dab,
        "S1": S1,
        "grad_h1": grad_h1,
        "grad_eri": grad_eri,
    }


def _build_fock_state_t1_blocks(K: int, na: int, nb: int, psi_vec: np.ndarray, space: SpinDeterminantSpace):
    """
    Independently construct exact Fock-space D3 and Q3 directly by applying
    a^+_p a^+_q a^+_r a_w a_v a_u and a_p a_q a_r a^+_w a^+_v a^+_u to psi_vec.
    No closed-form RDM contractions used here! Pure brute-force Fock operator expectation values.
    """
    psi = psi_vec.reshape(space.na_det, space.nb_det)
    triples = list(itertools.combinations(range(K), 3))
    pairs = list(itertools.combinations(range(K), 2))
    aab_idx = [(p, q, r) for (p, q) in pairs for r in range(K)]

    n_aaa = len(triples)
    n_aab = len(aab_idx)

    # Helper: compute <J| op_a_col |I> or ket vectors in alpha space
    def eval_single_spin_3op(strings, op_builder):
        """Return dict mapping multi-index -> sparse ket action vectors on determinant basis."""
        pass

    D3_aaa = np.zeros((n_aaa, n_aaa), dtype=np.float64)
    Q3_aaa = np.zeros((n_aaa, n_aaa), dtype=np.float64)
    # Build ket vectors C_u = a_u a_v a_w |I_a> and A_u = a^+_w a^+_v a^+_u |I_a>
    # Wait: D3_aaa[(p<q<r),(u<v<w)] = <(a_p a_q a_r |psi>) | (a_u a_v a_w |psi>)>
    # Since (a^+_p a^+_q a^+_r)^dagger = a_r a_q a_p = - a_p a_q a_r !
    # Let's check sign directly via full 6-op string:
    # (a^+_p a^+_q a^+_r a_w a_v a_u) = (a_r a_q a_p)^dagger (a_w a_v a_u) !
    # Check: (a_r a_q a_p)^dagger = a^+_p a^+_q a^+_r . Wait: a_w a_v a_u = - a_u a_v a_w,
    # whereas (a_w a_v a_u) has adjoint a^+_u a^+_v a^+_w.
    # For D3: <(a_w a_v a_u) psi | (a_w' a_v' a_u') psi> with ket op = a_w a_v a_u, its bra op^dag = a^+_p a^+_q a^+_r ONLY if bra op = a_r a_q a_p!
    # Wait: (a_r a_q a_p) -> ket vector has opposite sign of (a_p a_q a_r), wait: on BOTH row and col if we define B_{pqr} = a_r a_q a_p,
    # then B_{pqr}^dag B_{uvw} = a^+_p a^+_q a^+_r a_w a_v a_u !! Exact identity!
    # And for Q3: F_{pqr} = a^+_r a^+_q a^+_p -> F_{pqr}^dag F_{uvw} = a_p a_q a_r a^+_w a^+_v a^+_u !! Exact identity!
    # Wait: check single-spin Hermitian factor cleanly via dot products of transformed wavefunctions!

    # For alpha-alpha-alpha: row matrix B_a[i, I, L] = (a_{r} a_{q} a_{p} |psi>)[new_I, L]
    def act_a_kets(op_list):
        """Return dense tensor (len(op_list), na_det, nb_det) after acting on alpha bits only."""
        out = np.zeros((len(op_list), space.na_det, space.nb_det), dtype=np.float64)
        for idx_op, ops in enumerate(op_list):
            for I, m_a in enumerate(space.a_strings):
                res = apply_seq_single_spin(m_a, ops)
                if res is not None:
                    new_m, sgn = res
                    # map new_m to arbitrary integer mask index via basis? Wait: 3 annihilations land in na-3 determinants!
                    pass
        return out

    # Instead of assuming target determinant basis size is inside na_det, accumulate via full Fock mask dict!
    def compute_fock_gram_single_spin(op_tuples, psi_mat, on_alpha=True):
        """Given list of operator chains O_k acting on one spin (leaving target Fock mask),
        compute Gram matrix G[j, k] = <O_j psi | O_k psi> directly in full Fock space!"""
        vecs = []
        strings = space.a_strings if on_alpha else space.b_strings
        for ops in op_tuples:
            acc = {}
            for idx, m in enumerate(strings):
                res = apply_seq_single_spin(m, ops)
                if res is None:
                    continue
                new_m, sgn = res
                row = psi_mat[idx, :] if on_alpha else psi_mat[:, idx]
                if new_m in acc:
                    acc[new_m] = acc[new_m] + sgn * row
                else:
                    acc[new_m] = sgn * row
            vecs.append(acc)
        n = len(op_tuples)
        G = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            vi = vecs[i]
            if not vi:
                continue
            for j in range(i, n):
                vj = vecs[j]
                if not vj:
                    continue
                val = 0.0
                for m, a in vi.items():
                    if m in vj:
                        val += float(np.dot(a, vj[m]))
                G[i, j] = G[j, i] = val
        return G

    def compute_fock_gram_aab(op_tuples, psi_mat, alpha_first=True):
        """
        op_tuples: list of ((op_a0, op_a1), op_b) pairs.
        If alpha_first=True: compute Gram of operator B_{(pq,r)} = a_{r,b} a_{q,a} a_{p,a}
          so B^dag B = a^+_{p,a} a^+_{q,a} a^+_{r,b} a_{w,b} a_{v,a} a_{u,a}  (= D3_aab!)
        And for F_{(pq,r)} = a^+_{r,b} a^+_{q,a} a^+_{p,a}:
          F^dag F = a_{p,a} a_{q,a} a_{r,b} a^+_{w,b} a^+_{v,a} a^+_{u,a} (= Q3_aab!)
        Wait: careful about sign of moving a_{r,b} past alpha! Since B acts directly on |a> (x) |b>,
        if we treat Fock ket |m_a, m_b> = prod_a a^+_a prod_b a^+_b |0>, ANY single beta operator
        moving left past m_a = na electrons flips sign by (-1)^na, WHICH IS IDENTICAL for row B^dag and col B
        since BOTH act at the same orbital ordering! Wait: (-1)^(2*na) = +1 !!
        """
        a_strings = space.a_strings if alpha_first else space.b_strings
        b_strings = space.b_strings if alpha_first else space.a_strings
        vecs = []
        for ops_a, op_b in op_tuples:
            acc = {}
            for ia, ma in enumerate(a_strings):
                ra = apply_seq_single_spin(ma, ops_a)
                if ra is None:
                    continue
                new_ma, sgna = ra
                for ib, mb in enumerate(b_strings):
                    rb = apply_seq_single_spin(mb, [op_b])
                    if rb is None:
                        continue
                    new_mb, sgnb = rb
                    amp = psi_mat[ia, ib] if alpha_first else psi_mat[ib, ia]
                    key = (new_ma, new_mb)
                    acc[key] = acc.get(key, 0.0) + sgna * sgnb * amp
            vecs.append(acc)
        n = len(op_tuples)
        G = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            vi = vecs[i]
            if not vi:
                continue
            for j in range(i, n):
                vj = vecs[j]
                if not vj:
                    continue
                val = 0.0
                for key, a in vi.items():
                    if key in vj:
                        val += a * vj[key]
                G[i, j] = G[j, i] = val
        return G

    # aaa: B = a_r a_q a_p  =>  B^dag B = a^+_p a^+_q a^+_r a_w a_v a_u (D3)
    #      F = a^+_r a^+_q a^+_p => F^dag F = a_p a_q a_r a^+_w a^+_v a^+_u (Q3)
    d_ops_aaa = [[("a", p[2]), ("a", p[1]), ("a", p[0])] for p in triples]
    q_ops_aaa = [[("c", p[2]), ("c", p[1]), ("c", p[0])] for p in triples]

    D3_aaa = compute_fock_gram_single_spin(d_ops_aaa, psi, on_alpha=True)
    Q3_aaa = compute_fock_gram_single_spin(q_ops_aaa, psi, on_alpha=True)
    D3_bbb = compute_fock_gram_single_spin(d_ops_aaa, psi, on_alpha=False)
    Q3_bbb = compute_fock_gram_single_spin(q_ops_aaa, psi, on_alpha=False)

    # aab: B = (a_q a_p, a_r) => a^+_p a^+_q a^+_r a_w a_v a_u (note: a_r is beta, past 2 alpha ops commutes!)
    d_ops_aab = [([("a", q), ("a", p)], ("a", r)) for (p, q, r) in aab_idx]
    q_ops_aab = [([("c", q), ("c", p)], ("c", r)) for (p, q, r) in aab_idx]
    D3_aab = compute_fock_gram_aab(d_ops_aab, psi, alpha_first=True)
    Q3_aab = compute_fock_gram_aab(q_ops_aab, psi, alpha_first=True)
    D3_bba = compute_fock_gram_aab(d_ops_aab, psi, alpha_first=False)
    Q3_bba = compute_fock_gram_aab(q_ops_aab, psi, alpha_first=False)

    return {
        "D3_aaa": D3_aaa, "Q3_aaa": Q3_aaa, "T1_aaa": D3_aaa + Q3_aaa,
        "D3_aab": D3_aab, "Q3_aab": Q3_aab, "T1_aab": D3_aab + Q3_aab,
        "D3_bbb": D3_bbb, "Q3_bbb": Q3_bbb, "T1_bbb": D3_bbb + Q3_bbb,
        "D3_bba": D3_bba, "Q3_bba": Q3_bba, "T1_bba": D3_bba + Q3_bba,
    }


def _build_map_t1_blocks(K: int, ga: np.ndarray, gb: np.ndarray,
                         Daa: np.ndarray, Dbb: np.ndarray, Dab: np.ndarray):
    """Evaluate closed-form T1 = T1(gamma, D) linear map formulas (no 3-RDM required!)."""
    triples = list(itertools.combinations(range(K), 3))
    pairs = list(itertools.combinations(range(K), 2))
    aab_idx = [(p, q, r) for (p, q) in pairs for r in range(K)]

    def t1_same(g, D2):
        n = len(triples)
        T = np.zeros((n, n), dtype=np.float64)
        # Note D2 has full 4-index antisymmetric tensor D2[p, q, r, s] = <a^+_p a^+_q a_s a_r>
        for row, (p0, p1, p2) in enumerate(triples):
            p_arr = (p0, p1, p2)
            for col, (q0, q1, q2) in enumerate(triples):
                q_arr = (q0, q1, q2)
                val = 1.0 if row == col else 0.0
                for i in range(3):
                    a1, a2 = p_arr[(i + 1) % 3], p_arr[(i + 2) % 3]
                    if a1 > a2:
                        a1, a2 = a2, a1
                    for j in range(3):
                        b1, b2 = q_arr[(j + 1) % 3], q_arr[(j + 2) % 3]
                        if b1 > b2:
                            b1, b2 = b2, b1
                        # Careful: cyclic pair preserves parity since (0,1,2):
                        # i=0 -> (1,2) +; i=1 -> (2,0) -, flipped (0,2) cancels (-1)^1 -> let's use exact remaining 2 ordered!
                        pa = tuple(p_arr[k] for k in range(3) if k != i)
                        qb = tuple(q_arr[k] for k in range(3) if k != j)
                        sgn = -1.0 if ((i + j) & 1) else 1.0
                        if pa == qb:
                            val -= sgn * g[p_arr[i], q_arr[j]]
                        if p_arr[i] == q_arr[j]:
                            val += sgn * D2[pa[0], pa[1], qb[0], qb[1]]
                T[row, col] = val
        return T

    def t1_mixed(g_same, g_diff, D_same, D_cross):
        n = len(aab_idx)
        T = np.zeros((n, n), dtype=np.float64)
        for row, (p, q, r) in enumerate(aab_idx):
            for col, (u, v, w) in enumerate(aab_idx):
                val = 0.0
                if r == w:
                    if p == u and q == v:
                        val += 1.0
                    if p == u:
                        val -= g_same[q, v]
                    if p == v:
                        val += g_same[q, u]
                    if q == u:
                        val += g_same[p, v]
                    if q == v:
                        val -= g_same[p, u]
                    val += D_same[p, q, u, v]
                if p == u and q == v:
                    val -= g_diff[r, w]
                if p == u:
                    val += D_cross[q, r, v, w]
                if p == v:
                    val -= D_cross[q, r, u, w]
                if q == u:
                    val -= D_cross[p, r, v, w]
                if q == v:
                    val += D_cross[p, r, u, w]
                T[row, col] = val
        return T

    return {
        "T1_aaa": t1_same(ga, Daa),
        "T1_bbb": t1_same(gb, Dbb),
        "T1_aab": t1_mixed(ga, gb, Daa, Dab),
        "T1_bba": t1_mixed(gb, ga, Dbb, Dab.transpose(1, 0, 3, 2)),
    }


def verify_t1_fock_oracle(K: int = 4, na: int = 2, nb: int = 2, two_s: int = 0, seed: int = 7):
    """
    Generate a strongly correlated asymmetric random Hamiltonian (with nonzero 1-body + 2-body),
    compute its FCI eigenstate psi0 in exact spin sector 2S, and assert:
      max |T1_fock - T1_map| < 1e-13, min_eig(T1) > -1e-13, trace exact < 1e-13.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((K, K))
    h1 = 0.5 * (A + A.T)
    B = rng.standard_normal((K, K, K, K))
    eri = 0.125 * (
        B + B.transpose(1, 0, 2, 3) + B.transpose(0, 1, 3, 2) + B.transpose(1, 0, 3, 2)
        + B.transpose(2, 3, 0, 1) + B.transpose(3, 2, 0, 1) + B.transpose(2, 3, 1, 0) + B.transpose(3, 2, 1, 0)
    )
    res = solve_fci_sector(K, na, nb, 0.0, h1, eri, two_s=two_s, n_eigs=2)
    rdms = compute_fci_12rdm_and_grad(res, state_idx=0)
    fock = _build_fock_state_t1_blocks(K, na, nb, res["psi0"], res["space"])
    mp = _build_map_t1_blocks(K, rdms["ga"], rdms["gb"], rdms["Daa"], rdms["Dbb"], rdms["Dab"])

    def comb(n, k):
        return math.comb(n, k) if (0 <= k <= n) else 0

    exp_traces = {
        "T1_aaa": float(comb(na, 3) + comb(K - na, 3)),
        "T1_bbb": float(comb(nb, 3) + comb(K - nb, 3)),
        "T1_aab": float(comb(na, 2) * nb + comb(K - na, 2) * (K - nb)),
        "T1_bba": float(comb(nb, 2) * na + comb(K - nb, 2) * (K - na)),
    }

    report = {}
    max_err = 0.0
    min_eig = 1e9
    for key in ("T1_aaa", "T1_aab", "T1_bbb", "T1_bba"):
        diff = float(np.max(np.abs(fock[key] - mp[key])))
        ev = float(np.min(np.linalg.eigvalsh(mp[key])))
        tr_diff = abs(float(np.trace(mp[key])) - exp_traces[key])
        max_err = max(max_err, diff, tr_diff)
        min_eig = min(min_eig, ev)
        report[key] = {
            "map_vs_fock_err": diff,
            "trace_err": tr_diff,
            "min_eig": ev,
            "D3_min_eig": float(np.min(np.linalg.eigvalsh(fock[key.replace("T1", "D3")]))),
            "Q3_min_eig": float(np.min(np.linalg.eigvalsh(fock[key.replace("T1", "Q3")]))),
        }

    return {"K": K, "na": na, "nb": nb, "two_s": two_s, "E0": res["E0"], "max_err": max_err, "min_eig": min_eig, "blocks": report}


if __name__ == "__main__":
    print("=== Independent Brute-Force Fock-Space T1 = D3 + Q3 Verification ===")
    for label, na, nb, two_s, seed in [("Singlet (4e,4o)", 2, 2, 0, 11), ("Triplet (4e,4o)", 3, 1, 2, 23), ("Doublet (5e,5o)", 3, 2, 1, 31)]:
        K = 5 if "5o" in label else 4
        rep = verify_t1_fock_oracle(K=K, na=na, nb=nb, two_s=two_s, seed=seed)
        print(f"  {label}: max_err={rep['max_err']:.3e}  min_eig(T1)={rep['min_eig']:.3e}")
        assert rep["max_err"] < 1e-13, f"T1 mismatch on {label}: {rep}"
        assert rep["min_eig"] > -1e-13, f"T1 non-PSD on {label}: {rep}"
    print("ALL FOCK T1=D3+Q3 ORACLE VERIFICATIONS PASSED (< 1e-13).")
