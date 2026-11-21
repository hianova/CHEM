#!/usr/bin/env python3
"""
Generate canonical Molpro/PySCF FCIDUMP benchmark suite for CHEM in benchmarks/.
Integrals are computed independently via McMurchie-Davidson Hermite Gaussians
(STO-3G Hehre-Stewart-Pople 1969) followed by canonical RHF MO transformation
and active-space frozen-core projection.
"""
import argparse
import math
import os
import numpy as np


def boys(mmax, t):
    if t < 15.0:
        mtop = mmax + 48
        f = 0.0
        et = math.exp(-t)
        out = [0.0] * (mmax + 1)
        for m in range(mtop - 1, -1, -1):
            f = (2.0 * t * f + et) / (2 * m + 1)
            if m <= mmax:
                out[m] = f
        return out
    et = math.exp(-t)
    f0 = 0.5 * math.sqrt(math.pi / t) * math.erf(math.sqrt(t))
    out = [f0]
    for m in range(mmax):
        out.append(((2 * m + 1) * out[-1] - et) / (2.0 * t))
    return out


def md_E(i, j, t, xA, xB, a, b):
    if t < 0 or t > i + j:
        return 0.0
    p = a + b
    q = a * b / p
    xAB = xA - xB
    if i == 0 and j == 0 and t == 0:
        return math.exp(-q * xAB * xAB)
    P = (a * xA + b * xB) / p
    if j == 0:
        return ((1.0 / (2.0 * p)) * md_E(i - 1, 0, t - 1, xA, xB, a, b)
                + (P - xA) * md_E(i - 1, 0, t, xA, xB, a, b)
                + (t + 1) * md_E(i - 1, 0, t + 1, xA, xB, a, b))
    return ((1.0 / (2.0 * p)) * md_E(i, j - 1, t - 1, xA, xB, a, b)
            + (P - xB) * md_E(i, j - 1, t, xA, xB, a, b)
            + (t + 1) * md_E(i, j - 1, t + 1, xA, xB, a, b))


def md_R_table(max_l, p, PCx, PCy, PCz):
    U = p * (PCx * PCx + PCy * PCy + PCz * PCz)
    F = boys(max_l, U)
    R = np.zeros((max_l + 1, max_l + 1, max_l + 1, max_l + 1), dtype=np.float64)
    fac = 1.0
    for n in range(max_l + 1):
        R[0, 0, 0, n] = fac * F[n]
        fac *= -2.0 * p
    for L in range(1, max_l + 1):
        for t in range(L + 1):
            for u in range(L - t + 1):
                v = L - t - u
                for n in range(max_l - L + 1):
                    if t > 0:
                        val = PCx * R[t - 1, u, v, n + 1]
                        if t > 1:
                            val += (t - 1) * R[t - 2, u, v, n + 1]
                    elif u > 0:
                        val = PCy * R[t, u - 1, v, n + 1]
                        if u > 1:
                            val += (u - 1) * R[t, u - 2, v, n + 1]
                    else:
                        val = PCz * R[t, u, v - 1, n + 1]
                        if v > 1:
                            val += (v - 1) * R[t, u, v - 2, n + 1]
                    R[t, u, v, n] = val
    return R


def build_sto3g(Z_list, coords_list, charge_coords=None):
    ch_coords = coords_list if charge_coords is None else charge_coords
    a1s = [2.227660584, 0.4057711562, 0.1098175104]
    d1s = [0.1543289673, 0.5353281423, 0.4446345422]
    a2sp = [0.994203258, 0.231030315, 0.075138553]
    d2s = [-0.09996722919, 0.3995128261, 0.7001154689]
    d2p = [0.1559162750, 0.6076837186, 0.3919573931]
    zeta_map = {
        1: (1.24, None),
        2: (2.0925, None),
        3: (2.69, 0.80),
        6: (5.67, 1.72),
        7: (6.67, 1.95),
        8: (7.66, 2.25),
    }

    aos = []
    for Z, pos in zip(Z_list, coords_list):
        z1s, z2sp = zeta_map[Z]
        aos.append({"pos": pos, "lmn": (0, 0, 0),
                    "exps": [a * z1s * z1s for a in a1s],
                    "coefs": [c * (2.0 * a * z1s * z1s / math.pi) ** 0.75 for a, c in zip(a1s, d1s)]})
        if z2sp is not None:
            aos.append({"pos": pos, "lmn": (0, 0, 0),
                        "exps": [a * z2sp * z2sp for a in a2sp],
                        "coefs": [c * (2.0 * a * z2sp * z2sp / math.pi) ** 0.75 for a, c in zip(a2sp, d2s)]})
            for lmn in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                aos.append({"pos": pos, "lmn": lmn,
                            "exps": [a * z2sp * z2sp for a in a2sp],
                            "coefs": [c * 2.0 * math.sqrt(a * z2sp * z2sp) * (2.0 * a * z2sp * z2sp / math.pi) ** 0.75
                                      for a, c in zip(a2sp, d2p)]})

    for ao in aos:
        l, m, n = ao["lmn"]
        selfS = 0.0
        for a, ca in zip(ao["exps"], ao["coefs"]):
            for b, cb in zip(ao["exps"], ao["coefs"]):
                p = a + b
                selfS += (ca * cb
                          * md_E(l, l, 0, ao["pos"][0], ao["pos"][0], a, b)
                          * md_E(m, m, 0, ao["pos"][1], ao["pos"][1], a, b)
                          * md_E(n, n, 0, ao["pos"][2], ao["pos"][2], a, b)
                          * (math.pi / p) ** 1.5)
        inv = 1.0 / math.sqrt(selfS)
        ao["coefs"] = [c * inv for c in ao["coefs"]]

    K = len(aos)
    S = np.zeros((K, K))
    T = np.zeros((K, K))
    V = np.zeros((K, K))
    ERI = np.zeros((K, K, K, K))

    pair_cache = {}
    for i in range(K):
        ai = aos[i]
        la, ma, na = ai["lmn"]
        for j in range(i + 1):
            aj = aos[j]
            lb, mb, nb = aj["lmn"]
            sij = tij = vij = 0.0
            prims = []
            for pa in range(3):
                a, ca = ai["exps"][pa], ai["coefs"][pa]
                for pb in range(3):
                    b, cb = aj["exps"][pb], aj["coefs"][pb]
                    cc = ca * cb
                    p = a + b
                    pref_st = (math.pi / p) ** 1.5
                    ex0 = md_E(la, lb, 0, ai["pos"][0], aj["pos"][0], a, b)
                    ey0 = md_E(ma, mb, 0, ai["pos"][1], aj["pos"][1], a, b)
                    ez0 = md_E(na, nb, 0, ai["pos"][2], aj["pos"][2], a, b)
                    sij += cc * ex0 * ey0 * ez0 * pref_st

                    dx = -0.5 * (lb * (lb - 1) * md_E(la, lb - 2, 0, ai["pos"][0], aj["pos"][0], a, b)
                                 - 2 * b * (2 * lb + 1) * ex0
                                 + 4 * b * b * md_E(la, lb + 2, 0, ai["pos"][0], aj["pos"][0], a, b))
                    dy = -0.5 * (mb * (mb - 1) * md_E(ma, mb - 2, 0, ai["pos"][1], aj["pos"][1], a, b)
                                 - 2 * b * (2 * mb + 1) * ey0
                                 + 4 * b * b * md_E(ma, mb + 2, 0, ai["pos"][1], aj["pos"][1], a, b))
                    dz = -0.5 * (nb * (nb - 1) * md_E(na, nb - 2, 0, ai["pos"][2], aj["pos"][2], a, b)
                                 - 2 * b * (2 * nb + 1) * ez0
                                 + 4 * b * b * md_E(na, nb + 2, 0, ai["pos"][2], aj["pos"][2], a, b))
                    tij += cc * (dx * ey0 * ez0 + ex0 * dy * ez0 + ex0 * ey0 * dz) * pref_st

                    Px = (a * ai["pos"][0] + b * aj["pos"][0]) / p
                    Py = (a * ai["pos"][1] + b * aj["pos"][1]) / p
                    Pz = (a * ai["pos"][2] + b * aj["pos"][2]) / p
                    Ex = [md_E(la, lb, t, ai["pos"][0], aj["pos"][0], a, b) for t in range(la + lb + 1)]
                    Ey = [md_E(ma, mb, u, ai["pos"][1], aj["pos"][1], a, b) for u in range(ma + mb + 1)]
                    Ez = [md_E(na, nb, v, ai["pos"][2], aj["pos"][2], a, b) for v in range(na + nb + 1)]
                    for Zc, posC in zip(Z_list, ch_coords):
                        Rt = md_R_table(la + lb + ma + mb + na + nb, p,
                                        Px - posC[0], Py - posC[1], Pz - posC[2])
                        acc = 0.0
                        for t, ext in enumerate(Ex):
                            for u, eyu in enumerate(Ey):
                                for v, ezv in enumerate(Ez):
                                    acc += ext * eyu * ezv * Rt[t, u, v, 0]
                        vij -= Zc * cc * (2.0 * math.pi / p) * acc
                    prims.append((cc, p, Px, Py, Pz, Ex, Ey, Ez))
            S[i, j] = S[j, i] = sij
            T[i, j] = T[j, i] = tij
            V[i, j] = V[j, i] = vij
            pair_cache[(i, j)] = prims

    all_s = all(ao["lmn"] == (0, 0, 0) for ao in aos)
    if all_s:
        # Fast NumPy vectorized s-only ERI path (for H12, H16 chains)
        n_pairs = K * (K + 1) // 2
        p_arr = np.zeros((n_pairs, 9))
        w_arr = np.zeros((n_pairs, 9))
        Px_arr = np.zeros((n_pairs, 9))
        Py_arr = np.zeros((n_pairs, 9))
        Pz_arr = np.zeros((n_pairs, 9))
        pair_ij = []
        idx = 0
        for i in range(K):
            for j in range(i + 1):
                pair_ij.append((i, j))
                for m, (cc, p, Px, Py, Pz, Ex, Ey, Ez) in enumerate(pair_cache[(i, j)]):
                    w_arr[idx, m] = cc * Ex[0] * Ey[0] * Ez[0]
                    p_arr[idx, m] = p
                    Px_arr[idx, m] = Px
                    Py_arr[idx, m] = Py
                    Pz_arr[idx, m] = Pz
                idx += 1

        erf_vec = np.vectorize(math.erf, otypes=[np.float64])
        for ij in range(n_pairs):
            i, j = pair_ij[ij]
            p1 = p_arr[ij, :, None]        # (9, 1)
            w1 = w_arr[ij, :, None]
            Px1, Py1, Pz1 = Px_arr[ij, :, None], Py_arr[ij, :, None], Pz_arr[ij, :, None]
            # Vectorize over all kl <= ij
            p2 = p_arr[:ij + 1, None, :]   # (ij+1, 1, 9)
            w2 = w_arr[:ij + 1, None, :]
            dx = Px1[None, :, :] - Px_arr[:ij + 1, None, :]
            dy = Py1[None, :, :] - Py_arr[:ij + 1, None, :]
            dz = Pz1[None, :, :] - Pz_arr[:ij + 1, None, :]
            dist2 = dx * dx + dy * dy + dz * dz
            alpha = (p1[None, :, :] * p2) / (p1[None, :, :] + p2)
            U = alpha * dist2
            F0 = np.where(U < 1e-12, 1.0 - U / 3.0 + U * U / 10.0,
                          0.5 * np.sqrt(math.pi / np.maximum(U, 1e-30)) * erf_vec(np.sqrt(np.maximum(U, 0.0))))
            pref = 2.0 * (math.pi ** 2.5) / (p1[None, :, :] * p2 * np.sqrt(p1[None, :, :] + p2))
            vals = np.sum(w1[None, :, :] * w2 * pref * F0, axis=(1, 2))
            for kl in range(ij + 1):
                k, l = pair_ij[kl]
                val = float(vals[kl])
                for a, b in ((i, j), (j, i)):
                    for c, d in ((k, l), (l, k)):
                        ERI[a, b, c, d] = val
                        ERI[c, d, a, b] = val
    else:
        for i in range(K):
            for j in range(i + 1):
                ij = i * (i + 1) // 2 + j
                prims_ab = pair_cache[(i, j)]
                max_ab = sum(aos[i]["lmn"]) + sum(aos[j]["lmn"])
                for k in range(K):
                    for l in range(k + 1):
                        kl = k * (k + 1) // 2 + l
                        if kl > ij:
                            continue
                        prims_cd = pair_cache[(k, l)]
                        max_cd = sum(aos[k]["lmn"]) + sum(aos[l]["lmn"])
                        val = 0.0
                        for cc1, p, Px, Py, Pz, Ex1, Ey1, Ez1 in prims_ab:
                            for cc2, q, Qx, Qy, Qz, Ex2, Ey2, Ez2 in prims_cd:
                                alpha = p * q / (p + q)
                                Rt = md_R_table(max_ab + max_cd, alpha, Px - Qx, Py - Qy, Pz - Qz)
                                pref = 2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))
                                acc = 0.0
                                for t, et1 in enumerate(Ex1):
                                    for tau, et2 in enumerate(Ex2):
                                        sx = (-1.0 if (tau & 1) else 1.0) * et1 * et2
                                        for u, eu1 in enumerate(Ey1):
                                            for ups, eu2 in enumerate(Ey2):
                                                sy = sx * (-1.0 if (ups & 1) else 1.0) * eu1 * eu2
                                                for v, ev1 in enumerate(Ez1):
                                                    for phi, ev2 in enumerate(Ez2):
                                                        sz = (-1.0 if (phi & 1) else 1.0) * ev1 * ev2
                                                        acc += sy * sz * Rt[t + tau, u + ups, v + phi, 0]
                                val += cc1 * cc2 * pref * acc
                        for a, b in ((i, j), (j, i)):
                            for c, d in ((k, l), (l, k)):
                                ERI[a, b, c, d] = val
                                ERI[c, d, a, b] = val

    enuc = 0.0
    for a in range(len(Z_list)):
        for b in range(a + 1, len(Z_list)):
            dx = ch_coords[a][0] - ch_coords[b][0]
            dy = ch_coords[a][1] - ch_coords[b][1]
            dz = ch_coords[a][2] - ch_coords[b][2]
            enuc += (Z_list[a] * Z_list[b]) / math.sqrt(dx * dx + dy * dy + dz * dz)

    nel = sum(Z_list)
    return K, nel, enuc, S, T + V, ERI


def rhf_canonical_mo(K, nel, enuc, S, Hcore, ERI, return_C=False):
    nocc = nel // 2
    seval, sevec = np.linalg.eigh(S)
    X = sevec @ np.diag(1.0 / np.sqrt(seval)) @ sevec.T
    P = np.zeros((K, K))
    E_prev = 0.0
    C = X.copy()
    for _ in range(150):
        G = np.einsum("ls,mnls->mn", P, ERI) - 0.5 * np.einsum("ls,mlns->mn", P, ERI)
        F = Hcore + G
        E_elec = 0.5 * np.sum(P * (Hcore + F))
        _, Cp = np.linalg.eigh(X.T @ F @ X)
        C = X @ Cp
        P_new = 2.0 * C[:, :nocc] @ C[:, :nocc].T
        if abs(E_elec - E_prev) < 1e-13 and np.linalg.norm(P_new - P) < 1e-12:
            P = P_new
            break
        P = 0.5 * P_new + 0.5 * P if abs(E_elec - E_prev) > 1e-3 else P_new
        E_prev = E_elec
    G = np.einsum("ls,mnls->mn", P, ERI) - 0.5 * np.einsum("ls,mlns->mn", P, ERI)
    F = Hcore + G
    _, Cp = np.linalg.eigh(X.T @ F @ X)
    C = X @ Cp
    H_mo = C.T @ Hcore @ C
    ERI_mo = np.einsum("ip,jq,kr,ls,ijkl->pqrs", C, C, C, C, ERI, optimize=True)
    if return_C:
        return enuc, H_mo, ERI_mo, C
    return enuc, H_mo, ERI_mo


def project_active_space(enuc, H_mo, ERI_mo, frozen_idx, active_idx, nel_active):
    e_core = float(enuc)
    for c in frozen_idx:
        e_core += 2.0 * float(H_mo[c, c])
        for c2 in frozen_idx:
            e_core += 2.0 * float(ERI_mo[c, c, c2, c2]) - float(ERI_mo[c, c2, c, c2])

    Ka = len(active_idx)
    h_act = np.zeros((Ka, Ka))
    for i, pi in enumerate(active_idx):
        for j, pj in enumerate(active_idx):
            v = float(H_mo[pi, pj])
            for c in frozen_idx:
                v += 2.0 * float(ERI_mo[pi, pj, c, c]) - float(ERI_mo[pi, c, pj, c])
            h_act[i, j] = v

    eri_act = ERI_mo[np.ix_(active_idx, active_idx, active_idx, active_idx)].copy()
    return Ka, nel_active, e_core, h_act, eri_act


def write_fcidump(path, K, nelec, ms2, e_core, h1, eri):
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


def make_tm_spin_model(K=8, nelec=8, ms2=0):
    """
    Model transition-metal d6 + 3-ligand (8e,8o) Kanamori-Hubbard active space:
    Orbital 0: closed ligand donor (eps = -2.4 Ha)
    Orbitals 1,2,3: metal t2g (eps = -0.35, -0.32, -0.29 Ha)
    Orbitals 4,5: metal eg (eps = -0.15, -0.12 Ha -> small Delta_cf = 0.20 Ha, weak-field!)
    Orbitals 6,7: empty ligand virtuals (eps = +1.10, +1.25 Ha)
    Strong Hund's J_H = 0.28 Ha vs U = 1.15 Ha favors high-spin S=2 quintet over S=0 singlet.
    """
    h1 = np.zeros((K, K))
    eps = [-2.40, -0.35, -0.32, -0.29, -0.15, -0.12, 1.10, 1.25]
    for p in range(K):
        h1[p, p] = eps[p]
    # Small ligand-metal hybridization
    for p, q, t in [(0, 4, -0.12), (0, 5, -0.10), (4, 6, -0.18), (5, 7, -0.18), (1, 6, -0.06)]:
        h1[p, q] = h1[q, p] = t

    eri = np.zeros((K, K, K, K))
    U_d = 1.15
    J_H = 0.28
    U_p = U_d - 2.0 * J_H
    U_L = 0.85

    # On-site repulsion
    for p in range(K):
        eri[p, p, p, p] = U_d if (1 <= p <= 5) else U_L

    # Inter-orbital Coulomb U' and Hund exchange J_H inside the 5 d-orbitals
    for p in range(1, 6):
        for q in range(p + 1, 6):
            for a, b, c, d in [(p, p, q, q), (q, q, p, p)]:
                eri[a, b, c, d] = U_p
            for a, b, c, d in [
                (p, q, p, q), (q, p, p, q), (p, q, q, p), (q, p, q, p),
                (p, p, q, q)  # keep U' on (pp|qq), J_H on exchange (pq|pq) and pair-hop (pq|qp)
            ]:
                if not (a == b and c == d):
                    eri[a, b, c, d] = J_H
    # Weak inter-site repulsion with ligands
    for p in range(1, 6):
        for l in (0, 6, 7):
            eri[p, p, l, l] = eri[l, l, p, p] = 0.15
    return K, nelec, ms2, 0.0, h1, eri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="benchmarks")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def gen_mol(name, Zs, coords, ms2=0, frozen=None, active=None, nel_act=None):
        K, nel, enuc, S, Hcore, ERI = build_sto3g(Zs, coords)
        # Note: for odd-electron systems (e.g. H3 doublet), use (nel - 1) closed-shell reference Fock or nel-1+1
        nel_ref = nel if (nel % 2 == 0) else (nel - 1)
        enuc, H_mo, ERI_mo = rhf_canonical_mo(K, nel_ref, enuc, S, Hcore, ERI)
        if frozen is None:
            frozen = []
        if active is None:
            active = list(range(K))
        if nel_act is None:
            nel_act = nel - 2 * len(frozen)
        Ka, nela, ecore, ha, eria = project_active_space(enuc, H_mo, ERI_mo, frozen, active, nel_act)
        path = os.path.join(args.out_dir, f"{name}.fcidump")
        write_fcidump(path, Ka, nela, ms2, ecore, ha, eria)
        print(f"Generated {path} (K={Ka}, nelec={nela}, MS2={ms2})")

    # 1. H2 dissociation points
    for r in (1.4, 5.0):
        gen_mol(f"h2_r{r:.1f}", [1, 1], [[0.0, 0.0, -0.5 * r], [0.0, 0.0, 0.5 * r]], ms2=0)

    # 2. Open-shell H3 stretched doublet (3e, 3o, MS2=1)
    gen_mol("h3_doublet", [1, 1, 1], [[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [1.2, 2.08, 0.0]], ms2=1)

    # 3. H4 square-to-rectangle strong correlation curve & square triplet
    for ry in (2.0, 2.2, 2.6, 4.5):
        gen_mol(f"h4_ry{ry:.1f}", [1, 1, 1, 1],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, ry, 0.0], [0.0, ry, 0.0]], ms2=0)
    gen_mol("h4_ry2.0_triplet", [1, 1, 1, 1],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]], ms2=2)

    # 4. H2O eq (10e, 7o) and stretched frozen-1s (8e, 6o)
    th = 104.52 * math.pi / 180.0
    for name, r0, frz in [("h2o_eq", 1.809, []), ("h2o_stretched", 3.2, [0])]:
        coords = [[0.0, 0.0, 0.0],
                  [r0 * math.sin(0.5 * th), 0.0, r0 * math.cos(0.5 * th)],
                  [-r0 * math.sin(0.5 * th), 0.0, r0 * math.cos(0.5 * th)]]
        act = [i for i in range(7) if i not in frz]
        gen_mol(name, [8, 1, 1], coords, ms2=0, frozen=frz, active=act)

    # 5. N2 triple-bond breaking: (6e, 6o) at R=2.07 & 4.2, and (10e, 8o) at R=4.0
    gen_mol("n2_eq_6e6o", [7, 7], [[0.0, 0.0, -1.035], [0.0, 0.0, 1.035]],
            ms2=0, frozen=[0, 1, 2, 3], active=[4, 5, 6, 7, 8, 9])
    gen_mol("n2_stretched_6e6o", [7, 7], [[0.0, 0.0, -2.10], [0.0, 0.0, 2.10]],
            ms2=0, frozen=[0, 1, 2, 3], active=[4, 5, 6, 7, 8, 9])
    gen_mol("n2_stretched_10e8o", [7, 7], [[0.0, 0.0, -2.00], [0.0, 0.0, 2.00]],
            ms2=0, frozen=[0, 1], active=list(range(2, 10)))

    # 6. Transition-metal spin-state ordering (8e, 8o) S=0 vs S=2
    for ms2_val, sname in [(0, "tm_spin_s0_8e8o"), (4, "tm_spin_s2_8e8o")]:
        K, ne, ms2, ec, h1, eri = make_tm_spin_model(8, 8, ms2_val)
        path = os.path.join(args.out_dir, f"{sname}.fcidump")
        write_fcidump(path, K, ne, ms2, ec, h1, eri)
        print(f"Generated {path} (K={K}, nelec={ne}, MS2={ms2})")

    # 7. Beyond-FCI Hydrogen chains: H12 (12e, 12o) & H16 (16e, 16o) singlet & triplet
    for n_h in (12, 16):
        coords = [[0.0, 0.0, 2.4 * i] for i in range(n_h)]
        Zs = [1] * n_h
        gen_mol(f"h{n_h}_chain_{n_h}e{n_h}o", Zs, coords, ms2=0)
        gen_mol(f"h{n_h}_chain_triplet_{n_h}e{n_h}o", Zs, coords, ms2=2)

    # 8. Fixed-MO Hellmann-Feynman nuclear force response for H4 Ry=2.0 (dR_y = +-0.005 Bohr)
    c_ref_geom = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
    K4, nel4, en4, S4, H4, ERI4 = build_sto3g([1, 1, 1, 1], c_ref_geom)
    _, _, _, C_h4 = rhf_canonical_mo(K4, nel4, en4, S4, H4, ERI4, return_C=True)
    for tag, dy in [("fplus", 0.005), ("fminus", -0.005)]:
        c_ch = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0 + dy, 0.0], [0.0, 2.0 + dy, 0.0]]
        _, _, ep, _, Hp, ERIp = build_sto3g([1, 1, 1, 1], c_ref_geom, charge_coords=c_ch)
        Hm = C_h4.T @ Hp @ C_h4
        Em = np.einsum("ip,jq,kr,ls,ijkl->pqrs", C_h4, C_h4, C_h4, C_h4, ERIp, optimize=True)
        path = os.path.join(args.out_dir, f"h4_ry2.0_{tag}.fcidump")
        write_fcidump(path, K4, nel4, 0, ep, Hm, Em)
        print(f"Generated {path} (K={K4}, nelec={nel4}, fixed-MO Hellmann-Feynman dy={dy:+.3f})")


if __name__ == "__main__":
    main()
