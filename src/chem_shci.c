#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include "chem_shci.h"
#include "chem_rdm2.h"
#include "flow.h"
#include "searu.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHEM_ISQRT2 0.70710678118654752440084436210484904

static double shci_now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void shci_set_err(char *err, size_t err_sz, const char *fmt, ...) {
    if (!err || err_sz == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_sz, fmt, ap);
    va_end(ap);
}

ChemVarOpts chem_var_opts_default(void) {
    ChemVarOpts o;
    memset(&o, 0, sizeof(o));
    o.max_dets = 1024;
    o.run_searu = 1;
    o.target_two_s = -1;
    o.n_eps1 = 5;
    o.eps1_schedule[0] = 0.05;
    o.eps1_schedule[1] = 0.015;
    o.eps1_schedule[2] = 0.005;
    o.eps1_schedule[3] = 0.0015;
    o.eps1_schedule[4] = 0.0004;
    o.seed = UINT64_C(0x517CC1B727220A95);
    return o;
}

void chem_var_result_free(ChemVarResult *vr) {
    if (!vr) return;
    free(vr->dets);
    free(vr->psi0);
    memset(vr, 0, sizeof(*vr));
}

/* ============================================================================
 * Jordan-Wigner Determinant & Slater-Condon Rules
 * ============================================================================ */

static inline int jw_par(uint64_t det, int bit) {
    if (bit == 0) return 1;
    uint64_t mask = (UINT64_C(1) << (unsigned)bit) - UINT64_C(1);
    return (searu_popcnt64(det & mask) & 1U) ? -1 : 1;
}

static inline double h1_so(const ChemHam *ham, int p, int q) {
    if ((p & 1) != (q & 1)) return 0.0;
    return chem_ham_h1(ham, p >> 1, q >> 1);
}

static inline double e2_so(const ChemHam *ham, int p, int q, int r, int s) {
    if ((p & 1) != (r & 1) || (q & 1) != (s & 1)) return 0.0;
    return chem_ham_eri(ham, p >> 1, r >> 1, q >> 1, s >> 1);
}

static double det_diag_energy(const ChemHam *ham, uint64_t d) {
    int occ[64];
    int nocc = 0;
    uint64_t tmp = d;
    while (tmp) {
        int b = (int)searu_ctz64(tmp);
        occ[nocc++] = b;
        tmp &= tmp - UINT64_C(1);
    }
    double e = ham->E_core;
    for (int i = 0; i < nocc; i++) {
        int p = occ[i];
        e += h1_so(ham, p, p);
        for (int j = i + 1; j < nocc; j++) {
            int q = occ[j];
            e += e2_so(ham, p, q, p, q) - e2_so(ham, p, q, q, p);
        }
    }
    return e;
}

static double slater_condon_mel(const ChemHam *ham, uint64_t dI, uint64_t dJ) {
    if (dI == dJ) return det_diag_energy(ham, dI);
    uint64_t diff = dI ^ dJ;
    uint32_t nd = searu_popcnt64(diff);
    if (nd == 2U) {
        int r = (int)searu_ctz64(dI & diff);
        int c = (int)searu_ctz64(dJ & diff);
        if ((r & 1) != (c & 1)) return 0.0;
        int sgn = jw_par(dI, r) * jw_par(dI ^ (UINT64_C(1) << (unsigned)r), c);
        double mat = h1_so(ham, r, c);
        uint64_t common = dI & dJ;
        while (common) {
            int p = (int)searu_ctz64(common);
            mat += e2_so(ham, r, p, c, p) - e2_so(ham, r, p, p, c);
            common &= common - UINT64_C(1);
        }
        return (double)sgn * mat;
    }
    if (nd == 4U) {
        uint64_t rem = dI & diff;
        int r1 = (int)searu_ctz64(rem);
        int r2 = (int)searu_ctz64(rem ^ (UINT64_C(1) << (unsigned)r1));
        uint64_t add = dJ & diff;
        int c1 = (int)searu_ctz64(add);
        int c2 = (int)searu_ctz64(add ^ (UINT64_C(1) << (unsigned)c1));
        uint64_t t1 = dI ^ (UINT64_C(1) << (unsigned)r1);
        uint64_t t2 = t1 ^ (UINT64_C(1) << (unsigned)r2);
        uint64_t t3 = t2 ^ (UINT64_C(1) << (unsigned)c2);
        int sgn = jw_par(dI, r1) * jw_par(t1, r2) * jw_par(t2, c2) * jw_par(t3, c1);
        double mat = e2_so(ham, r1, r2, c1, c2) - e2_so(ham, r1, r2, c2, c1);
        return (double)sgn * mat;
    }
    return 0.0;
}

static double spin_sq_mel(int na, int nb, uint64_t dI, uint64_t dJ) {
    static const uint64_t EVEN = UINT64_C(0x5555555555555555);
    if (dI == dJ) {
        uint64_t a = dI & EVEN;
        uint64_t b = (dI >> 1U) & EVEN;
        uint32_t ndocc = searu_popcnt64(a & b);
        double sz = 0.5 * (double)(na - nb);
        return sz * (sz + 1.0) + (double)nb - (double)ndocc;
    }
    uint64_t aI = dI & EVEN, bI = (dI >> 1U) & EVEN;
    uint64_t aJ = dJ & EVEN, bJ = (dJ >> 1U) & EVEN;
    uint64_t a_in  = aI & ~aJ;
    uint64_t a_out = aJ & ~aI;
    uint64_t b_in  = bI & ~bJ;
    uint64_t b_out = bJ & ~bI;
    if (a_in == b_out && a_out == b_in &&
        searu_popcnt64(a_in) == 1U && searu_popcnt64(a_out) == 1U) {
        return 1.0;
    }
    return 0.0;
}

static uint64_t binom_sat(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    uint64_t res = 1;
    for (int i = 1; i <= k; i++) {
        uint64_t num = (uint64_t)(n - k + i);
        if (res > (UINT64_MAX / num)) return UINT64_MAX;
        res = (res * num) / (uint64_t)i;
    }
    return res;
}

static uint64_t mul_sat_u64(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > UINT64_MAX / b) return UINT64_MAX;
    return a * b;
}

static uint64_t canonical_hf_det(int na, int nb) {
    uint64_t d = 0;
    for (int p = 0; p < na; p++) d |= (UINT64_C(1) << (unsigned)(2 * p));
    for (int p = 0; p < nb; p++) d |= (UINT64_C(1) << (unsigned)(2 * p + 1));
    return d;
}

static uint64_t spin_flip_det(uint64_t d, int K) {
    uint64_t out = 0;
    for (int p = 0; p < K; p++) {
        uint64_t a = (d >> (unsigned)(2 * p)) & UINT64_C(1);
        uint64_t b = (d >> (unsigned)(2 * p + 1)) & UINT64_C(1);
        out |= (b << (unsigned)(2 * p)) | (a << (unsigned)(2 * p + 1));
    }
    return out;
}

/* ============================================================================
 * RHF / UHF + Pulay DIIS in Orthonormal MO Basis (S = I_K)
 * ============================================================================ */

static int sym_eigh_small(int n, double *A, double *evals) {
    __LAPACK_int nn = (__LAPACK_int)n;
    __LAPACK_int lwork = -1, liwork = -1, info = 0;
    double wq = 0.0;
    __LAPACK_int iwq = 0;
    dsyevd_("V", "U", &nn, A, &nn, evals, &wq, &lwork, &iwq, &liwork, &info);
    if (info != 0) return -1;
    lwork = (__LAPACK_int)wq + 16;
    liwork = iwq + 8;
    double *wk = (double *)malloc((size_t)lwork * sizeof(double));
    __LAPACK_int *iwk = (__LAPACK_int *)malloc((size_t)liwork * sizeof(__LAPACK_int));
    if (!wk || !iwk) {
        free(wk);
        free(iwk);
        return -1;
    }
    dsyevd_("V", "U", &nn, A, &nn, evals, wk, &lwork, iwk, &liwork, &info);
    free(wk);
    free(iwk);
    return (info == 0) ? 0 : -1;
}

static void build_uhf_fock(const ChemHam *ham, const double *Pa, const double *Pb,
                           double *Fa, double *Fb) {
    int K = ham->K;
    size_t ksz = (size_t)K;
    for (int u = 0; u < K; u++) {
        for (int v = u; v < K; v++) {
            double ga = chem_ham_h1(ham, u, v);
            double gb = ga;
            for (int l = 0; l < K; l++) {
                for (int s = 0; s < K; s++) {
                    double pa = Pa[(size_t)l * ksz + (size_t)s];
                    double pb = Pb[(size_t)l * ksz + (size_t)s];
                    double j = chem_ham_eri(ham, u, v, l, s);
                    double k = chem_ham_eri(ham, u, l, v, s);
                    ga += (pa + pb) * j - pa * k;
                    gb += (pa + pb) * j - pb * k;
                }
            }
            Fa[(size_t)u * ksz + (size_t)v] = Fa[(size_t)v * ksz + (size_t)u] = ga;
            Fb[(size_t)u * ksz + (size_t)v] = Fb[(size_t)v * ksz + (size_t)u] = gb;
        }
    }
}

static void density_from_c_rows(int K, int nocc, const double *V_rows, double *P) {
    /* sym_eigh_small returns eigenvectors in C rows: MO k is V_rows[k*K + mu] */
    size_t ksz = (size_t)K;
    memset(P, 0, ksz * ksz * sizeof(double));
    for (int k = 0; k < nocc; k++) {
        const double *ck = V_rows + (size_t)k * ksz;
        for (int i = 0; i < K; i++) {
            for (int j = i; j < K; j++) {
                double v = ck[i] * ck[j];
                P[(size_t)i * ksz + (size_t)j] += v;
                if (i != j) P[(size_t)j * ksz + (size_t)i] += v;
            }
        }
    }
}

static double scf_comm_norm(int K, const double *F, const double *P) {
    size_t ksz = (size_t)K;
    double n2 = 0.0;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            double fp = 0.0, pf = 0.0;
            for (int m = 0; m < K; m++) {
                fp += F[(size_t)i * ksz + (size_t)m] * P[(size_t)m * ksz + (size_t)j];
                pf += P[(size_t)i * ksz + (size_t)m] * F[(size_t)m * ksz + (size_t)j];
            }
            double d = fp - pf;
            n2 += d * d;
        }
    }
    return sqrt(n2);
}

#define MAX_DIIS_HIST 6

static double run_scf_diis(const ChemHam *ham, int restricted, double *C_a_out, double *C_b_out,
                           int *iters_out, double *comm_out) {
    int K = ham->K, na = ham->na, nb = ham->nb;
    if (na != nb) restricted = 0;
    size_t ksz = (size_t)K;
    size_t k2 = ksz * ksz;

    double *Pa = (double *)calloc(k2, sizeof(double));
    double *Pb = (double *)calloc(k2, sizeof(double));
    double *Fa = (double *)calloc(k2, sizeof(double));
    double *Fb = (double *)calloc(k2, sizeof(double));
    double *Va = (double *)calloc(k2, sizeof(double));
    double *Vb = (double *)calloc(k2, sizeof(double));
    double *eva = (double *)calloc(ksz, sizeof(double));
    double *evb = (double *)calloc(ksz, sizeof(double));

    double *hist_Fa[MAX_DIIS_HIST] = {0}, *hist_Fb[MAX_DIIS_HIST] = {0};
    double *hist_ea[MAX_DIIS_HIST] = {0}, *hist_eb[MAX_DIIS_HIST] = {0};
    int alloc_ok = (Pa && Pb && Fa && Fb && Va && Vb && eva && evb);
    for (int i = 0; i < MAX_DIIS_HIST && alloc_ok; i++) {
        hist_Fa[i] = (double *)calloc(k2, sizeof(double));
        hist_Fb[i] = (double *)calloc(k2, sizeof(double));
        hist_ea[i] = (double *)calloc(k2, sizeof(double));
        hist_eb[i] = (double *)calloc(k2, sizeof(double));
        if (!hist_Fa[i] || !hist_Fb[i] || !hist_ea[i] || !hist_eb[i]) alloc_ok = 0;
    }
    if (!alloc_ok) {
        free(Pa); free(Pb); free(Fa); free(Fb); free(Va); free(Vb); free(eva); free(evb);
        for (int i = 0; i < MAX_DIIS_HIST; i++) {
            free(hist_Fa[i]); free(hist_Fb[i]); free(hist_ea[i]); free(hist_eb[i]);
        }
        return det_diag_energy(ham, canonical_hf_det(na, nb));
    }

    /* Core-Hamiltonian guess (plus canonical HF det check) */
    memcpy(Va, ham->h, k2 * sizeof(double));
    sym_eigh_small(K, Va, eva);
    memcpy(Vb, Va, k2 * sizeof(double));
    density_from_c_rows(K, na, Va, Pa);
    density_from_c_rows(K, nb, Vb, Pb);

    double E_best = det_diag_energy(ham, canonical_hf_det(na, nb));
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            C_a_out[(size_t)i * ksz + (size_t)j] = (i == j) ? 1.0 : 0.0;
            C_b_out[(size_t)i * ksz + (size_t)j] = (i == j) ? 1.0 : 0.0;
        }
    }

    double E_prev = 0.0;
    int nh = 0;
    int iters = 0;
    double cnorm = 0.0;

    for (int it = 1; it <= 120; it++) {
        iters = it;
        build_uhf_fock(ham, Pa, Pb, Fa, Fb);

        double E_elec = 0.0;
        for (size_t i = 0; i < k2; i++) {
            double hij = ham->h[i];
            E_elec += 0.5 * (Pa[i] * (hij + Fa[i]) + Pb[i] * (hij + Fb[i]));
        }
        double E_tot = ham->E_core + E_elec;

        /* Store DIIS error e = F*P - P*F */
        int slot = nh % MAX_DIIS_HIST;
        memcpy(hist_Fa[slot], Fa, k2 * sizeof(double));
        memcpy(hist_Fb[slot], Fb, k2 * sizeof(double));
        double ea_norm = 0.0, eb_norm = 0.0;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                double fpa = 0.0, pfa = 0.0, fpb = 0.0, pfb = 0.0;
                for (int m = 0; m < K; m++) {
                    fpa += Fa[(size_t)i * ksz + (size_t)m] * Pa[(size_t)m * ksz + (size_t)j];
                    pfa += Pa[(size_t)i * ksz + (size_t)m] * Fa[(size_t)m * ksz + (size_t)j];
                    fpb += Fb[(size_t)i * ksz + (size_t)m] * Pb[(size_t)m * ksz + (size_t)j];
                    pfb += Pb[(size_t)i * ksz + (size_t)m] * Fb[(size_t)m * ksz + (size_t)j];
                }
                double da = fpa - pfa;
                double db = fpb - pfb;
                hist_ea[slot][(size_t)i * ksz + (size_t)j] = da;
                hist_eb[slot][(size_t)i * ksz + (size_t)j] = db;
                ea_norm += da * da;
                eb_norm += db * db;
            }
        }
        cnorm = sqrt(ea_norm + eb_norm);
        if (nh < MAX_DIIS_HIST) nh++;
        else {
            /* Shift history FIFO cleanly */
            double *tmpFa = hist_Fa[0], *tmpFb = hist_Fb[0];
            double *tmpea = hist_ea[0], *tmpeb = hist_eb[0];
            for (int q = 0; q < MAX_DIIS_HIST - 1; q++) {
                hist_Fa[q] = hist_Fa[q + 1];
                hist_Fb[q] = hist_Fb[q + 1];
                hist_ea[q] = hist_ea[q + 1];
                hist_eb[q] = hist_eb[q + 1];
            }
            hist_Fa[MAX_DIIS_HIST - 1] = tmpFa;
            hist_Fb[MAX_DIIS_HIST - 1] = tmpFb;
            hist_ea[MAX_DIIS_HIST - 1] = tmpea;
            hist_eb[MAX_DIIS_HIST - 1] = tmpeb;
            memcpy(hist_Fa[MAX_DIIS_HIST - 1], Fa, k2 * sizeof(double));
            memcpy(hist_Fb[MAX_DIIS_HIST - 1], Fb, k2 * sizeof(double));
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    double fpa = 0.0, pfa = 0.0, fpb = 0.0, pfb = 0.0;
                    for (int m = 0; m < K; m++) {
                        fpa += Fa[(size_t)i * ksz + (size_t)m] * Pa[(size_t)m * ksz + (size_t)j];
                        pfa += Pa[(size_t)i * ksz + (size_t)m] * Fa[(size_t)m * ksz + (size_t)j];
                        fpb += Fb[(size_t)i * ksz + (size_t)m] * Pb[(size_t)m * ksz + (size_t)j];
                        pfb += Pb[(size_t)i * ksz + (size_t)m] * Fb[(size_t)m * ksz + (size_t)j];
                    }
                    hist_ea[MAX_DIIS_HIST - 1][(size_t)i * ksz + (size_t)j] = fpa - pfa;
                    hist_eb[MAX_DIIS_HIST - 1][(size_t)i * ksz + (size_t)j] = fpb - pfb;
                }
            }
        }

        if (it >= 2 && fabs(E_tot - E_prev) < 1e-11 && cnorm < 1e-9) {
            if (E_tot < E_best) {
                E_best = E_tot;
                memcpy(C_a_out, Va, k2 * sizeof(double));
                memcpy(C_b_out, Vb, k2 * sizeof(double));
            }
            break;
        }
        E_prev = E_tot;

        if (nh >= 2 && cnorm < 1.0) {
            int m = nh;
            int nsys = m + 1;
            double B[(MAX_DIIS_HIST + 1) * (MAX_DIIS_HIST + 1)];
            double rhs[MAX_DIIS_HIST + 1];
            __LAPACK_int ipiv[MAX_DIIS_HIST + 1];
            memset(B, 0, sizeof(B));
            memset(rhs, 0, sizeof(rhs));
            for (int i = 0; i < m; i++) {
                B[i * nsys + m] = B[m * nsys + i] = -1.0;
                for (int j = i; j < m; j++) {
                    double dot = 0.0;
                    for (size_t q = 0; q < k2; q++) {
                        dot += hist_ea[i][q] * hist_ea[j][q] + hist_eb[i][q] * hist_eb[j][q];
                    }
                    B[i * nsys + j] = B[j * nsys + i] = dot;
                }
            }
            rhs[m] = -1.0;
            __LAPACK_int nn = (__LAPACK_int)nsys, nrhs = 1, info = 0;
            dgesv_(&nn, &nrhs, B, &nn, ipiv, rhs, &nn, &info);
            if (info == 0) {
                memset(Fa, 0, k2 * sizeof(double));
                memset(Fb, 0, k2 * sizeof(double));
                for (int i = 0; i < m; i++) {
                    for (size_t q = 0; q < k2; q++) {
                        Fa[q] += rhs[i] * hist_Fa[i][q];
                        Fb[q] += rhs[i] * hist_Fb[i][q];
                    }
                }
            }
        }

        memcpy(Va, Fa, k2 * sizeof(double));
        sym_eigh_small(K, Va, eva);
        if (restricted) {
            memcpy(Vb, Va, k2 * sizeof(double));
        } else {
            memcpy(Vb, Fb, k2 * sizeof(double));
            sym_eigh_small(K, Vb, evb);
        }

        double *Pa_new = (double *)malloc(k2 * sizeof(double));
        double *Pb_new = (double *)malloc(k2 * sizeof(double));
        if (!Pa_new || !Pb_new) {
            free(Pa_new);
            free(Pb_new);
            break;
        }
        density_from_c_rows(K, na, Va, Pa_new);
        density_from_c_rows(K, nb, Vb, Pb_new);

        /* Evaluate exact single-det energy at new density */
        build_uhf_fock(ham, Pa_new, Pb_new, Fa, Fb);
        double E_new = ham->E_core;
        for (size_t i = 0; i < k2; i++) {
            E_new += 0.5 * (Pa_new[i] * (ham->h[i] + Fa[i]) + Pb_new[i] * (ham->h[i] + Fb[i]));
        }
        if (E_new < E_best) {
            E_best = E_new;
            memcpy(C_a_out, Va, k2 * sizeof(double));
            memcpy(C_b_out, Vb, k2 * sizeof(double));
        }

        double damp = (it < 4) ? 0.5 : 1.0;
        for (size_t i = 0; i < k2; i++) {
            Pa[i] = damp * Pa_new[i] + (1.0 - damp) * Pa[i];
            Pb[i] = damp * Pb_new[i] + (1.0 - damp) * Pb[i];
        }
        free(Pa_new);
        free(Pb_new);
    }

    density_from_c_rows(K, na, C_a_out, Pa);
    density_from_c_rows(K, nb, C_b_out, Pb);
    build_uhf_fock(ham, Pa, Pb, Fa, Fb);
    cnorm = sqrt(scf_comm_norm(K, Fa, Pa) * scf_comm_norm(K, Fa, Pa) +
                 scf_comm_norm(K, Fb, Pb) * scf_comm_norm(K, Fb, Pb));

    for (int i = 0; i < MAX_DIIS_HIST; i++) {
        free(hist_Fa[i]);
        free(hist_Fb[i]);
        free(hist_ea[i]);
        free(hist_eb[i]);
    }
    free(Pa); free(Pb); free(Fa); free(Fb);
    free(Va); free(Vb); free(eva); free(evb);

    if (iters_out) *iters_out = iters;
    if (comm_out) *comm_out = cnorm;
    return E_best;
}

/* ============================================================================
 * SEARU x 2-Jet Orbital Rotation Escape + Discrete Determinant Discovery
 * ============================================================================ */

static double eval_sd_energy_from_C(const ChemHam *ham, const double *Ca, const double *Cb,
                                    double *Pa, double *Pb, double *Fa, double *Fb) {
    int K = ham->K;
    size_t k2 = (size_t)K * (size_t)K;
    density_from_c_rows(K, ham->na, Ca, Pa);
    density_from_c_rows(K, ham->nb, Cb, Pb);
    build_uhf_fock(ham, Pa, Pb, Fa, Fb);
    double e = ham->E_core;
    for (size_t i = 0; i < k2; i++) {
        e += 0.5 * (Pa[i] * (ham->h[i] + Fa[i]) + Pb[i] * (ham->h[i] + Fb[i]));
    }
    return e;
}

static void apply_jacobi_row_rot(int K, double *C, int r1, int r2, double theta) {
    double c = cos(theta), s = sin(theta);
    size_t ksz = (size_t)K;
    double *row1 = C + (size_t)r1 * ksz;
    double *row2 = C + (size_t)r2 * ksz;
    for (int m = 0; m < K; m++) {
        double x = row1[m], y = row2[m];
        row1[m] = c * x + s * y;
        row2[m] = -s * x + c * y;
    }
}

static double searu_2jet_orbital_rot(const ChemHam *ham, int restricted, const double *Ca_in,
                                     const double *Cb_in, uint64_t seed) {
    int K = ham->K, na = ham->na, nb = ham->nb;
    if (na != nb) restricted = 0;
    size_t k2 = (size_t)K * (size_t)K;
    double *Ca = (double *)malloc(k2 * sizeof(double));
    double *Cb = (double *)malloc(k2 * sizeof(double));
    double *Ca_try = (double *)malloc(k2 * sizeof(double));
    double *Cb_try = (double *)malloc(k2 * sizeof(double));
    double *Pa = (double *)malloc(k2 * sizeof(double));
    double *Pb = (double *)malloc(k2 * sizeof(double));
    double *Fa = (double *)malloc(k2 * sizeof(double));
    double *Fb = (double *)malloc(k2 * sizeof(double));
    if (!Ca || !Cb || !Ca_try || !Cb_try || !Pa || !Pb || !Fa || !Fb) {
        free(Ca); free(Cb); free(Ca_try); free(Cb_try);
        free(Pa); free(Pb); free(Fa); free(Fb);
        return 0.0;
    }
    memcpy(Ca, Ca_in, k2 * sizeof(double));
    memcpy(Cb, Cb_in, k2 * sizeof(double));

    double E_best = eval_sd_energy_from_C(ham, Ca, Cb, Pa, Pb, Fa, Fb);
    uint64_t rng = seed ? seed : UINT64_C(0x9E3779B97F4A7C15);

    int max_trials = (K <= 8) ? 24 : 36;
    for (int tr = 0; tr < max_trials; tr++) {
        uint64_t swan = 0;
        uint64_t k_flips = searu_zipf_jump(&rng, SEARU_CHAOS_DEFAULT_Q8, &swan);
        int n_rot = (int)(1U + (uint32_t)((k_flips - 1U) & 3U));

        memcpy(Ca_try, Ca, k2 * sizeof(double));
        memcpy(Cb_try, Cb, k2 * sizeof(double));

        int last_spin = 0, last_occ = 0, last_vir = 0;
        double last_ang = 0.0;
        for (int f = 0; f < n_rot; f++) {
            int spin = (int)(searu_rng_next(&rng) & 1U);
            int nocc = (spin == 0) ? na : nb;
            if (nocc <= 0 || nocc >= K) {
                spin = 1 - spin;
                nocc = (spin == 0) ? na : nb;
            }
            if (nocc <= 0 || nocc >= K) continue;
            int i_occ = (int)searu_Bounded_u32(searu_rng_next(&rng), (uint32_t)nocc);
            int a_vir = nocc + (int)searu_Bounded_u32(searu_rng_next(&rng), (uint32_t)(K - nocc));
            double ang = (swan ? 0.45 : 0.18) * ((searu_rng_next(&rng) & 1U) ? 1.0 : -1.0);
            if (restricted) {
                apply_jacobi_row_rot(K, Ca_try, i_occ, a_vir, ang);
                apply_jacobi_row_rot(K, Cb_try, i_occ, a_vir, ang);
            } else {
                apply_jacobi_row_rot(K, (spin == 0) ? Ca_try : Cb_try, i_occ, a_vir, ang);
            }
            last_spin = spin;
            last_occ = i_occ;
            last_vir = a_vir;
            last_ang = ang;
        }

        /* 2-Jet continuous angle refinement via FLOW flow_step on last rotation coordinate */
        int nocc_l = (last_spin == 0) ? na : nb;
        if (nocc_l > 0 && nocc_l < K) {
            double h = 1e-3;
            double E0 = eval_sd_energy_from_C(ham, Ca_try, Cb_try, Pa, Pb, Fa, Fb);
            double *C_p = (last_spin == 0) ? Ca_try : Cb_try;
            apply_jacobi_row_rot(K, C_p, last_occ, last_vir, h);
            if (restricted) apply_jacobi_row_rot(K, Cb_try, last_occ, last_vir, h);
            double Ep = eval_sd_energy_from_C(ham, Ca_try, Cb_try, Pa, Pb, Fa, Fb);
            apply_jacobi_row_rot(K, C_p, last_occ, last_vir, -2.0 * h);
            if (restricted) apply_jacobi_row_rot(K, Cb_try, last_occ, last_vir, -2.0 * h);
            double Em = eval_sd_energy_from_C(ham, Ca_try, Cb_try, Pa, Pb, Fa, Fb);
            apply_jacobi_row_rot(K, C_p, last_occ, last_vir, h);
            if (restricted) apply_jacobi_row_rot(K, Cb_try, last_occ, last_vir, h);

            double g1 = (Ep - Em) / (2.0 * h);
            double g2 = (Ep - 2.0 * E0 + Em) / (h * h);
            if (g2 > 1e-6) {
                uint64_t st[8] = {
                    (uint64_t)(int64_t)llround(fmax(-500.0, fmin(500.0, g1)) * 65536.0),
                    (uint64_t)(int64_t)llround(fmax(0.01, fmin(500.0, g2)) * 65536.0),
                    UINT64_C(0),
                    (uint64_t)(int64_t)llround(last_ang * 65536.0),
                    UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)
                };
                int32_t dt_max = (int32_t)llround(0.35 * 65536.0);
                uint64_t bd[8] = {
                    UINT64_C(0), UINT64_C(0), UINT64_C(0),
                    ((uint64_t)(uint32_t)(-2 * 65536) << 32U) | (uint64_t)(uint32_t)(2 * 65536),
                    ((uint64_t)(uint32_t)dt_max << 32U),
                    UINT64_C(0), UINT64_C(0), UINT64_C(0)
                };
                uint64_t out[8] = {0};
                flow_step(out, st, bd);
                double ang_new = (double)(int64_t)out[3] / 65536.0;
                double d_ang = ang_new - last_ang;
                apply_jacobi_row_rot(K, C_p, last_occ, last_vir, d_ang);
                if (restricted) apply_jacobi_row_rot(K, Cb_try, last_occ, last_vir, d_ang);
            }
        }

        double E_cand = eval_sd_energy_from_C(ham, Ca_try, Cb_try, Pa, Pb, Fa, Fb);
        if (E_cand < E_best - 1e-12) {
            E_best = E_cand;
            memcpy(Ca, Ca_try, k2 * sizeof(double));
            memcpy(Cb, Cb_try, k2 * sizeof(double));
        }
    }

    free(Ca); free(Cb); free(Ca_try); free(Cb_try);
    free(Pa); free(Pb); free(Fa); free(Fb);
    return E_best;
}

/* ============================================================================
 * Dense & Davidson Eigensolvers for Variational CI Subspace
 * ============================================================================ */

static void apply_H_dets(const ChemHam *ham, int ndet, const uint64_t *dets,
                         const double *diag, double s2_pen, const double *v, double *Hv) {
    int na = ham->na, nb = ham->nb;
    for (int i = 0; i < ndet; i++) Hv[i] = diag[i] * v[i];
    for (int i = 0; i < ndet; i++) {
        uint64_t dI = dets[i];
        double vi = v[i];
        double acc = 0.0;
        for (int j = 0; j < i; j++) {
            uint64_t diff = dI ^ dets[j];
            uint32_t nd = searu_popcnt64(diff);
            if (nd == 2U || nd == 4U) {
                double hij = slater_condon_mel(ham, dI, dets[j]);
                if (s2_pen != 0.0 && nd == 4U) {
                    hij += s2_pen * spin_sq_mel(na, nb, dI, dets[j]);
                }
                if (hij != 0.0) {
                    acc += hij * v[j];
                    Hv[j] += hij * vi;
                }
            }
        }
        Hv[i] += acc;
    }
}

static int solve_ci_subspace(const ChemHam *ham, int ndet, const uint64_t *dets,
                             int target_two_s, int need_excited,
                             double *e0_out, double *e1_out, double *psi0_out) {
    if (ndet <= 0) return -1;
    if (ndet == 1) {
        *e0_out = det_diag_energy(ham, dets[0]);
        if (e1_out) *e1_out = *e0_out;
        psi0_out[0] = 1.0;
        return 0;
    }

    if (ndet <= 512) {
        size_t nn = (size_t)ndet;
        double *H = (double *)calloc(nn * nn, sizeof(double));
        double *evals = (double *)malloc(nn * sizeof(double));
        if (!H || !evals) {
            free(H);
            free(evals);
            return -1;
        }
        for (int i = 0; i < ndet; i++) {
            H[(size_t)i * nn + (size_t)i] = det_diag_energy(ham, dets[i]);
            for (int j = 0; j < i; j++) {
                double hij = slater_condon_mel(ham, dets[i], dets[j]);
                H[(size_t)i * nn + (size_t)j] = hij;
                H[(size_t)j * nn + (size_t)i] = hij;
            }
        }

        if (target_two_s >= 0) {
            double *S2 = (double *)calloc(nn * nn, sizeof(double));
            double *ev_s2 = (double *)malloc(nn * sizeof(double));
            if (S2 && ev_s2) {
                for (int i = 0; i < ndet; i++) {
                    S2[(size_t)i * nn + (size_t)i] = spin_sq_mel(ham->na, ham->nb, dets[i], dets[i]);
                    for (int j = 0; j < i; j++) {
                        double s2ij = spin_sq_mel(ham->na, ham->nb, dets[i], dets[j]);
                        S2[(size_t)i * nn + (size_t)j] = s2ij;
                        S2[(size_t)j * nn + (size_t)i] = s2ij;
                    }
                }
                if (sym_eigh_small(ndet, S2, ev_s2) == 0) {
                    double s_target = 0.25 * (double)target_two_s * (double)(target_two_s + 2);
                    int n_s = 0;
                    for (int k = 0; k < ndet; k++) {
                        if (fabs(ev_s2[k] - s_target) < 1e-5) n_s++;
                    }
                    if (n_s > 0) {
                        double *Q_S = (double *)malloc((size_t)n_s * nn * sizeof(double));
                        double *T_m = (double *)malloc((size_t)n_s * nn * sizeof(double));
                        double *H_S = (double *)malloc((size_t)n_s * (size_t)n_s * sizeof(double));
                        double *ev_h = (double *)malloc((size_t)n_s * sizeof(double));
                        if (Q_S && T_m && H_S && ev_h) {
                            int r = 0;
                            for (int k = 0; k < ndet; k++) {
                                if (fabs(ev_s2[k] - s_target) < 1e-5) {
                                    memcpy(Q_S + (size_t)r * nn, S2 + (size_t)k * nn,
                                           nn * sizeof(double));
                                    r++;
                                }
                            }
                            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                        n_s, ndet, ndet, 1.0, Q_S, ndet, H, ndet, 0.0, T_m, ndet);
                            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                        n_s, n_s, ndet, 1.0, T_m, ndet, Q_S, ndet, 0.0, H_S, n_s);
                            if (sym_eigh_small(n_s, H_S, ev_h) == 0) {
                                *e0_out = ev_h[0];
                                if (e1_out) *e1_out = (n_s >= 2) ? ev_h[1] : ev_h[0];
                                cblas_dgemv(CblasRowMajor, CblasTrans, n_s, ndet, 1.0, Q_S, ndet,
                                            H_S, 1, 0.0, psi0_out, 1);
                                free(Q_S); free(T_m); free(H_S); free(ev_h);
                                free(S2); free(ev_s2);
                                free(H); free(evals);
                                return 0;
                            }
                        }
                        free(Q_S); free(T_m); free(H_S); free(ev_h);
                    }
                }
            }
            free(S2);
            free(ev_s2);
        }

        int rc = sym_eigh_small(ndet, H, evals);
        if (rc == 0) {
            *e0_out = evals[0];
            if (e1_out) *e1_out = (ndet >= 2) ? evals[1] : evals[0];
            memcpy(psi0_out, H, nn * sizeof(double)); /* Row 0 of V_rows = eigenvector 0 */
        }
        free(H);
        free(evals);
        return rc;
    }

    /* Davidson eigensolver for ndet > 512 */
    double s2_pen = (target_two_s == 0) ? 2.0 : 0.0;
    int max_sub = 24;
    if (max_sub > ndet) max_sub = ndet;
    int n_seed = (ndet < 64) ? ndet : 64;

    double *diag = (double *)malloc((size_t)ndet * sizeof(double));
    int *order = (int *)malloc((size_t)ndet * sizeof(int));
    double *V = (double *)calloc((size_t)max_sub * (size_t)ndet, sizeof(double));
    double *HV = (double *)calloc((size_t)max_sub * (size_t)ndet, sizeof(double));
    double *ritz_vec = (double *)malloc((size_t)ndet * sizeof(double));
    double *res_vec = (double *)malloc((size_t)ndet * sizeof(double));
    double *Hsub = (double *)calloc((size_t)max_sub * (size_t)max_sub, sizeof(double));
    double *ev_sub = (double *)calloc((size_t)max_sub, sizeof(double));

    if (!diag || !order || !V || !HV || !ritz_vec || !res_vec || !Hsub || !ev_sub) {
        free(diag); free(order); free(V); free(HV);
        free(ritz_vec); free(res_vec); free(Hsub); free(ev_sub);
        return -1;
    }

    for (int i = 0; i < ndet; i++) {
        diag[i] = det_diag_energy(ham, dets[i]);
        if (s2_pen != 0.0) {
            diag[i] += s2_pen * spin_sq_mel(ham->na, ham->nb, dets[i], dets[i]);
        }
        order[i] = i;
    }
    /* Selection sort lowest n_seed diagonal indices */
    for (int i = 0; i < n_seed; i++) {
        int best = i;
        for (int j = i + 1; j < ndet; j++) {
            if (diag[order[j]] < diag[order[best]]) best = j;
        }
        int t = order[i];
        order[i] = order[best];
        order[best] = t;
    }

    /* Seed initial vector from small dense diag of lowest n_seed determinants */
    double *H0 = (double *)calloc((size_t)n_seed * (size_t)n_seed, sizeof(double));
    double *ev0 = (double *)malloc((size_t)n_seed * sizeof(double));
    if (!H0 || !ev0) {
        free(H0);
        free(ev0);
        free(V); free(HV); free(ritz_vec); free(res_vec); free(Hsub); free(ev_sub);
        return -1;
    }
    for (int i = 0; i < n_seed; i++) {
        H0[(size_t)i * (size_t)n_seed + (size_t)i] = diag[order[i]];
        for (int j = 0; j < i; j++) {
            double hij = slater_condon_mel(ham, dets[order[i]], dets[order[j]]);
            if (s2_pen != 0.0 && searu_popcnt64(dets[order[i]] ^ dets[order[j]]) == 4U) {
                hij += s2_pen * spin_sq_mel(ham->na, ham->nb, dets[order[i]], dets[order[j]]);
            }
            H0[(size_t)i * (size_t)n_seed + (size_t)j] = hij;
            H0[(size_t)j * (size_t)n_seed + (size_t)i] = hij;
        }
    }
    sym_eigh_small(n_seed, H0, ev0);
    for (int i = 0; i < n_seed; i++) V[order[i]] = H0[i];
    free(H0);
    free(ev0);

    int m = 1;
    apply_H_dets(ham, ndet, dets, diag, s2_pen, V, HV);

    double E_cur = 0.0, E_exc = 0.0;
    for (int iter = 0; iter < 60; iter++) {
        memset(Hsub, 0, (size_t)m * (size_t)m * sizeof(double));
        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                double s = 0.0;
                const double *vi = V + (size_t)i * (size_t)ndet;
                const double *hvj = HV + (size_t)j * (size_t)ndet;
                for (int k = 0; k < ndet; k++) s += vi[k] * hvj[k];
                Hsub[(size_t)i * (size_t)m + (size_t)j] = s;
                Hsub[(size_t)j * (size_t)m + (size_t)i] = s;
            }
        }
        if (sym_eigh_small(m, Hsub, ev_sub) != 0) break;

        E_cur = ev_sub[0];
        E_exc = (m >= 2) ? ev_sub[1] : E_cur;

        /* Hsub row 0 holds eigenvector 0 coefficients in basis V[0..m-1] */
        memset(ritz_vec, 0, (size_t)ndet * sizeof(double));
        memset(res_vec, 0, (size_t)ndet * sizeof(double));
        for (int j = 0; j < m; j++) {
            double c = Hsub[j];
            const double *vj = V + (size_t)j * (size_t)ndet;
            const double *hvj = HV + (size_t)j * (size_t)ndet;
            for (int k = 0; k < ndet; k++) {
                ritz_vec[k] += c * vj[k];
                res_vec[k] += c * (hvj[k] - E_cur * vj[k]);
            }
        }

        double rnorm2 = 0.0;
        for (int k = 0; k < ndet; k++) rnorm2 += res_vec[k] * res_vec[k];
        if (sqrt(rnorm2) < 1e-9 && (!need_excited || m >= 2)) break;

        /* Preconditioned correction vector (targeting root 1 if root 0 converged & excited requested) */
        int target_root = (sqrt(rnorm2) < 1e-8 && need_excited && m >= 2) ? 1 : 0;
        if (target_root == 1) {
            double E1 = ev_sub[1];
            memset(res_vec, 0, (size_t)ndet * sizeof(double));
            for (int j = 0; j < m; j++) {
                double c = Hsub[(size_t)m + (size_t)j];
                const double *vj = V + (size_t)j * (size_t)ndet;
                const double *hvj = HV + (size_t)j * (size_t)ndet;
                for (int k = 0; k < ndet; k++) res_vec[k] += c * (hvj[k] - E1 * vj[k]);
            }
            double r1n2 = 0.0;
            for (int k = 0; k < ndet; k++) r1n2 += res_vec[k] * res_vec[k];
            if (sqrt(r1n2) < 1e-7) break;
        }
        double E_t = ev_sub[target_root];

        if (m >= max_sub) {
            /* Restart with top 2 Ritz vectors */
            int keep = (need_excited && m >= 2) ? 2 : 1;
            double *Vnew = (double *)calloc((size_t)keep * (size_t)ndet, sizeof(double));
            double *HVnew = (double *)calloc((size_t)keep * (size_t)ndet, sizeof(double));
            if (!Vnew || !HVnew) {
                free(Vnew);
                free(HVnew);
                break;
            }
            for (int r = 0; r < keep; r++) {
                for (int j = 0; j < m; j++) {
                    double c = Hsub[(size_t)r * (size_t)m + (size_t)j];
                    const double *vj = V + (size_t)j * (size_t)ndet;
                    const double *hvj = HV + (size_t)j * (size_t)ndet;
                    double *vr = Vnew + (size_t)r * (size_t)ndet;
                    double *hvr = HVnew + (size_t)r * (size_t)ndet;
                    for (int k = 0; k < ndet; k++) {
                        vr[k] += c * vj[k];
                        hvr[k] += c * hvj[k];
                    }
                }
            }
            memcpy(V, Vnew, (size_t)keep * (size_t)ndet * sizeof(double));
            memcpy(HV, HVnew, (size_t)keep * (size_t)ndet * sizeof(double));
            free(Vnew);
            free(HVnew);
            m = keep;
        }

        double *w = V + (size_t)m * (size_t)ndet;
        for (int k = 0; k < ndet; k++) {
            double den = E_t - diag[k];
            if (fabs(den) < 1e-3) den = (den >= 0.0) ? 1e-3 : -1e-3;
            w[k] = res_vec[k] / den;
        }
        /* Two-pass modified Gram-Schmidt orthogonalization */
        for (int pass = 0; pass < 2; pass++) {
            for (int j = 0; j < m; j++) {
                const double *vj = V + (size_t)j * (size_t)ndet;
                double dot = 0.0;
                for (int k = 0; k < ndet; k++) dot += vj[k] * w[k];
                for (int k = 0; k < ndet; k++) w[k] -= dot * vj[k];
            }
        }
        double wn2 = 0.0;
        for (int k = 0; k < ndet; k++) wn2 += w[k] * w[k];
        double wn = sqrt(wn2);
        if (wn < 1e-12) break;
        double inv = 1.0 / wn;
        for (int k = 0; k < ndet; k++) w[k] *= inv;
        apply_H_dets(ham, ndet, dets, diag, s2_pen, w, HV + (size_t)m * (size_t)ndet);
        m++;
    }

    memcpy(psi0_out, ritz_vec, (size_t)ndet * sizeof(double));
    if (s2_pen != 0.0) {
        double e_pure = 0.0;
        char ebuf[64] = {0};
        if (chem_var_eval_witness(ham, ndet, dets, psi0_out, &e_pure, ebuf, sizeof(ebuf)) == 0) {
            E_cur = e_pure;
        }
    }
    *e0_out = E_cur;
    if (e1_out) *e1_out = E_exc;

    free(diag); free(order); free(V); free(HV);
    free(ritz_vec); free(res_vec); free(Hsub); free(ev_sub);
    return 0;
}

/* ============================================================================
 * Determinant Enumeration (Exact FCI) & Deterministic Heat-Bath SHCI
 * ============================================================================ */

static void gen_spin_combs(int K, int n, int start, int shift_bit, uint64_t cur,
                           uint64_t *out, int *cnt) {
    if (n == 0) {
        out[(*cnt)++] = cur;
        return;
    }
    for (int i = start; i <= K - n; i++) {
        gen_spin_combs(K, n - 1, i + 1, shift_bit,
                       cur | (UINT64_C(1) << (unsigned)(2 * i + shift_bit)), out, cnt);
    }
}

typedef struct {
    int r, s;
    double abs_h;
} HBEntry;

typedef struct {
    uint64_t *keys;
    uint8_t *used;
    int cap_mask;
} DetSet;

static int detset_init(DetSet *ds, int min_cap) {
    int cap = 64;
    while (cap < min_cap * 2) cap *= 2;
    ds->keys = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    ds->used = (uint8_t *)calloc((size_t)cap, sizeof(uint8_t));
    if (!ds->keys || !ds->used) {
        free(ds->keys);
        free(ds->used);
        return -1;
    }
    ds->cap_mask = cap - 1;
    return 0;
}

static void detset_free(DetSet *ds) {
    free(ds->keys);
    free(ds->used);
    memset(ds, 0, sizeof(*ds));
}

static inline int detset_insert(DetSet *ds, uint64_t d) {
    uint64_t h = d * UINT64_C(0x9E3779B97F4A7C15);
    h ^= h >> 30U;
    int idx = (int)(h & (uint64_t)ds->cap_mask);
    while (ds->used[idx]) {
        if (ds->keys[idx] == d) return 0; /* Already present */
        idx = (idx + 1) & ds->cap_mask;
    }
    ds->used[idx] = 1;
    ds->keys[idx] = d;
    return 1;
}

static int cmp_hb_desc(const void *a, const void *b) {
    const HBEntry *x = (const HBEntry *)a;
    const HBEntry *y = (const HBEntry *)b;
    if (x->abs_h < y->abs_h) return 1;
    if (x->abs_h > y->abs_h) return -1;
    return 0;
}

static void add_det_and_partner(DetSet *ds, uint64_t *dets, int *ndet, int max_dets,
                                uint64_t d, int K, int sym_flip) {
    if (*ndet >= max_dets) return;
    if (detset_insert(ds, d)) {
        dets[(*ndet)++] = d;
    }
    if (sym_flip && *ndet < max_dets) {
        uint64_t df = spin_flip_det(d, K);
        if (detset_insert(ds, df)) {
            dets[(*ndet)++] = df;
        }
    }
}

static void add_det_spin_closure(DetSet *ds, uint64_t *dets, int *ndet, int max_dets,
                                 uint64_t d, int K, int sym_flip, int do_closure) {
    if (*ndet >= max_dets) return;
    if (do_closure) {
        static const uint64_t EVEN = UINT64_C(0x5555555555555555);
        uint64_t a = d & EVEN;
        uint64_t b = (d >> 1U) & EVEN;
        uint64_t docc = a & b;
        uint64_t open_m = a ^ b;
        int open_orbs[16];
        int n_open = 0, n_oa = 0;
        uint64_t tmp = open_m;
        while (tmp && n_open < 16) {
            int p = (int)(searu_ctz64(tmp) >> 1U);
            open_orbs[n_open++] = p;
            if ((a >> (unsigned)(2 * p)) & 1U) n_oa++;
            tmp &= tmp - UINT64_C(1);
        }
        if (n_open == 0) {
            if (detset_insert(ds, d)) dets[(*ndet)++] = d;
            return;
        }
        if (n_open <= 6) {
            uint64_t base = 0;
            for (int p = 0; p < K; p++) {
                if ((docc >> (unsigned)(2 * p)) & 1U) {
                    base |= (UINT64_C(3) << (unsigned)(2 * p));
                }
            }
            uint64_t fam[24];
            int fam_cnt = 0;
            uint64_t lim = UINT64_C(1) << (unsigned)n_open;
            for (uint64_t mask = 0; mask < lim; mask++) {
                if ((int)searu_popcnt64(mask) != n_oa) continue;
                uint64_t df = base;
                for (int i = 0; i < n_open; i++) {
                    int p = open_orbs[i];
                    if ((mask >> (unsigned)i) & 1U) df |= (UINT64_C(1) << (unsigned)(2 * p));
                    else df |= (UINT64_C(1) << (unsigned)(2 * p + 1));
                }
                if (fam_cnt < 24) fam[fam_cnt++] = df;
            }
            if (*ndet + fam_cnt <= max_dets) {
                for (int i = 0; i < fam_cnt; i++) {
                    if (detset_insert(ds, fam[i])) dets[(*ndet)++] = fam[i];
                }
            }
            return;
        }
        if (n_open > 6) return;
    }
    add_det_and_partner(ds, dets, ndet, max_dets, d, K, sym_flip);
}

static int run_heatbath_shci(const ChemHam *ham, const ChemVarOpts *opts,
                             double *e0_out, int *ndet_out, uint64_t **dets_out,
                             double **psi0_out) {
    int K = ham->K, na = ham->na, nb = ham->nb;
    int max_dets = (opts->max_dets > 16) ? opts->max_dets : 1024;
    int sym_flip = (na == nb) ? 1 : 0;
    int do_closure = (opts->target_two_s >= 0 && max_dets <= 1024) ? 1 : 0;

    uint64_t *dets = (uint64_t *)malloc((size_t)max_dets * sizeof(uint64_t));
    DetSet ds;
    if (!dets || detset_init(&ds, max_dets * 2) != 0) {
        free(dets);
        return -1;
    }

    int ndet = 0;
    uint64_t d_hf = canonical_hf_det(na, nb);
    add_det_spin_closure(&ds, dets, &ndet, max_dets, d_hf, K, sym_flip, do_closure);

    /* When targeting S=0 on closed-shell reference, seed seniority-0 pair doubles first */
    if (opts->target_two_s == 0 && na == nb) {
        for (int i = 0; i < na && ndet < max_dets; i++) {
            for (int a = na; a < K && ndet < max_dets; a++) {
                uint64_t d_pair = d_hf ^ (UINT64_C(3) << (unsigned)(2 * i))
                                       ^ (UINT64_C(3) << (unsigned)(2 * a));
                add_det_spin_closure(&ds, dets, &ndet, max_dets, d_pair, K, sym_flip, do_closure);
            }
        }
    }

    /* Build Heat-Bath sorted double excitation tables for same-spin (p<q) & opp-spin (p,q) */
    int naa = K * (K - 1) / 2;
    int nab = K * K;
    HBEntry *hb_ss = (naa > 0) ? (HBEntry *)malloc((size_t)naa * (size_t)naa * sizeof(HBEntry)) : NULL;
    int *hb_ss_len = (naa > 0) ? (int *)calloc((size_t)naa, sizeof(int)) : NULL;
    HBEntry *hb_os = (HBEntry *)malloc((size_t)nab * (size_t)nab * sizeof(HBEntry));
    int *hb_os_len = (int *)calloc((size_t)nab, sizeof(int));
    if ((naa > 0 && (!hb_ss || !hb_ss_len)) || !hb_os || !hb_os_len) {
        free(hb_ss); free(hb_ss_len); free(hb_os); free(hb_os_len);
        free(dets); detset_free(&ds);
        return -1;
    }

    int pair_ij[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int np = 0;
    for (int p = 0; p < K; p++) {
        for (int q = p + 1; q < K; q++) {
            pair_ij[np][0] = p;
            pair_ij[np][1] = q;
            np++;
        }
    }

    for (int pq = 0; pq < naa; pq++) {
        int p = pair_ij[pq][0], q = pair_ij[pq][1];
        HBEntry *row = hb_ss + (size_t)pq * (size_t)naa;
        int cnt = 0;
        for (int rs = 0; rs < naa; rs++) {
            if (rs == pq) continue;
            int r = pair_ij[rs][0], s = pair_ij[rs][1];
            double val = fabs(chem_ham_eri(ham, p, r, q, s) - chem_ham_eri(ham, p, s, q, r));
            if (val > 1e-8) {
                row[cnt].r = r;
                row[cnt].s = s;
                row[cnt].abs_h = val;
                cnt++;
            }
        }
        qsort(row, (size_t)cnt, sizeof(HBEntry), cmp_hb_desc);
        hb_ss_len[pq] = cnt;
    }

    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
            int pq = p * K + q;
            HBEntry *row = hb_os + (size_t)pq * (size_t)nab;
            int cnt = 0;
            for (int r = 0; r < K; r++) {
                for (int s = 0; s < K; s++) {
                    if (r == p && s == q) continue;
                    double val = fabs(chem_ham_eri(ham, p, r, q, s));
                    if (val > 1e-8) {
                        row[cnt].r = r;
                        row[cnt].s = s;
                        row[cnt].abs_h = val;
                        cnt++;
                    }
                }
            }
            qsort(row, (size_t)cnt, sizeof(HBEntry), cmp_hb_desc);
            hb_os_len[pq] = cnt;
        }
    }

    /* Seed top-coupled singles & doubles from HF det + SEARU multi-reference proposals */
    for (int spin = 0; spin < 2; spin++) {
        int nocc = (spin == 0) ? na : nb;
        for (int i = 0; i < nocc && ndet < max_dets; i++) {
            for (int a = nocc; a < K && ndet < max_dets; a++) {
                uint64_t d1 = d_hf ^ (UINT64_C(1) << (unsigned)(2 * i + spin))
                                   ^ (UINT64_C(1) << (unsigned)(2 * a + spin));
                if (fabs(slater_condon_mel(ham, d_hf, d1)) > 1e-6) {
                    add_det_spin_closure(&ds, dets, &ndet, max_dets, d1, K, sym_flip, do_closure);
                }
            }
        }
    }
    /* Seed top-64 opp-spin & same-spin doubles from HF det */
    double e_hf = det_diag_energy(ham, d_hf);
    if (opts->run_searu) {
        uint64_t rng = opts->seed;
        for (int t = 0; t < 256 && ndet < max_dets / 4; t++) {
            uint64_t d = d_hf;
            uint64_t swan = 0;
            uint64_t k_jump = searu_zipf_jump(&rng, SEARU_CHAOS_DEFAULT_Q8, &swan);
            int nflips = (int)(1U + (uint32_t)((k_jump - 1U) & 3U));
            for (int f = 0; f < nflips; f++) {
                int sp = (int)(searu_rng_next(&rng) & 1U);
                int nocc = (sp == 0) ? na : nb;
                if (nocc <= 0 || nocc >= K) continue;
                int p = (int)searu_Bounded_u32(searu_rng_next(&rng), (uint32_t)K);
                int q = (int)searu_Bounded_u32(searu_rng_next(&rng), (uint32_t)K);
                uint64_t bp = UINT64_C(1) << (unsigned)(2 * p + sp);
                uint64_t bq = UINT64_C(1) << (unsigned)(2 * q + sp);
                if (((d & bp) != 0) && ((d & bq) == 0)) {
                    d ^= bp | bq;
                }
            }
            if (d != d_hf && det_diag_energy(ham, d) < e_hf + 4.0) {
                add_det_spin_closure(&ds, dets, &ndet, max_dets, d, K, sym_flip, do_closure);
            }
        }
    }

    double *psi = (double *)calloc((size_t)max_dets, sizeof(double));
    if (!psi) {
        free(hb_ss); free(hb_ss_len); free(hb_os); free(hb_os_len);
        free(dets); detset_free(&ds);
        return -1;
    }
    double E_cur = e_hf;

    /* Precompute pair index table for fast same-spin lookup */
    int pair_idx_tbl[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];
    memset(pair_idx_tbl, -1, sizeof(pair_idx_tbl));
    for (int pq = 0; pq < naa; pq++) {
        pair_idx_tbl[pair_ij[pq][0]][pair_ij[pq][1]] = pq;
        pair_idx_tbl[pair_ij[pq][1]][pair_ij[pq][0]] = pq;
    }

    for (int st = 0; st < opts->n_eps1 && ndet < max_dets; st++) {
        double eps1 = opts->eps1_schedule[st];
        solve_ci_subspace(ham, ndet, dets, opts->target_two_s, 0, &E_cur, NULL, psi);
        int ndet_cur = ndet;

        for (int idet = 0; idet < ndet_cur && ndet < max_dets; idet++) {
            double ac = fabs(psi[idet]);
            if (ac < 1e-4) continue;
            uint64_t dI = dets[idet];

            /* Extract occupied alpha and beta spatial orbitals */
            int occ_a[CHEM_MAX_ORBITALS], occ_b[CHEM_MAX_ORBITALS];
            int ca = 0, cb = 0;
            for (int p = 0; p < K; p++) {
                if ((dI >> (unsigned)(2 * p)) & 1U) occ_a[ca++] = p;
                if ((dI >> (unsigned)(2 * p + 1)) & 1U) occ_b[cb++] = p;
            }

            /* 1. Single excitations */
            for (int ii = 0; ii < ca && ndet < max_dets; ii++) {
                int p = occ_a[ii];
                for (int r = 0; r < K && ndet < max_dets; r++) {
                    if ((dI >> (unsigned)(2 * r)) & 1U) continue;
                    uint64_t d1 = dI ^ (UINT64_C(1) << (unsigned)(2 * p))
                                     ^ (UINT64_C(1) << (unsigned)(2 * r));
                    if (ac * fabs(slater_condon_mel(ham, dI, d1)) >= eps1) {
                        add_det_spin_closure(&ds, dets, &ndet, max_dets, d1, K, sym_flip,
                                             do_closure);
                    }
                }
            }
            for (int ii = 0; ii < cb && ndet < max_dets; ii++) {
                int p = occ_b[ii];
                for (int r = 0; r < K && ndet < max_dets; r++) {
                    if ((dI >> (unsigned)(2 * r + 1)) & 1U) continue;
                    uint64_t d1 = dI ^ (UINT64_C(1) << (unsigned)(2 * p + 1))
                                     ^ (UINT64_C(1) << (unsigned)(2 * r + 1));
                    if (ac * fabs(slater_condon_mel(ham, dI, d1)) >= eps1) {
                        add_det_spin_closure(&ds, dets, &ndet, max_dets, d1, K, sym_flip,
                                             do_closure);
                    }
                }
            }

            /* 2. Opposite-spin Heat-Bath doubles */
            for (int ia = 0; ia < ca && ndet < max_dets; ia++) {
                int p = occ_a[ia];
                for (int ib = 0; ib < cb && ndet < max_dets; ib++) {
                    int q = occ_b[ib];
                    int pq = p * K + q;
                    const HBEntry *row = hb_os + (size_t)pq * (size_t)nab;
                    int len = hb_os_len[pq];
                    for (int k = 0; k < len && ndet < max_dets; k++) {
                        if (ac * row[k].abs_h < eps1) break;
                        int r = row[k].r, s = row[k].s;
                        if (((dI >> (unsigned)(2 * r)) & 1U) ||
                            ((dI >> (unsigned)(2 * s + 1)) & 1U)) {
                            continue;
                        }
                        uint64_t d2 = dI ^ (UINT64_C(1) << (unsigned)(2 * p))
                                         ^ (UINT64_C(1) << (unsigned)(2 * q + 1))
                                         ^ (UINT64_C(1) << (unsigned)(2 * r))
                                         ^ (UINT64_C(1) << (unsigned)(2 * s + 1));
                        add_det_spin_closure(&ds, dets, &ndet, max_dets, d2, K, sym_flip,
                                             do_closure);
                    }
                }
            }

            /* 3. Same-spin Heat-Bath doubles (alpha-alpha and beta-beta) */
            for (int sp = 0; sp < 2 && ndet < max_dets; sp++) {
                const int *occ = (sp == 0) ? occ_a : occ_b;
                int nsp = (sp == 0) ? ca : cb;
                for (int i1 = 0; i1 < nsp && ndet < max_dets; i1++) {
                    for (int i2 = i1 + 1; i2 < nsp && ndet < max_dets; i2++) {
                        int p = occ[i1], q = occ[i2];
                        int pq = pair_idx_tbl[p][q];
                        if (pq < 0) continue;
                        const HBEntry *row = hb_ss + (size_t)pq * (size_t)naa;
                        int len = hb_ss_len[pq];
                        for (int k = 0; k < len && ndet < max_dets; k++) {
                            if (ac * row[k].abs_h < eps1) break;
                            int r = row[k].r, s = row[k].s;
                            if (((dI >> (unsigned)(2 * r + sp)) & 1U) ||
                                ((dI >> (unsigned)(2 * s + sp)) & 1U)) {
                                continue;
                            }
                            uint64_t d2 = dI ^ (UINT64_C(1) << (unsigned)(2 * p + sp))
                                             ^ (UINT64_C(1) << (unsigned)(2 * q + sp))
                                             ^ (UINT64_C(1) << (unsigned)(2 * r + sp))
                                             ^ (UINT64_C(1) << (unsigned)(2 * s + sp));
                            add_det_spin_closure(&ds, dets, &ndet, max_dets, d2, K, sym_flip,
                                                 do_closure);
                        }
                    }
                }
            }
        }
    }

    solve_ci_subspace(ham, ndet, dets, opts->target_two_s, 0, &E_cur, NULL, psi);

    free(hb_ss); free(hb_ss_len);
    free(hb_os); free(hb_os_len);
    detset_free(&ds);

    *e0_out = E_cur;
    *ndet_out = ndet;
    *dets_out = dets;
    *psi0_out = psi;
    return 0;
}

/* Exact full-space variance of single determinant d_ref: sum_{J != d_ref} |<J|H|d_ref>|^2 */
static double single_det_full_variance(const ChemHam *ham, uint64_t d_ref) {
    int K = ham->K;
    int occ_a[CHEM_MAX_ORBITALS], occ_b[CHEM_MAX_ORBITALS];
    int vir_a[CHEM_MAX_ORBITALS], vir_b[CHEM_MAX_ORBITALS];
    int ca = 0, cb = 0, va = 0, vb = 0;
    for (int p = 0; p < K; p++) {
        if ((d_ref >> (unsigned)(2 * p)) & 1U) occ_a[ca++] = p;
        else vir_a[va++] = p;
        if ((d_ref >> (unsigned)(2 * p + 1)) & 1U) occ_b[cb++] = p;
        else vir_b[vb++] = p;
    }

    double var = 0.0;
    /* Singles (alpha & beta) */
    for (int ii = 0; ii < ca; ii++) {
        int i = occ_a[ii];
        for (int aa = 0; aa < va; aa++) {
            int a = vir_a[aa];
            uint64_t d1 = d_ref ^ (UINT64_C(1) << (unsigned)(2 * i))
                                ^ (UINT64_C(1) << (unsigned)(2 * a));
            double h = slater_condon_mel(ham, d_ref, d1);
            var += h * h;
        }
    }
    for (int ii = 0; ii < cb; ii++) {
        int i = occ_b[ii];
        for (int aa = 0; aa < vb; aa++) {
            int a = vir_b[aa];
            uint64_t d1 = d_ref ^ (UINT64_C(1) << (unsigned)(2 * i + 1))
                                ^ (UINT64_C(1) << (unsigned)(2 * a + 1));
            double h = slater_condon_mel(ham, d_ref, d1);
            var += h * h;
        }
    }
    /* Opp-spin doubles */
    for (int ii = 0; ii < ca; ii++) {
        int i = occ_a[ii];
        for (int jj = 0; jj < cb; jj++) {
            int j = occ_b[jj];
            for (int aa = 0; aa < va; aa++) {
                int a = vir_a[aa];
                for (int bb = 0; bb < vb; bb++) {
                    int b = vir_b[bb];
                    double h = chem_ham_eri(ham, i, a, j, b);
                    var += h * h;
                }
            }
        }
    }
    /* Same-spin doubles */
    for (int i1 = 0; i1 < ca; i1++) {
        for (int i2 = i1 + 1; i2 < ca; i2++) {
            int i = occ_a[i1], j = occ_a[i2];
            for (int a1 = 0; a1 < va; a1++) {
                for (int a2 = a1 + 1; a2 < va; a2++) {
                    int a = vir_a[a1], b = vir_a[a2];
                    double h = chem_ham_eri(ham, i, a, j, b) - chem_ham_eri(ham, i, b, j, a);
                    var += h * h;
                }
            }
        }
    }
    for (int i1 = 0; i1 < cb; i1++) {
        for (int i2 = i1 + 1; i2 < cb; i2++) {
            int i = occ_b[i1], j = occ_b[i2];
            for (int a1 = 0; a1 < vb; a1++) {
                for (int a2 = a1 + 1; a2 < vb; a2++) {
                    int a = vir_b[a1], b = vir_b[a2];
                    double h = chem_ham_eri(ham, i, a, j, b) - chem_ham_eri(ham, i, b, j, a);
                    var += h * h;
                }
            }
        }
    }
    return var;
}

/* ============================================================================
 * Public Variational Funnel & Independent Witness Evaluator
 * ============================================================================ */

int chem_var_solve(const ChemHam *ham, const ChemVarOpts *opts_in, ChemVarResult *out,
                   char *err, size_t err_sz) {
    if (!ham || !out) {
        shci_set_err(err, err_sz, "null argument to chem_var_solve");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    double t0 = shci_now_sec();
    ChemVarOpts opts = opts_in ? *opts_in : chem_var_opts_default();

    int K = ham->K, na = ham->na, nb = ham->nb;
    uint64_t d_hf0 = canonical_hf_det(na, nb);
    out->E_hf0 = det_diag_energy(ham, d_hf0);

    size_t k2 = (size_t)K * (size_t)K;
    double *Ca = (double *)malloc(k2 * sizeof(double));
    double *Cb = (double *)malloc(k2 * sizeof(double));
    if (!Ca || !Cb) {
        free(Ca);
        free(Cb);
        shci_set_err(err, err_sz, "out of memory in chem_var_solve");
        return -1;
    }

    int restricted = (opts.target_two_s == 0 && na == nb) ? 1 : 0;
    out->E_scf = run_scf_diis(ham, restricted, Ca, Cb, &out->diis_iters, &out->comm_norm);
    out->E_searu_rot = opts.run_searu ? searu_2jet_orbital_rot(ham, restricted, Ca, Cb, opts.seed)
                                      : out->E_scf;
    free(Ca);
    free(Cb);

    uint64_t na_comb = binom_sat(K, na);
    uint64_t nb_comb = binom_sat(K, nb);
    uint64_t ndet_space = mul_sat_u64(na_comb, nb_comb);
    out->ndet_space = ndet_space;

    if (ndet_space <= 4096U) {
        out->is_exact_fci = 1;
        int n_a = (int)na_comb, n_b = (int)nb_comb;
        int ndet = n_a * n_b;
        uint64_t *ac = (uint64_t *)malloc((size_t)n_a * sizeof(uint64_t));
        uint64_t *bc = (uint64_t *)malloc((size_t)n_b * sizeof(uint64_t));
        uint64_t *dets = (uint64_t *)malloc((size_t)ndet * sizeof(uint64_t));
        double *psi0 = (double *)calloc((size_t)ndet, sizeof(double));
        if (!ac || !bc || !dets || !psi0) {
            free(ac); free(bc); free(dets); free(psi0);
            shci_set_err(err, err_sz, "out of memory in exact FCI");
            return -1;
        }
        int ca = 0, cb = 0;
        gen_spin_combs(K, na, 0, 0, UINT64_C(0), ac, &ca);
        gen_spin_combs(K, nb, 0, 1, UINT64_C(0), bc, &cb);
        int idx = 0;
        for (int i = 0; i < ca; i++) {
            for (int j = 0; j < cb; j++) dets[idx++] = ac[i] | bc[j];
        }
        free(ac);
        free(bc);

        double e0 = 0.0, e1 = 0.0;
        int want_exc = (ndet >= 2) ? 1 : 0;
        if (solve_ci_subspace(ham, ndet, dets, opts.target_two_s, want_exc, &e0, &e1, psi0) != 0) {
            free(dets);
            free(psi0);
            shci_set_err(err, err_sz, "FCI eigensolver failed");
            return -1;
        }
        out->E_ci = e0;
        out->E_ci_excited = e1;
        out->has_excited = (ndet >= 2) ? 1 : 0;
        out->ndet_var = ndet;
        out->dets = dets;
        out->psi0 = psi0;
        out->var_H_best = 0.0;
    } else {
        out->is_exact_fci = 0;
        if (run_heatbath_shci(ham, &opts, &out->E_ci, &out->ndet_var, &out->dets, &out->psi0) !=
            0) {
            shci_set_err(err, err_sz, "Heat-Bath SHCI solver failed");
            return -1;
        }
    }

    double best_ub = out->E_hf0;
    if (out->E_scf < best_ub) best_ub = out->E_scf;
    if (out->E_searu_rot < best_ub) best_ub = out->E_searu_rot;
    if (out->E_ci < best_ub) best_ub = out->E_ci;
    out->E_ub = best_ub;

    out->var_H_hf = single_det_full_variance(ham, d_hf0);
    out->E_weinstein = out->E_hf0 - sqrt(fmax(0.0, out->var_H_hf));
    out->has_weinstein = 1;
    if (out->has_excited && out->E_ci_excited > out->E_hf0 + 1e-12) {
        out->E_temple = out->E_hf0 - out->var_H_hf / (out->E_ci_excited - out->E_hf0);
        out->has_temple = 1;
    }

    out->elapsed_sec = shci_now_sec() - t0;
    return 0;
}

int chem_var_eval_witness(const ChemHam *ham, int ndet, const uint64_t *dets, const double *coeffs,
                          double *energy_out, char *err, size_t err_sz) {
    if (!ham || ndet <= 0 || !dets || !coeffs || !energy_out) {
        shci_set_err(err, err_sz, "invalid argument to chem_var_eval_witness");
        return -1;
    }
    uint64_t mask_all = (ham->K == 32) ? UINT64_MAX : ((UINT64_C(1) << (unsigned)(2 * ham->K)) - 1ULL);
    uint64_t mask_a = 0, mask_b = 0;
    for (int p = 0; p < ham->K; p++) {
        mask_a |= (UINT64_C(1) << (unsigned)(2 * p));
        mask_b |= (UINT64_C(1) << (unsigned)(2 * p + 1));
    }

    double norm2 = 0.0;
    for (int i = 0; i < ndet; i++) {
        uint64_t d = dets[i];
        if ((d & ~mask_all) != 0ULL) {
            shci_set_err(err, err_sz, "witness determinant %d out of orbital range", i);
            return -1;
        }
        if ((int)searu_popcnt64(d & mask_a) != ham->na ||
            (int)searu_popcnt64(d & mask_b) != ham->nb) {
            shci_set_err(err, err_sz, "witness determinant %d wrong (na, nb)", i);
            return -1;
        }
        norm2 += coeffs[i] * coeffs[i];
    }
    if (norm2 <= 1e-30) {
        shci_set_err(err, err_sz, "witness wavefunction has zero norm");
        return -1;
    }

    double num = 0.0;
    for (int i = 0; i < ndet; i++) {
        double ci = coeffs[i];
        if (fabs(ci) < 1e-16) continue;
        num += ci * ci * det_diag_energy(ham, dets[i]);
        for (int j = 0; j < i; j++) {
            double cj = coeffs[j];
            if (fabs(cj) < 1e-16) continue;
            uint32_t nd = searu_popcnt64(dets[i] ^ dets[j]);
            if (nd == 2U || nd == 4U) {
                num += 2.0 * ci * cj * slater_condon_mel(ham, dets[i], dets[j]);
            }
        }
    }
    *energy_out = num / norm2;
    return 0;
}

int chem_var_eval_spin_sq(int na, int nb, int ndet, const uint64_t *dets, const double *coeffs,
                          double *s2_out) {
    if (ndet <= 0 || !dets || !coeffs || !s2_out) return -1;
    double n2 = 0.0, num = 0.0;
    for (int i = 0; i < ndet; i++) {
        double ci = coeffs[i];
        n2 += ci * ci;
        if (fabs(ci) < 1e-16) continue;
        num += ci * ci * spin_sq_mel(na, nb, dets[i], dets[i]);
        for (int j = 0; j < i; j++) {
            double cj = coeffs[j];
            if (fabs(cj) < 1e-16) continue;
            double v = spin_sq_mel(na, nb, dets[i], dets[j]);
            if (v != 0.0) num += 2.0 * ci * cj * v;
        }
    }
    if (n2 <= 1e-30) return -1;
    *s2_out = num / n2;
    return 0;
}

int chem_var_compute_rdm1_diag(const ChemHam *ham, int ndet, const uint64_t *dets,
                               const double *coeffs, double *occ_out) {
    if (!ham || ndet <= 0 || !dets || !coeffs || !occ_out) return -1;
    int K = ham->K;
    memset(occ_out, 0, (size_t)K * sizeof(double));
    double n2 = 0.0;
    for (int i = 0; i < ndet; i++) {
        double w = coeffs[i] * coeffs[i];
        n2 += w;
        if (w == 0.0) continue;
        uint64_t d = dets[i];
        for (int p = 0; p < K; p++) {
            double o = (double)(((d >> (unsigned)(2 * p)) & UINT64_C(1)) +
                                ((d >> (unsigned)(2 * p + 1)) & UINT64_C(1)));
            occ_out[p] += w * o;
        }
    }
    if (n2 <= 1e-30) return -1;
    for (int p = 0; p < K; p++) occ_out[p] /= n2;
    return 0;
}

static const char *shci_json_find(const char *s, const char *e, const char *key) {
    size_t klen = strlen(key);
    const char *p = s;
    while (p < e) {
        const char *q = strchr(p, '"');
        if (!q || q >= e) return NULL;
        const char *q2 = q + 1;
        while (q2 < e && *q2 != '"') {
            if (*q2 == '\\' && q2 + 1 < e) q2 += 2;
            else q2++;
        }
        if (q2 >= e) return NULL;
        if ((size_t)(q2 - (q + 1)) == klen && memcmp(q + 1, key, klen) == 0) {
            const char *c = q2 + 1;
            while (c < e && isspace((unsigned char)*c)) c++;
            if (c < e && *c == ':') {
                c++;
                while (c < e && isspace((unsigned char)*c)) c++;
                return c;
            }
        }
        p = q2 + 1;
    }
    return NULL;
}

int chem_var_load_witness_json(const char *path, const ChemHam *ham, int target_two_s,
                               ChemVarResult *out, char *err, size_t err_sz) {
    if (!path || !ham || !out) {
        shci_set_err(err, err_sz, "null argument to chem_var_load_witness_json");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        shci_set_err(err, err_sz, "cannot open witness JSON: %s", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long fsz = ftell(fp);
    if (fsz <= 0) {
        fclose(fp);
        shci_set_err(err, err_sz, "empty witness file: %s", path);
        return -1;
    }
    fseek(fp, 0, SEEK_SET);
    char *txt = (char *)malloc((size_t)fsz + 1U);
    if (!txt) {
        fclose(fp);
        shci_set_err(err, err_sz, "out of memory reading witness");
        return -1;
    }
    size_t nr = fread(txt, 1, (size_t)fsz, fp);
    fclose(fp);
    txt[nr] = '\0';

    const char *s = txt, *e = txt + nr;
    const char *vs = shci_json_find(s, e, "variational");
    if (vs && *vs == '{') s = vs;

    const char *da = shci_json_find(s, e, "dets");
    const char *ca = shci_json_find(s, e, "coeffs");
    if (!ca) ca = shci_json_find(s, e, "psi0");
    if (!da || *da != '[' || !ca || *ca != '[') {
        free(txt);
        shci_set_err(err, err_sz, "witness JSON missing dets or coeffs/psi0 arrays");
        return -1;
    }

    /* Count elements in dets */
    int cap = 16, ndet = 0;
    uint64_t *dets = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
    double *psi0 = (double *)malloc((size_t)cap * sizeof(double));
    if (!dets || !psi0) {
        free(dets); free(psi0); free(txt);
        return -1;
    }

    const char *p = da + 1;
    while (p < e) {
        while (p < e && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (p < e && *p == ']') break;
        uint64_t val = 0;
        if (*p == '"') {
            p++;
            char *ep = NULL;
            val = (uint64_t)strtoull(p, &ep, 0);
            if (ep == p || *ep != '"') { ndet = -1; break; }
            p = ep + 1;
        } else {
            char *ep = NULL;
            val = (uint64_t)strtoull(p, &ep, 0);
            if (ep == p) { ndet = -1; break; }
            p = ep;
        }
        if (ndet >= cap) {
            cap *= 2;
            uint64_t *d2 = (uint64_t *)realloc(dets, (size_t)cap * sizeof(uint64_t));
            double *c2 = (double *)realloc(psi0, (size_t)cap * sizeof(double));
            if (!d2 || !c2) { free(d2 ? d2 : dets); free(c2 ? c2 : psi0); free(txt); return -1; }
            dets = d2; psi0 = c2;
        }
        dets[ndet++] = val;
    }
    if (ndet <= 0) {
        free(dets); free(psi0); free(txt);
        shci_set_err(err, err_sz, "invalid dets array in witness JSON");
        return -1;
    }

    int nc = 0;
    p = ca + 1;
    while (p < e) {
        while (p < e && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (p < e && *p == ']') break;
        if (nc >= ndet) { nc = -1; break; }
        char *ep = NULL;
        psi0[nc++] = strtod(p, &ep);
        if (ep == p) { nc = -1; break; }
        p = ep;
    }
    free(txt);
    if (nc != ndet) {
        free(dets); free(psi0);
        shci_set_err(err, err_sz, "coeffs length mismatch (%d != %d)", nc, ndet);
        return -1;
    }

    double e_eval = 0.0;
    if (chem_var_eval_witness(ham, ndet, dets, psi0, &e_eval, err, err_sz) != 0) {
        free(dets); free(psi0);
        return -1;
    }
    if (target_two_s >= 0) {
        double s2 = 0.0;
        double s2_t = 0.25 * (double)target_two_s * (double)(target_two_s + 2);
        if (chem_var_eval_spin_sq(ham->na, ham->nb, ndet, dets, psi0, &s2) != 0 ||
            fabs(s2 - s2_t) > 1e-3) {
            free(dets); free(psi0);
            shci_set_err(err, err_sz,
                         "external witness fails spin-purity target 2S=%d (<S^2>=%.6f != %.6f)",
                         target_two_s, s2, s2_t);
            return -1;
        }
    }

    double nrm = 0.0;
    for (int i = 0; i < ndet; i++) nrm += psi0[i] * psi0[i];
    nrm = sqrt(nrm);
    for (int i = 0; i < ndet; i++) psi0[i] /= nrm;

    out->E_hf0 = e_eval;
    out->E_scf = e_eval;
    out->E_searu_rot = e_eval;
    out->E_ci = e_eval;
    out->E_ub = e_eval;
    out->ndet_var = ndet;
    out->dets = dets;
    out->psi0 = psi0;
    return 0;
}

static inline double obs_h1_so(const ChemObservable *obs, int p, int q) {
    if ((p & 1) != (q & 1)) return 0.0;
    int K = obs->K;
    const double *mat = ((p & 1) == 0) ? obs->o1_a : obs->o1_b;
    return mat ? mat[(size_t)(p >> 1) * (size_t)K + (size_t)(q >> 1)] : 0.0;
}

static inline double obs_e2_so(const ChemObservable *obs, int p, int q, int r, int s) {
    if (!obs->has_o2) return 0.0;
    if ((p & 1) != (r & 1) || (q & 1) != (s & 1)) return 0.0;
    size_t ksz = (size_t)obs->K;
    size_t idx = ((((size_t)(p >> 1) * ksz + (size_t)(r >> 1)) * ksz + (size_t)(q >> 1)) * ksz +
                  (size_t)(s >> 1));
    const double *t = ((p & 1) == (q & 1)) ? obs->o2_aa : obs->o2_ab;
    return t ? t[idx] : 0.0;
}

static double obs_slater_condon_mel(const ChemObservable *obs, uint64_t dI, uint64_t dJ) {
    if (dI == dJ) {
        int occ[64];
        int nocc = 0;
        uint64_t tmp = dI;
        while (tmp) {
            occ[nocc++] = (int)searu_ctz64(tmp);
            tmp &= tmp - UINT64_C(1);
        }
        double e = obs->O_core;
        for (int i = 0; i < nocc; i++) {
            int p = occ[i];
            e += obs_h1_so(obs, p, p);
            for (int j = i + 1; j < nocc; j++) {
                int q = occ[j];
                e += obs_e2_so(obs, p, q, p, q) - obs_e2_so(obs, p, q, q, p);
            }
        }
        return e;
    }
    uint64_t diff = dI ^ dJ;
    uint32_t nd = searu_popcnt64(diff);
    if (nd == 2U) {
        int r = (int)searu_ctz64(dI & diff);
        int c = (int)searu_ctz64(dJ & diff);
        if ((r & 1) != (c & 1)) return 0.0;
        int sgn = jw_par(dI, r) * jw_par(dI ^ (UINT64_C(1) << (unsigned)r), c);
        double mat = obs_h1_so(obs, r, c);
        uint64_t common = dI & dJ;
        while (common) {
            int p = (int)searu_ctz64(common);
            mat += obs_e2_so(obs, r, p, c, p) - obs_e2_so(obs, r, p, p, c);
            common &= common - UINT64_C(1);
        }
        return (double)sgn * mat;
    }
    if (nd == 4U && obs->has_o2) {
        uint64_t rem = dI & diff;
        int r1 = (int)searu_ctz64(rem);
        int r2 = (int)searu_ctz64(rem ^ (UINT64_C(1) << (unsigned)r1));
        uint64_t add = dJ & diff;
        int c1 = (int)searu_ctz64(add);
        int c2 = (int)searu_ctz64(add ^ (UINT64_C(1) << (unsigned)c1));
        uint64_t t1 = dI ^ (UINT64_C(1) << (unsigned)r1);
        uint64_t t2 = t1 ^ (UINT64_C(1) << (unsigned)r2);
        uint64_t t3 = t2 ^ (UINT64_C(1) << (unsigned)c2);
        int sgn = jw_par(dI, r1) * jw_par(t1, r2) * jw_par(t2, c2) * jw_par(t3, c1);
        return (double)sgn * (obs_e2_so(obs, r1, r2, c1, c2) - obs_e2_so(obs, r1, r2, c2, c1));
    }
    return 0.0;
}

int chem_eval_observable_witness(const ChemHam *ham, const ChemObservable *obs,
                                 int ndet, const uint64_t *dets, const double *coeffs,
                                 double *val_out, char *err, size_t err_sz) {
    if (!ham || !obs || obs->K != ham->K || ndet <= 0 || !dets || !coeffs || !val_out) {
        shci_set_err(err, err_sz, "invalid argument to chem_eval_observable_witness");
        return -1;
    }
    double norm2 = 0.0, num = 0.0;
    for (int i = 0; i < ndet; i++) {
        double ci = coeffs[i];
        norm2 += ci * ci;
        if (fabs(ci) < 1e-16) continue;
        num += ci * ci * obs_slater_condon_mel(obs, dets[i], dets[i]);
        for (int j = 0; j < i; j++) {
            double cj = coeffs[j];
            if (fabs(cj) < 1e-16) continue;
            uint32_t nd = searu_popcnt64(dets[i] ^ dets[j]);
            if (nd == 2U || (nd == 4U && obs->has_o2)) {
                num += 2.0 * ci * cj * obs_slater_condon_mel(obs, dets[i], dets[j]);
            }
        }
    }
    if (norm2 <= 1e-30) {
        shci_set_err(err, err_sz, "zero-norm witness in chem_eval_observable_witness");
        return -1;
    }
    *val_out = num / norm2;
    return 0;
}

typedef struct {
    double val;
    int spin;
    int eig_idx;
} GpcOccEntry;

static int cmp_gpc_desc(const void *a, const void *b) {
    const GpcOccEntry *ea = (const GpcOccEntry *)a;
    const GpcOccEntry *eb = (const GpcOccEntry *)b;
    if (ea->val > eb->val) return -1;
    if (ea->val < eb->val) return 1;
    return 0;
}

int chem_var_analyze_gpc(const ChemHam *ham, int ndet, const uint64_t *dets, const double *coeffs,
                         ChemGPCResult *out, char *err, size_t err_sz) {
    if (!ham || ndet <= 0 || !dets || !coeffs || !out) {
        shci_set_err(err, err_sz, "invalid argument to chem_var_analyze_gpc");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    int K = ham->K;
    out->K = K;
    out->na = ham->na;
    out->nb = ham->nb;
    out->nelec = ham->nelec;

    double Ga[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS] = {0};
    double Gb[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS] = {0};
    double norm2 = 0.0;
    for (int i = 0; i < ndet; i++) {
        double ci = coeffs[i];
        norm2 += ci * ci;
        if (fabs(ci) < 1e-16) continue;
        double w = ci * ci;
        uint64_t dI = dets[i];
        for (int p = 0; p < K; p++) {
            if ((dI >> (unsigned)(2 * p)) & UINT64_C(1)) Ga[p * K + p] += w;
            if ((dI >> (unsigned)(2 * p + 1)) & UINT64_C(1)) Gb[p * K + p] += w;
        }
        for (int j = 0; j < i; j++) {
            double cj = coeffs[j];
            if (fabs(cj) < 1e-16) continue;
            uint64_t dJ = dets[j];
            uint64_t diff = dI ^ dJ;
            if (searu_popcnt64(diff) == 2U) {
                int p_so = (int)searu_ctz64(dI & diff);
                int q_so = (int)searu_ctz64(dJ & diff);
                if ((p_so & 1) == (q_so & 1)) {
                    int sgn = jw_par(dJ, q_so) * jw_par(dJ ^ (UINT64_C(1) << (unsigned)q_so), p_so);
                    double val = (double)sgn * ci * cj;
                    int p = p_so >> 1, q = q_so >> 1;
                    double *G = ((p_so & 1) == 0) ? Ga : Gb;
                    G[p * K + q] += val;
                    G[q * K + p] += val;
                }
            }
        }
    }
    if (norm2 <= 1e-30) {
        shci_set_err(err, err_sz, "zero-norm witness in chem_var_analyze_gpc");
        return -1;
    }
    for (int i = 0; i < K * K; i++) {
        Ga[i] /= norm2;
        Gb[i] /= norm2;
    }

    double eva[CHEM_MAX_ORBITALS] = {0}, evb[CHEM_MAX_ORBITALS] = {0};
    if (sym_eigh_small(K, Ga, eva) != 0 || sym_eigh_small(K, Gb, evb) != 0) {
        shci_set_err(err, err_sz, "1-RDM diagonalization failed");
        return -1;
    }

    /* Reorder descending: eigenvector e_idx for eigenvalue occ[e_idx] sits at row e_idx of U */
    GpcOccEntry entries[2 * CHEM_MAX_ORBITALS];
    for (int e_idx = 0; e_idx < K; e_idx++) {
        int rev = K - 1 - e_idx;
        out->occ_a[e_idx] = eva[rev];
        out->occ_b[e_idx] = evb[rev];
        for (int p = 0; p < K; p++) {
            out->U_a[e_idx * K + p] = Ga[rev * K + p];
            out->U_b[e_idx * K + p] = Gb[rev * K + p];
        }
        entries[2 * e_idx].val = eva[rev];
        entries[2 * e_idx].spin = 0;
        entries[2 * e_idx].eig_idx = e_idx;
        entries[2 * e_idx + 1].val = evb[rev];
        entries[2 * e_idx + 1].spin = 1;
        entries[2 * e_idx + 1].eig_idx = e_idx;
    }
    qsort(entries, (size_t)(2 * K), sizeof(GpcOccEntry), cmp_gpc_desc);

    double s1 = 0.0;
    for (int i = 0; i < 2 * K; i++) {
        double l = entries[i].val;
        out->occ_sorted[i] = l;
        if (l > 1e-14 && l < 1.0 - 1e-14) {
            s1 -= l * log(l) + (1.0 - l) * log(1.0 - l);
        }
    }
    out->entanglement_entropy = s1;

    if (ham->nelec == 3 && K == 3) {
        out->has_bd_3e3o = 1;
        double e0 = fabs(out->occ_sorted[0] + out->occ_sorted[5] - 1.0);
        double e1 = fabs(out->occ_sorted[1] + out->occ_sorted[4] - 1.0);
        double e2 = fabs(out->occ_sorted[2] + out->occ_sorted[3] - 1.0);
        out->bd_pair_err = fmax(e0, fmax(e1, e2));
        out->bd_slack = out->occ_sorted[4] + out->occ_sorted[5] - out->occ_sorted[3];
        out->bd_pinned = (fabs(out->bd_slack) < 1e-6) ? 1 : 0;
        for (int m = 0; m < 3; m++) {
            out->bd_obs_spin[m] = entries[3 + m].spin;
            out->bd_obs_idx[m] = entries[3 + m].eig_idx;
        }
    } else if (ham->nelec == 2 && K == 2) {
        out->has_bd_2e2o = 1;
        double e0 = fabs(out->occ_sorted[0] + out->occ_sorted[3] - 1.0);
        double e1 = fabs(out->occ_sorted[1] + out->occ_sorted[2] - 1.0);
        out->bd24_pair_err = fmax(e0, e1);
    }
    return 0;
}

typedef struct {
    size_t cap;
    size_t size;
    uint64_t *keys;
    double *vals;
    uint8_t *used;
} VarHashTable;

static inline size_t var_hash_det(uint64_t x, size_t mask) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return (size_t)x & mask;
}

static int var_ht_init(VarHashTable *ht, size_t cap) {
    ht->cap = cap;
    ht->size = 0;
    ht->keys = (uint64_t *)calloc(cap, sizeof(uint64_t));
    ht->vals = (double *)calloc(cap, sizeof(double));
    ht->used = (uint8_t *)calloc(cap, sizeof(uint8_t));
    if (!ht->keys || !ht->vals || !ht->used) {
        free(ht->keys); free(ht->vals); free(ht->used);
        return -1;
    }
    return 0;
}

static void var_ht_free(VarHashTable *ht) {
    free(ht->keys);
    free(ht->vals);
    free(ht->used);
    memset(ht, 0, sizeof(*ht));
}

static int var_ht_add(VarHashTable *ht, uint64_t key, double delta) {
    if (fabs(delta) < 1e-18) return 0;
    if (ht->size * 10 >= ht->cap * 6) {
        size_t new_cap = ht->cap * 2;
        VarHashTable nh;
        if (var_ht_init(&nh, new_cap) != 0) return -1;
        size_t nmask = new_cap - 1;
        for (size_t i = 0; i < ht->cap; i++) {
            if (ht->used[i]) {
                size_t slot = var_hash_det(ht->keys[i], nmask);
                while (nh.used[slot]) slot = (slot + 1) & nmask;
                nh.used[slot] = 1;
                nh.keys[slot] = ht->keys[i];
                nh.vals[slot] = ht->vals[i];
                nh.size++;
            }
        }
        var_ht_free(ht);
        *ht = nh;
    }
    size_t mask = ht->cap - 1;
    size_t slot = var_hash_det(key, mask);
    while (ht->used[slot]) {
        if (ht->keys[slot] == key) {
            ht->vals[slot] += delta;
            return 0;
        }
        slot = (slot + 1) & mask;
    }
    ht->used[slot] = 1;
    ht->keys[slot] = key;
    ht->vals[slot] = delta;
    ht->size++;
    return 0;
}

int chem_var_eval_variance(const ChemHam *ham, int ndet, const uint64_t *dets, const double *coeffs,
                           double *e_var_out, double *var_out, char *err, size_t err_sz) {
    if (!ham || ndet <= 0 || !dets || !coeffs || !e_var_out || !var_out) {
        shci_set_err(err, err_sz, "invalid argument to chem_var_eval_variance");
        return -1;
    }
    double e_var = 0.0;
    if (chem_var_eval_witness(ham, ndet, dets, coeffs, &e_var, err, err_sz) != 0) return -1;

    double norm2 = 0.0;
    for (int i = 0; i < ndet; i++) norm2 += coeffs[i] * coeffs[i];
    if (norm2 <= 1e-30) {
        shci_set_err(err, err_sz, "zero-norm witness in chem_var_eval_variance");
        return -1;
    }

    VarHashTable ht;
    if (var_ht_init(&ht, 4096) != 0) {
        shci_set_err(err, err_sz, "out of memory in variance hash table");
        return -1;
    }

    int K = ham->K;
    int K2 = 2 * K;
    for (int id = 0; id < ndet; id++) {
        double cJ = coeffs[id];
        if (fabs(cJ) < 1e-15) continue;
        uint64_t dJ = dets[id];

        if (var_ht_add(&ht, dJ, cJ * (det_diag_energy(ham, dJ) - e_var)) != 0) {
            var_ht_free(&ht);
            shci_set_err(err, err_sz, "OOM in variance hash");
            return -1;
        }

        int occ_list[64], vir_a[32], vir_b[32];
        int nocc = 0, nva = 0, nvb = 0;
        for (int so = 0; so < K2; so++) {
            if ((dJ >> (unsigned)so) & UINT64_C(1)) {
                occ_list[nocc++] = so;
            } else {
                if ((so & 1) == 0) vir_a[nva++] = so;
                else vir_b[nvb++] = so;
            }
        }

        /* All spin-conserving single excitations */
        for (int i = 0; i < nocc; i++) {
            int r = occ_list[i];
            int *vlist = ((r & 1) == 0) ? vir_a : vir_b;
            int nv = ((r & 1) == 0) ? nva : nvb;
            for (int v = 0; v < nv; v++) {
                int a = vlist[v];
                uint64_t dI = dJ ^ (UINT64_C(1) << (unsigned)r) ^ (UINT64_C(1) << (unsigned)a);
                double mel = slater_condon_mel(ham, dI, dJ);
                if (fabs(mel) > 1e-15 && var_ht_add(&ht, dI, cJ * mel) != 0) {
                    var_ht_free(&ht);
                    shci_set_err(err, err_sz, "OOM in variance hash");
                    return -1;
                }
            }
        }

        /* All spin-conserving double excitations */
        for (int i = 0; i < nocc; i++) {
            int r1 = occ_list[i];
            int s1 = r1 & 1;
            for (int j = i + 1; j < nocc; j++) {
                int r2 = occ_list[j];
                int s2 = r2 & 1;
                if (s1 == s2) {
                    int *vlist = (s1 == 0) ? vir_a : vir_b;
                    int nv = (s1 == 0) ? nva : nvb;
                    for (int u = 0; u < nv; u++) {
                        int a1 = vlist[u];
                        for (int w = u + 1; w < nv; w++) {
                            int a2 = vlist[w];
                            double g = chem_ham_eri(ham, a1 >> 1, r1 >> 1, a2 >> 1, r2 >> 1) -
                                       chem_ham_eri(ham, a1 >> 1, r2 >> 1, a2 >> 1, r1 >> 1);
                            if (fabs(g) <= 1e-15) continue;
                            uint64_t dI = dJ ^ (UINT64_C(1) << (unsigned)r1) ^
                                          (UINT64_C(1) << (unsigned)r2) ^
                                          (UINT64_C(1) << (unsigned)a1) ^
                                          (UINT64_C(1) << (unsigned)a2);
                            double mel = slater_condon_mel(ham, dI, dJ);
                            if (var_ht_add(&ht, dI, cJ * mel) != 0) {
                                var_ht_free(&ht);
                                shci_set_err(err, err_sz, "OOM in variance hash");
                                return -1;
                            }
                        }
                    }
                } else {
                    int ra = (s1 == 0) ? r1 : r2;
                    int rb = (s1 == 1) ? r1 : r2;
                    for (int ua = 0; ua < nva; ua++) {
                        int aa = vir_a[ua];
                        for (int ub = 0; ub < nvb; ub++) {
                            int ab = vir_b[ub];
                            double g = chem_ham_eri(ham, aa >> 1, ra >> 1, ab >> 1, rb >> 1);
                            if (fabs(g) <= 1e-15) continue;
                            uint64_t dI = dJ ^ (UINT64_C(1) << (unsigned)ra) ^
                                          (UINT64_C(1) << (unsigned)rb) ^
                                          (UINT64_C(1) << (unsigned)aa) ^
                                          (UINT64_C(1) << (unsigned)ab);
                            double mel = slater_condon_mel(ham, dI, dJ);
                            if (var_ht_add(&ht, dI, cJ * mel) != 0) {
                                var_ht_free(&ht);
                                shci_set_err(err, err_sz, "OOM in variance hash");
                                return -1;
                            }
                        }
                    }
                }
            }
        }
    }

    double var_sum = 0.0;
    for (size_t i = 0; i < ht.cap; i++) {
        if (ht.used[i]) var_sum += ht.vals[i] * ht.vals[i];
    }
    var_ht_free(&ht);

    *e_var_out = e_var;
    *var_out = var_sum / norm2;
    return 0;
}

/* ============================================================================
 * Exact FCI 2-RDM Block Builders (Mode A & Mode B) for Self-Test Verification
 * ============================================================================ */

static void svec_pack_shci(int n, const double *M, double *v) {
    int p = 0;
    for (int i = 0; i < n; i++) {
        v[p++] = M[(size_t)i * (size_t)n + (size_t)i];
        for (int j = i + 1; j < n; j++) {
            v[p++] = CHEM_ISQRT2 * (M[(size_t)i * (size_t)n + (size_t)j] +
                                    M[(size_t)j * (size_t)n + (size_t)i]);
        }
    }
}

/* Computes <psi | op_str | psi> by applying 4-fermion term (or linear combo) via JW bit ops.
 * Each term is (create=1 / annih=0, spin_orb) applied right-to-left: op[3], op[2], op[1], op[0]. */
static double fci_expect_4op(int ndet, const uint64_t *dets, const double *psi0,
                             const int is_create[4], const int so[4]) {
    double acc = 0.0;
    for (int J = 0; J < ndet; J++) {
        double cJ = psi0[J];
        if (fabs(cJ) < 1e-15) continue;
        uint64_t t = dets[J];
        int s = 1;
        int ok = 1;
        for (int step = 3; step >= 0; step--) {
            int b = so[step];
            int occ = (int)((t >> (unsigned)b) & 1U);
            if (is_create[step] ? occ : !occ) {
                ok = 0;
                break;
            }
            s *= jw_par(t, b);
            t ^= (UINT64_C(1) << (unsigned)b);
        }
        if (!ok) continue;
        for (int idet = 0; idet < ndet; idet++) {
            if (dets[idet] == t) {
                acc += (double)s * cJ * psi0[idet];
                break;
            }
        }
    }
    return acc;
}

int chem_fci_compute_dqg_mode_a(const ChemHam *ham, int ndet, const uint64_t *dets, const double *psi0,
                                double **D_svec_out, double **Q_svec_out, double **G_svec_out,
                                char *err, size_t err_sz) {
    int K = ham->K;
    int m0 = K * (K + 1) / 2;
    int m1 = K * (K - 1) / 2;
    int k2 = K * K;
    int szD0 = m0 * (m0 + 1) / 2, szD1 = m1 * (m1 + 1) / 2, szG = k2 * (k2 + 1) / 2;

    double *Dsv = (double *)calloc((size_t)(szD0 + szD1), sizeof(double));
    double *Qsv = (double *)calloc((size_t)(szD0 + szD1), sizeof(double));
    double *Gsv = (double *)calloc((size_t)(2 * szG), sizeof(double));
    double *D0 = (double *)calloc((size_t)m0 * (size_t)m0, sizeof(double));
    double *Q0 = (double *)calloc((size_t)m0 * (size_t)m0, sizeof(double));
    double *D1 = (double *)calloc((size_t)(m1 > 0 ? m1 : 1) * (size_t)(m1 > 0 ? m1 : 1), sizeof(double));
    double *Q1 = (double *)calloc((size_t)(m1 > 0 ? m1 : 1) * (size_t)(m1 > 0 ? m1 : 1), sizeof(double));
    double *G0 = (double *)calloc((size_t)k2 * (size_t)k2, sizeof(double));
    double *G1 = (double *)calloc((size_t)k2 * (size_t)k2, sizeof(double));
    if (!Dsv || !Qsv || !Gsv || !D0 || !Q0 || !D1 || !Q1 || !G0 || !G1) {
        free(Dsv); free(Qsv); free(Gsv);
        free(D0); free(Q0); free(D1); free(Q1); free(G0); free(G1);
        shci_set_err(err, err_sz, "out of memory in chem_fci_compute_dqg_mode_a");
        return -1;
    }

    int p0[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int p1[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int n0 = 0, n1 = 0;
    for (int p = 0; p < K; p++) {
        for (int q = p; q < K; q++) {
            p0[n0][0] = p;
            p0[n0][1] = q;
            n0++;
            if (p < q) {
                p1[n1][0] = p;
                p1[n1][1] = q;
                n1++;
            }
        }
    }

    const int ic_D[4] = {1, 1, 0, 0};
    const int ic_Q[4] = {0, 0, 1, 1};
    const int ic_G[4] = {1, 0, 1, 0};

    /* D0 and Q0 */
    for (int P = 0; P < m0; P++) {
        int p = p0[P][0], q = p0[P][1];
        double ap = (p == q) ? CHEM_ISQRT2 : 1.0;
        for (int R = 0; R <= P; R++) {
            int r = p0[R][0], s = p0[R][1];
            double ar = (r == s) ? CHEM_ISQRT2 : 1.0;
            double fac = 0.5 * ap * ar;
            double dv = 0.0, qv = 0.0;
            int pp[2] = {p, q}, qq[2] = {q, p};
            int rr[2] = {r, s}, ss[2] = {s, r};
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) {
                    int so_d[4] = {2 * pp[a], 2 * qq[a] + 1, 2 * ss[b] + 1, 2 * rr[b]};
                    int so_q[4] = {2 * qq[a] + 1, 2 * pp[a], 2 * rr[b], 2 * ss[b] + 1};
                    dv += fci_expect_4op(ndet, dets, psi0, ic_D, so_d);
                    qv += fci_expect_4op(ndet, dets, psi0, ic_Q, so_q);
                }
            }
            D0[(size_t)P * (size_t)m0 + (size_t)R] = D0[(size_t)R * (size_t)m0 + (size_t)P] = fac * dv;
            Q0[(size_t)P * (size_t)m0 + (size_t)R] = Q0[(size_t)R * (size_t)m0 + (size_t)P] = fac * qv;
        }
    }

    /* D1 and Q1 (m=+1 alpha-alpha representative) */
    for (int P = 0; P < m1; P++) {
        int p = p1[P][0], q = p1[P][1];
        for (int R = 0; R <= P; R++) {
            int r = p1[R][0], s = p1[R][1];
            int so_d[4] = {2 * p, 2 * q, 2 * s, 2 * r};
            int so_q[4] = {2 * q, 2 * p, 2 * r, 2 * s};
            double dv = fci_expect_4op(ndet, dets, psi0, ic_D, so_d);
            double qv = fci_expect_4op(ndet, dets, psi0, ic_Q, so_q);
            D1[(size_t)P * (size_t)m1 + (size_t)R] = D1[(size_t)R * (size_t)m1 + (size_t)P] = dv;
            Q1[(size_t)P * (size_t)m1 + (size_t)R] = Q1[(size_t)R * (size_t)m1 + (size_t)P] = qv;
        }
    }

    /* G0 and G1 */
    for (int P = 0; P < k2; P++) {
        int p = P / K, q = P % K;
        for (int R = 0; R <= P; R++) {
            int r = R / K, s = R % K;
            double g0v = 0.0;
            for (int sig = 0; sig < 2; sig++) {
                for (int tau = 0; tau < 2; tau++) {
                    int so_g0[4] = {2 * p + sig, 2 * q + sig, 2 * s + tau, 2 * r + tau};
                    g0v += 0.5 * fci_expect_4op(ndet, dets, psi0, ic_G, so_g0);
                }
            }
            int so_g1[4] = {2 * p, 2 * q + 1, 2 * s + 1, 2 * r};
            double g1v = fci_expect_4op(ndet, dets, psi0, ic_G, so_g1);
            G0[(size_t)P * (size_t)k2 + (size_t)R] = G0[(size_t)R * (size_t)k2 + (size_t)P] = g0v;
            G1[(size_t)P * (size_t)k2 + (size_t)R] = G1[(size_t)R * (size_t)k2 + (size_t)P] = g1v;
        }
    }

    svec_pack_shci(m0, D0, Dsv);
    if (m1 > 0) svec_pack_shci(m1, D1, Dsv + szD0);
    svec_pack_shci(m0, Q0, Qsv);
    if (m1 > 0) svec_pack_shci(m1, Q1, Qsv + szD0);
    svec_pack_shci(k2, G0, Gsv);
    svec_pack_shci(k2, G1, Gsv + szG);

    free(D0); free(Q0); free(D1); free(Q1); free(G0); free(G1);
    *D_svec_out = Dsv;
    *Q_svec_out = Qsv;
    *G_svec_out = Gsv;
    return 0;
}

int chem_fci_compute_dqg_mode_b(const ChemHam *ham, int ndet, const uint64_t *dets, const double *psi0,
                                double **D_svec_out, double **Q_svec_out, double **G_svec_out,
                                char *err, size_t err_sz) {
    int K = ham->K;
    int naa = K * (K - 1) / 2;
    int nab = K * K;
    int ngd = 2 * K * K;
    int szDaa = naa * (naa + 1) / 2;
    int szDab = nab * (nab + 1) / 2;
    int szGd = ngd * (ngd + 1) / 2;

    double *Dsv = (double *)calloc((size_t)(2 * szDaa + szDab), sizeof(double));
    double *Qsv = (double *)calloc((size_t)(2 * szDaa + szDab), sizeof(double));
    double *Gsv = (double *)calloc((size_t)(szGd + 2 * szDab), sizeof(double));
    double *Daa = (double *)calloc((size_t)(naa > 0 ? naa : 1) * (size_t)(naa > 0 ? naa : 1), sizeof(double));
    double *Dbb = (double *)calloc((size_t)(naa > 0 ? naa : 1) * (size_t)(naa > 0 ? naa : 1), sizeof(double));
    double *Qaa = (double *)calloc((size_t)(naa > 0 ? naa : 1) * (size_t)(naa > 0 ? naa : 1), sizeof(double));
    double *Qbb = (double *)calloc((size_t)(naa > 0 ? naa : 1) * (size_t)(naa > 0 ? naa : 1), sizeof(double));
    double *Dab = (double *)calloc((size_t)nab * (size_t)nab, sizeof(double));
    double *Qab = (double *)calloc((size_t)nab * (size_t)nab, sizeof(double));
    double *Gd  = (double *)calloc((size_t)ngd * (size_t)ngd, sizeof(double));
    double *Gab = (double *)calloc((size_t)nab * (size_t)nab, sizeof(double));
    double *Gba = (double *)calloc((size_t)nab * (size_t)nab, sizeof(double));

    if (!Dsv || !Qsv || !Gsv || !Daa || !Dbb || !Qaa || !Qbb || !Dab || !Qab || !Gd || !Gab ||
        !Gba) {
        free(Dsv); free(Qsv); free(Gsv);
        free(Daa); free(Dbb); free(Qaa); free(Qbb);
        free(Dab); free(Qab); free(Gd); free(Gab); free(Gba);
        shci_set_err(err, err_sz, "out of memory in chem_fci_compute_dqg_mode_b");
        return -1;
    }

    int p1[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int np = 0;
    for (int i = 0; i < K; i++) {
        for (int j = i + 1; j < K; j++) {
            p1[np][0] = i;
            p1[np][1] = j;
            np++;
        }
    }

    const int ic_D[4] = {1, 1, 0, 0};
    const int ic_Q[4] = {0, 0, 1, 1};
    const int ic_G[4] = {1, 0, 1, 0};

    for (int spin = 0; spin < 2; spin++) {
        double *Dss = (spin == 0) ? Daa : Dbb;
        double *Qss = (spin == 0) ? Qaa : Qbb;
        for (int p = 0; p < naa; p++) {
            int i = p1[p][0], j = p1[p][1];
            int si = 2 * i + spin, sj = 2 * j + spin;
            for (int q = 0; q <= p; q++) {
                int k = p1[q][0], l = p1[q][1];
                int sk = 2 * k + spin, sl = 2 * l + spin;
                int so_d[4] = {si, sj, sl, sk};
                int so_q[4] = {sj, si, sk, sl};
                double dv = fci_expect_4op(ndet, dets, psi0, ic_D, so_d);
                double qv = fci_expect_4op(ndet, dets, psi0, ic_Q, so_q);
                Dss[(size_t)p * (size_t)naa + (size_t)q] = Dss[(size_t)q * (size_t)naa + (size_t)p] = dv;
                Qss[(size_t)p * (size_t)naa + (size_t)q] = Qss[(size_t)q * (size_t)naa + (size_t)p] = qv;
            }
        }
    }

    for (int p = 0; p < nab; p++) {
        int i = p / K, j = p % K;
        for (int q = 0; q <= p; q++) {
            int k = q / K, l = q % K;
            int so_d[4] = {2 * i, 2 * j + 1, 2 * l + 1, 2 * k};
            int so_q[4] = {2 * j + 1, 2 * i, 2 * k, 2 * l + 1};
            int so_gab[4] = {2 * i, 2 * j + 1, 2 * l + 1, 2 * k};
            int so_gba[4] = {2 * i + 1, 2 * j, 2 * l, 2 * k + 1};
            double dv = fci_expect_4op(ndet, dets, psi0, ic_D, so_d);
            double qv = fci_expect_4op(ndet, dets, psi0, ic_Q, so_q);
            double abv = fci_expect_4op(ndet, dets, psi0, ic_G, so_gab);
            double bav = fci_expect_4op(ndet, dets, psi0, ic_G, so_gba);
            Dab[(size_t)p * (size_t)nab + (size_t)q] = Dab[(size_t)q * (size_t)nab + (size_t)p] = dv;
            Qab[(size_t)p * (size_t)nab + (size_t)q] = Qab[(size_t)q * (size_t)nab + (size_t)p] = qv;
            Gab[(size_t)p * (size_t)nab + (size_t)q] = Gab[(size_t)q * (size_t)nab + (size_t)p] = abv;
            Gba[(size_t)p * (size_t)nab + (size_t)q] = Gba[(size_t)q * (size_t)nab + (size_t)p] = bav;
        }
    }

    for (int r = 0; r < ngd; r++) {
        int sig = (r >= nab) ? 1 : 0;
        int rp = r % nab;
        int i = rp / K, j = rp % K;
        for (int c = 0; c <= r; c++) {
            int tau = (c >= nab) ? 1 : 0;
            int cp = c % nab;
            int k = cp / K, l = cp % K;
            int so_gd[4] = {2 * i + sig, 2 * j + sig, 2 * l + tau, 2 * k + tau};
            double v = fci_expect_4op(ndet, dets, psi0, ic_G, so_gd);
            Gd[(size_t)r * (size_t)ngd + (size_t)c] = Gd[(size_t)c * (size_t)ngd + (size_t)r] = v;
        }
    }

    if (naa > 0) {
        svec_pack_shci(naa, Daa, Dsv);
        svec_pack_shci(naa, Dbb, Dsv + szDaa);
        svec_pack_shci(naa, Qaa, Qsv);
        svec_pack_shci(naa, Qbb, Qsv + szDaa);
    }
    svec_pack_shci(nab, Dab, Dsv + 2 * szDaa);
    svec_pack_shci(nab, Qab, Qsv + 2 * szDaa);
    svec_pack_shci(ngd, Gd, Gsv);
    svec_pack_shci(nab, Gab, Gsv + szGd);
    svec_pack_shci(nab, Gba, Gsv + szGd + szDab);

    free(Daa); free(Dbb); free(Qaa); free(Qbb);
    free(Dab); free(Qab); free(Gd); free(Gab); free(Gba);
    *D_svec_out = Dsv;
    *Q_svec_out = Qsv;
    *G_svec_out = Gsv;
    return 0;
}
