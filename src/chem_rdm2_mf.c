#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include "chem_rdm2.h"
#include "chem_shci.h"
#include "flow.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHEM_SQRT2  1.41421356237309504880168872420969808
#define CHEM_ISQRT2 0.70710678118654752440084436210484904

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void set_err(char *err, size_t err_sz, const char *fmt, ...) {
    if (!err || err_sz == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_sz, fmt, ap);
    va_end(ap);
}

/* ============================================================================
 * Public Option & Metadata Helpers
 * ============================================================================ */

ChemSolveOpts chem_solve_opts_default(const ChemHam *ham) {
    ChemSolveOpts o;
    memset(&o, 0, sizeof(o));
    o.mode = CHEM_MODE_AUTO;
    o.two_s = ham ? (ham->ms2 >= 0 ? ham->ms2 : -ham->ms2) : 0;
    o.constrain_s2 = 1;
    o.max_iters = 800;
    o.tol = 1e-6;
    o.alpha = 1.6;
    o.check_cert_every = 0;
    o.veto_enabled = 0;
    o.E_ai = 0.0;
    o.veto_tol = 1e-4;
    o.enable_t2 = 0;
    o.t2_max_orbs = 6;
    o.t2_norbs = 0;
    o.num_threads = 0;
    o.verbose = 0;
    return o;
}

ChemMode chem_resolve_mode(const ChemHam *ham, const ChemSolveOpts *opts) {
    if (opts && opts->mode != CHEM_MODE_AUTO) return opts->mode;
    int two_s = opts ? opts->two_s : (ham->ms2 >= 0 ? ham->ms2 : -ham->ms2);
    int cs2 = opts ? opts->constrain_s2 : 1;
    if (ham->na == ham->nb && two_s == 0 && cs2) {
        return CHEM_MODE_A_SINGLET;
    }
    return CHEM_MODE_B_GENERAL;
}

const char *chem_block_name(ChemMode mode, int blk_idx) {
    static const char *MODE_A_NAMES[8] = {"D0", "D1", "Q0", "Q1", "G0", "G1", "T2aab", "T2mix"};
    static const char *MODE_B_NAMES[CHEM_MAX_BLOCKS] = {
        "Daa", "Dbb", "Dab", "Qaa", "Qbb", "Qab", "Gdiag", "Gab", "Gba",
        "T2aab", "T2mixA", "T2bba", "T2mixB"
    };
    if (mode == CHEM_MODE_A_SINGLET) {
        return (blk_idx >= 0 && blk_idx < 8) ? MODE_A_NAMES[blk_idx] : "?";
    }
    return (blk_idx >= 0 && blk_idx < CHEM_MAX_BLOCKS) ? MODE_B_NAMES[blk_idx] : "?";
}

void chem_dual_witness_free(ChemDualWitness *w) {
    if (!w) return;
    free(w->y_eq);
    free(w->Z_Q);
    free(w->Z_G);
    free(w->Z_T2);
    memset(w, 0, sizeof(*w));
}

void chem_solve_result_free(ChemSolveResult *res) {
    if (!res) return;
    chem_dual_witness_free(&res->witness);
    memset(res, 0, sizeof(*res));
}

ChemVetoResult chem_make_veto_result(const ChemSolveResult *solve, double E_ub_variational) {
    ChemVetoResult v;
    v.vetoed = solve->vetoed;
    v.iter_stopped = solve->iter_stopped;
    v.E_lb_cert = solve->E_lb_cert;
    v.E_ub_variational = E_ub_variational;
    v.min_cert_eig = solve->min_eig_cert;
    v.interval_width = E_ub_variational - solve->E_lb_cert;
    return v;
}

/* ============================================================================
 * Isometric Symmetric Vector Packing & Indexing
 * ============================================================================ */

static inline int svec_len(int n) {
    return n * (n + 1) / 2;
}

static void svec_unpack(int n, const double *v, double *M) {
    int p = 0;
    for (int i = 0; i < n; i++) {
        M[(size_t)i * (size_t)n + (size_t)i] = v[p++];
        for (int j = i + 1; j < n; j++) {
            double val = CHEM_ISQRT2 * v[p++];
            M[(size_t)i * (size_t)n + (size_t)j] = val;
            M[(size_t)j * (size_t)n + (size_t)i] = val;
        }
    }
}

/* Adjoint-exact packing for arbitrary (possibly unsymmetrized) n x n matrix M */
static void svec_pack_general(int n, const double *M, double *v) {
    int p = 0;
    for (int i = 0; i < n; i++) {
        v[p++] = M[(size_t)i * (size_t)n + (size_t)i];
        for (int j = i + 1; j < n; j++) {
            double a = M[(size_t)i * (size_t)n + (size_t)j];
            double b = M[(size_t)j * (size_t)n + (size_t)i];
            v[p++] = CHEM_ISQRT2 * (a + b);
        }
    }
}

static inline int idx_sym_pair(int K, int p, int q) {
    if (p > q) {
        int t = p;
        p = q;
        q = t;
    }
    return p * K - p * (p - 1) / 2 + (q - p);
}

static inline int idx_asym_pair(int K, int p, int q, int *sgn) {
    if (p == q) {
        *sgn = 0;
        return -1;
    }
    if (p > q) {
        int t = p;
        p = q;
        q = t;
        *sgn = -1;
    } else {
        *sgn = 1;
    }
    return p * (K - 1) - p * (p - 1) / 2 + (q - p - 1);
}

static inline double dot_vec(int n, const double *a, const double *b) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline double norm2_vec(int n, const double *a) {
    return sqrt(dot_vec(n, a, a));
}

/* ============================================================================
 * Descriptor & Workspace for Matrix-Free Mode A / Mode B Operators
 * ============================================================================ */

typedef struct {
    ChemMode mode;
    int K, na, nb, two_s, constrain_s2;
    int t2_enabled;
    int t2_norbs;
    int t2_orbs[CHEM_MAX_ORBITALS];
    int t2_naa, t2_nab, t2_naab, t2_nabb, t2_nmix;
    int t2_P1[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];
    double t2_sgn1[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];

    int nblocks;
    int block_dims[CHEM_MAX_BLOCKS];
    int block_szs[CHEM_MAX_BLOCKS];
    double block_traces[CHEM_MAX_BLOCKS];
    int nD, nQ, nG, nT2, nAll, meq;
    int max_dim;
    int second_max_dim;
    int num_threads;

    /* Pre-mapped dimensions */
    int m0, m1, k2;    /* Mode A: K(K+1)/2, K(K-1)/2, K^2 */
    int naa, nab, ngd; /* Mode B: K(K-1)/2, K^2, 2*K^2 */

    /* O(1) index lookup tables (populated once in mf_ctx_init) */
    int P0[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];
    int P1[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];
    double sgn1[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];

    /* RHS constants */
    double *b_eq;  /* [meq] */
    double *q0;    /* [nQ] */
    double *C_D;   /* [nD] */

    /* Reusable scratch buffers for O(K^4) matrix-free evaluation */
    double *scr_St;    /* [K^4] Mode A */
    double *scr_Tt;    /* [K^4] Mode A */
    double *scr_ga;    /* [K^2] */
    double *scr_gb;    /* [K^2] Mode B */
    double *scr_M0;    /* [max_dim^2] */
    double *scr_M1;    /* [K^4] */
    double *scr_M2;    /* [k2^2] */
    double *scr_D0;    /* Mode A [m0^2] or Mode B Daa [naa^2] */
    double *scr_D1;    /* Mode A [m1^2] or Mode B Dbb [naa^2] */
    double *scr_D2;    /* Mode B Dab [nab^2] */

    /* Optional T2 sub-space buffers (allocated only when t2_enabled == 1) */
    double *scr_t2_Daa;
    double *scr_t2_Dbb;
    double *scr_t2_Dab;
    double *scr_t2_Dab_sw;
    double *scr_t2_ga;
    double *scr_t2_gb;
    double *scr_t2_Mat; /* [max(t2_naab, t2_nmix)^2] */

    /* LAPACK dsyevd_ scratch (thread 0 handles up to max_dim) */
    double *ev_val;    /* [max_dim] */
    double *ev_work;
    __LAPACK_int *ev_iwork;
    __LAPACK_int ev_lwork;
    __LAPACK_int ev_liwork;

    /* Optional worker scratch for parallel block eigendecomposition (up to 3 helpers for second_max_dim) */
    struct {
        double *V;
        double *ev_val;
        double *ev_work;
        __LAPACK_int *ev_iwork;
    } thr_ws[3];
    int thr_nworkers;
    int thr_dim;
    __LAPACK_int thr_lwork;
    __LAPACK_int thr_liwork;
} ChemMFContext;

static void mf_ctx_free(ChemMFContext *ctx) {
    if (!ctx) return;
    free(ctx->b_eq);
    free(ctx->q0);
    free(ctx->C_D);
    free(ctx->scr_St);
    free(ctx->scr_Tt);
    free(ctx->scr_ga);
    free(ctx->scr_gb);
    free(ctx->scr_M0);
    free(ctx->scr_M1);
    free(ctx->scr_M2);
    free(ctx->scr_D0);
    free(ctx->scr_D1);
    free(ctx->scr_D2);
    free(ctx->scr_t2_Daa);
    free(ctx->scr_t2_Dbb);
    free(ctx->scr_t2_Dab);
    free(ctx->scr_t2_Dab_sw);
    free(ctx->scr_t2_ga);
    free(ctx->scr_t2_gb);
    free(ctx->scr_t2_Mat);
    free(ctx->ev_val);
    free(ctx->ev_work);
    free(ctx->ev_iwork);
    for (int t = 0; t < 3; t++) {
        free(ctx->thr_ws[t].V);
        free(ctx->thr_ws[t].ev_val);
        free(ctx->thr_ws[t].ev_work);
        free(ctx->thr_ws[t].ev_iwork);
    }
    memset(ctx, 0, sizeof(*ctx));
}

/* ============================================================================
 * Mode A Exact Matrix-Free Forward & Adjoint Maps (6 PSD Blocks)
 * ============================================================================ */

static void mode_a_expand_st_tt(const ChemMFContext *ctx, const double *d_svec,
                                double *St, double *Tt) {
    int K = ctx->K;
    int m0 = ctx->m0;
    int m1 = ctx->m1;
    double *D0 = ctx->scr_D0;
    double *D1 = ctx->scr_D1;
    svec_unpack(m0, d_svec, D0);
    if (m1 > 0) svec_unpack(m1, d_svec + ctx->block_szs[0], D1);

    size_t ksz = (size_t)K;
    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
            double epq = (p == q) ? 1.0 : CHEM_ISQRT2;
            int P = ctx->P0[p][q];
            int P1 = ctx->P1[p][q];
            double zpq = ctx->sgn1[p][q] * CHEM_ISQRT2;
            for (int r = 0; r < K; r++) {
                for (int s = 0; s < K; s++) {
                    double ers = (r == s) ? 1.0 : CHEM_ISQRT2;
                    int R = ctx->P0[r][s];
                    size_t off4 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz +
                                   (size_t)s);
                    St[off4] = epq * ers * D0[(size_t)P * (size_t)m0 + (size_t)R];
                    int R1 = ctx->P1[r][s];
                    if (P1 >= 0 && R1 >= 0) {
                        double zrs = ctx->sgn1[r][s] * CHEM_ISQRT2;
                        Tt[off4] = zpq * zrs * D1[(size_t)P1 * (size_t)m1 + (size_t)R1];
                    } else {
                        Tt[off4] = 0.0;
                    }
                }
            }
        }
    }
}

static void mode_a_contract_sh_th(const ChemMFContext *ctx, const double *Sh, const double *Th,
                                  double *d_hat_svec) {
    int K = ctx->K;
    int m0 = ctx->m0;
    int m1 = ctx->m1;
    double *D0h = ctx->scr_D0;
    double *D1h = ctx->scr_D1;
    memset(D0h, 0, (size_t)m0 * (size_t)m0 * sizeof(double));
    if (m1 > 0) memset(D1h, 0, (size_t)m1 * (size_t)m1 * sizeof(double));

    size_t ksz = (size_t)K;
    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
            double epq = (p == q) ? 1.0 : CHEM_ISQRT2;
            int P = ctx->P0[p][q];
            int P1 = ctx->P1[p][q];
            double zpq = ctx->sgn1[p][q] * CHEM_ISQRT2;
            for (int r = 0; r < K; r++) {
                for (int s = 0; s < K; s++) {
                    double ers = (r == s) ? 1.0 : CHEM_ISQRT2;
                    int R = ctx->P0[r][s];
                    size_t off4 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz +
                                   (size_t)s);
                    D0h[(size_t)P * (size_t)m0 + (size_t)R] += epq * ers * Sh[off4];
                    if (P1 >= 0 && Th) {
                        int R1 = ctx->P1[r][s];
                        if (R1 >= 0) {
                            double zrs = ctx->sgn1[r][s] * CHEM_ISQRT2;
                            D1h[(size_t)P1 * (size_t)m1 + (size_t)R1] += zpq * zrs * Th[off4];
                        }
                    }
                }
            }
        }
    }

    svec_pack_general(m0, D0h, d_hat_svec);
    if (m1 > 0) svec_pack_general(m1, D1h, d_hat_svec + ctx->block_szs[0]);
}

static void mode_a_gamma_from_st(int K, int ne, const double *St, double *ga) {
    double fac = 2.0 / (double)(ne + 1);
    size_t ksz = (size_t)K;
    for (int p = 0; p < K; p++) {
        for (int r = 0; r < K; r++) {
            double sum = 0.0;
            for (int q = 0; q < K; q++) {
                sum += St[((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz + (size_t)q)];
            }
            ga[(size_t)p * ksz + (size_t)r] = fac * sum;
        }
    }
}

static void mode_a_gamma_adjoint_to_sh(int K, int ne, const double *ga_hat, double *Sh) {
    double fac = 2.0 / (double)(ne + 1);
    size_t ksz = (size_t)K;
    for (int p = 0; p < K; p++) {
        for (int r = 0; r < K; r++) {
            double v = fac * ga_hat[(size_t)p * ksz + (size_t)r];
            for (int q = 0; q < K; q++) {
                Sh[((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz + (size_t)q)] += v;
            }
        }
    }
}

static void mode_a_LQ_fwd(const ChemMFContext *ctx, const double *d_in, double *q_out) {
    int K = ctx->K;
    int m0 = ctx->m0;
    int m1 = ctx->m1;
    int sz0 = ctx->block_szs[0];
    int sz1 = ctx->block_szs[1];

    mode_a_expand_st_tt(ctx, d_in, ctx->scr_St, ctx->scr_Tt);
    mode_a_gamma_from_st(K, ctx->na, ctx->scr_St, ctx->scr_ga);

    double *Q0 = ctx->scr_M0;
    double *Q1 = ctx->scr_M1;
    svec_unpack(m0, d_in, Q0);
    if (m1 > 0) svec_unpack(m1, d_in + sz0, Q1);

    const double *ga = ctx->scr_ga;
    size_t ksz = (size_t)K;
    for (int a = 0; a < K; a++) {
        for (int b = 0; b < K; b++) {
            double gba = ga[(size_t)b * ksz + (size_t)a];
            for (int c = 0; c < K; c++) {
                double ba = (a == c) ? CHEM_SQRT2 : 1.0;
                double bb = (b == c) ? CHEM_SQRT2 : 1.0;
                int P0 = ctx->P0[a][c];
                int R0 = ctx->P0[b][c];
                Q0[(size_t)P0 * (size_t)m0 + (size_t)R0] -= ba * bb * gba;

                if (m1 > 0 && a != c && b != c) {
                    int P1 = ctx->P1[a][c];
                    int R1 = ctx->P1[b][c];
                    Q1[(size_t)P1 * (size_t)m1 + (size_t)R1] -= ctx->sgn1[a][c] * ctx->sgn1[b][c] * gba;
                }
            }
        }
    }
    svec_pack_general(m0, Q0, q_out);
    if (m1 > 0) svec_pack_general(m1, Q1, q_out + sz0);
    (void)sz1;
}

static void mode_a_LQ_adj(const ChemMFContext *ctx, const double *yq_in, double *d_out) {
    int K = ctx->K;
    int m0 = ctx->m0;
    int m1 = ctx->m1;
    int sz0 = ctx->block_szs[0];

    double *Y0 = ctx->scr_M0;
    double *Y1 = ctx->scr_M1;
    svec_unpack(m0, yq_in, Y0);
    if (m1 > 0) svec_unpack(m1, yq_in + sz0, Y1);

    double *ga_hat = ctx->scr_ga;
    memset(ga_hat, 0, (size_t)K * (size_t)K * sizeof(double));
    size_t ksz = (size_t)K;

    for (int a = 0; a < K; a++) {
        for (int b = 0; b < K; b++) {
            double acc = 0.0;
            for (int c = 0; c < K; c++) {
                double ba = (a == c) ? CHEM_SQRT2 : 1.0;
                double bb = (b == c) ? CHEM_SQRT2 : 1.0;
                int P0 = ctx->P0[a][c];
                int R0 = ctx->P0[b][c];
                acc -= ba * bb * Y0[(size_t)P0 * (size_t)m0 + (size_t)R0];
                if (m1 > 0 && a != c && b != c) {
                    int P1 = ctx->P1[a][c];
                    int R1 = ctx->P1[b][c];
                    acc -= ctx->sgn1[a][c] * ctx->sgn1[b][c] * Y1[(size_t)P1 * (size_t)m1 + (size_t)R1];
                }
            }
            ga_hat[(size_t)b * ksz + (size_t)a] = acc;
        }
    }

    size_t k4 = ksz * ksz * ksz * ksz;
    memset(ctx->scr_St, 0, k4 * sizeof(double));
    mode_a_gamma_adjoint_to_sh(K, ctx->na, ga_hat, ctx->scr_St);
    mode_a_contract_sh_th(ctx, ctx->scr_St, NULL, d_out);

    for (int i = 0; i < ctx->nD; i++) d_out[i] += yq_in[i];
}

static void mode_a_LG_fwd(const ChemMFContext *ctx, const double *d_in, double *g_out) {
    int K = ctx->K;
    int k2 = ctx->k2;
    size_t ksz = (size_t)K;

    mode_a_expand_st_tt(ctx, d_in, ctx->scr_St, ctx->scr_Tt);
    mode_a_gamma_from_st(K, ctx->na, ctx->scr_St, ctx->scr_ga);

    const double *St = ctx->scr_St;
    const double *Tt = ctx->scr_Tt;
    const double *ga = ctx->scr_ga;
    double *G0 = ctx->scr_M0;
    double *G1 = ctx->scr_M1;

    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
            int row = p * K + q;
            for (int r = 0; r < K; r++) {
                double gpr = ga[(size_t)p * ksz + (size_t)r];
                for (int s = 0; s < K; s++) {
                    int col = r * K + s;
                    double dg = (q == s) ? gpr : 0.0;
                    size_t i_psqr = ((((size_t)p * ksz + (size_t)s) * ksz + (size_t)q) * ksz +
                                     (size_t)r);
                    size_t i_psrq = ((((size_t)p * ksz + (size_t)s) * ksz + (size_t)r) * ksz +
                                     (size_t)q);
                    G0[(size_t)row * (size_t)k2 + (size_t)col] = dg + St[i_psqr] + 3.0 * Tt[i_psqr];
                    G1[(size_t)row * (size_t)k2 + (size_t)col] = dg - St[i_psrq] - Tt[i_psrq];
                }
            }
        }
    }

    int szg = ctx->block_szs[4];
    svec_pack_general(k2, G0, g_out);
    svec_pack_general(k2, G1, g_out + szg);
}

static void mode_a_LG_adj(const ChemMFContext *ctx, const double *yg_in, double *d_out) {
    int K = ctx->K;
    int k2 = ctx->k2;
    size_t ksz = (size_t)K;
    int szg = ctx->block_szs[4];

    double *Y0 = ctx->scr_M0;
    double *Y1 = ctx->scr_M1;
    svec_unpack(k2, yg_in, Y0);
    svec_unpack(k2, yg_in + szg, Y1);

    size_t k4 = ksz * ksz * ksz * ksz;
    double *Sh = ctx->scr_St;
    double *Th = ctx->scr_Tt;
    double *ga_hat = ctx->scr_ga;
    memset(Sh, 0, k4 * sizeof(double));
    memset(Th, 0, k4 * sizeof(double));
    memset(ga_hat, 0, ksz * ksz * sizeof(double));

    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
            int row = p * K + q;
            for (int r = 0; r < K; r++) {
                for (int s = 0; s < K; s++) {
                    int col = r * K + s;
                    double y0 = Y0[(size_t)row * (size_t)k2 + (size_t)col];
                    double y1 = Y1[(size_t)row * (size_t)k2 + (size_t)col];
                    if (q == s) {
                        ga_hat[(size_t)p * ksz + (size_t)r] += y0 + y1;
                    }
                    size_t i_psqr = ((((size_t)p * ksz + (size_t)s) * ksz + (size_t)q) * ksz +
                                     (size_t)r);
                    size_t i_psrq = ((((size_t)p * ksz + (size_t)s) * ksz + (size_t)r) * ksz +
                                     (size_t)q);
                    Sh[i_psqr] += y0;
                    Th[i_psqr] += 3.0 * y0;
                    Sh[i_psrq] -= y1;
                    Th[i_psrq] -= y1;
                }
            }
        }
    }

    mode_a_gamma_adjoint_to_sh(K, ctx->na, ga_hat, Sh);
    mode_a_contract_sh_th(ctx, Sh, Th, d_out);
}

static void mode_a_Aeq_fwd(const ChemMFContext *ctx, const double *d_in, double *eq_out) {
    int K = ctx->K;
    int ne = ctx->na;
    int m0 = ctx->m0;
    int m1 = ctx->m1;
    int sz0 = ctx->block_szs[0];

    double *D0 = ctx->scr_D0;
    svec_unpack(m0, d_in, D0);
    double tr0 = 0.0;
    for (int p = 0; p < m0; p++) tr0 += D0[(size_t)p * (size_t)m0 + (size_t)p];
    eq_out[0] = tr0;

    if (ne == 1) {
        double tr1 = 0.0;
        if (m1 > 0) {
            double *D1 = ctx->scr_D1;
            svec_unpack(m1, d_in + sz0, D1);
            for (int p = 0; p < m1; p++) tr1 += D1[(size_t)p * (size_t)m1 + (size_t)p];
        }
        eq_out[1] = tr1;
        return;
    }

    mode_a_expand_st_tt(ctx, d_in, ctx->scr_St, ctx->scr_Tt);
    const double *St = ctx->scr_St;
    const double *Tt = ctx->scr_Tt;
    double cs = (double)(ne - 1);
    double ct = (double)(ne + 1);
    size_t ksz = (size_t)K;

    for (int p = 0; p < K; p++) {
        for (int r = p; r < K; r++) {
            double mu = (p == r) ? 1.0 : CHEM_SQRT2;
            double s_sum = 0.0, t_sum = 0.0;
            for (int q = 0; q < K; q++) {
                size_t i1 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz + (size_t)q);
                size_t i2 = ((((size_t)r * ksz + (size_t)q) * ksz + (size_t)p) * ksz + (size_t)q);
                s_sum += St[i1] + St[i2];
                t_sum += Tt[i1] + Tt[i2];
            }
            eq_out[1 + idx_sym_pair(K, p, r)] = 0.5 * mu * (cs * s_sum - ct * t_sum);
        }
    }
}

static void mode_a_Aeq_adj(const ChemMFContext *ctx, const double *lam_in, double *d_out) {
    int K = ctx->K;
    int ne = ctx->na;
    int m0 = ctx->m0;
    int m1 = ctx->m1;
    int sz0 = ctx->block_szs[0];

    if (ne == 1) {
        memset(d_out, 0, (size_t)ctx->nD * sizeof(double));
        double *D0h = ctx->scr_D0;
        memset(D0h, 0, (size_t)m0 * (size_t)m0 * sizeof(double));
        for (int p = 0; p < m0; p++) D0h[(size_t)p * (size_t)m0 + (size_t)p] = lam_in[0];
        svec_pack_general(m0, D0h, d_out);
        if (m1 > 0) {
            double *D1h = ctx->scr_D1;
            memset(D1h, 0, (size_t)m1 * (size_t)m1 * sizeof(double));
            for (int p = 0; p < m1; p++) D1h[(size_t)p * (size_t)m1 + (size_t)p] = lam_in[1];
            svec_pack_general(m1, D1h, d_out + sz0);
        }
        return;
    }

    size_t ksz = (size_t)K;
    size_t k4 = ksz * ksz * ksz * ksz;
    double *Sh = ctx->scr_St;
    double *Th = ctx->scr_Tt;
    memset(Sh, 0, k4 * sizeof(double));
    memset(Th, 0, k4 * sizeof(double));

    double cs = (double)(ne - 1);
    double ct = (double)(ne + 1);
    for (int p = 0; p < K; p++) {
        for (int r = p; r < K; r++) {
            double mu = (p == r) ? 1.0 : CHEM_SQRT2;
            double w = 0.5 * mu * lam_in[1 + idx_sym_pair(K, p, r)];
            for (int q = 0; q < K; q++) {
                size_t i1 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz + (size_t)q);
                size_t i2 = ((((size_t)r * ksz + (size_t)q) * ksz + (size_t)p) * ksz + (size_t)q);
                Sh[i1] += w * cs;
                Sh[i2] += w * cs;
                Th[i1] -= w * ct;
                Th[i2] -= w * ct;
            }
        }
    }
    mode_a_contract_sh_th(ctx, Sh, Th, d_out);

    /* Add trace term lam_in[0] * I_m0 directly to packed diagonal of D0 */
    double *D0h = ctx->scr_D0;
    svec_unpack(m0, d_out, D0h);
    for (int p = 0; p < m0; p++) D0h[(size_t)p * (size_t)m0 + (size_t)p] += lam_in[0];
    svec_pack_general(m0, D0h, d_out);
}

/* ============================================================================
 * Mode B Exact Matrix-Free Forward & Adjoint Maps (9 PSD Blocks)
 * ============================================================================ */

static void mode_b_unpack_D(const ChemMFContext *ctx, const double *d_in,
                            double *Daa, double *Dbb, double *Dab) {
    int naa = ctx->naa;
    int nab = ctx->nab;
    int off0 = 0;
    int off1 = ctx->block_szs[0];
    int off2 = off1 + ctx->block_szs[1];
    if (naa > 0) {
        svec_unpack(naa, d_in + off0, Daa);
        svec_unpack(naa, d_in + off1, Dbb);
    }
    svec_unpack(nab, d_in + off2, Dab);
}

static void mode_b_pack_D(const ChemMFContext *ctx, const double *Daa, const double *Dbb,
                          const double *Dab, double *d_out) {
    int naa = ctx->naa;
    int nab = ctx->nab;
    int off0 = 0;
    int off1 = ctx->block_szs[0];
    int off2 = off1 + ctx->block_szs[1];
    if (naa > 0) {
        svec_pack_general(naa, Daa, d_out + off0);
        svec_pack_general(naa, Dbb, d_out + off1);
    }
    svec_pack_general(nab, Dab, d_out + off2);
}

static void mode_b_gamma_fwd(const ChemMFContext *ctx, const double *Daa, const double *Dbb,
                             const double *Dab, double *ga, double *gb) {
    int K = ctx->K;
    int na = ctx->na;
    int nb = ctx->nb;
    size_t ksz = (size_t)K;
    size_t nab = (size_t)ctx->nab;
    size_t naa = (size_t)ctx->naa;

    if (na >= 1 && nb >= 1) {
        double inv_nb = 1.0 / (double)nb;
        double inv_na = 1.0 / (double)na;
        for (int i = 0; i < K; i++) {
            for (int k = 0; k < K; k++) {
                double sa = 0.0, sb = 0.0;
                for (int m = 0; m < K; m++) {
                    sa += Dab[(size_t)(i * K + m) * nab + (size_t)(k * K + m)];
                    sb += Dab[(size_t)(m * K + i) * nab + (size_t)(m * K + k)];
                }
                ga[(size_t)i * ksz + (size_t)k] = inv_nb * sa;
                gb[(size_t)i * ksz + (size_t)k] = inv_na * sb;
            }
        }
        return;
    }

    memset(ga, 0, ksz * ksz * sizeof(double));
    memset(gb, 0, ksz * ksz * sizeof(double));
    if (nb == 0 && na >= 2) {
        double inv = 1.0 / (double)(na - 1);
        for (int i = 0; i < K; i++) {
            for (int k = 0; k < K; k++) {
                double sa = 0.0;
                for (int m = 0; m < K; m++) {
                    int s1 = 0, s2 = 0;
                    int p = idx_asym_pair(K, i, m, &s1);
                    int q = idx_asym_pair(K, k, m, &s2);
                    if (p >= 0 && q >= 0) sa += (double)(s1 * s2) * Daa[(size_t)p * naa + (size_t)q];
                }
                ga[(size_t)i * ksz + (size_t)k] = inv * sa;
            }
        }
    } else if (na == 0 && nb >= 2) {
        double inv = 1.0 / (double)(nb - 1);
        for (int i = 0; i < K; i++) {
            for (int k = 0; k < K; k++) {
                double sb = 0.0;
                for (int m = 0; m < K; m++) {
                    int s1 = 0, s2 = 0;
                    int p = idx_asym_pair(K, i, m, &s1);
                    int q = idx_asym_pair(K, k, m, &s2);
                    if (p >= 0 && q >= 0) sb += (double)(s1 * s2) * Dbb[(size_t)p * naa + (size_t)q];
                }
                gb[(size_t)i * ksz + (size_t)k] = inv * sb;
            }
        }
    }
}

static void mode_b_gamma_adj(const ChemMFContext *ctx, const double *ga_hat, const double *gb_hat,
                             double *Daa_h, double *Dbb_h, double *Dab_h) {
    int K = ctx->K;
    int na = ctx->na;
    int nb = ctx->nb;
    size_t ksz = (size_t)K;
    size_t nab = (size_t)ctx->nab;
    size_t naa = (size_t)ctx->naa;

    if (na >= 1 && nb >= 1) {
        double inv_nb = 1.0 / (double)nb;
        double inv_na = 1.0 / (double)na;
        for (int i = 0; i < K; i++) {
            for (int k = 0; k < K; k++) {
                double va = inv_nb * ga_hat[(size_t)i * ksz + (size_t)k];
                double vb = inv_na * gb_hat[(size_t)i * ksz + (size_t)k];
                for (int m = 0; m < K; m++) {
                    Dab_h[(size_t)(i * K + m) * nab + (size_t)(k * K + m)] += va;
                    Dab_h[(size_t)(m * K + i) * nab + (size_t)(m * K + k)] += vb;
                }
            }
        }
        return;
    }

    if (nb == 0 && na >= 2) {
        double inv = 1.0 / (double)(na - 1);
        for (int i = 0; i < K; i++) {
            for (int k = 0; k < K; k++) {
                double va = inv * ga_hat[(size_t)i * ksz + (size_t)k];
                for (int m = 0; m < K; m++) {
                    int s1 = 0, s2 = 0;
                    int p = idx_asym_pair(K, i, m, &s1);
                    int q = idx_asym_pair(K, k, m, &s2);
                    if (p >= 0 && q >= 0) {
                        Daa_h[(size_t)p * naa + (size_t)q] += (double)(s1 * s2) * va;
                    }
                }
            }
        }
    } else if (na == 0 && nb >= 2) {
        double inv = 1.0 / (double)(nb - 1);
        for (int i = 0; i < K; i++) {
            for (int k = 0; k < K; k++) {
                double vb = inv * gb_hat[(size_t)i * ksz + (size_t)k];
                for (int m = 0; m < K; m++) {
                    int s1 = 0, s2 = 0;
                    int p = idx_asym_pair(K, i, m, &s1);
                    int q = idx_asym_pair(K, k, m, &s2);
                    if (p >= 0 && q >= 0) {
                        Dbb_h[(size_t)p * naa + (size_t)q] += (double)(s1 * s2) * vb;
                    }
                }
            }
        }
    }
}

static void mode_b_LQ_fwd(const ChemMFContext *ctx, const double *d_in, double *q_out) {
    int K = ctx->K;
    int naa = ctx->naa;
    int nab = ctx->nab;
    size_t ksz = (size_t)K;

    double *Daa = ctx->scr_D0;
    double *Dbb = ctx->scr_D1;
    double *Dab = ctx->scr_D2;
    mode_b_unpack_D(ctx, d_in, Daa, Dbb, Dab);

    double *ga = ctx->scr_ga;
    double *gb = ctx->scr_gb;
    mode_b_gamma_fwd(ctx, Daa, Dbb, Dab, ga, gb);

    double *Qaa = ctx->scr_M0;
    double *Qbb = ctx->scr_M1;
    double *Qab = ctx->scr_M2;
    if (naa > 0) {
        memcpy(Qaa, Daa, (size_t)naa * (size_t)naa * sizeof(double));
        memcpy(Qbb, Dbb, (size_t)naa * (size_t)naa * sizeof(double));
    }
    memcpy(Qab, Dab, (size_t)nab * (size_t)nab * sizeof(double));

    for (int a = 0; a < K; a++) {
        for (int b = 0; b < K; b++) {
            double gba_a = ga[(size_t)b * ksz + (size_t)a];
            double gba_b = gb[(size_t)b * ksz + (size_t)a];
            for (int c = 0; c < K; c++) {
                if (naa > 0 && a != c && b != c) {
                    int P = ctx->P1[a][c];
                    int R = ctx->P1[b][c];
                    double sgn = ctx->sgn1[a][c] * ctx->sgn1[b][c];
                    Qaa[(size_t)P * (size_t)naa + (size_t)R] -= sgn * gba_a;
                    Qbb[(size_t)P * (size_t)naa + (size_t)R] -= sgn * gba_b;
                }
                /* Qab[(a,c),(b,c)] -= ga[b,a], Qab[(c,a),(c,b)] -= gb[b,a] */
                Qab[(size_t)(a * K + c) * (size_t)nab + (size_t)(b * K + c)] -= gba_a;
                Qab[(size_t)(c * K + a) * (size_t)nab + (size_t)(c * K + b)] -= gba_b;
            }
        }
    }

    mode_b_pack_D(ctx, Qaa, Qbb, Qab, q_out);
}

static void mode_b_LQ_adj(const ChemMFContext *ctx, const double *yq_in, double *d_out) {
    int K = ctx->K;
    int naa = ctx->naa;
    int nab = ctx->nab;
    size_t ksz = (size_t)K;

    double *Yaa = ctx->scr_M0;
    double *Ybb = ctx->scr_M1;
    double *Yab = ctx->scr_M2;
    mode_b_unpack_D(ctx, yq_in, Yaa, Ybb, Yab);

    double *ga_h = ctx->scr_ga;
    double *gb_h = ctx->scr_gb;
    for (int a = 0; a < K; a++) {
        for (int b = 0; b < K; b++) {
            double sa = 0.0, sb = 0.0;
            for (int c = 0; c < K; c++) {
                if (naa > 0 && a != c && b != c) {
                    int P = ctx->P1[a][c];
                    int R = ctx->P1[b][c];
                    double sgn = ctx->sgn1[a][c] * ctx->sgn1[b][c];
                    sa -= sgn * Yaa[(size_t)P * (size_t)naa + (size_t)R];
                    sb -= sgn * Ybb[(size_t)P * (size_t)naa + (size_t)R];
                }
                sa -= Yab[(size_t)(a * K + c) * (size_t)nab + (size_t)(b * K + c)];
                sb -= Yab[(size_t)(c * K + a) * (size_t)nab + (size_t)(c * K + b)];
            }
            ga_h[(size_t)b * ksz + (size_t)a] = sa;
            gb_h[(size_t)b * ksz + (size_t)a] = sb;
        }
    }

    double *Daa_h = ctx->scr_D0;
    double *Dbb_h = ctx->scr_D1;
    double *Dab_h = ctx->scr_D2;
    if (naa > 0) {
        memcpy(Daa_h, Yaa, (size_t)naa * (size_t)naa * sizeof(double));
        memcpy(Dbb_h, Ybb, (size_t)naa * (size_t)naa * sizeof(double));
    }
    memcpy(Dab_h, Yab, (size_t)nab * (size_t)nab * sizeof(double));

    mode_b_gamma_adj(ctx, ga_h, gb_h, Daa_h, Dbb_h, Dab_h);
    mode_b_pack_D(ctx, Daa_h, Dbb_h, Dab_h, d_out);
}

static void mode_b_LG_fwd(const ChemMFContext *ctx, const double *d_in, double *g_out) {
    int K = ctx->K;
    int naa = ctx->naa;
    int nab = ctx->nab;
    int ngd = ctx->ngd;
    size_t ksz = (size_t)K;

    double *Daa = ctx->scr_D0;
    double *Dbb = ctx->scr_D1;
    double *Dab = ctx->scr_D2;
    mode_b_unpack_D(ctx, d_in, Daa, Dbb, Dab);

    double *ga = ctx->scr_ga;
    double *gb = ctx->scr_gb;
    mode_b_gamma_fwd(ctx, Daa, Dbb, Dab, ga, gb);

    double *Gd  = ctx->scr_M0;
    double *Gab = ctx->scr_M1;
    double *Gba = ctx->scr_M2;
    memset(Gd, 0, (size_t)ngd * (size_t)ngd * sizeof(double));

    /* Gdiag: alpha-alpha (top-left), beta-beta (bottom-right), alpha-beta & beta-alpha */
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            int p_ij = i * K + j;
            for (int k = 0; k < K; k++) {
                double ga_ik = ga[(size_t)i * ksz + (size_t)k];
                double gb_ik = gb[(size_t)i * ksz + (size_t)k];
                int qa = ctx->P1[k][j];
                double s_kj = ctx->sgn1[k][j];
                for (int l = 0; l < K; l++) {
                    int q_kl = k * K + l;
                    double vaa = (j == l) ? ga_ik : 0.0;
                    double vbb = (j == l) ? gb_ik : 0.0;
                    if (naa > 0) {
                        int pa = ctx->P1[i][l];
                        if (pa >= 0 && qa >= 0) {
                            double sgn = ctx->sgn1[i][l] * s_kj;
                            vaa -= sgn * Daa[(size_t)pa * (size_t)naa + (size_t)qa];
                            vbb -= sgn * Dbb[(size_t)pa * (size_t)naa + (size_t)qa];
                        }
                    }
                    Gd[(size_t)p_ij * (size_t)ngd + (size_t)q_kl] = vaa;
                    Gd[(size_t)(nab + p_ij) * (size_t)ngd + (size_t)(nab + q_kl)] = vbb;

                    double vab = Dab[(size_t)(i * K + l) * (size_t)nab + (size_t)(j * K + k)];
                    double vba = Dab[(size_t)(l * K + i) * (size_t)nab + (size_t)(k * K + j)];
                    Gd[(size_t)p_ij * (size_t)ngd + (size_t)(nab + q_kl)] = vab;
                    Gd[(size_t)(nab + p_ij) * (size_t)ngd + (size_t)q_kl] = vba;

                    /* Gab and Gba */
                    Gab[(size_t)p_ij * (size_t)nab + (size_t)q_kl] =
                        ((j == l) ? ga_ik : 0.0) -
                        Dab[(size_t)(i * K + l) * (size_t)nab + (size_t)(k * K + j)];
                    Gba[(size_t)p_ij * (size_t)nab + (size_t)q_kl] =
                        ((j == l) ? gb_ik : 0.0) -
                        Dab[(size_t)(l * K + i) * (size_t)nab + (size_t)(j * K + k)];
                }
            }
        }
    }

    int off0 = 0;
    int off1 = ctx->block_szs[6];
    int off2 = off1 + ctx->block_szs[7];
    svec_pack_general(ngd, Gd, g_out + off0);
    svec_pack_general(nab, Gab, g_out + off1);
    svec_pack_general(nab, Gba, g_out + off2);
}

static void mode_b_LG_adj(const ChemMFContext *ctx, const double *yg_in, double *d_out) {
    int K = ctx->K;
    int naa = ctx->naa;
    int nab = ctx->nab;
    int ngd = ctx->ngd;
    size_t ksz = (size_t)K;

    int off0 = 0;
    int off1 = ctx->block_szs[6];
    int off2 = off1 + ctx->block_szs[7];

    double *Yd  = ctx->scr_M0;
    double *Yab = ctx->scr_M1;
    double *Yba = ctx->scr_M2;
    svec_unpack(ngd, yg_in + off0, Yd);
    svec_unpack(nab, yg_in + off1, Yab);
    svec_unpack(nab, yg_in + off2, Yba);

    double *Daa_h = ctx->scr_D0;
    double *Dbb_h = ctx->scr_D1;
    double *Dab_h = ctx->scr_D2;
    double *ga_h  = ctx->scr_ga;
    double *gb_h  = ctx->scr_gb;
    if (naa > 0) {
        memset(Daa_h, 0, (size_t)naa * (size_t)naa * sizeof(double));
        memset(Dbb_h, 0, (size_t)naa * (size_t)naa * sizeof(double));
    }
    memset(Dab_h, 0, (size_t)nab * (size_t)nab * sizeof(double));
    memset(ga_h, 0, ksz * ksz * sizeof(double));
    memset(gb_h, 0, ksz * ksz * sizeof(double));

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            int p_ij = i * K + j;
            for (int k = 0; k < K; k++) {
                int qa = ctx->P1[k][j];
                double s_kj = ctx->sgn1[k][j];
                for (int l = 0; l < K; l++) {
                    int q_kl = k * K + l;
                    double yaa = Yd[(size_t)p_ij * (size_t)ngd + (size_t)q_kl];
                    double ybb = Yd[(size_t)(nab + p_ij) * (size_t)ngd + (size_t)(nab + q_kl)];
                    double yd_ab = Yd[(size_t)p_ij * (size_t)ngd + (size_t)(nab + q_kl)];
                    double yd_ba = Yd[(size_t)(nab + p_ij) * (size_t)ngd + (size_t)q_kl];
                    double yg_ab = Yab[(size_t)p_ij * (size_t)nab + (size_t)q_kl];
                    double yg_ba = Yba[(size_t)p_ij * (size_t)nab + (size_t)q_kl];

                    if (j == l) {
                        ga_h[(size_t)i * ksz + (size_t)k] += yaa + yg_ab;
                        gb_h[(size_t)i * ksz + (size_t)k] += ybb + yg_ba;
                    }
                    if (naa > 0) {
                        int pa = ctx->P1[i][l];
                        if (pa >= 0 && qa >= 0) {
                            double sgn = ctx->sgn1[i][l] * s_kj;
                            Daa_h[(size_t)pa * (size_t)naa + (size_t)qa] -= sgn * yaa;
                            Dbb_h[(size_t)pa * (size_t)naa + (size_t)qa] -= sgn * ybb;
                        }
                    }

                    Dab_h[(size_t)(i * K + l) * (size_t)nab + (size_t)(j * K + k)] += yd_ab;
                    Dab_h[(size_t)(l * K + i) * (size_t)nab + (size_t)(k * K + j)] += yd_ba;
                    Dab_h[(size_t)(i * K + l) * (size_t)nab + (size_t)(k * K + j)] -= yg_ab;
                    Dab_h[(size_t)(l * K + i) * (size_t)nab + (size_t)(j * K + k)] -= yg_ba;
                }
            }
        }
    }

    mode_b_gamma_adj(ctx, ga_h, gb_h, Daa_h, Dbb_h, Dab_h);
    mode_b_pack_D(ctx, Daa_h, Dbb_h, Dab_h, d_out);
}

static void mode_b_Aeq_fwd(const ChemMFContext *ctx, const double *d_in, double *eq_out) {
    int K = ctx->K;
    int na = ctx->na;
    int nb = ctx->nb;
    int naa = ctx->naa;
    int nab = ctx->nab;
    size_t ksz = (size_t)K;

    double *Daa = ctx->scr_D0;
    double *Dbb = ctx->scr_D1;
    double *Dab = ctx->scr_D2;
    mode_b_unpack_D(ctx, d_in, Daa, Dbb, Dab);

    double tr_aa = 0.0, tr_bb = 0.0, tr_ab = 0.0;
    for (int p = 0; p < naa; p++) {
        tr_aa += Daa[(size_t)p * (size_t)naa + (size_t)p];
        tr_bb += Dbb[(size_t)p * (size_t)naa + (size_t)p];
    }
    for (int p = 0; p < nab; p++) tr_ab += Dab[(size_t)p * (size_t)nab + (size_t)p];

    int r = 0;
    eq_out[r++] = tr_aa;
    eq_out[r++] = tr_bb;
    eq_out[r++] = tr_ab;

    if (ctx->constrain_s2 && na >= 1 && nb >= 1) {
        double sw = 0.0;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                sw += Dab[(size_t)(p * K + q) * (size_t)nab + (size_t)(q * K + p)];
            }
        }
        eq_out[r++] = sw;
    }

    if (na >= 1 && nb >= 1) {
        double *ga = ctx->scr_ga;
        double *gb = ctx->scr_gb;
        mode_b_gamma_fwd(ctx, Daa, Dbb, Dab, ga, gb);

        if (na >= 2) {
            double cna = (double)(na - 1);
            for (int i = 0; i < K; i++) {
                for (int k = i; k < K; k++) {
                    if (i == K - 1 && k == K - 1) continue;
                    double mu = (i == k) ? 1.0 : CHEM_SQRT2;
                    double da = 0.0;
                    for (int m = 0; m < K; m++) {
                        int s1 = 0, s2 = 0;
                        int p = idx_asym_pair(K, i, m, &s1);
                        int q = idx_asym_pair(K, k, m, &s2);
                        if (p >= 0 && q >= 0) {
                            da += (double)(s1 * s2) * Daa[(size_t)p * (size_t)naa + (size_t)q];
                        }
                    }
                    double ga_sym = 0.5 * (ga[(size_t)i * ksz + (size_t)k] +
                                           ga[(size_t)k * ksz + (size_t)i]);
                    eq_out[r++] = mu * (da - cna * ga_sym);
                }
            }
        }
        if (nb >= 2) {
            double cnb = (double)(nb - 1);
            for (int i = 0; i < K; i++) {
                for (int k = i; k < K; k++) {
                    if (i == K - 1 && k == K - 1) continue;
                    double mu = (i == k) ? 1.0 : CHEM_SQRT2;
                    double db = 0.0;
                    for (int m = 0; m < K; m++) {
                        int s1 = 0, s2 = 0;
                        int p = idx_asym_pair(K, i, m, &s1);
                        int q = idx_asym_pair(K, k, m, &s2);
                        if (p >= 0 && q >= 0) {
                            db += (double)(s1 * s2) * Dbb[(size_t)p * (size_t)naa + (size_t)q];
                        }
                    }
                    double gb_sym = 0.5 * (gb[(size_t)i * ksz + (size_t)k] +
                                           gb[(size_t)k * ksz + (size_t)i]);
                    eq_out[r++] = mu * (db - cnb * gb_sym);
                }
            }
        }
    }
}

static void mode_b_Aeq_adj(const ChemMFContext *ctx, const double *lam_in, double *d_out) {
    int K = ctx->K;
    int na = ctx->na;
    int nb = ctx->nb;
    int naa = ctx->naa;
    int nab = ctx->nab;
    size_t ksz = (size_t)K;

    double *Daa_h = ctx->scr_D0;
    double *Dbb_h = ctx->scr_D1;
    double *Dab_h = ctx->scr_D2;
    double *ga_h  = ctx->scr_ga;
    double *gb_h  = ctx->scr_gb;
    if (naa > 0) {
        memset(Daa_h, 0, (size_t)naa * (size_t)naa * sizeof(double));
        memset(Dbb_h, 0, (size_t)naa * (size_t)naa * sizeof(double));
    }
    memset(Dab_h, 0, (size_t)nab * (size_t)nab * sizeof(double));
    memset(ga_h, 0, ksz * ksz * sizeof(double));
    memset(gb_h, 0, ksz * ksz * sizeof(double));

    int r = 0;
    double l_aa = lam_in[r++];
    double l_bb = lam_in[r++];
    double l_ab = lam_in[r++];
    for (int p = 0; p < naa; p++) {
        Daa_h[(size_t)p * (size_t)naa + (size_t)p] += l_aa;
        Dbb_h[(size_t)p * (size_t)naa + (size_t)p] += l_bb;
    }
    for (int p = 0; p < nab; p++) Dab_h[(size_t)p * (size_t)nab + (size_t)p] += l_ab;

    if (ctx->constrain_s2 && na >= 1 && nb >= 1) {
        double l_s2 = lam_in[r++];
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                Dab_h[(size_t)(p * K + q) * (size_t)nab + (size_t)(q * K + p)] += l_s2;
            }
        }
    }

    if (na >= 1 && nb >= 1) {
        if (na >= 2) {
            double cna = (double)(na - 1);
            for (int i = 0; i < K; i++) {
                for (int k = i; k < K; k++) {
                    if (i == K - 1 && k == K - 1) continue;
                    double mu = (i == k) ? 1.0 : CHEM_SQRT2;
                    double w = mu * lam_in[r++];
                    for (int m = 0; m < K; m++) {
                        int s1 = 0, s2 = 0;
                        int p = idx_asym_pair(K, i, m, &s1);
                        int q = idx_asym_pair(K, k, m, &s2);
                        if (p >= 0 && q >= 0) {
                            Daa_h[(size_t)p * (size_t)naa + (size_t)q] += (double)(s1 * s2) * w;
                        }
                    }
                    ga_h[(size_t)i * ksz + (size_t)k] -= 0.5 * cna * w;
                    ga_h[(size_t)k * ksz + (size_t)i] -= 0.5 * cna * w;
                }
            }
        }
        if (nb >= 2) {
            double cnb = (double)(nb - 1);
            for (int i = 0; i < K; i++) {
                for (int k = i; k < K; k++) {
                    if (i == K - 1 && k == K - 1) continue;
                    double mu = (i == k) ? 1.0 : CHEM_SQRT2;
                    double w = mu * lam_in[r++];
                    for (int m = 0; m < K; m++) {
                        int s1 = 0, s2 = 0;
                        int p = idx_asym_pair(K, i, m, &s1);
                        int q = idx_asym_pair(K, k, m, &s2);
                        if (p >= 0 && q >= 0) {
                            Dbb_h[(size_t)p * (size_t)naa + (size_t)q] += (double)(s1 * s2) * w;
                        }
                    }
                    gb_h[(size_t)i * ksz + (size_t)k] -= 0.5 * cnb * w;
                    gb_h[(size_t)k * ksz + (size_t)i] -= 0.5 * cnb * w;
                }
            }
        }
    }

    mode_b_gamma_adj(ctx, ga_h, gb_h, Daa_h, Dbb_h, Dab_h);
    mode_b_pack_D(ctx, Daa_h, Dbb_h, Dab_h, d_out);
}

/* ============================================================================
 * Matrix-Free Spin-Complete Partial 3-RDM T2 Operators (Mode A & Mode B)
 * ============================================================================ */

static inline double t2_get_daa_asym(const ChemMFContext *ctx, const double *Dss,
                                     int a, int b, int c, int d) {
    int pa = ctx->t2_P1[a][b];
    int pc = ctx->t2_P1[c][d];
    if (pa < 0 || pc < 0) return 0.0;
    return ctx->t2_sgn1[a][b] * ctx->t2_sgn1[c][d] *
           Dss[(size_t)pa * (size_t)ctx->t2_naa + (size_t)pc];
}

static inline void t2_add_daa_asym(const ChemMFContext *ctx, double *Dss_h,
                                   int a, int b, int c, int d, double val) {
    int pa = ctx->t2_P1[a][b];
    int pc = ctx->t2_P1[c][d];
    if (pa < 0 || pc < 0) return;
    Dss_h[(size_t)pa * (size_t)ctx->t2_naa + (size_t)pc] +=
        ctx->t2_sgn1[a][b] * ctx->t2_sgn1[c][d] * val;
}

static void t2_core_aab_fwd(const ChemMFContext *ctx, const double *gb,
                            const double *Daa, const double *Dab, double *M) {
    int Ks = ctx->t2_norbs;
    int naa = ctx->t2_naa;
    int nab = ctx->t2_nab;
    int naab = ctx->t2_naab;
    size_t stride = (size_t)naab;
    memset(M, 0, stride * stride * sizeof(double));

    int pairs[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int np = 0;
    for (int p = 0; p < Ks; p++) {
        for (int q = p + 1; q < Ks; q++) {
            pairs[np][0] = p;
            pairs[np][1] = q;
            np++;
        }
    }

    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[pq][0], q = pairs[pq][1];
        /* delta_pu * delta_qv * gb[w, r] */
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                M[(size_t)ri * stride + (size_t)cj] += gb[(size_t)w * (size_t)Ks + (size_t)r];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[uv][0], v = pairs[uv][1];
            double daa_v = Daa[(size_t)pq * (size_t)naa + (size_t)uv];
            for (int r = 0; r < Ks; r++) {
                M[(size_t)(pq * Ks + r) * stride + (size_t)(uv * Ks + r)] += daa_v;
            }
            int m_pu = (p == u), m_qu = (q == u), m_pv = (p == v), m_qv = (q == v);
            if (!(m_pu || m_qu || m_pv || m_qv)) continue;
            for (int r = 0; r < Ks; r++) {
                int ri = pq * Ks + r;
                for (int w = 0; w < Ks; w++) {
                    int cj = uv * Ks + w;
                    double acc = 0.0;
                    if (m_pu) acc -= Dab[(size_t)(q * Ks + w) * (size_t)nab + (size_t)(v * Ks + r)];
                    if (m_qu) acc += Dab[(size_t)(p * Ks + w) * (size_t)nab + (size_t)(v * Ks + r)];
                    if (m_pv) acc += Dab[(size_t)(q * Ks + w) * (size_t)nab + (size_t)(u * Ks + r)];
                    if (m_qv) acc -= Dab[(size_t)(p * Ks + w) * (size_t)nab + (size_t)(u * Ks + r)];
                    M[(size_t)ri * stride + (size_t)cj] += acc;
                }
            }
        }
    }
}

static void t2_core_aab_adj(const ChemMFContext *ctx, const double *Y,
                            double *gb_h, double *Daa_h, double *Dab_h) {
    int Ks = ctx->t2_norbs;
    int naa = ctx->t2_naa;
    int nab = ctx->t2_nab;
    int naab = ctx->t2_naab;
    size_t stride = (size_t)naab;

    int pairs[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int np = 0;
    for (int p = 0; p < Ks; p++) {
        for (int q = p + 1; q < Ks; q++) {
            pairs[np][0] = p;
            pairs[np][1] = q;
            np++;
        }
    }

    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[pq][0], q = pairs[pq][1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                gb_h[(size_t)w * (size_t)Ks + (size_t)r] += Y[(size_t)ri * stride + (size_t)cj];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[uv][0], v = pairs[uv][1];
            double acc_daa = 0.0;
            for (int r = 0; r < Ks; r++) {
                acc_daa += Y[(size_t)(pq * Ks + r) * stride + (size_t)(uv * Ks + r)];
            }
            Daa_h[(size_t)pq * (size_t)naa + (size_t)uv] += acc_daa;

            int m_pu = (p == u), m_qu = (q == u), m_pv = (p == v), m_qv = (q == v);
            if (!(m_pu || m_qu || m_pv || m_qv)) continue;
            for (int r = 0; r < Ks; r++) {
                int ri = pq * Ks + r;
                for (int w = 0; w < Ks; w++) {
                    int cj = uv * Ks + w;
                    double y = Y[(size_t)ri * stride + (size_t)cj];
                    if (m_pu) Dab_h[(size_t)(q * Ks + w) * (size_t)nab + (size_t)(v * Ks + r)] -= y;
                    if (m_qu) Dab_h[(size_t)(p * Ks + w) * (size_t)nab + (size_t)(v * Ks + r)] += y;
                    if (m_pv) Dab_h[(size_t)(q * Ks + w) * (size_t)nab + (size_t)(u * Ks + r)] += y;
                    if (m_qv) Dab_h[(size_t)(p * Ks + w) * (size_t)nab + (size_t)(u * Ks + r)] -= y;
                }
            }
        }
    }
}

static void t2_core_mix_fwd(const ChemMFContext *ctx, const double *ga, const double *gb,
                            const double *Daa, const double *Dbb, const double *Dab, double *M) {
    int Ks = ctx->t2_norbs;
    int naa = ctx->t2_naa;
    int nab = ctx->t2_nab;
    int off = ctx->t2_naab;
    int nmix = ctx->t2_nmix;
    size_t stride = (size_t)nmix;
    memset(M, 0, stride * stride * sizeof(double));

    int pairs[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int np = 0;
    for (int p = 0; p < Ks; p++) {
        for (int q = p + 1; q < Ks; q++) {
            pairs[np][0] = p;
            pairs[np][1] = q;
            np++;
        }
    }

    /* 1. T2_aaa subblock */
    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[pq][0], q = pairs[pq][1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                M[(size_t)ri * stride + (size_t)cj] += ga[(size_t)w * (size_t)Ks + (size_t)r];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[uv][0], v = pairs[uv][1];
            double daa_v = Daa[(size_t)pq * (size_t)naa + (size_t)uv];
            for (int r = 0; r < Ks; r++) {
                M[(size_t)(pq * Ks + r) * stride + (size_t)(uv * Ks + r)] += daa_v;
            }
            int m_pu = (p == u), m_qu = (q == u), m_pv = (p == v), m_qv = (q == v);
            if (!(m_pu || m_qu || m_pv || m_qv)) continue;
            for (int r = 0; r < Ks; r++) {
                int ri = pq * Ks + r;
                for (int w = 0; w < Ks; w++) {
                    int cj = uv * Ks + w;
                    double acc = 0.0;
                    if (m_pu) acc -= t2_get_daa_asym(ctx, Daa, q, w, v, r);
                    if (m_qu) acc += t2_get_daa_asym(ctx, Daa, p, w, v, r);
                    if (m_pv) acc += t2_get_daa_asym(ctx, Daa, q, w, u, r);
                    if (m_qv) acc -= t2_get_daa_asym(ctx, Daa, p, w, u, r);
                    M[(size_t)ri * stride + (size_t)cj] += acc;
                }
            }
        }
    }

    /* 2. T2_abb subblock: O(Ks^5) sparse delta structure */
    for (int p = 0; p < Ks; p++) {
        for (int q = 0; q < Ks; q++) {
            /* pu && qv */
            for (int r = 0; r < Ks; r++) {
                int ri = off + (p * Ks + q) * Ks + r;
                for (int w = 0; w < Ks; w++) {
                    int cj = off + (p * Ks + q) * Ks + w;
                    M[(size_t)ri * stride + (size_t)cj] += gb[(size_t)w * (size_t)Ks + (size_t)r];
                }
            }
            /* r == w: + Dab[(p,q),(u,v)] */
            for (int u = 0; u < Ks; u++) {
                for (int v = 0; v < Ks; v++) {
                    double dv = Dab[(size_t)(p * Ks + q) * (size_t)nab + (size_t)(u * Ks + v)];
                    for (int r = 0; r < Ks; r++) {
                        int ri = off + (p * Ks + q) * Ks + r;
                        int cj = off + (u * Ks + v) * Ks + r;
                        M[(size_t)ri * stride + (size_t)cj] += dv;
                    }
                }
            }
            /* p == u: - Dbb[(w,q),(r,v)] */
            for (int r = 0; r < Ks; r++) {
                int ri = off + (p * Ks + q) * Ks + r;
                for (int v = 0; v < Ks; v++) {
                    for (int w = 0; w < Ks; w++) {
                        int cj = off + (p * Ks + v) * Ks + w;
                        M[(size_t)ri * stride + (size_t)cj] -= t2_get_daa_asym(ctx, Dbb, w, q, r, v);
                    }
                }
            }
            /* q == v: - Dab[(p,w),(u,r)] */
            for (int r = 0; r < Ks; r++) {
                int ri = off + (p * Ks + q) * Ks + r;
                for (int u = 0; u < Ks; u++) {
                    for (int w = 0; w < Ks; w++) {
                        int cj = off + (u * Ks + q) * Ks + w;
                        M[(size_t)ri * stride + (size_t)cj] -=
                            Dab[(size_t)(p * Ks + w) * (size_t)nab + (size_t)(u * Ks + r)];
                    }
                }
            }
        }
    }

    /* 3. Symmetric T2_cross subblock between (p<q, r) and (u, v, w) */
    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[pq][0], q = pairs[pq][1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int v = 0; v < Ks; v++) {
                for (int w = 0; w < Ks; w++) {
                    int cjp = off + (p * Ks + v) * Ks + w;
                    double vp = Dab[(size_t)(q * Ks + w) * (size_t)nab + (size_t)(r * Ks + v)];
                    M[(size_t)ri * stride + (size_t)cjp] += vp;
                    M[(size_t)cjp * stride + (size_t)ri] += vp;

                    int cjq = off + (q * Ks + v) * Ks + w;
                    double vq = Dab[(size_t)(p * Ks + w) * (size_t)nab + (size_t)(r * Ks + v)];
                    M[(size_t)ri * stride + (size_t)cjq] -= vq;
                    M[(size_t)cjq * stride + (size_t)ri] -= vq;
                }
            }
        }
    }
}

static void t2_core_mix_adj(const ChemMFContext *ctx, const double *Y,
                            double *ga_h, double *gb_h,
                            double *Daa_h, double *Dbb_h, double *Dab_h) {
    int Ks = ctx->t2_norbs;
    int naa = ctx->t2_naa;
    int nab = ctx->t2_nab;
    int off = ctx->t2_naab;
    int nmix = ctx->t2_nmix;
    size_t stride = (size_t)nmix;

    int pairs[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int np = 0;
    for (int p = 0; p < Ks; p++) {
        for (int q = p + 1; q < Ks; q++) {
            pairs[np][0] = p;
            pairs[np][1] = q;
            np++;
        }
    }

    /* 1. Adjoint of T2_aaa */
    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[pq][0], q = pairs[pq][1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                ga_h[(size_t)w * (size_t)Ks + (size_t)r] += Y[(size_t)ri * stride + (size_t)cj];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[uv][0], v = pairs[uv][1];
            double acc_daa = 0.0;
            for (int r = 0; r < Ks; r++) {
                acc_daa += Y[(size_t)(pq * Ks + r) * stride + (size_t)(uv * Ks + r)];
            }
            Daa_h[(size_t)pq * (size_t)naa + (size_t)uv] += acc_daa;

            int m_pu = (p == u), m_qu = (q == u), m_pv = (p == v), m_qv = (q == v);
            if (!(m_pu || m_qu || m_pv || m_qv)) continue;
            for (int r = 0; r < Ks; r++) {
                int ri = pq * Ks + r;
                for (int w = 0; w < Ks; w++) {
                    int cj = uv * Ks + w;
                    double y = Y[(size_t)ri * stride + (size_t)cj];
                    if (m_pu) t2_add_daa_asym(ctx, Daa_h, q, w, v, r, -y);
                    if (m_qu) t2_add_daa_asym(ctx, Daa_h, p, w, v, r, y);
                    if (m_pv) t2_add_daa_asym(ctx, Daa_h, q, w, u, r, y);
                    if (m_qv) t2_add_daa_asym(ctx, Daa_h, p, w, u, r, -y);
                }
            }
        }
    }

    /* 2. Adjoint of T2_abb */
    for (int p = 0; p < Ks; p++) {
        for (int q = 0; q < Ks; q++) {
            for (int r = 0; r < Ks; r++) {
                int ri = off + (p * Ks + q) * Ks + r;
                for (int w = 0; w < Ks; w++) {
                    int cj = off + (p * Ks + q) * Ks + w;
                    gb_h[(size_t)w * (size_t)Ks + (size_t)r] += Y[(size_t)ri * stride + (size_t)cj];
                }
            }
            for (int u = 0; u < Ks; u++) {
                for (int v = 0; v < Ks; v++) {
                    double acc = 0.0;
                    for (int r = 0; r < Ks; r++) {
                        int ri = off + (p * Ks + q) * Ks + r;
                        int cj = off + (u * Ks + v) * Ks + r;
                        acc += Y[(size_t)ri * stride + (size_t)cj];
                    }
                    Dab_h[(size_t)(p * Ks + q) * (size_t)nab + (size_t)(u * Ks + v)] += acc;
                }
            }
            for (int r = 0; r < Ks; r++) {
                int ri = off + (p * Ks + q) * Ks + r;
                for (int v = 0; v < Ks; v++) {
                    for (int w = 0; w < Ks; w++) {
                        int cj = off + (p * Ks + v) * Ks + w;
                        t2_add_daa_asym(ctx, Dbb_h, w, q, r, v,
                                        -Y[(size_t)ri * stride + (size_t)cj]);
                    }
                }
            }
            for (int r = 0; r < Ks; r++) {
                int ri = off + (p * Ks + q) * Ks + r;
                for (int u = 0; u < Ks; u++) {
                    for (int w = 0; w < Ks; w++) {
                        int cj = off + (u * Ks + q) * Ks + w;
                        Dab_h[(size_t)(p * Ks + w) * (size_t)nab + (size_t)(u * Ks + r)] -=
                            Y[(size_t)ri * stride + (size_t)cj];
                    }
                }
            }
        }
    }

    /* 3. Adjoint of symmetric T2_cross (factor 2 from upper-right + lower-left mirror) */
    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[pq][0], q = pairs[pq][1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int v = 0; v < Ks; v++) {
                for (int w = 0; w < Ks; w++) {
                    int cjp = off + (p * Ks + v) * Ks + w;
                    double yp = Y[(size_t)ri * stride + (size_t)cjp] +
                                Y[(size_t)cjp * stride + (size_t)ri];
                    Dab_h[(size_t)(q * Ks + w) * (size_t)nab + (size_t)(r * Ks + v)] += yp;

                    int cjq = off + (q * Ks + v) * Ks + w;
                    double yq = Y[(size_t)ri * stride + (size_t)cjq] +
                                Y[(size_t)cjq * stride + (size_t)ri];
                    Dab_h[(size_t)(p * Ks + w) * (size_t)nab + (size_t)(r * Ks + v)] -= yq;
                }
            }
        }
    }
}

static void mf_LT2_fwd(const ChemMFContext *ctx, const double *d_in, double *t2_out) {
    if (!ctx->t2_enabled) return;
    int K = ctx->K;
    int Ks = ctx->t2_norbs;
    int naa_s = ctx->t2_naa;
    int nab_s = ctx->t2_nab;
    size_t ksz = (size_t)K;
    size_t kssz = (size_t)Ks;

    double *ga_s = ctx->scr_t2_ga;
    double *gb_s = ctx->scr_t2_gb;
    double *Daa_s = ctx->scr_t2_Daa;
    double *Dbb_s = ctx->scr_t2_Dbb;
    double *Dab_s = ctx->scr_t2_Dab;
    double *Mat = ctx->scr_t2_Mat;

    if (ctx->mode == CHEM_MODE_A_SINGLET) {
        mode_a_expand_st_tt(ctx, d_in, ctx->scr_St, ctx->scr_Tt);
        mode_a_gamma_from_st(K, ctx->na, ctx->scr_St, ctx->scr_ga);
        const double *St = ctx->scr_St;
        const double *Tt = ctx->scr_Tt;
        const double *ga = ctx->scr_ga;
        const double *D1 = ctx->scr_D1;
        int m1 = ctx->m1;

        for (int a = 0; a < Ks; a++) {
            int oa = ctx->t2_orbs[a];
            for (int b = 0; b < Ks; b++) {
                int ob = ctx->t2_orbs[b];
                double gv = ga[(size_t)oa * ksz + (size_t)ob];
                ga_s[(size_t)a * kssz + (size_t)b] = gv;
                gb_s[(size_t)a * kssz + (size_t)b] = gv;
            }
        }
        for (int ab = 0; ab < naa_s; ab++) {
            int a = -1, b = -1;
            for (int i = 0; i < Ks && a < 0; i++) {
                for (int j = i + 1; j < Ks; j++) {
                    if (ctx->t2_P1[i][j] == ab) { a = i; b = j; break; }
                }
            }
            int oa = ctx->t2_orbs[a], ob = ctx->t2_orbs[b];
            int Pab = ctx->P1[oa][ob];
            double sab = ctx->sgn1[oa][ob];
            for (int cd = 0; cd < naa_s; cd++) {
                int c = -1, d = -1;
                for (int i = 0; i < Ks && c < 0; i++) {
                    for (int j = i + 1; j < Ks; j++) {
                        if (ctx->t2_P1[i][j] == cd) { c = i; d = j; break; }
                    }
                }
                int oc = ctx->t2_orbs[c], od = ctx->t2_orbs[d];
                int Pcd = ctx->P1[oc][od];
                double scd = ctx->sgn1[oc][od];
                double v = (m1 > 0 && Pab >= 0 && Pcd >= 0)
                               ? (sab * scd * D1[(size_t)Pab * (size_t)m1 + (size_t)Pcd])
                               : 0.0;
                Daa_s[(size_t)ab * (size_t)naa_s + (size_t)cd] = v;
                Dbb_s[(size_t)ab * (size_t)naa_s + (size_t)cd] = v;
            }
        }
        for (int a = 0; a < Ks; a++) {
            int oa = ctx->t2_orbs[a];
            for (int b = 0; b < Ks; b++) {
                int ob = ctx->t2_orbs[b];
                for (int c = 0; c < Ks; c++) {
                    int oc = ctx->t2_orbs[c];
                    for (int d = 0; d < Ks; d++) {
                        int od = ctx->t2_orbs[d];
                        size_t off4 = ((((size_t)oa * ksz + (size_t)ob) * ksz + (size_t)oc) * ksz +
                                       (size_t)od);
                        Dab_s[(size_t)(a * Ks + b) * (size_t)nab_s + (size_t)(c * Ks + d)] =
                            St[off4] + Tt[off4];
                    }
                }
            }
        }
        t2_core_aab_fwd(ctx, gb_s, Daa_s, Dab_s, Mat);
        svec_pack_general(ctx->t2_naab, Mat, t2_out);
        t2_core_mix_fwd(ctx, ga_s, gb_s, Daa_s, Dbb_s, Dab_s, Mat);
        svec_pack_general(ctx->t2_nmix, Mat, t2_out + ctx->block_szs[6]);
        return;
    }

    /* Mode B */
    int naa = ctx->naa;
    int nab = ctx->nab;
    double *Daa = ctx->scr_D0;
    double *Dbb = ctx->scr_D1;
    double *Dab = ctx->scr_D2;
    double *ga = ctx->scr_ga;
    double *gb = ctx->scr_gb;
    mode_b_unpack_D(ctx, d_in, Daa, Dbb, Dab);
    mode_b_gamma_fwd(ctx, Daa, Dbb, Dab, ga, gb);

    for (int a = 0; a < Ks; a++) {
        int oa = ctx->t2_orbs[a];
        for (int b = 0; b < Ks; b++) {
            int ob = ctx->t2_orbs[b];
            ga_s[(size_t)a * kssz + (size_t)b] = ga[(size_t)oa * ksz + (size_t)ob];
            gb_s[(size_t)a * kssz + (size_t)b] = gb[(size_t)oa * ksz + (size_t)ob];
        }
    }
    int s_pairs[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int snp = 0;
    for (int i = 0; i < Ks; i++) {
        for (int j = i + 1; j < Ks; j++) {
            s_pairs[snp][0] = i;
            s_pairs[snp][1] = j;
            snp++;
        }
    }
    for (int ab = 0; ab < naa_s; ab++) {
        int oa = ctx->t2_orbs[s_pairs[ab][0]], ob = ctx->t2_orbs[s_pairs[ab][1]];
        int Pab = ctx->P1[oa][ob];
        double sab = ctx->sgn1[oa][ob];
        for (int cd = 0; cd < naa_s; cd++) {
            int oc = ctx->t2_orbs[s_pairs[cd][0]], od = ctx->t2_orbs[s_pairs[cd][1]];
            int Pcd = ctx->P1[oc][od];
            double sgn = sab * ctx->sgn1[oc][od];
            Daa_s[(size_t)ab * (size_t)naa_s + (size_t)cd] =
                (naa > 0 && Pab >= 0 && Pcd >= 0) ? sgn * Daa[(size_t)Pab * (size_t)naa + (size_t)Pcd] : 0.0;
            Dbb_s[(size_t)ab * (size_t)naa_s + (size_t)cd] =
                (naa > 0 && Pab >= 0 && Pcd >= 0) ? sgn * Dbb[(size_t)Pab * (size_t)naa + (size_t)Pcd] : 0.0;
        }
    }
    double *Dab_sw = ctx->scr_t2_Dab_sw;
    for (int a = 0; a < Ks; a++) {
        int oa = ctx->t2_orbs[a];
        for (int b = 0; b < Ks; b++) {
            int ob = ctx->t2_orbs[b];
            for (int c = 0; c < Ks; c++) {
                int oc = ctx->t2_orbs[c];
                for (int d = 0; d < Ks; d++) {
                    int od = ctx->t2_orbs[d];
                    double v = Dab[(size_t)(oa * K + ob) * (size_t)nab + (size_t)(oc * K + od)];
                    Dab_s[(size_t)(a * Ks + b) * (size_t)nab_s + (size_t)(c * Ks + d)] = v;
                    Dab_sw[(size_t)(b * Ks + a) * (size_t)nab_s + (size_t)(d * Ks + c)] = v;
                }
            }
        }
    }

    int off = 0;
    t2_core_aab_fwd(ctx, gb_s, Daa_s, Dab_s, Mat);
    svec_pack_general(ctx->t2_naab, Mat, t2_out + off);
    off += ctx->block_szs[9];

    t2_core_mix_fwd(ctx, ga_s, gb_s, Daa_s, Dbb_s, Dab_s, Mat);
    svec_pack_general(ctx->t2_nmix, Mat, t2_out + off);
    off += ctx->block_szs[10];

    t2_core_aab_fwd(ctx, ga_s, Dbb_s, Dab_sw, Mat);
    svec_pack_general(ctx->t2_naab, Mat, t2_out + off);
    off += ctx->block_szs[11];

    t2_core_mix_fwd(ctx, gb_s, ga_s, Dbb_s, Daa_s, Dab_sw, Mat);
    svec_pack_general(ctx->t2_nmix, Mat, t2_out + off);
}

static void mf_LT2_adj(const ChemMFContext *ctx, const double *yt2_in, double *d_out) {
    if (!ctx->t2_enabled) return;
    int K = ctx->K;
    int Ks = ctx->t2_norbs;
    int naa_s = ctx->t2_naa;
    int nab_s = ctx->t2_nab;
    size_t ksz = (size_t)K;
    size_t kssz = (size_t)Ks;

    double *ga_s_h = ctx->scr_t2_ga;
    double *gb_s_h = ctx->scr_t2_gb;
    double *Daa_s_h = ctx->scr_t2_Daa;
    double *Dbb_s_h = ctx->scr_t2_Dbb;
    double *Dab_s_h = ctx->scr_t2_Dab;
    double *Dab_sw_h = ctx->scr_t2_Dab_sw;
    double *Mat = ctx->scr_t2_Mat;

    memset(ga_s_h, 0, kssz * kssz * sizeof(double));
    memset(gb_s_h, 0, kssz * kssz * sizeof(double));
    memset(Daa_s_h, 0, (size_t)naa_s * (size_t)naa_s * sizeof(double));
    memset(Dbb_s_h, 0, (size_t)naa_s * (size_t)naa_s * sizeof(double));
    memset(Dab_s_h, 0, (size_t)nab_s * (size_t)nab_s * sizeof(double));

    int s_pairs[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    int snp = 0;
    for (int i = 0; i < Ks; i++) {
        for (int j = i + 1; j < Ks; j++) {
            s_pairs[snp][0] = i;
            s_pairs[snp][1] = j;
            snp++;
        }
    }

    if (ctx->mode == CHEM_MODE_A_SINGLET) {
        svec_unpack(ctx->t2_naab, yt2_in, Mat);
        t2_core_aab_adj(ctx, Mat, gb_s_h, Daa_s_h, Dab_s_h);

        svec_unpack(ctx->t2_nmix, yt2_in + ctx->block_szs[6], Mat);
        t2_core_mix_adj(ctx, Mat, ga_s_h, gb_s_h, Daa_s_h, Dbb_s_h, Dab_s_h);

        size_t k4 = ksz * ksz * ksz * ksz;
        double *Sh = ctx->scr_St;
        double *Th = ctx->scr_Tt;
        double *ga_h = ctx->scr_ga;
        memset(Sh, 0, k4 * sizeof(double));
        memset(Th, 0, k4 * sizeof(double));
        memset(ga_h, 0, ksz * ksz * sizeof(double));

        for (int a = 0; a < Ks; a++) {
            int oa = ctx->t2_orbs[a];
            for (int b = 0; b < Ks; b++) {
                int ob = ctx->t2_orbs[b];
                ga_h[(size_t)oa * ksz + (size_t)ob] +=
                    ga_s_h[(size_t)a * kssz + (size_t)b] + gb_s_h[(size_t)a * kssz + (size_t)b];
            }
        }
        mode_a_gamma_adjoint_to_sh(K, ctx->na, ga_h, Sh);

        for (int a = 0; a < Ks; a++) {
            int oa = ctx->t2_orbs[a];
            for (int b = 0; b < Ks; b++) {
                int ob = ctx->t2_orbs[b];
                for (int c = 0; c < Ks; c++) {
                    int oc = ctx->t2_orbs[c];
                    for (int d = 0; d < Ks; d++) {
                        int od = ctx->t2_orbs[d];
                        double v = Dab_s_h[(size_t)(a * Ks + b) * (size_t)nab_s +
                                           (size_t)(c * Ks + d)];
                        size_t off4 = ((((size_t)oa * ksz + (size_t)ob) * ksz + (size_t)oc) * ksz +
                                       (size_t)od);
                        Sh[off4] += v;
                        Th[off4] += v;
                    }
                }
            }
        }

        /* Contract Sh, Th into D0h, D1h + direct D1_h scatter before svec packing */
        int m0 = ctx->m0;
        int m1 = ctx->m1;
        double *D0h = ctx->scr_D0;
        double *D1h = ctx->scr_D1;
        memset(D0h, 0, (size_t)m0 * (size_t)m0 * sizeof(double));
        if (m1 > 0) memset(D1h, 0, (size_t)m1 * (size_t)m1 * sizeof(double));
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                double epq = (p == q) ? 1.0 : CHEM_ISQRT2;
                int P = ctx->P0[p][q];
                int P1 = ctx->P1[p][q];
                double zpq = ctx->sgn1[p][q] * CHEM_ISQRT2;
                for (int r = 0; r < K; r++) {
                    for (int s = 0; s < K; s++) {
                        double ers = (r == s) ? 1.0 : CHEM_ISQRT2;
                        int R = ctx->P0[r][s];
                        size_t off4 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz +
                                       (size_t)s);
                        D0h[(size_t)P * (size_t)m0 + (size_t)R] += epq * ers * Sh[off4];
                        if (P1 >= 0) {
                            int R1 = ctx->P1[r][s];
                            if (R1 >= 0) {
                                double zrs = ctx->sgn1[r][s] * CHEM_ISQRT2;
                                D1h[(size_t)P1 * (size_t)m1 + (size_t)R1] += zpq * zrs * Th[off4];
                            }
                        }
                    }
                }
            }
        }
        if (m1 > 0) {
            for (int ab = 0; ab < naa_s; ab++) {
                int oa = ctx->t2_orbs[s_pairs[ab][0]], ob = ctx->t2_orbs[s_pairs[ab][1]];
                int Pab = ctx->P1[oa][ob];
                double sab = ctx->sgn1[oa][ob];
                for (int cd = 0; cd < naa_s; cd++) {
                    int oc = ctx->t2_orbs[s_pairs[cd][0]], od = ctx->t2_orbs[s_pairs[cd][1]];
                    int Pcd = ctx->P1[oc][od];
                    if (Pab >= 0 && Pcd >= 0) {
                        double sgn = sab * ctx->sgn1[oc][od];
                        double v = Daa_s_h[(size_t)ab * (size_t)naa_s + (size_t)cd] +
                                   Dbb_s_h[(size_t)ab * (size_t)naa_s + (size_t)cd];
                        D1h[(size_t)Pab * (size_t)m1 + (size_t)Pcd] += sgn * v;
                    }
                }
            }
        }
        svec_pack_general(m0, D0h, d_out);
        if (m1 > 0) svec_pack_general(m1, D1h, d_out + ctx->block_szs[0]);
        return;
    }

    /* Mode B: blocks 9 (aab), 10 (mix_a), 11 (bba), 12 (mix_b) */
    memset(Dab_sw_h, 0, (size_t)nab_s * (size_t)nab_s * sizeof(double));
    int off = 0;
    svec_unpack(ctx->t2_naab, yt2_in + off, Mat);
    t2_core_aab_adj(ctx, Mat, gb_s_h, Daa_s_h, Dab_s_h);
    off += ctx->block_szs[9];

    svec_unpack(ctx->t2_nmix, yt2_in + off, Mat);
    t2_core_mix_adj(ctx, Mat, ga_s_h, gb_s_h, Daa_s_h, Dbb_s_h, Dab_s_h);
    off += ctx->block_szs[10];

    svec_unpack(ctx->t2_naab, yt2_in + off, Mat);
    t2_core_aab_adj(ctx, Mat, ga_s_h, Dbb_s_h, Dab_sw_h);
    off += ctx->block_szs[11];

    svec_unpack(ctx->t2_nmix, yt2_in + off, Mat);
    t2_core_mix_adj(ctx, Mat, gb_s_h, ga_s_h, Dbb_s_h, Daa_s_h, Dab_sw_h);

    int naa = ctx->naa;
    int nab = ctx->nab;
    double *Daa_h = ctx->scr_D0;
    double *Dbb_h = ctx->scr_D1;
    double *Dab_h = ctx->scr_D2;
    double *ga_h = ctx->scr_ga;
    double *gb_h = ctx->scr_gb;
    if (naa > 0) {
        memset(Daa_h, 0, (size_t)naa * (size_t)naa * sizeof(double));
        memset(Dbb_h, 0, (size_t)naa * (size_t)naa * sizeof(double));
    }
    memset(Dab_h, 0, (size_t)nab * (size_t)nab * sizeof(double));
    memset(ga_h, 0, ksz * ksz * sizeof(double));
    memset(gb_h, 0, ksz * ksz * sizeof(double));

    for (int a = 0; a < Ks; a++) {
        int oa = ctx->t2_orbs[a];
        for (int b = 0; b < Ks; b++) {
            int ob = ctx->t2_orbs[b];
            ga_h[(size_t)oa * ksz + (size_t)ob] = ga_s_h[(size_t)a * kssz + (size_t)b];
            gb_h[(size_t)oa * ksz + (size_t)ob] = gb_s_h[(size_t)a * kssz + (size_t)b];
        }
    }
    if (naa > 0) {
        for (int ab = 0; ab < naa_s; ab++) {
            int oa = ctx->t2_orbs[s_pairs[ab][0]], ob = ctx->t2_orbs[s_pairs[ab][1]];
            int Pab = ctx->P1[oa][ob];
            double sab = ctx->sgn1[oa][ob];
            for (int cd = 0; cd < naa_s; cd++) {
                int oc = ctx->t2_orbs[s_pairs[cd][0]], od = ctx->t2_orbs[s_pairs[cd][1]];
                int Pcd = ctx->P1[oc][od];
                if (Pab >= 0 && Pcd >= 0) {
                    double sgn = sab * ctx->sgn1[oc][od];
                    Daa_h[(size_t)Pab * (size_t)naa + (size_t)Pcd] +=
                        sgn * Daa_s_h[(size_t)ab * (size_t)naa_s + (size_t)cd];
                    Dbb_h[(size_t)Pab * (size_t)naa + (size_t)Pcd] +=
                        sgn * Dbb_s_h[(size_t)ab * (size_t)naa_s + (size_t)cd];
                }
            }
        }
    }
    for (int a = 0; a < Ks; a++) {
        int oa = ctx->t2_orbs[a];
        for (int b = 0; b < Ks; b++) {
            int ob = ctx->t2_orbs[b];
            for (int c = 0; c < Ks; c++) {
                int oc = ctx->t2_orbs[c];
                for (int d = 0; d < Ks; d++) {
                    int od = ctx->t2_orbs[d];
                    Dab_h[(size_t)(oa * K + ob) * (size_t)nab + (size_t)(oc * K + od)] +=
                        Dab_s_h[(size_t)(a * Ks + b) * (size_t)nab_s + (size_t)(c * Ks + d)] +
                        Dab_sw_h[(size_t)(b * Ks + a) * (size_t)nab_s + (size_t)(d * Ks + c)];
                }
            }
        }
    }

    mode_b_gamma_adj(ctx, ga_h, gb_h, Daa_h, Dbb_h, Dab_h);
    mode_b_pack_D(ctx, Daa_h, Dbb_h, Dab_h, d_out);
}

/* ============================================================================
 * Unified Dispatch & Context Setup
 * ============================================================================ */

static void mf_LQ_fwd(const ChemMFContext *c, const double *d, double *q) {
    if (c->mode == CHEM_MODE_A_SINGLET) mode_a_LQ_fwd(c, d, q);
    else mode_b_LQ_fwd(c, d, q);
}
static void mf_LQ_adj(const ChemMFContext *c, const double *yq, double *d) {
    if (c->mode == CHEM_MODE_A_SINGLET) mode_a_LQ_adj(c, yq, d);
    else mode_b_LQ_adj(c, yq, d);
}
static void mf_LG_fwd(const ChemMFContext *c, const double *d, double *g) {
    if (c->mode == CHEM_MODE_A_SINGLET) mode_a_LG_fwd(c, d, g);
    else mode_b_LG_fwd(c, d, g);
}
static void mf_LG_adj(const ChemMFContext *c, const double *yg, double *d) {
    if (c->mode == CHEM_MODE_A_SINGLET) mode_a_LG_adj(c, yg, d);
    else mode_b_LG_adj(c, yg, d);
}
static void mf_Aeq_fwd(const ChemMFContext *c, const double *d, double *eq) {
    if (c->mode == CHEM_MODE_A_SINGLET) mode_a_Aeq_fwd(c, d, eq);
    else mode_b_Aeq_fwd(c, d, eq);
}
static void mf_Aeq_adj(const ChemMFContext *c, const double *lam, double *d) {
    if (c->mode == CHEM_MODE_A_SINGLET) mode_a_Aeq_adj(c, lam, d);
    else mode_b_Aeq_adj(c, lam, d);
}

static void mf_build_hamiltonian_cost(const ChemMFContext *ctx, const ChemHam *ham, double *C_D) {
    int K = ctx->K;
    size_t ksz = (size_t)K;

    if (ctx->mode == CHEM_MODE_A_SINGLET) {
        size_t k4 = ksz * ksz * ksz * ksz;
        double *Sh = ctx->scr_St;
        double *Th = ctx->scr_Tt;
        double *ga_h = ctx->scr_ga;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                for (int r = 0; r < K; r++) {
                    for (int s = 0; s < K; s++) {
                        double e = chem_ham_eri(ham, p, r, q, s);
                        size_t off4 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz +
                                       (size_t)s);
                        Sh[off4] = e;
                        Th[off4] = 3.0 * e;
                    }
                }
            }
        }
        for (int p = 0; p < K; p++) {
            for (int r = 0; r < K; r++) {
                ga_h[(size_t)p * ksz + (size_t)r] = 2.0 * chem_ham_h1(ham, p, r);
            }
        }
        mode_a_gamma_adjoint_to_sh(K, ctx->na, ga_h, Sh);
        mode_a_contract_sh_th(ctx, Sh, Th, C_D);
        (void)k4;
        return;
    }

    /* Mode B cost */
    int naa = ctx->naa;
    int nab = ctx->nab;
    double *Daa_h = ctx->scr_D0;
    double *Dbb_h = ctx->scr_D1;
    double *Dab_h = ctx->scr_D2;
    double *ga_h  = ctx->scr_ga;
    double *gb_h  = ctx->scr_gb;

    for (int i = 0; i < K; i++) {
        for (int k = 0; k < K; k++) {
            double h_ik = chem_ham_h1(ham, i, k);
            ga_h[(size_t)i * ksz + (size_t)k] = h_ik;
            gb_h[(size_t)i * ksz + (size_t)k] = h_ik;
        }
    }

    int p_idx = 0;
    int pair_ij[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
    for (int i = 0; i < K; i++) {
        for (int j = i + 1; j < K; j++) {
            pair_ij[p_idx][0] = i;
            pair_ij[p_idx][1] = j;
            p_idx++;
        }
    }
    for (int p = 0; p < naa; p++) {
        int i = pair_ij[p][0], j = pair_ij[p][1];
        for (int q = 0; q < naa; q++) {
            int k = pair_ij[q][0], l = pair_ij[q][1];
            double val = chem_ham_eri(ham, i, k, j, l) - chem_ham_eri(ham, i, l, j, k);
            Daa_h[(size_t)p * (size_t)naa + (size_t)q] = val;
            Dbb_h[(size_t)p * (size_t)naa + (size_t)q] = val;
        }
    }
    for (int p = 0; p < nab; p++) {
        int i = p / K, j = p % K;
        for (int q = 0; q < nab; q++) {
            int k = q / K, l = q % K;
            Dab_h[(size_t)p * (size_t)nab + (size_t)q] = chem_ham_eri(ham, i, k, j, l);
        }
    }

    mode_b_gamma_adj(ctx, ga_h, gb_h, Daa_h, Dbb_h, Dab_h);
    mode_b_pack_D(ctx, Daa_h, Dbb_h, Dab_h, C_D);
}

static int cmp_int_asc(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);
}

static int mf_ctx_init_ext(ChemMFContext *ctx, int K, int na, int nb, int two_s, int constrain_s2,
                           ChemMode mode, const ChemHam *ham_opt,
                           int enable_t2, int t2_max_orbs, int t2_norbs_in, const int *t2_orbs_in,
                           int num_threads, char *err, size_t err_sz) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->mode = mode;
    ctx->K = K;
    ctx->na = na;
    ctx->nb = nb;
    ctx->two_s = two_s;
    ctx->constrain_s2 = constrain_s2;

    if (enable_t2 && K >= 2) {
        int cap = (t2_max_orbs > 0) ? t2_max_orbs : 6;
        if (cap > K) cap = K;
        if (cap > CHEM_MAX_ORBITALS) cap = CHEM_MAX_ORBITALS;
        int Ks = 0;
        if (t2_norbs_in > 0 && t2_orbs_in) {
            Ks = (t2_norbs_in > K) ? K : t2_norbs_in;
            for (int i = 0; i < Ks; i++) {
                if (t2_orbs_in[i] < 0 || t2_orbs_in[i] >= K) {
                    set_err(err, err_sz, "t2_orbs[%d]=%d out of range [0,%d)", i, t2_orbs_in[i], K);
                    return -1;
                }
                ctx->t2_orbs[i] = t2_orbs_in[i];
            }
            qsort(ctx->t2_orbs, (size_t)Ks, sizeof(int), cmp_int_asc);
            for (int i = 1; i < Ks; i++) {
                if (ctx->t2_orbs[i] == ctx->t2_orbs[i - 1]) {
                    set_err(err, err_sz, "duplicate orbital %d in t2_orbs", ctx->t2_orbs[i]);
                    return -1;
                }
            }
        } else {
            Ks = cap;
            int center = (na > 0) ? (na - 1) : (K / 2 - 1);
            int half = Ks / 2;
            int start = center - half + 1;
            if (start < 0) start = 0;
            if (start + Ks > K) start = K - Ks;
            for (int i = 0; i < Ks; i++) ctx->t2_orbs[i] = start + i;
        }
        ctx->t2_enabled = 1;
        ctx->t2_norbs = Ks;
        ctx->t2_naa = Ks * (Ks - 1) / 2;
        ctx->t2_nab = Ks * Ks;
        ctx->t2_naab = ctx->t2_naa * Ks;
        ctx->t2_nabb = Ks * Ks * Ks;
        ctx->t2_nmix = ctx->t2_naab + ctx->t2_nabb;
        for (int a = 0; a < Ks; a++) {
            for (int b = 0; b < Ks; b++) {
                int sg = 0;
                ctx->t2_P1[a][b] = idx_asym_pair(Ks, a, b, &sg);
                ctx->t2_sgn1[a][b] = (double)sg;
            }
        }
    }

    double tr_aab_full = 0.5 * (double)(K * (K - 1)) * (double)nb +
                         (double)K * 0.5 * (double)(na * (na - 1)) -
                         (double)(K - 1) * (double)(na * nb);
    double tr_mix_a_full = 0.5 * (double)(K * (K - 1)) * (double)na -
                           0.5 * (double)(K - 2) * (double)(na * (na - 1)) +
                           (double)(K * nb * (K - nb + 1));
    double tr_bba_full = 0.5 * (double)(K * (K - 1)) * (double)na +
                         (double)K * 0.5 * (double)(nb * (nb - 1)) -
                         (double)(K - 1) * (double)(na * nb);
    double tr_mix_b_full = 0.5 * (double)(K * (K - 1)) * (double)nb -
                           0.5 * (double)(K - 2) * (double)(nb * (nb - 1)) +
                           (double)(K * na * (K - na + 1));
    if (tr_aab_full < 0.0) tr_aab_full = 0.0;
    if (tr_mix_a_full < 0.0) tr_mix_a_full = 0.0;
    if (tr_bba_full < 0.0) tr_bba_full = 0.0;
    if (tr_mix_b_full < 0.0) tr_mix_b_full = 0.0;

    if (mode == CHEM_MODE_A_SINGLET) {
        if (na != nb) {
            set_err(err, err_sz, "Mode A requires na == nb (got %d != %d)", na, nb);
            return -1;
        }
        int ne = na;
        ctx->nblocks = ctx->t2_enabled ? 8 : 6;
        ctx->m0 = K * (K + 1) / 2;
        ctx->m1 = K * (K - 1) / 2;
        ctx->k2 = K * K;
        ctx->block_dims[0] = ctx->m0;
        ctx->block_dims[1] = ctx->m1;
        ctx->block_dims[2] = ctx->m0;
        ctx->block_dims[3] = ctx->m1;
        ctx->block_dims[4] = ctx->k2;
        ctx->block_dims[5] = ctx->k2;

        ctx->block_traces[0] = 0.5 * (double)(ne * (ne + 1));
        ctx->block_traces[1] = 0.5 * (double)(ne * (ne - 1));
        ctx->block_traces[2] = 0.5 * (double)((K - ne) * (K - ne + 1));
        ctx->block_traces[3] = 0.5 * (double)((K - ne) * (K - ne - 1));
        ctx->block_traces[4] = (double)(ne * (K - ne + 2));
        ctx->block_traces[5] = (double)(ne * (K - ne));

        if (ctx->t2_enabled) {
            ctx->block_dims[6] = ctx->t2_naab;
            ctx->block_dims[7] = ctx->t2_nmix;
            ctx->block_traces[6] = (ctx->t2_norbs == K)
                                       ? tr_aab_full
                                       : fmin((double)ctx->t2_naab, tr_aab_full);
            ctx->block_traces[7] = (ctx->t2_norbs == K)
                                       ? tr_mix_a_full
                                       : fmin((double)ctx->t2_nmix, tr_mix_a_full);
        }

        for (int b = 0; b < ctx->nblocks; b++) ctx->block_szs[b] = svec_len(ctx->block_dims[b]);
        ctx->nD = ctx->block_szs[0] + ctx->block_szs[1];
        ctx->nQ = ctx->block_szs[2] + ctx->block_szs[3];
        ctx->nG = ctx->block_szs[4] + ctx->block_szs[5];
        ctx->nT2 = ctx->t2_enabled ? (ctx->block_szs[6] + ctx->block_szs[7]) : 0;
        ctx->nAll = ctx->nD + ctx->nQ + ctx->nG + ctx->nT2;
        ctx->meq = (ne == 1) ? 2 : (1 + ctx->m0);
    } else {
        ctx->nblocks = ctx->t2_enabled ? 13 : 9;
        ctx->naa = K * (K - 1) / 2;
        ctx->nab = K * K;
        ctx->ngd = 2 * K * K;

        ctx->block_dims[0] = ctx->naa;
        ctx->block_dims[1] = ctx->naa;
        ctx->block_dims[2] = ctx->nab;
        ctx->block_dims[3] = ctx->naa;
        ctx->block_dims[4] = ctx->naa;
        ctx->block_dims[5] = ctx->nab;
        ctx->block_dims[6] = ctx->ngd;
        ctx->block_dims[7] = ctx->nab;
        ctx->block_dims[8] = ctx->nab;

        ctx->block_traces[0] = 0.5 * (double)(na * (na - 1));
        ctx->block_traces[1] = 0.5 * (double)(nb * (nb - 1));
        ctx->block_traces[2] = (double)(na * nb);
        ctx->block_traces[3] = 0.5 * (double)((K - na) * (K - na - 1));
        ctx->block_traces[4] = 0.5 * (double)((K - nb) * (K - nb - 1));
        ctx->block_traces[5] = (double)((K - na) * (K - nb));
        ctx->block_traces[6] = (double)(na * (K - na + 1) + nb * (K - nb + 1));
        ctx->block_traces[7] = (double)(na * (K - nb));
        ctx->block_traces[8] = (double)(nb * (K - na));

        if (ctx->t2_enabled) {
            ctx->block_dims[9]  = ctx->t2_naab;
            ctx->block_dims[10] = ctx->t2_nmix;
            ctx->block_dims[11] = ctx->t2_naab;
            ctx->block_dims[12] = ctx->t2_nmix;
            ctx->block_traces[9]  = (ctx->t2_norbs == K)
                                        ? tr_aab_full
                                        : fmin((double)ctx->t2_naab, tr_aab_full);
            ctx->block_traces[10] = (ctx->t2_norbs == K)
                                        ? tr_mix_a_full
                                        : fmin((double)ctx->t2_nmix, tr_mix_a_full);
            ctx->block_traces[11] = (ctx->t2_norbs == K)
                                        ? tr_bba_full
                                        : fmin((double)ctx->t2_naab, tr_bba_full);
            ctx->block_traces[12] = (ctx->t2_norbs == K)
                                        ? tr_mix_b_full
                                        : fmin((double)ctx->t2_nmix, tr_mix_b_full);
        }

        for (int b = 0; b < ctx->nblocks; b++) ctx->block_szs[b] = svec_len(ctx->block_dims[b]);
        ctx->nD = ctx->block_szs[0] + ctx->block_szs[1] + ctx->block_szs[2];
        ctx->nQ = ctx->block_szs[3] + ctx->block_szs[4] + ctx->block_szs[5];
        ctx->nG = ctx->block_szs[6] + ctx->block_szs[7] + ctx->block_szs[8];
        ctx->nT2 = ctx->t2_enabled ? (ctx->block_szs[9] + ctx->block_szs[10] +
                                      ctx->block_szs[11] + ctx->block_szs[12])
                                   : 0;
        ctx->nAll = ctx->nD + ctx->nQ + ctx->nG + ctx->nT2;

        int meq = 3;
        if (constrain_s2 && na >= 1 && nb >= 1) meq += 1;
        if (na >= 1 && nb >= 1) {
            int nsym_m1 = K * (K + 1) / 2 - 1;
            if (na >= 2) meq += nsym_m1;
            if (nb >= 2) meq += nsym_m1;
        }
        ctx->meq = meq;
    }

    int d1 = 0, d2 = 0;
    for (int b = 0; b < ctx->nblocks; b++) {
        int d = ctx->block_dims[b];
        if (d >= d1) {
            d2 = d1;
            d1 = d;
        } else if (d > d2) {
            d2 = d;
        }
    }
    ctx->max_dim = d1;
    ctx->second_max_dim = d2;

    int req_thr = (num_threads <= 0) ? 2 : num_threads;
    if (req_thr > 4) req_thr = 4;
    if (d1 < 120 || d2 <= 0 || req_thr <= 1) {
        ctx->num_threads = 1;
        ctx->thr_nworkers = 0;
        ctx->thr_dim = 0;
    } else {
        ctx->num_threads = req_thr;
        ctx->thr_nworkers = req_thr - 1;
        ctx->thr_dim = d2;
    }

    size_t ksz = (size_t)K;
    size_t k2sz = ksz * ksz;
    size_t md2 = (size_t)ctx->max_dim * (size_t)ctx->max_dim;
    size_t k2_2 = k2sz * k2sz;

    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
            ctx->P0[p][q] = idx_sym_pair(K, p, q);
            int sg = 0;
            ctx->P1[p][q] = idx_asym_pair(K, p, q, &sg);
            ctx->sgn1[p][q] = (double)sg;
        }
    }

    ctx->b_eq = (double *)calloc((size_t)ctx->meq, sizeof(double));
    ctx->q0   = (double *)calloc((size_t)ctx->nQ, sizeof(double));
    ctx->C_D  = (double *)calloc((size_t)ctx->nD, sizeof(double));
    ctx->scr_ga = (double *)calloc(k2sz, sizeof(double));
    ctx->scr_gb = (double *)calloc(k2sz, sizeof(double));
    ctx->scr_M0 = (double *)calloc(md2, sizeof(double));
    ctx->scr_M1 = (double *)calloc(k2_2, sizeof(double));
    ctx->scr_M2 = (double *)calloc(k2_2, sizeof(double));
    ctx->ev_val = (double *)calloc((size_t)ctx->max_dim, sizeof(double));

    if (mode == CHEM_MODE_A_SINGLET) {
        size_t k4 = k2sz * k2sz;
        ctx->scr_St = (double *)calloc(k4, sizeof(double));
        ctx->scr_Tt = (double *)calloc(k4, sizeof(double));
        ctx->scr_D0 = (double *)calloc((size_t)ctx->m0 * (size_t)ctx->m0, sizeof(double));
        ctx->scr_D1 = (double *)calloc((size_t)(ctx->m1 > 0 ? ctx->m1 : 1) *
                                           (size_t)(ctx->m1 > 0 ? ctx->m1 : 1),
                                       sizeof(double));
    } else {
        size_t naa_sz = (size_t)(ctx->naa > 0 ? ctx->naa : 1);
        size_t nab_sz = (size_t)ctx->nab;
        ctx->scr_D0 = (double *)calloc(naa_sz * naa_sz, sizeof(double));
        ctx->scr_D1 = (double *)calloc(naa_sz * naa_sz, sizeof(double));
        ctx->scr_D2 = (double *)calloc(nab_sz * nab_sz, sizeof(double));
    }

    if (ctx->t2_enabled) {
        size_t kssz = (size_t)ctx->t2_norbs;
        size_t naa_s_sz = (size_t)(ctx->t2_naa > 0 ? ctx->t2_naa : 1);
        size_t nab_s_sz = (size_t)ctx->t2_nab;
        size_t mat_d = (size_t)((ctx->t2_nmix > ctx->t2_naab) ? ctx->t2_nmix : ctx->t2_naab);
        ctx->scr_t2_Daa = (double *)calloc(naa_s_sz * naa_s_sz, sizeof(double));
        ctx->scr_t2_Dbb = (double *)calloc(naa_s_sz * naa_s_sz, sizeof(double));
        ctx->scr_t2_Dab = (double *)calloc(nab_s_sz * nab_s_sz, sizeof(double));
        ctx->scr_t2_Dab_sw = (double *)calloc(nab_s_sz * nab_s_sz, sizeof(double));
        ctx->scr_t2_ga = (double *)calloc(kssz * kssz, sizeof(double));
        ctx->scr_t2_gb = (double *)calloc(kssz * kssz, sizeof(double));
        ctx->scr_t2_Mat = (double *)calloc(mat_d * mat_d, sizeof(double));
    }

    /* Query LAPACK workspace size for dsyevd_ */
    __LAPACK_int n_q = (__LAPACK_int)ctx->max_dim;
    __LAPACK_int lwork_q = -1, liwork_q = -1, info_q = 0;
    double wq = 0.0;
    __LAPACK_int iwq = 0;
    dsyevd_("V", "U", &n_q, ctx->scr_M0, &n_q, ctx->ev_val, &wq, &lwork_q, &iwq, &liwork_q,
            &info_q);
    if (info_q != 0) {
        mf_ctx_free(ctx);
        set_err(err, err_sz, "dsyevd_ workspace query failed (info=%d)", (int)info_q);
        return -1;
    }
    ctx->ev_lwork = (__LAPACK_int)wq + 64;
    size_t min_ev = md2 + (size_t)(ctx->max_dim / 2 + 2) * (size_t)ctx->max_dim;
    if ((size_t)ctx->ev_lwork < min_ev) ctx->ev_lwork = (__LAPACK_int)min_ev;
    ctx->ev_liwork = iwq + 16;
    ctx->ev_work = (double *)malloc((size_t)ctx->ev_lwork * sizeof(double));
    ctx->ev_iwork = (__LAPACK_int *)malloc((size_t)ctx->ev_liwork * sizeof(__LAPACK_int));

    if (ctx->thr_nworkers > 0) {
        __LAPACK_int nw = (__LAPACK_int)ctx->thr_dim;
        __LAPACK_int lq2 = -1, liq2 = -1, iq2 = 0;
        double wq2 = 0.0;
        __LAPACK_int iwq2 = 0;
        dsyevd_("V", "U", &nw, ctx->scr_M0, &nw, ctx->ev_val, &wq2, &lq2, &iwq2, &liq2, &iq2);
        if (iq2 == 0) {
            ctx->thr_lwork = (__LAPACK_int)wq2 + 64;
            size_t td = (size_t)ctx->thr_dim;
            size_t td2 = td * td;
            size_t min_th = td2 + (size_t)(ctx->thr_dim / 2 + 2) * td;
            if ((size_t)ctx->thr_lwork < min_th) ctx->thr_lwork = (__LAPACK_int)min_th;
            ctx->thr_liwork = iwq2 + 16;
            for (int t = 0; t < ctx->thr_nworkers; t++) {
                ctx->thr_ws[t].V = (double *)malloc(td2 * sizeof(double));
                ctx->thr_ws[t].ev_val = (double *)malloc(td * sizeof(double));
                ctx->thr_ws[t].ev_work = (double *)malloc((size_t)ctx->thr_lwork * sizeof(double));
                ctx->thr_ws[t].ev_iwork =
                    (__LAPACK_int *)malloc((size_t)ctx->thr_liwork * sizeof(__LAPACK_int));
                if (!ctx->thr_ws[t].V || !ctx->thr_ws[t].ev_val || !ctx->thr_ws[t].ev_work ||
                    !ctx->thr_ws[t].ev_iwork) {
                    ctx->thr_nworkers = t;
                    if (t == 0) ctx->num_threads = 1;
                    break;
                }
            }
        } else {
            ctx->thr_nworkers = 0;
            ctx->num_threads = 1;
        }
    }

    if (!ctx->b_eq || !ctx->q0 || !ctx->C_D || !ctx->scr_ga || !ctx->scr_M0 || !ctx->scr_M2 ||
        !ctx->ev_work || !ctx->ev_iwork) {
        mf_ctx_free(ctx);
        set_err(err, err_sz, "out of memory initializing matrix-free RDM2 context");
        return -1;
    }

    /* Populate b_eq and q0 */
    if (mode == CHEM_MODE_A_SINGLET) {
        int ne = na;
        ctx->b_eq[0] = 0.5 * (double)(ne * (ne + 1));
        /* q0 = (I_m0, I_m1) */
        double *Q0 = ctx->scr_M0;
        memset(Q0, 0, (size_t)ctx->m0 * (size_t)ctx->m0 * sizeof(double));
        for (int i = 0; i < ctx->m0; i++) Q0[(size_t)i * (size_t)ctx->m0 + (size_t)i] = 1.0;
        svec_pack_general(ctx->m0, Q0, ctx->q0);
        if (ctx->m1 > 0) {
            double *Q1 = ctx->scr_M1;
            memset(Q1, 0, (size_t)ctx->m1 * (size_t)ctx->m1 * sizeof(double));
            for (int i = 0; i < ctx->m1; i++) Q1[(size_t)i * (size_t)ctx->m1 + (size_t)i] = 1.0;
            svec_pack_general(ctx->m1, Q1, ctx->q0 + ctx->block_szs[2]);
        }
    } else {
        int r = 0;
        ctx->b_eq[r++] = 0.5 * (double)(na * (na - 1));
        ctx->b_eq[r++] = 0.5 * (double)(nb * (nb - 1));
        ctx->b_eq[r++] = (double)(na * nb);
        if (constrain_s2 && na >= 1 && nb >= 1) {
            double dnm = (double)(na - nb);
            double ts = (double)two_s;
            ctx->b_eq[r++] = 0.5 * (double)(na + nb) + 0.25 * dnm * dnm - 0.25 * ts * (ts + 2.0);
        }
        /* q0 = (I_naa, I_naa, I_nab) */
        if (ctx->naa > 0) {
            double *Iaa = ctx->scr_D0;
            memset(Iaa, 0, (size_t)ctx->naa * (size_t)ctx->naa * sizeof(double));
            for (int i = 0; i < ctx->naa; i++) Iaa[(size_t)i * (size_t)ctx->naa + (size_t)i] = 1.0;
            svec_pack_general(ctx->naa, Iaa, ctx->q0);
            svec_pack_general(ctx->naa, Iaa, ctx->q0 + ctx->block_szs[3]);
        }
        double *Iab = ctx->scr_D2;
        memset(Iab, 0, (size_t)ctx->nab * (size_t)ctx->nab * sizeof(double));
        for (int i = 0; i < ctx->nab; i++) Iab[(size_t)i * (size_t)ctx->nab + (size_t)i] = 1.0;
        svec_pack_general(ctx->nab, Iab, ctx->q0 + ctx->block_szs[3] + ctx->block_szs[4]);
    }

    if (ham_opt) mf_build_hamiltonian_cost(ctx, ham_opt, ctx->C_D);
    return 0;
}

static int mf_ctx_init(ChemMFContext *ctx, int K, int na, int nb, int two_s, int constrain_s2,
                       ChemMode mode, const ChemHam *ham_opt, char *err, size_t err_sz) {
    return mf_ctx_init_ext(ctx, K, na, nb, two_s, constrain_s2, mode, ham_opt,
                           0, 0, 0, NULL, 1, err, err_sz);
}

/* ============================================================================
 * Accelerate Eigensolver, Half-Rank dsyrk & Parallel Spectral Split
 * ============================================================================ */

static int syevd_with_ws(int n, const double *v_svec, int need_vecs,
                         double *mat_out, double *evals_out, double *frob_out,
                         double *work, __LAPACK_int lwork,
                         __LAPACK_int *iwork, __LAPACK_int liwork) {
    if (n <= 0) {
        if (frob_out) *frob_out = 0.0;
        return 0;
    }
    svec_unpack(n, v_svec, mat_out);
    if (frob_out) {
        *frob_out = norm2_vec(svec_len(n), v_svec);
    }
    __LAPACK_int nn = (__LAPACK_int)n;
    __LAPACK_int info = 0;
    dsyevd_(need_vecs ? "V" : "N", "U", &nn, mat_out, &nn, evals_out,
            work, &lwork, iwork, &liwork, &info);
    return (info == 0) ? 0 : -1;
}

static int accel_syevd(ChemMFContext *ctx, int n, const double *v_svec, int need_vecs,
                       double *mat_out, double *evals_out, double *frob_out) {
    return syevd_with_ws(n, v_svec, need_vecs, mat_out, evals_out, frob_out,
                         ctx->ev_work, ctx->ev_lwork, ctx->ev_iwork, ctx->ev_liwork);
}

/* Forms M_sub = sum_{k in [k0, k1)} weight_k * v_k v_k^T via cblas_dsyrk (Upper triangle) */
static void reconstruct_sym_from_eigs(int n, const double *V_rows, const double *evals,
                                      int k0, int k1, double scale, double *A_tmp, double *M_out) {
    int cnt = k1 - k0;
    size_t nn = (size_t)n;
    if (cnt <= 0) {
        memset(M_out, 0, nn * nn * sizeof(double));
        return;
    }
    for (int r = 0; r < cnt; r++) {
        double w = sqrt(fabs(evals[k0 + r]) * scale);
        const double *src = V_rows + (size_t)(k0 + r) * nn;
        double *dst = A_tmp + (size_t)r * nn;
        for (int i = 0; i < n; i++) dst[i] = w * src[i];
    }
    /* Row-major C := A^T * A where A is (cnt x n) -> C is (n x n), upper triangle populated */
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, n, cnt, 1.0, A_tmp, n, 0.0, M_out, n);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            M_out[(size_t)j * nn + (size_t)i] = M_out[(size_t)i * nn + (size_t)j];
        }
    }
}

static int split_one_block(int dim, int sz, double sigma, const double *m_blk,
                           double *z_blk, double *x_blk,
                           double *V, double *evals,
                           double *work, __LAPACK_int lwork,
                           __LAPACK_int *iwork, __LAPACK_int liwork) {
    if (dim <= 0) return 0;
    if (syevd_with_ws(dim, m_blk, 1, V, evals, NULL, work, lwork, iwork, liwork) != 0) {
        return -1;
    }
    /* work (>= dim^2 + (dim/2 + 2)*dim doubles) is idle after dsyevd_; reuse it for Mrec/A_tmp */
    double *Mrec = work;
    double *A_tmp = work + (size_t)dim * (size_t)dim;
    int k_pos = 0;
    while (k_pos < dim && evals[k_pos] <= 0.0) k_pos++;

    /* Half-rank dsyrk: reconstruct whichever spectral tail has <= dim/2 eigenpairs,
     * then recover the complementary half directly in svec space via M = Z - (1/sigma)*X. */
    if (k_pos <= dim / 2) {
        reconstruct_sym_from_eigs(dim, V, evals, 0, k_pos, sigma, A_tmp, Mrec);
        svec_pack_general(dim, Mrec, x_blk);
        double inv_sig = 1.0 / sigma;
        for (int i = 0; i < sz; i++) z_blk[i] = m_blk[i] + inv_sig * x_blk[i];
    } else {
        reconstruct_sym_from_eigs(dim, V, evals, k_pos, dim, 1.0, A_tmp, Mrec);
        svec_pack_general(dim, Mrec, z_blk);
        for (int i = 0; i < sz; i++) x_blk[i] = sigma * (z_blk[i] - m_blk[i]);
    }
    return 0;
}

typedef struct {
    ChemMFContext *ctx;
    double sigma;
    const double *M_svec;
    double *Z_out;
    double *X_out;
    int blk_offs[CHEM_MAX_BLOCKS];
    int blk_order[CHEM_MAX_BLOCKS];
    int done[CHEM_MAX_BLOCKS];
    pthread_mutex_t mu;
    int fail;
} ParallelSplitState;

typedef struct {
    ParallelSplitState *st;
    int wid; /* 0 = main thread, 1..thr_nworkers = helper threads */
} SplitWorkerArg;

static void *split_worker_fn(void *arg) {
    SplitWorkerArg *wa = (SplitWorkerArg *)arg;
    ParallelSplitState *st = wa->st;
    ChemMFContext *ctx = st->ctx;
    int wid = wa->wid;
    int max_cap = (wid == 0) ? ctx->max_dim : ctx->thr_dim;

    double *V = (wid == 0) ? ctx->scr_M0 : ctx->thr_ws[wid - 1].V;
    double *evals = (wid == 0) ? ctx->ev_val : ctx->thr_ws[wid - 1].ev_val;
    double *work = (wid == 0) ? ctx->ev_work : ctx->thr_ws[wid - 1].ev_work;
    __LAPACK_int lwork = (wid == 0) ? ctx->ev_lwork : ctx->thr_lwork;
    __LAPACK_int *iwork = (wid == 0) ? ctx->ev_iwork : ctx->thr_ws[wid - 1].ev_iwork;
    __LAPACK_int liwork = (wid == 0) ? ctx->ev_liwork : ctx->thr_liwork;

    for (;;) {
        int b_pick = -1;
        pthread_mutex_lock(&st->mu);
        if (!st->fail) {
            for (int k = 0; k < ctx->nblocks; k++) {
                int b = st->blk_order[k];
                if (!st->done[b] && ctx->block_dims[b] <= max_cap) {
                    st->done[b] = 1;
                    b_pick = b;
                    break;
                }
            }
        }
        pthread_mutex_unlock(&st->mu);
        if (b_pick < 0) break;

        int dim = ctx->block_dims[b_pick];
        int sz = ctx->block_szs[b_pick];
        int off = st->blk_offs[b_pick];
        if (split_one_block(dim, sz, st->sigma, st->M_svec + off, st->Z_out + off,
                            st->X_out + off, V, evals, work, lwork, iwork, liwork) != 0) {
            pthread_mutex_lock(&st->mu);
            st->fail = 1;
            pthread_mutex_unlock(&st->mu);
            break;
        }
    }
    return NULL;
}

/* Splits M = Z_new - (1/sigma)*X_new -> Z_out = M_+, X_out = sigma * (-M_-) */
static int spectral_split_all(ChemMFContext *ctx, double sigma, const double *M_svec,
                              double *Z_out, double *X_out) {
    if (ctx->thr_nworkers <= 0 || ctx->nblocks <= 1) {
        int off = 0;
        for (int b = 0; b < ctx->nblocks; b++) {
            int dim = ctx->block_dims[b];
            int sz = ctx->block_szs[b];
            if (dim > 0) {
                if (split_one_block(dim, sz, sigma, M_svec + off, Z_out + off, X_out + off,
                                    ctx->scr_M0, ctx->ev_val,
                                    ctx->ev_work, ctx->ev_lwork, ctx->ev_iwork, ctx->ev_liwork) !=
                    0) {
                    return -1;
                }
            }
            off += sz;
        }
        return 0;
    }

    ParallelSplitState st;
    memset(&st, 0, sizeof(st));
    st.ctx = ctx;
    st.sigma = sigma;
    st.M_svec = M_svec;
    st.Z_out = Z_out;
    st.X_out = X_out;
    int off = 0;
    for (int b = 0; b < ctx->nblocks; b++) {
        st.blk_offs[b] = off;
        st.blk_order[b] = b;
        off += ctx->block_szs[b];
    }
    /* Sort block indices in descending order of dimension so main grabs max_dim first */
    for (int i = 0; i < ctx->nblocks; i++) {
        int best = i;
        for (int j = i + 1; j < ctx->nblocks; j++) {
            if (ctx->block_dims[st.blk_order[j]] > ctx->block_dims[st.blk_order[best]]) best = j;
        }
        int tmp = st.blk_order[i];
        st.blk_order[i] = st.blk_order[best];
        st.blk_order[best] = tmp;
    }

    pthread_mutex_init(&st.mu, NULL);
    pthread_t thrs[3];
    SplitWorkerArg args[4];
    args[0].st = &st;
    args[0].wid = 0;
    int spawned = 0;
    for (int t = 1; t <= ctx->thr_nworkers; t++) {
        args[t].st = &st;
        args[t].wid = t;
        if (pthread_create(&thrs[t - 1], NULL, split_worker_fn, &args[t]) == 0) {
            spawned++;
        } else {
            break;
        }
    }
    split_worker_fn(&args[0]);
    for (int t = 0; t < spawned; t++) {
        pthread_join(thrs[t], NULL);
    }
    pthread_mutex_destroy(&st.mu);
    return st.fail ? -1 : 0;
}

/* ============================================================================
 * Normal-Equation Solver G G*(y) = r via PCG + Schur Complement Cholesky
 * ============================================================================ */

static void apply_H_op(const ChemMFContext *ctx, const double *d_in, double *d_out,
                       double *buf_Q, double *buf_G, double *buf_T2, double *buf_d) {
    mf_LQ_fwd(ctx, d_in, buf_Q);
    mf_LQ_adj(ctx, buf_Q, d_out);
    mf_LG_fwd(ctx, d_in, buf_G);
    mf_LG_adj(ctx, buf_G, buf_d);
    for (int i = 0; i < ctx->nD; i++) {
        d_out[i] += d_in[i] + buf_d[i];
    }
    if (ctx->t2_enabled && buf_T2) {
        mf_LT2_fwd(ctx, d_in, buf_T2);
        mf_LT2_adj(ctx, buf_T2, buf_d);
        for (int i = 0; i < ctx->nD; i++) {
            d_out[i] += buf_d[i];
        }
    }
}

static void apply_H_precond(const ChemMFContext *ctx, const double *r, double *z) {
    double t2_extra = ctx->t2_enabled ? (3.5 * (double)ctx->t2_norbs) : 0.0;
    if (ctx->mode == CHEM_MODE_A_SINGLET) {
        int s0 = ctx->block_szs[0];
        double w0 = 1.0 / (4.0 + t2_extra);
        double w1 = 1.0 / (12.0 + t2_extra);
        for (int i = 0; i < s0; i++) z[i] = w0 * r[i];
        for (int i = s0; i < ctx->nD; i++) z[i] = w1 * r[i];
    } else {
        int s01 = ctx->block_szs[0] + ctx->block_szs[1];
        double w01 = 1.0 / (4.0 + t2_extra);
        double w2 = 1.0 / (6.0 + t2_extra);
        for (int i = 0; i < s01; i++) z[i] = w01 * r[i];
        for (int i = s01; i < ctx->nD; i++) z[i] = w2 * r[i];
    }
}

typedef struct {
    double *S_eq;  /* [meq x meq] Cholesky factor (upper) */
    double *r_cg, *z_cg, *p_cg, *Ap_cg;
    double *buf_Q, *buf_G, *buf_T2, *buf_d;
    double *v0, *v_D, *f_D, *aeq_rhs, *tmp_d;
} NormalEqSolver;

static void normal_eq_free(NormalEqSolver *ns) {
    if (!ns) return;
    free(ns->S_eq);
    free(ns->r_cg);
    free(ns->z_cg);
    free(ns->p_cg);
    free(ns->Ap_cg);
    free(ns->buf_Q);
    free(ns->buf_G);
    free(ns->buf_T2);
    free(ns->buf_d);
    free(ns->v0);
    free(ns->v_D);
    free(ns->f_D);
    free(ns->aeq_rhs);
    free(ns->tmp_d);
    memset(ns, 0, sizeof(*ns));
}

static int pcg_solve_H(const ChemMFContext *ctx, NormalEqSolver *ns, const double *rhs,
                       double *x, int zero_init, double tol_rel) {
    int nD = ctx->nD;
    double b_norm = norm2_vec(nD, rhs);
    if (b_norm == 0.0) {
        memset(x, 0, (size_t)nD * sizeof(double));
        return 0;
    }

    if (zero_init) {
        memset(x, 0, (size_t)nD * sizeof(double));
        memcpy(ns->r_cg, rhs, (size_t)nD * sizeof(double));
    } else {
        apply_H_op(ctx, x, ns->Ap_cg, ns->buf_Q, ns->buf_G, ns->buf_T2, ns->buf_d);
        for (int i = 0; i < nD; i++) ns->r_cg[i] = rhs[i] - ns->Ap_cg[i];
    }

    double r_norm = norm2_vec(nD, ns->r_cg);
    if (r_norm <= tol_rel * b_norm) return 0;

    apply_H_precond(ctx, ns->r_cg, ns->z_cg);
    memcpy(ns->p_cg, ns->z_cg, (size_t)nD * sizeof(double));
    double rz = dot_vec(nD, ns->r_cg, ns->z_cg);

    for (int it = 1; it <= 180; it++) {
        apply_H_op(ctx, ns->p_cg, ns->Ap_cg, ns->buf_Q, ns->buf_G, ns->buf_T2, ns->buf_d);
        double pAp = dot_vec(nD, ns->p_cg, ns->Ap_cg);
        if (pAp <= 0.0) break;
        double alpha = rz / pAp;
        for (int i = 0; i < nD; i++) {
            x[i] += alpha * ns->p_cg[i];
            ns->r_cg[i] -= alpha * ns->Ap_cg[i];
        }
        r_norm = norm2_vec(nD, ns->r_cg);
        if (r_norm <= tol_rel * b_norm) return it;
        apply_H_precond(ctx, ns->r_cg, ns->z_cg);
        double rz_new = dot_vec(nD, ns->r_cg, ns->z_cg);
        double beta = rz_new / (rz + 1e-300);
        for (int i = 0; i < nD; i++) ns->p_cg[i] = ns->z_cg[i] + beta * ns->p_cg[i];
        rz = rz_new;
    }
    return 180;
}

static int normal_eq_init(const ChemMFContext *ctx, NormalEqSolver *ns, char *err, size_t err_sz) {
    memset(ns, 0, sizeof(*ns));
    int nD = ctx->nD, nQ = ctx->nQ, nG = ctx->nG, nT2 = ctx->nT2, meq = ctx->meq;
    ns->S_eq    = (double *)calloc((size_t)meq * (size_t)meq, sizeof(double));
    ns->r_cg    = (double *)malloc((size_t)nD * sizeof(double));
    ns->z_cg    = (double *)malloc((size_t)nD * sizeof(double));
    ns->p_cg    = (double *)malloc((size_t)nD * sizeof(double));
    ns->Ap_cg   = (double *)malloc((size_t)nD * sizeof(double));
    ns->buf_Q   = (double *)malloc((size_t)nQ * sizeof(double));
    ns->buf_G   = (double *)malloc((size_t)nG * sizeof(double));
    ns->buf_T2  = ctx->t2_enabled ? (double *)malloc((size_t)(nT2 > 0 ? nT2 : 1) * sizeof(double))
                                  : NULL;
    ns->buf_d   = (double *)malloc((size_t)nD * sizeof(double));
    ns->v0      = (double *)calloc((size_t)nD, sizeof(double));
    ns->v_D     = (double *)calloc((size_t)nD, sizeof(double));
    ns->f_D     = (double *)malloc((size_t)nD * sizeof(double));
    ns->aeq_rhs = (double *)malloc((size_t)meq * sizeof(double));
    ns->tmp_d   = (double *)malloc((size_t)nD * sizeof(double));

    if (!ns->S_eq || !ns->r_cg || !ns->buf_G || !ns->v0 || !ns->aeq_rhs ||
        (ctx->t2_enabled && !ns->buf_T2)) {
        normal_eq_free(ns);
        set_err(err, err_sz, "out of memory allocating normal equation workspace");
        return -1;
    }

    double *e_col = (double *)calloc((size_t)meq, sizeof(double));
    double *a_m   = (double *)malloc((size_t)nD * sizeof(double));
    double *u_m   = (double *)malloc((size_t)nD * sizeof(double));
    double *su_m  = (double *)malloc((size_t)meq * sizeof(double));

    for (int m = 0; m < meq; m++) {
        if (m > 0) e_col[m - 1] = 0.0;
        e_col[m] = 1.0;
        mf_Aeq_adj(ctx, e_col, a_m);
        pcg_solve_H(ctx, ns, a_m, u_m, 1, 1e-13);
        mf_Aeq_fwd(ctx, u_m, su_m);
        for (int i = 0; i < meq; i++) {
            ns->S_eq[(size_t)i * (size_t)meq + (size_t)m] = su_m[i];
        }
    }
    free(e_col);
    free(a_m);
    free(u_m);
    free(su_m);

    /* Symmetrize and Cholesky factor S_eq */
    for (int i = 0; i < meq; i++) {
        for (int j = i + 1; j < meq; j++) {
            double avg = 0.5 * (ns->S_eq[(size_t)i * (size_t)meq + (size_t)j] +
                                ns->S_eq[(size_t)j * (size_t)meq + (size_t)i]);
            ns->S_eq[(size_t)i * (size_t)meq + (size_t)j] = avg;
            ns->S_eq[(size_t)j * (size_t)meq + (size_t)i] = avg;
        }
    }

    __LAPACK_int n_s = (__LAPACK_int)meq;
    __LAPACK_int info = 0;
    dpotrf_("U", &n_s, ns->S_eq, &n_s, &info);
    if (info != 0) {
        normal_eq_free(ns);
        set_err(err, err_sz, "Schur complement S_eq is not SPD (dpotrf info=%d)", (int)info);
        return -1;
    }
    return 0;
}

/* Solves G G*(y) = r and outputs Gt_y = G*(y) as well as y_eq */
static int normal_eq_solve(const ChemMFContext *ctx, NormalEqSolver *ns,
                           const double *r_eq, const double *r_Q, const double *r_G,
                           const double *r_T2, double *y_eq_out, double *Gt_y_out,
                           int warm_start, double cg_tol) {
    int nD = ctx->nD, nQ = ctx->nQ, nG = ctx->nG, nT2 = ctx->nT2, meq = ctx->meq;

    mf_LQ_adj(ctx, r_Q, ns->f_D);
    mf_LG_adj(ctx, r_G, ns->tmp_d);
    for (int i = 0; i < nD; i++) ns->f_D[i] += ns->tmp_d[i];
    if (ctx->t2_enabled && r_T2) {
        mf_LT2_adj(ctx, r_T2, ns->tmp_d);
        for (int i = 0; i < nD; i++) ns->f_D[i] += ns->tmp_d[i];
    }

    pcg_solve_H(ctx, ns, ns->f_D, ns->v0, !warm_start, cg_tol);
    mf_Aeq_fwd(ctx, ns->v0, ns->aeq_rhs);
    for (int m = 0; m < meq; m++) y_eq_out[m] = r_eq[m] - ns->aeq_rhs[m];

    __LAPACK_int n_s = (__LAPACK_int)meq;
    __LAPACK_int nrhs = 1;
    __LAPACK_int info = 0;
    dpotrs_("U", &n_s, &nrhs, ns->S_eq, &n_s, y_eq_out, &n_s, &info);
    if (info != 0) return -1;

    mf_Aeq_adj(ctx, y_eq_out, ns->tmp_d);
    for (int i = 0; i < nD; i++) ns->tmp_d[i] += ns->f_D[i];
    pcg_solve_H(ctx, ns, ns->tmp_d, ns->v_D, !warm_start, cg_tol);

    memcpy(Gt_y_out, ns->v_D, (size_t)nD * sizeof(double));
    mf_LQ_fwd(ctx, ns->v_D, ns->buf_Q);
    mf_LG_fwd(ctx, ns->v_D, ns->buf_G);
    for (int i = 0; i < nQ; i++) Gt_y_out[nD + i] = ns->buf_Q[i] - r_Q[i];
    for (int i = 0; i < nG; i++) Gt_y_out[nD + nQ + i] = ns->buf_G[i] - r_G[i];
    if (ctx->t2_enabled && r_T2) {
        mf_LT2_fwd(ctx, ns->v_D, ns->buf_T2);
        for (int i = 0; i < nT2; i++) Gt_y_out[nD + nQ + nG + i] = ns->buf_T2[i] - r_T2[i];
    }
    return 0;
}

/* ============================================================================
 * Unconditional Anytime Dual Lower Bound Certifier (Serial & Parallel)
 * ============================================================================ */

typedef struct {
    ChemMFContext *ctx;
    const double *Z_D;
    const double *Z_Q;
    const double *Z_G;
    const double *Z_T2;
    double w_scale;
    int blk_offs[CHEM_MAX_BLOCKS];
    int blk_src[CHEM_MAX_BLOCKS]; /* 0=D, 1=Q, 2=G, 3=T2 */
    double fp_arr[CHEM_MAX_BLOCKS];
    double frob_arr[CHEM_MAX_BLOCKS];
    int done[CHEM_MAX_BLOCKS];
    pthread_mutex_t mu;
    int fail;
} ParallelCertState;

typedef struct {
    ParallelCertState *st;
    int wid;
} CertWorkerArg;

static void *cert_worker_fn(void *arg) {
    CertWorkerArg *wa = (CertWorkerArg *)arg;
    ParallelCertState *st = wa->st;
    ChemMFContext *ctx = st->ctx;
    int wid = wa->wid;
    int max_cap = (wid == 0) ? ctx->max_dim : ctx->thr_dim;

    double *M0 = (wid == 0) ? ctx->scr_M0 : ctx->thr_ws[wid - 1].V;
    double *evals = (wid == 0) ? ctx->ev_val : ctx->thr_ws[wid - 1].ev_val;
    double *work = (wid == 0) ? ctx->ev_work : ctx->thr_ws[wid - 1].ev_work;
    __LAPACK_int lwork = (wid == 0) ? ctx->ev_lwork : ctx->thr_lwork;
    __LAPACK_int *iwork = (wid == 0) ? ctx->ev_iwork : ctx->thr_ws[wid - 1].ev_iwork;
    __LAPACK_int liwork = (wid == 0) ? ctx->ev_liwork : ctx->thr_liwork;

    for (;;) {
        int b_pick = -1;
        pthread_mutex_lock(&st->mu);
        if (!st->fail) {
            for (int b = 0; b < ctx->nblocks; b++) {
                if (!st->done[b] && ctx->block_dims[b] <= max_cap) {
                    st->done[b] = 1;
                    b_pick = b;
                    break;
                }
            }
        }
        pthread_mutex_unlock(&st->mu);
        if (b_pick < 0) break;

        int dim = ctx->block_dims[b_pick];
        if (dim <= 0) continue;
        const double *ptr = NULL;
        double sc = 1.0;
        if (st->blk_src[b_pick] == 0) {
            ptr = st->Z_D + st->blk_offs[b_pick];
        } else if (st->blk_src[b_pick] == 1) {
            ptr = st->Z_Q + st->blk_offs[b_pick];
            sc = st->w_scale;
        } else if (st->blk_src[b_pick] == 2) {
            ptr = st->Z_G + st->blk_offs[b_pick];
            sc = st->w_scale;
        } else {
            ptr = st->Z_T2 + st->blk_offs[b_pick];
            sc = st->w_scale;
        }
        double frob_raw = 0.0;
        if (syevd_with_ws(dim, ptr, 0, M0, evals, &frob_raw, work, lwork, iwork, liwork) != 0) {
            pthread_mutex_lock(&st->mu);
            st->fail = 1;
            pthread_mutex_unlock(&st->mu);
            break;
        }
        st->fp_arr[b_pick] = sc * evals[0];
        st->frob_arr[b_pick] = sc * frob_raw;
    }
    return NULL;
}

static int certify_with_ctx(ChemMFContext *ctx, double e_core,
                            const double *y_eq, const double *Z_Q, const double *Z_G,
                            const double *Z_T2, double w_scale, ChemSolveResult *res_out,
                            char *err, size_t err_sz) {
    int nD = ctx->nD, nQ = ctx->nQ, meq = ctx->meq;

    double *Z_D = (double *)malloc((size_t)nD * sizeof(double));
    double *tmp_d = (double *)malloc((size_t)nD * sizeof(double));
    if (!Z_D || !tmp_d) {
        free(Z_D);
        free(tmp_d);
        set_err(err, err_sz, "out of memory in certifier");
        return -1;
    }

    /* Z_D = C_D - w_scale * (A_eq*(y_eq) + L_Q*(Z_Q) + L_G*(Z_G) + L_T2*(Z_T2)) */
    mf_Aeq_adj(ctx, y_eq, tmp_d);
    for (int i = 0; i < nD; i++) Z_D[i] = ctx->C_D[i] - w_scale * tmp_d[i];
    mf_LQ_adj(ctx, Z_Q, tmp_d);
    for (int i = 0; i < nD; i++) Z_D[i] -= w_scale * tmp_d[i];
    mf_LG_adj(ctx, Z_G, tmp_d);
    for (int i = 0; i < nD; i++) Z_D[i] -= w_scale * tmp_d[i];
    if (ctx->t2_enabled && Z_T2) {
        mf_LT2_adj(ctx, Z_T2, tmp_d);
        for (int i = 0; i < nD; i++) Z_D[i] -= w_scale * tmp_d[i];
    }
    free(tmp_d);

    double e_dual = e_core + w_scale * (dot_vec(meq, y_eq, ctx->b_eq) - dot_vec(nQ, Z_Q, ctx->q0));

    int nD_blocks = (ctx->mode == CHEM_MODE_A_SINGLET) ? 2 : 3;
    int nQ_blocks = nD_blocks;
    int nG_blocks = (ctx->mode == CHEM_MODE_A_SINGLET) ? 2 : 3;

    ParallelCertState st;
    memset(&st, 0, sizeof(st));
    st.ctx = ctx;
    st.Z_D = Z_D;
    st.Z_Q = Z_Q;
    st.Z_G = Z_G;
    st.Z_T2 = Z_T2;
    st.w_scale = w_scale;

    int off = 0;
    for (int b = 0; b < ctx->nblocks; b++) {
        int sz = ctx->block_szs[b];
        st.blk_offs[b] = off;
        if (b < nD_blocks) {
            st.blk_src[b] = 0;
            off = (b == nD_blocks - 1) ? 0 : off + sz;
        } else if (b < nD_blocks + nQ_blocks) {
            st.blk_src[b] = 1;
            off = (b == nD_blocks + nQ_blocks - 1) ? 0 : off + sz;
        } else if (b < nD_blocks + nQ_blocks + nG_blocks) {
            st.blk_src[b] = 2;
            off = (b == nD_blocks + nQ_blocks + nG_blocks - 1) ? 0 : off + sz;
        } else {
            st.blk_src[b] = 3;
            off += sz;
        }
    }

    if (ctx->thr_nworkers > 0) {
        pthread_mutex_init(&st.mu, NULL);
        pthread_t thrs[3];
        CertWorkerArg args[4];
        args[0].st = &st;
        args[0].wid = 0;
        int spawned = 0;
        for (int t = 1; t <= ctx->thr_nworkers; t++) {
            args[t].st = &st;
            args[t].wid = t;
            if (pthread_create(&thrs[t - 1], NULL, cert_worker_fn, &args[t]) == 0) {
                spawned++;
            } else {
                break;
            }
        }
        cert_worker_fn(&args[0]);
        for (int t = 0; t < spawned; t++) pthread_join(thrs[t], NULL);
        pthread_mutex_destroy(&st.mu);
    } else {
        pthread_mutex_init(&st.mu, NULL);
        CertWorkerArg a0 = {&st, 0};
        cert_worker_fn(&a0);
        pthread_mutex_destroy(&st.mu);
    }

    if (st.fail) {
        free(Z_D);
        set_err(err, err_sz, "dsyevd_ failed during certification");
        return -1;
    }

    double min_fp_all = 0.0;
    double min_cert_all = 0.0;
    int first_blk = 1;
    double floor_corr = 0.0;
    double sum_frob_tr = 0.0;

    for (int b = 0; b < ctx->nblocks; b++) {
        int dim = ctx->block_dims[b];
        res_out->block_dims[b] = dim;
        res_out->block_traces[b] = ctx->block_traces[b];
        if (dim <= 0) {
            res_out->block_min_fp[b] = 0.0;
            res_out->block_min_cert[b] = 0.0;
            continue;
        }
        double lam_fp = st.fp_arr[b];
        double frob = st.frob_arr[b];
        double c_mach = 64.0 * DBL_EPSILON * sqrt((double)dim);
        double lam_cert = lam_fp - c_mach * frob;
        res_out->block_min_fp[b] = lam_fp;
        res_out->block_min_cert[b] = lam_cert;

        if (first_blk || lam_fp < min_fp_all) min_fp_all = lam_fp;
        if (first_blk || lam_cert < min_cert_all) min_cert_all = lam_cert;
        first_blk = 0;

        if (lam_cert < 0.0) {
            floor_corr += lam_cert * ctx->block_traces[b];
        }
        sum_frob_tr += frob * ctx->block_traces[b];
    }

    double tr_D_sum = 0.0;
    for (int b = 0; b < nD_blocks; b++) tr_D_sum += ctx->block_traces[b];
    double c_norm = norm2_vec(nD, ctx->C_D);
    double y_norm = w_scale * norm2_vec(meq, y_eq);
    double beq_norm = norm2_vec(meq, ctx->b_eq);
    double delta_round =
        64.0 * DBL_EPSILON * (fabs(e_core) + c_norm * tr_D_sum + y_norm * beq_norm + sum_frob_tr);

    res_out->mode_used = ctx->mode;
    res_out->nblocks = ctx->nblocks;
    res_out->t2_enabled = ctx->t2_enabled;
    res_out->t2_norbs = ctx->t2_norbs;
    if (ctx->t2_enabled) {
        memcpy(res_out->t2_orbs, ctx->t2_orbs, (size_t)ctx->t2_norbs * sizeof(int));
    }
    res_out->E_dual_raw = e_dual;
    res_out->min_eig_fp = min_fp_all;
    res_out->min_eig_cert = min_cert_all;
    res_out->delta_round = delta_round;
    res_out->E_lb_cert = e_dual + floor_corr - delta_round;

    free(Z_D);
    return 0;
}

int chem_certify_dual_witness(const ChemHam *ham, const ChemDualWitness *w,
                              ChemSolveResult *res_out, char *err, size_t err_sz) {
    if (!ham || !w || !res_out) {
        set_err(err, err_sz, "null argument to chem_certify_dual_witness");
        return -1;
    }
    memset(res_out, 0, sizeof(*res_out));
    ChemMFContext ctx;
    if (mf_ctx_init_ext(&ctx, w->K, w->na, w->nb, w->two_s, w->constrain_s2, w->mode, ham,
                        w->t2_enabled, w->t2_norbs, w->t2_norbs, w->t2_orbs, 0, err, err_sz) !=
        0) {
        return -1;
    }
    if (w->meq != ctx.meq || w->nD != ctx.nD || w->nQ != ctx.nQ || w->nG != ctx.nG ||
        w->nT2 != ctx.nT2) {
        mf_ctx_free(&ctx);
        set_err(err, err_sz, "dual witness dimensions mismatch context");
        return -1;
    }
    int rc = certify_with_ctx(&ctx, ham->E_core, w->y_eq, w->Z_Q, w->Z_G, w->Z_T2, 1.0, res_out,
                              err, err_sz);
    mf_ctx_free(&ctx);
    return rc;
}

/* ============================================================================
 * Trivial Active-Space Analytical Ground State (nelec <= 1, K == 1, nelec == 2K)
 * ============================================================================ */

static int solve_trivial_space_cost(const ChemHam *ham, double e_core, const double *custom_C_D,
                                    ChemMode mode, int two_s, int constrain_s2,
                                    const ChemSolveOpts *opts, ChemSolveResult *res, char *err,
                                    size_t err_sz) {
    int K = ham->K;
    int ne = ham->nelec;
    if (custom_C_D && !(ne == 0 || (ne == 1 && K == 1) || ne == 2 * K || (K == 1 && ne == 2))) {
        return 1;
    }
    if (!custom_C_D) {
        e_core = ham->E_core;
    }
    double E_exact = e_core;

    if (ne == 0) {
        E_exact = e_core;
    } else if (!custom_C_D && ne == 2 * K) {
        for (int p = 0; p < K; p++) E_exact += 2.0 * chem_ham_h1(ham, p, p);
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                E_exact += 2.0 * chem_ham_eri(ham, p, p, q, q) - chem_ham_eri(ham, p, q, p, q);
            }
        }
    } else if (!custom_C_D && ne == 1) {
        double *H1 = (double *)malloc((size_t)K * (size_t)K * sizeof(double));
        double *ev = (double *)malloc((size_t)K * sizeof(double));
        __LAPACK_int n = (__LAPACK_int)K, lwork = -1, liwork = -1, info = 0;
        double wq = 0.0;
        __LAPACK_int iwq = 0;
        memcpy(H1, ham->h, (size_t)K * (size_t)K * sizeof(double));
        dsyevd_("N", "U", &n, H1, &n, ev, &wq, &lwork, &iwq, &liwork, &info);
        lwork = (__LAPACK_int)wq + 16;
        liwork = iwq + 8;
        double *wk = (double *)malloc((size_t)lwork * sizeof(double));
        __LAPACK_int *iwk = (__LAPACK_int *)malloc((size_t)liwork * sizeof(__LAPACK_int));
        dsyevd_("N", "U", &n, H1, &n, ev, wk, &lwork, iwk, &liwork, &info);
        E_exact += ev[0];
        free(H1);
        free(ev);
        free(wk);
        free(iwk);
    } else if (!custom_C_D && K == 1 && ne == 2) {
        /* 1 spatial orbital, 2 electrons (na=1, nb=1) */
        E_exact += 2.0 * chem_ham_h1(ham, 0, 0) + chem_ham_eri(ham, 0, 0, 0, 0);
    } else {
        return 1; /* Not trivial */
    }

    if (custom_C_D) {
        /* For custom cost on fully filled/empty 1-state space (K==1 && ne==2 or ne==2*K) */
        ChemMFContext tctx;
        if (mf_ctx_init_ext(&tctx, K, ham->na, ham->nb, two_s, constrain_s2, mode, NULL,
                            0, 0, 0, NULL, 1, err, err_sz) == 0) {
            if (tctx.nD > 0) {
                /* Wait until full BPSDP if nD > 1 unless ne == 0 */
                mf_ctx_free(&tctx);
                if (ne > 0) return 1;
            } else {
                mf_ctx_free(&tctx);
            }
        }
    }

    (void)err;
    (void)err_sz;
    res->mode_used = mode;
    res->nblocks = (mode == CHEM_MODE_A_SINGLET) ? 6 : 9;
    res->iters = 0;
    res->converged = 1;
    res->E_primal = E_exact;
    res->E_dual_raw = E_exact;
    res->delta_round = 64.0 * DBL_EPSILON * (fabs(E_exact) + 1.0);
    res->E_lb_cert = E_exact - res->delta_round;
    if (opts && opts->veto_enabled && res->E_lb_cert > opts->E_ai - opts->veto_tol) {
        res->vetoed = 1;
        res->iter_stopped = 0;
    }
    res->witness.mode = mode;
    res->witness.K = K;
    res->witness.na = ham->na;
    res->witness.nb = ham->nb;
    res->witness.two_s = two_s;
    res->witness.constrain_s2 = constrain_s2;
    return 0;
}

/* ============================================================================
 * FLOW 2-Jet Adaptive Penalty Controller
 * ============================================================================ */

static double flow_adapt_sigma(double sigma, double pres, double dres,
                               double *xi_prev, double *g0_prev, double *g1_est) {
    double xi = log(sigma);
    double g0 = log((dres + 1e-15) / (pres + 1e-15));

    double dxi = xi - *xi_prev;
    if (fabs(dxi) > 0.02) {
        double sec = (g0 - *g0_prev) / dxi;
        if (sec < -0.05 && sec > -10.0) {
            *g1_est = 0.5 * (*g1_est) + 0.5 * sec;
        }
    }
    double g1 = (*g1_est < -0.1) ? *g1_est : -1.0;

    int64_t j0_q16 = (int64_t)llround(fmax(-500.0, fmin(500.0, g0)) * 65536.0);
    int64_t j1_q16 = (int64_t)llround(fmax(-50.0, fmin(-0.1, g1)) * 65536.0);
    int64_t x_q16  = (int64_t)llround(fmax(-5.0, fmin(5.0, xi)) * 65536.0);

    uint64_t st[8] = {
        (uint64_t)j0_q16,
        (uint64_t)j1_q16,
        UINT64_C(0),
        (uint64_t)x_q16,
        UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)
    };
    int32_t b_exit = (int32_t)(-5 * 65536);
    int32_t b_enter = (int32_t)(5 * 65536);
    int32_t dt_max = (int32_t)llround(0.45 * 65536.0);
    uint64_t bd[8] = {
        UINT64_C(0), UINT64_C(0), UINT64_C(0),
        ((uint64_t)(uint32_t)b_exit << 32U) | (uint64_t)(uint32_t)b_enter,
        ((uint64_t)(uint32_t)dt_max << 32U),
        UINT64_C(0), UINT64_C(0), UINT64_C(0)
    };
    uint64_t out[8] = {0};
    flow_step(out, st, bd);

    double xi_next = (double)(int64_t)out[3] / 65536.0;
    if (xi_next < -5.0) xi_next = -5.0;
    if (xi_next > 5.0) xi_next = 5.0;

    *xi_prev = xi;
    *g0_prev = g0;
    return exp(xi_next);
}

/* ============================================================================
 * Matrix-Free BPSDP Dual Augmented Lagrangian Solver
 * ============================================================================ */

static int chem_solve_dqg_with_cost(const ChemHam *ham, double e_core, const double *custom_C_D,
                                    const ChemSolveOpts *opts_in, ChemSolveResult *res,
                                    char *err, size_t err_sz) {
    if (!ham || !res) {
        set_err(err, err_sz, "null argument to chem_solve_dqg");
        return -1;
    }
    memset(res, 0, sizeof(*res));
    double t0 = now_sec();

    ChemSolveOpts opts = opts_in ? *opts_in : chem_solve_opts_default(ham);
    ChemMode mode = chem_resolve_mode(ham, &opts);
    int two_s = opts.two_s;
    int cs2 = opts.constrain_s2;

    if (!custom_C_D) {
        e_core = ham->E_core;
        int triv = solve_trivial_space_cost(ham, e_core, NULL, mode, two_s, cs2, &opts, res, err,
                                            err_sz);
        if (triv == 0) {
            res->elapsed_sec = now_sec() - t0;
            return 0;
        }
    }

    ChemMFContext ctx;
    if (mf_ctx_init_ext(&ctx, ham->K, ham->na, ham->nb, two_s, cs2, mode,
                        custom_C_D ? NULL : ham, opts.enable_t2, opts.t2_max_orbs,
                        opts.t2_norbs, opts.t2_orbs, opts.num_threads, err, err_sz) != 0) {
        return -1;
    }
    if (custom_C_D) {
        memcpy(ctx.C_D, custom_C_D, (size_t)ctx.nD * sizeof(double));
    }

    NormalEqSolver ns;
    if (normal_eq_init(&ctx, &ns, err, err_sz) != 0) {
        mf_ctx_free(&ctx);
        return -1;
    }

    int nD = ctx.nD, nQ = ctx.nQ, nG = ctx.nG, nT2 = ctx.nT2, nAll = ctx.nAll, meq = ctx.meq;

    /* Scale cost vector for scale-invariant BPSDP conditioning */
    double c_norm = norm2_vec(nD, ctx.C_D);
    double scale_c = c_norm / sqrt((double)nD);
    if (scale_c < 1.0) scale_c = 1.0;

    double *C_sc   = (double *)calloc((size_t)nD, sizeof(double));
    double *X      = (double *)calloc((size_t)nAll, sizeof(double));
    double *Z      = (double *)calloc((size_t)nAll, sizeof(double));
    double *Gt_y   = (double *)calloc((size_t)nAll, sizeof(double));
    double *M      = (double *)calloc((size_t)nAll, sizeof(double));
    double *r_eq   = (double *)calloc((size_t)meq, sizeof(double));
    /* r_Q, r_G, r_T2 alias unused tail of M during residual construction and normal_eq_solve */
    double *r_Q    = M ? (M + nD) : NULL;
    double *r_G    = M ? (M + nD + nQ) : NULL;
    double *r_T2   = (M && ctx.t2_enabled) ? (M + nD + nQ + nG) : NULL;
    double *y_eq   = (double *)calloc((size_t)meq, sizeof(double));

    double *best_y   = (double *)calloc((size_t)meq, sizeof(double));
    double *best_ZQ  = (double *)calloc((size_t)nQ, sizeof(double));
    double *best_ZG  = (double *)calloc((size_t)nG, sizeof(double));
    double *best_ZT2 = ctx.t2_enabled ? (double *)calloc((size_t)(nT2 > 0 ? nT2 : 1), sizeof(double))
                                      : NULL;

    if (!C_sc || !X || !Z || !Gt_y || !M || !r_eq || !y_eq ||
        !best_y || !best_ZQ || !best_ZG || (ctx.t2_enabled && !best_ZT2)) {
        free(C_sc); free(X); free(Z); free(Gt_y); free(M);
        free(r_eq); free(y_eq);
        free(best_y); free(best_ZQ); free(best_ZG); free(best_ZT2);
        normal_eq_free(&ns);
        mf_ctx_free(&ctx);
        set_err(err, err_sz, "out of memory allocating BPSDP vectors");
        return -1;
    }

    for (int i = 0; i < nD; i++) C_sc[i] = ctx.C_D[i] / scale_c;
    double b_norm = sqrt(dot_vec(meq, ctx.b_eq, ctx.b_eq) + dot_vec(nQ, ctx.q0, ctx.q0));
    double c_sc_norm = norm2_vec(nD, C_sc);

    /* Initial feasible primal point estimate & dual start */
    double sigma = fmax(0.5, fmin(3.0, b_norm / fmax(1.0, c_sc_norm)));
    double xi_prev = log(sigma), g0_prev = 0.0, g1_est = -1.0;
    double alpha = (opts.alpha > 0.5 && opts.alpha < 1.99) ? opts.alpha : 1.6;
    int check_every = opts.check_cert_every;
    if (check_every <= 0) check_every = opts.veto_enabled ? 10 : 25;

    int has_best = 0;
    ChemSolveResult best_cert;
    memset(&best_cert, 0, sizeof(best_cert));

    int converged = 0;
    int iter_stopped = 0;
    double prim_res = 1.0, dual_res = 1.0;

    for (int it = 1; it <= opts.max_iters; it++) {
        iter_stopped = it;

        /* Compute r = G(C_sc - Z) + (1/sigma)*(b - G(X)) */
        double inv_sig = 1.0 / sigma;
        for (int i = 0; i < nD; i++) ns.tmp_d[i] = C_sc[i] - Z[i] - inv_sig * X[i];
        const double *z_q = Z + nD;
        const double *x_q = X + nD;
        const double *z_g = Z + nD + nQ;
        const double *x_g = X + nD + nQ;

        mf_Aeq_fwd(&ctx, ns.tmp_d, r_eq);
        mf_LQ_fwd(&ctx, ns.tmp_d, r_Q);
        mf_LG_fwd(&ctx, ns.tmp_d, r_G);
        if (ctx.t2_enabled) mf_LT2_fwd(&ctx, ns.tmp_d, r_T2);

        for (int m = 0; m < meq; m++) r_eq[m] += inv_sig * ctx.b_eq[m];
        for (int i = 0; i < nQ; i++) {
            r_Q[i] += z_q[i] - inv_sig * (ctx.q0[i] - x_q[i]);
        }
        for (int i = 0; i < nG; i++) {
            r_G[i] += z_g[i] + inv_sig * x_g[i];
        }
        if (ctx.t2_enabled) {
            const double *z_t2 = Z + nD + nQ + nG;
            const double *x_t2 = X + nD + nQ + nG;
            for (int i = 0; i < nT2; i++) r_T2[i] += z_t2[i] + inv_sig * x_t2[i];
        }

        double min_res = (prim_res < dual_res) ? prim_res : dual_res;
        double cg_tol = fmax(1e-11, fmin(1e-8, 0.01 * min_res));
        if (normal_eq_solve(&ctx, &ns, r_eq, r_Q, r_G, r_T2, y_eq, Gt_y, it > 1, cg_tol) != 0) {
            set_err(err, err_sz, "normal equation Cholesky solve failed at iter %d", it);
            break;
        }

        /* Over-relaxed dual step + spectral split (C_sc is zero on Q, G, T2) */
        for (int i = 0; i < nD; i++) {
            M[i] = alpha * (C_sc[i] - Gt_y[i]) + (1.0 - alpha) * Z[i] - inv_sig * X[i];
        }
        for (int i = nD; i < nAll; i++) {
            M[i] = -alpha * Gt_y[i] + (1.0 - alpha) * Z[i] - inv_sig * X[i];
        }

        if (spectral_split_all(&ctx, sigma, M, Z, X) != 0) {
            set_err(err, err_sz, "dsyevd_ failed during spectral split at iter %d", it);
            break;
        }

        /* Compute primal residual ||G(X) - b|| / (1 + ||b||) & dual residual ||G*(y) + Z - C|| / (1 + ||C||) */
        mf_Aeq_fwd(&ctx, X, ns.aeq_rhs);
        mf_LQ_fwd(&ctx, X, ns.buf_Q);
        mf_LG_fwd(&ctx, X, ns.buf_G);
        if (ctx.t2_enabled) mf_LT2_fwd(&ctx, X, ns.buf_T2);

        double p2 = 0.0;
        for (int m = 0; m < meq; m++) {
            double d = ns.aeq_rhs[m] - ctx.b_eq[m];
            p2 += d * d;
        }
        for (int i = 0; i < nQ; i++) {
            double d = ns.buf_Q[i] - X[nD + i] + ctx.q0[i];
            p2 += d * d;
        }
        for (int i = 0; i < nG; i++) {
            double d = ns.buf_G[i] - X[nD + nQ + i];
            p2 += d * d;
        }
        if (ctx.t2_enabled) {
            for (int i = 0; i < nT2; i++) {
                double d = ns.buf_T2[i] - X[nD + nQ + nG + i];
                p2 += d * d;
            }
        }
        prim_res = sqrt(p2) / (1.0 + b_norm);

        double d2 = 0.0;
        for (int i = 0; i < nD; i++) {
            double d = Gt_y[i] + Z[i] - C_sc[i];
            d2 += d * d;
        }
        for (int i = nD; i < nAll; i++) {
            double d = Gt_y[i] + Z[i];
            d2 += d * d;
        }
        dual_res = sqrt(d2) / (1.0 + c_sc_norm);

        int is_conv = (it >= 25 && prim_res < opts.tol && dual_res < opts.tol);
        int do_cert = is_conv || (it == opts.max_iters) || (it >= 10 && (it % check_every) == 0);

        if (do_cert) {
            ChemSolveResult cres;
            memset(&cres, 0, sizeof(cres));
            const double *z_t2_ptr = ctx.t2_enabled ? (Z + nD + nQ + nG) : NULL;
            if (certify_with_ctx(&ctx, e_core, y_eq, Z + nD, Z + nD + nQ, z_t2_ptr, scale_c,
                                 &cres, err, err_sz) != 0) {
                break;
            }

            if (!has_best || cres.E_lb_cert > best_cert.E_lb_cert) {
                has_best = 1;
                best_cert = cres;
                for (int m = 0; m < meq; m++) best_y[m] = scale_c * y_eq[m];
                for (int i = 0; i < nQ; i++) best_ZQ[i] = scale_c * Z[nD + i];
                for (int i = 0; i < nG; i++) best_ZG[i] = scale_c * Z[nD + nQ + i];
                if (ctx.t2_enabled) {
                    for (int i = 0; i < nT2; i++) best_ZT2[i] = scale_c * Z[nD + nQ + nG + i];
                }
            }

            if (opts.verbose && (it % (check_every * 2) == 0 || is_conv)) {
                printf("  [BPSDP it=%3d] sigma=%.3f pres=%.2e dres=%.2e E_lb_best=%.8f\n",
                       it, sigma, prim_res, dual_res, best_cert.E_lb_cert);
            }

            /* Anytime Early-Veto check */
            if (opts.veto_enabled && best_cert.E_lb_cert > opts.E_ai - opts.veto_tol) {
                res->vetoed = 1;
                break;
            }
        }

        if (is_conv) {
            converged = 1;
            break;
        }

        if (it % 20 == 0) {
            sigma = flow_adapt_sigma(sigma, prim_res, dual_res, &xi_prev, &g0_prev, &g1_est);
        }
    }

    if (!has_best) {
        for (int m = 0; m < meq; m++) best_y[m] = scale_c * y_eq[m];
        for (int i = 0; i < nQ; i++) best_ZQ[i] = scale_c * Z[nD + i];
        for (int i = 0; i < nG; i++) best_ZG[i] = scale_c * Z[nD + nQ + i];
        if (ctx.t2_enabled) {
            for (int i = 0; i < nT2; i++) best_ZT2[i] = scale_c * Z[nD + nQ + nG + i];
        }
        if (certify_with_ctx(&ctx, e_core, best_y, best_ZQ, best_ZG, best_ZT2, 1.0, &best_cert,
                             err, err_sz) != 0) {
            free(C_sc); free(X); free(Z); free(Gt_y); free(M);
            free(r_eq); free(y_eq);
            free(best_y); free(best_ZQ); free(best_ZG); free(best_ZT2);
            normal_eq_free(&ns);
            mf_ctx_free(&ctx);
            return -1;
        }
    }

    double e_primal = e_core + dot_vec(nD, ctx.C_D, X);
    int vetoed_flag = res->vetoed;
    *res = best_cert;
    res->iters = iter_stopped;
    res->converged = converged;
    res->vetoed = vetoed_flag;
    res->iter_stopped = iter_stopped;
    res->E_primal = e_primal;
    res->prim_res = prim_res;
    res->dual_res = dual_res;
    res->elapsed_sec = now_sec() - t0;

    /* Transfer best dual witness ownership */
    res->witness.mode = mode;
    res->witness.K = ham->K;
    res->witness.na = ham->na;
    res->witness.nb = ham->nb;
    res->witness.two_s = two_s;
    res->witness.constrain_s2 = cs2;
    res->witness.t2_enabled = ctx.t2_enabled;
    res->witness.t2_norbs = ctx.t2_norbs;
    if (ctx.t2_enabled) {
        memcpy(res->witness.t2_orbs, ctx.t2_orbs, (size_t)ctx.t2_norbs * sizeof(int));
    }
    res->witness.meq = meq;
    res->witness.nD = nD;
    res->witness.nQ = nQ;
    res->witness.nG = nG;
    res->witness.nT2 = nT2;
    res->witness.y_eq = best_y;
    res->witness.Z_Q = best_ZQ;
    res->witness.Z_G = best_ZG;
    res->witness.Z_T2 = best_ZT2;

    free(C_sc); free(X); free(Z); free(Gt_y); free(M);
    free(r_eq); free(y_eq);
    normal_eq_free(&ns);
    mf_ctx_free(&ctx);
    return 0;
}

int chem_solve_dqg(const ChemHam *ham, const ChemSolveOpts *opts_in, ChemSolveResult *res,
                   char *err, size_t err_sz) {
    return chem_solve_dqg_with_cost(ham, ham ? ham->E_core : 0.0, NULL, opts_in, res, err, err_sz);
}

/* ============================================================================
 * Certified Observable & Hellmann-Feynman Force Envelope Bounds
 * ============================================================================ */

void chem_observable_free(ChemObservable *obs) {
    if (!obs) return;
    free(obs->o1_a);
    free(obs->o1_b);
    free(obs->o2_ab);
    free(obs->o2_aa);
    memset(obs, 0, sizeof(*obs));
}

static int alloc_observable(ChemObservable *out, const char *name, int K, double o_core,
                            int has_o2, int spin_symmetric, char *err, size_t err_sz) {
    memset(out, 0, sizeof(*out));
    if (K <= 0 || K > CHEM_MAX_ORBITALS) {
        set_err(err, err_sz, "invalid K=%d for observable", K);
        return -1;
    }
    snprintf(out->name, sizeof(out->name), "%s", name ? name : "obs");
    out->K = K;
    out->O_core = o_core;
    out->has_o2 = has_o2;
    out->spin_symmetric = spin_symmetric;
    size_t k2 = (size_t)K * (size_t)K;
    size_t k4 = k2 * k2;
    out->o1_a = (double *)calloc(k2, sizeof(double));
    out->o1_b = (double *)calloc(k2, sizeof(double));
    if (has_o2) {
        out->o2_ab = (double *)calloc(k4, sizeof(double));
        out->o2_aa = (double *)calloc(k4, sizeof(double));
    }
    if (!out->o1_a || !out->o1_b || (has_o2 && (!out->o2_ab || !out->o2_aa))) {
        chem_observable_free(out);
        set_err(err, err_sz, "out of memory allocating observable");
        return -1;
    }
    return 0;
}

int chem_obs_orbital_occ(int K, int p, ChemObservable *out, char *err, size_t err_sz) {
    if (p < 0 || p >= K) {
        set_err(err, err_sz, "orbital index p=%d out of [0,%d)", p, K);
        return -1;
    }
    char nm[64];
    snprintf(nm, sizeof(nm), "occ:%d", p);
    if (alloc_observable(out, nm, K, 0.0, 0, 1, err, err_sz) != 0) return -1;
    out->o1_a[(size_t)p * (size_t)K + (size_t)p] = 1.0;
    out->o1_b[(size_t)p * (size_t)K + (size_t)p] = 1.0;
    return 0;
}

int chem_obs_double_occ(int K, int p, ChemObservable *out, char *err, size_t err_sz) {
    if (p < 0 || p >= K) {
        set_err(err, err_sz, "orbital index p=%d out of [0,%d)", p, K);
        return -1;
    }
    char nm[64];
    snprintf(nm, sizeof(nm), "docc:%d", p);
    if (alloc_observable(out, nm, K, 0.0, 1, 1, err, err_sz) != 0) return -1;
    size_t ksz = (size_t)K;
    size_t idx = ((((size_t)p * ksz + (size_t)p) * ksz + (size_t)p) * ksz + (size_t)p);
    out->o2_ab[idx] = 1.0;
    out->o2_aa[idx] = 1.0;
    return 0;
}

int chem_obs_spin_corr(int K, int p, int q, ChemObservable *out, char *err, size_t err_sz) {
    if (p < 0 || p >= K || q < 0 || q >= K) {
        set_err(err, err_sz, "orbital index (%d,%d) out of [0,%d)", p, q, K);
        return -1;
    }
    char nm[64];
    snprintf(nm, sizeof(nm), "sdot:%d:%d", p, q);
    if (alloc_observable(out, nm, K, 0.0, 1, 1, err, err_sz) != 0) return -1;
    size_t ksz = (size_t)K;
    if (p == q) {
        /* S_p^2 = 0.75 * (n_p - 2 * n_{p,a} n_{p,b}) */
        out->o1_a[(size_t)p * ksz + (size_t)p] = 0.75;
        out->o1_b[(size_t)p * ksz + (size_t)p] = 0.75;
        size_t idx = ((((size_t)p * ksz + (size_t)p) * ksz + (size_t)p) * ksz + (size_t)p);
        out->o2_ab[idx] = -1.5;
        out->o2_aa[idx] = -1.5;
    } else {
        /* Spin-free SU(2) form: S_p . S_q = -(1/4) n_p n_q - (1/2) sum_{st} a^+_{ps} a^+_{qt} a_{pt} a_{qs} */
        size_t ppqq = ((((size_t)p * ksz + (size_t)p) * ksz + (size_t)q) * ksz + (size_t)q);
        size_t qqpp = ((((size_t)q * ksz + (size_t)q) * ksz + (size_t)p) * ksz + (size_t)p);
        size_t pqqp = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)q) * ksz + (size_t)p);
        size_t qppq = ((((size_t)q * ksz + (size_t)p) * ksz + (size_t)p) * ksz + (size_t)q);
        out->o2_ab[ppqq] = out->o2_ab[qqpp] = -0.25;
        out->o2_ab[pqqp] = out->o2_ab[qppq] = -0.5;
        out->o2_aa[ppqq] = out->o2_aa[qqpp] = -0.25;
        out->o2_aa[pqqp] = out->o2_aa[qppq] = -0.5;
    }
    return 0;
}

int chem_obs_ham_diff(const ChemHam *h_plus, const ChemHam *h_minus, double scale,
                      const char *name, ChemObservable *out, char *err, size_t err_sz) {
    if (!h_plus || !h_minus || h_plus->K != h_minus->K) {
        set_err(err, err_sz, "incompatible Hamiltonians in chem_obs_ham_diff");
        return -1;
    }
    int K = h_plus->K;
    double oc = scale * (h_plus->E_core - h_minus->E_core);
    if (alloc_observable(out, name ? name : "hf-force", K, oc, 1, 1, err, err_sz) != 0) return -1;
    size_t ksz = (size_t)K;
    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
            double dh = scale * (chem_ham_h1(h_plus, p, q) - chem_ham_h1(h_minus, p, q));
            out->o1_a[(size_t)p * ksz + (size_t)q] = dh;
            out->o1_b[(size_t)p * ksz + (size_t)q] = dh;
        }
    }
    for (int p = 0; p < K; p++) {
        for (int r = 0; r < K; r++) {
            for (int q = 0; q < K; q++) {
                for (int s = 0; s < K; s++) {
                    size_t idx = ((((size_t)p * ksz + (size_t)r) * ksz + (size_t)q) * ksz +
                                  (size_t)s);
                    double de = scale * (chem_ham_eri(h_plus, p, r, q, s) -
                                         chem_ham_eri(h_minus, p, r, q, s));
                    out->o2_ab[idx] = de;
                    out->o2_aa[idx] = de;
                }
            }
        }
    }
    return 0;
}

static void mf_build_observable_cost(const ChemMFContext *ctx, const ChemObservable *obs,
                                     double *C_O) {
    int K = ctx->K;
    size_t ksz = (size_t)K;

    if (ctx->mode == CHEM_MODE_A_SINGLET) {
        double *Sh = ctx->scr_St;
        double *Th = ctx->scr_Tt;
        double *ga_h = ctx->scr_ga;
        size_t k4 = ksz * ksz * ksz * ksz;
        if (obs->has_o2) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    for (int r = 0; r < K; r++) {
                        for (int s = 0; s < K; s++) {
                            size_t idx_eri = ((((size_t)p * ksz + (size_t)r) * ksz + (size_t)q) *
                                              ksz + (size_t)s);
                            size_t off4 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz +
                                           (size_t)s);
                            double e = obs->o2_ab[idx_eri];
                            Sh[off4] = e;
                            Th[off4] = 3.0 * e;
                        }
                    }
                }
            }
        } else {
            memset(Sh, 0, k4 * sizeof(double));
            memset(Th, 0, k4 * sizeof(double));
        }
        for (int p = 0; p < K; p++) {
            for (int r = 0; r < K; r++) {
                size_t pr = (size_t)p * ksz + (size_t)r;
                ga_h[pr] = obs->o1_a[pr] + obs->o1_b[pr];
            }
        }
        mode_a_gamma_adjoint_to_sh(K, ctx->na, ga_h, Sh);
        mode_a_contract_sh_th(ctx, Sh, Th, C_O);
        return;
    }

    int naa = ctx->naa;
    int nab = ctx->nab;
    double *Daa_h = ctx->scr_D0;
    double *Dbb_h = ctx->scr_D1;
    double *Dab_h = ctx->scr_D2;
    double *ga_h = ctx->scr_ga;
    double *gb_h = ctx->scr_gb;
    if (naa > 0) {
        memset(Daa_h, 0, (size_t)naa * (size_t)naa * sizeof(double));
        memset(Dbb_h, 0, (size_t)naa * (size_t)naa * sizeof(double));
    }
    memset(Dab_h, 0, (size_t)nab * (size_t)nab * sizeof(double));
    memcpy(ga_h, obs->o1_a, ksz * ksz * sizeof(double));
    memcpy(gb_h, obs->o1_b, ksz * ksz * sizeof(double));

    if (obs->has_o2) {
        int p_idx = 0;
        int pair_ij[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS][2];
        for (int i = 0; i < K; i++) {
            for (int j = i + 1; j < K; j++) {
                pair_ij[p_idx][0] = i;
                pair_ij[p_idx][1] = j;
                p_idx++;
            }
        }
        for (int p = 0; p < naa; p++) {
            int i = pair_ij[p][0], j = pair_ij[p][1];
            for (int q = 0; q < naa; q++) {
                int k = pair_ij[q][0], l = pair_ij[q][1];
                size_t ikjl = ((((size_t)i * ksz + (size_t)k) * ksz + (size_t)j) * ksz + (size_t)l);
                size_t iljk = ((((size_t)i * ksz + (size_t)l) * ksz + (size_t)j) * ksz + (size_t)k);
                double val = obs->o2_aa[ikjl] - obs->o2_aa[iljk];
                Daa_h[(size_t)p * (size_t)naa + (size_t)q] = val;
                Dbb_h[(size_t)p * (size_t)naa + (size_t)q] = val;
            }
        }
        for (int p = 0; p < nab; p++) {
            int i = p / K, j = p % K;
            for (int q = 0; q < nab; q++) {
                int k = q / K, l = q % K;
                size_t ikjl = ((((size_t)i * ksz + (size_t)k) * ksz + (size_t)j) * ksz + (size_t)l);
                Dab_h[(size_t)p * (size_t)nab + (size_t)q] = obs->o2_ab[ikjl];
            }
        }
    }

    mode_b_gamma_adj(ctx, ga_h, gb_h, Daa_h, Dbb_h, Dab_h);
    mode_b_pack_D(ctx, Daa_h, Dbb_h, Dab_h, C_O);
}

static int eval_obs_lagrangian_lb(const ChemHam *ham, const ChemObservable *obs, double E_ub,
                                  const double *C_H, const double *C_O, double lambda, double sign_o,
                                  double e_core_o, int nD, const ChemSolveOpts *base_opts,
                                  double *bound_out, char *err, size_t err_sz) {
    (void)obs;
    double *C_mix = (double *)malloc((size_t)(nD > 0 ? nD : 1) * sizeof(double));
    if (!C_mix) {
        set_err(err, err_sz, "out of memory in eval_obs_lagrangian_lb");
        return -1;
    }
    for (int i = 0; i < nD; i++) {
        C_mix[i] = sign_o * C_O[i] + lambda * C_H[i];
    }
    double core_mix = sign_o * e_core_o + lambda * ham->E_core;
    ChemSolveOpts sopts = *base_opts;
    sopts.veto_enabled = 0;
    if (sopts.max_iters > 350) sopts.max_iters = 350;
    ChemSolveResult res;
    int rc = chem_solve_dqg_with_cost(ham, core_mix, C_mix, &sopts, &res, err, err_sz);
    free(C_mix);
    if (rc != 0) return -1;
    /* L(lambda) = E_lb(sign_o*O + lambda*H) - lambda * E_ub <= sign_o * <O> */
    double safety = 64.0 * DBL_EPSILON * (fabs(lambda) * fabs(E_ub) + 1.0);
    *bound_out = (res.E_lb_cert - lambda * E_ub) - safety;
    chem_solve_result_free(&res);
    return 0;
}

int chem_certify_observable(const ChemHam *ham, const ChemObservable *obs, double E_ub,
                            const ChemSolveOpts *opts_in, ChemObservableResult *out,
                            char *err, size_t err_sz) {
    if (!ham || !obs || !out) {
        set_err(err, err_sz, "null argument to chem_certify_observable");
        return -1;
    }
    if (obs->K != ham->K) {
        set_err(err, err_sz, "observable K=%d != Hamiltonian K=%d", obs->K, ham->K);
        return -1;
    }
    memset(out, 0, sizeof(*out));
    double t0 = now_sec();
    snprintf(out->name, sizeof(out->name), "%s", obs->name);
    out->E_ub = E_ub;

    ChemSolveOpts opts = opts_in ? *opts_in : chem_solve_opts_default(ham);
    ChemMode mode = chem_resolve_mode(ham, &opts);

    ChemMFContext ctx;
    if (mf_ctx_init_ext(&ctx, ham->K, ham->na, ham->nb, opts.two_s, opts.constrain_s2, mode,
                        ham, opts.enable_t2, opts.t2_max_orbs, opts.t2_norbs, opts.t2_orbs,
                        opts.num_threads, err, err_sz) != 0) {
        return -1;
    }
    int nD = ctx.nD;
    double *C_H = (double *)malloc((size_t)(nD > 0 ? nD : 1) * sizeof(double));
    double *C_O = (double *)malloc((size_t)(nD > 0 ? nD : 1) * sizeof(double));
    if (!C_H || !C_O) {
        free(C_H);
        free(C_O);
        mf_ctx_free(&ctx);
        set_err(err, err_sz, "out of memory allocating observable cost vectors");
        return -1;
    }
    memcpy(C_H, ctx.C_D, (size_t)nD * sizeof(double));
    mf_build_observable_cost(&ctx, obs, C_O);
    mf_ctx_free(&ctx);

    const double grid[7] = {0.0, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0};
    for (int side = 0; side < 2; side++) {
        double sign_o = (side == 0) ? 1.0 : -1.0;
        double best_b = -1e100;
        double best_lam = 0.0;
        int best_idx = 0;
        double vals[7];
        for (int g = 0; g < 7; g++) {
            double v = 0.0;
            if (eval_obs_lagrangian_lb(ham, obs, E_ub, C_H, C_O, grid[g], sign_o, obs->O_core, nD,
                                       &opts, &v, err, err_sz) != 0) {
                free(C_H);
                free(C_O);
                return -1;
            }
            vals[g] = v;
            if (g == 0 || v > best_b) {
                best_b = v;
                best_lam = grid[g];
                best_idx = g;
            }
        }
        /* Golden-section search on s = log(1 + lambda) around best grid index */
        double s_lo = (best_idx > 0) ? log(1.0 + grid[best_idx - 1]) : 0.0;
        double s_hi = (best_idx < 6) ? log(1.0 + grid[best_idx + 1]) : log(1.0 + grid[6]);
        if (s_hi > s_lo + 1e-6) {
            const double phi = 0.6180339887498949;
            double x1 = s_hi - phi * (s_hi - s_lo);
            double x2 = s_lo + phi * (s_hi - s_lo);
            double f1 = 0.0, f2 = 0.0;
            double l1 = exp(x1) - 1.0, l2 = exp(x2) - 1.0;
            if (eval_obs_lagrangian_lb(ham, obs, E_ub, C_H, C_O, l1, sign_o, obs->O_core, nD,
                                       &opts, &f1, err, err_sz) != 0 ||
                eval_obs_lagrangian_lb(ham, obs, E_ub, C_H, C_O, l2, sign_o, obs->O_core, nD,
                                       &opts, &f2, err, err_sz) != 0) {
                free(C_H);
                free(C_O);
                return -1;
            }
            if (f1 > best_b) { best_b = f1; best_lam = l1; }
            if (f2 > best_b) { best_b = f2; best_lam = l2; }
            for (int it = 0; it < 5; it++) {
                if (f1 < f2) {
                    s_lo = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = s_lo + phi * (s_hi - s_lo);
                    l2 = exp(x2) - 1.0;
                    if (eval_obs_lagrangian_lb(ham, obs, E_ub, C_H, C_O, l2, sign_o, obs->O_core,
                                               nD, &opts, &f2, err, err_sz) != 0) {
                        free(C_H);
                        free(C_O);
                        return -1;
                    }
                    if (f2 > best_b) { best_b = f2; best_lam = l2; }
                } else {
                    s_hi = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = s_hi - phi * (s_hi - s_lo);
                    l1 = exp(x1) - 1.0;
                    if (eval_obs_lagrangian_lb(ham, obs, E_ub, C_H, C_O, l1, sign_o, obs->O_core,
                                               nD, &opts, &f1, err, err_sz) != 0) {
                        free(C_H);
                        free(C_O);
                        return -1;
                    }
                    if (f1 > best_b) { best_b = f1; best_lam = l1; }
                }
            }
        }

        if (side == 0) {
            out->O_lb_cert = best_b;
            out->lam_lo = best_lam;
        } else {
            out->O_ub_cert = -best_b;
            out->lam_hi = best_lam;
        }
    }

    out->interval_width = out->O_ub_cert - out->O_lb_cert;
    out->elapsed_sec = now_sec() - t0;
    free(C_H);
    free(C_O);
    return 0;
}

/* ============================================================================
 * Adjoint Self-Test & FCI Consistency Audit
 * ============================================================================ */

static double rand_sym_double(uint64_t *rng) {
    uint64_t z = (*rng += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94D049BB133111EB);
    z ^= (z >> 31U);
    return ((double)(z >> 11U) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
}

int chem_test_adjoints_t2(int K, int na, int nb, int two_s, int constrain_s2, ChemMode mode,
                          int t2_norbs, const int *t2_orbs, uint64_t seed,
                          ChemAdjointReport *rep, char *err, size_t err_sz) {
    memset(rep, 0, sizeof(*rep));
    ChemMFContext ctx;
    if (mf_ctx_init_ext(&ctx, K, na, nb, two_s, constrain_s2, mode, NULL,
                        (t2_norbs > 0) ? 1 : 0, K, t2_norbs, t2_orbs, 1, err, err_sz) != 0) {
        return -1;
    }

    int nD = ctx.nD, nQ = ctx.nQ, nG = ctx.nG, nT2 = ctx.nT2, meq = ctx.meq;
    int max_out = nQ;
    if (nG > max_out) max_out = nG;
    if (nT2 > max_out) max_out = nT2;
    if (meq > max_out) max_out = meq;

    double *X = (double *)malloc((size_t)nD * sizeof(double));
    double *LX = (double *)malloc((size_t)max_out * sizeof(double));
    double *Y = (double *)malloc((size_t)max_out * sizeof(double));
    double *LsY = (double *)malloc((size_t)nD * sizeof(double));
    if (!X || !LX || !Y || !LsY) {
        free(X); free(LX); free(Y); free(LsY);
        mf_ctx_free(&ctx);
        set_err(err, err_sz, "out of memory in chem_test_adjoints");
        return -1;
    }

    uint64_t rng = seed ? seed : UINT64_C(0xCAFEBABE12345678);
    for (int i = 0; i < nD; i++) X[i] = rand_sym_double(&rng);

    /* 1. L_Q */
    for (int i = 0; i < nQ; i++) Y[i] = rand_sym_double(&rng);
    mf_LQ_fwd(&ctx, X, LX);
    mf_LQ_adj(&ctx, Y, LsY);
    double lhs = dot_vec(nQ, LX, Y);
    double rhs = dot_vec(nD, X, LsY);
    double sc = fmax(1.0, norm2_vec(nQ, LX) * norm2_vec(nQ, Y));
    rep->err_LQ = fabs(lhs - rhs) / sc;

    /* 2. L_G */
    for (int i = 0; i < nG; i++) Y[i] = rand_sym_double(&rng);
    mf_LG_fwd(&ctx, X, LX);
    mf_LG_adj(&ctx, Y, LsY);
    lhs = dot_vec(nG, LX, Y);
    rhs = dot_vec(nD, X, LsY);
    sc = fmax(1.0, norm2_vec(nG, LX) * norm2_vec(nG, Y));
    rep->err_LG = fabs(lhs - rhs) / sc;

    /* 3. A_eq */
    for (int i = 0; i < meq; i++) Y[i] = rand_sym_double(&rng);
    mf_Aeq_fwd(&ctx, X, LX);
    mf_Aeq_adj(&ctx, Y, LsY);
    lhs = dot_vec(meq, LX, Y);
    rhs = dot_vec(nD, X, LsY);
    sc = fmax(1.0, norm2_vec(meq, LX) * norm2_vec(meq, Y));
    rep->err_Aeq = fabs(lhs - rhs) / sc;

    /* 4. L_T2 */
    if (ctx.t2_enabled && nT2 > 0) {
        for (int i = 0; i < nT2; i++) Y[i] = rand_sym_double(&rng);
        mf_LT2_fwd(&ctx, X, LX);
        mf_LT2_adj(&ctx, Y, LsY);
        lhs = dot_vec(nT2, LX, Y);
        rhs = dot_vec(nD, X, LsY);
        sc = fmax(1.0, norm2_vec(nT2, LX) * norm2_vec(nT2, Y));
        rep->err_LT2 = fabs(lhs - rhs) / sc;
    } else {
        rep->err_LT2 = 0.0;
    }

    rep->max_err = fmax(fmax(rep->err_LQ, rep->err_LG), fmax(rep->err_Aeq, rep->err_LT2));

    free(X); free(LX); free(Y); free(LsY);
    mf_ctx_free(&ctx);
    return 0;
}

int chem_test_adjoints(int K, int na, int nb, int two_s, int constrain_s2, ChemMode mode,
                       uint64_t seed, ChemAdjointReport *rep, char *err, size_t err_sz) {
    return chem_test_adjoints_t2(K, na, nb, two_s, constrain_s2, mode, 0, NULL, seed, rep, err,
                                 err_sz);
}

int chem_verify_fci_t2_psd(const ChemHam *ham, ChemMode mode, int two_s,
                           int t2_norbs, const int *t2_orbs, const double *D_svec,
                           double *min_eig_out, double *trace_err_out,
                           char *err, size_t err_sz) {
    ChemMFContext ctx;
    if (mf_ctx_init_ext(&ctx, ham->K, ham->na, ham->nb, two_s, 1, mode, NULL,
                        1, ham->K, t2_norbs, t2_orbs, 1, err, err_sz) != 0) {
        return -1;
    }
    double *T2_sv = (double *)malloc((size_t)(ctx.nT2 > 0 ? ctx.nT2 : 1) * sizeof(double));
    if (!T2_sv) {
        mf_ctx_free(&ctx);
        set_err(err, err_sz, "out of memory in chem_verify_fci_t2_psd");
        return -1;
    }
    mf_LT2_fwd(&ctx, D_svec, T2_sv);

    int t2_start = (mode == CHEM_MODE_A_SINGLET) ? 6 : 9;
    int off = 0;
    double min_ev = 1e30;
    double max_tr_err = 0.0;
    for (int b = t2_start; b < ctx.nblocks; b++) {
        int dim = ctx.block_dims[b];
        int sz = ctx.block_szs[b];
        if (dim > 0) {
            if (accel_syevd(&ctx, dim, T2_sv + off, 0, ctx.scr_M0, ctx.ev_val, NULL) != 0) {
                free(T2_sv);
                mf_ctx_free(&ctx);
                set_err(err, err_sz, "dsyevd_ failed in chem_verify_fci_t2_psd");
                return -1;
            }
            if (ctx.ev_val[0] < min_ev) min_ev = ctx.ev_val[0];
            double tr = 0.0;
            for (int k = 0; k < dim; k++) tr += ctx.ev_val[k];
            double err_t = fabs(tr - ctx.block_traces[b]);
            if (err_t > max_tr_err) max_tr_err = err_t;
        }
        off += sz;
    }
    if (min_eig_out) *min_eig_out = min_ev;
    if (trace_err_out) *trace_err_out = max_tr_err;
    free(T2_sv);
    mf_ctx_free(&ctx);
    return 0;
}

int chem_verify_fci_dqg_consistency(const ChemHam *ham, ChemMode mode, int two_s,
                                    const double *D_svec, const double *Q_svec, const double *G_svec,
                                    double *car_map_err_out, double *eq_err_out,
                                    double *min_eig_out, double *energy_err_out,
                                    char *err, size_t err_sz) {
    ChemMFContext ctx;
    if (mf_ctx_init(&ctx, ham->K, ham->na, ham->nb, two_s, 1, mode, ham, err, err_sz) != 0) {
        return -1;
    }

    double *Q_map = (double *)malloc((size_t)ctx.nQ * sizeof(double));
    double *G_map = (double *)malloc((size_t)ctx.nG * sizeof(double));
    double *eq_v  = (double *)malloc((size_t)ctx.meq * sizeof(double));
    if (!Q_map || !G_map || !eq_v) {
        free(Q_map); free(G_map); free(eq_v);
        mf_ctx_free(&ctx);
        set_err(err, err_sz, "out of memory in FCI consistency check");
        return -1;
    }

    mf_LQ_fwd(&ctx, D_svec, Q_map);
    mf_LG_fwd(&ctx, D_svec, G_map);
    mf_Aeq_fwd(&ctx, D_svec, eq_v);

    double max_car = 0.0;
    for (int i = 0; i < ctx.nQ; i++) {
        double d = fabs((ctx.q0[i] + Q_map[i]) - Q_svec[i]);
        if (d > max_car) max_car = d;
    }
    for (int i = 0; i < ctx.nG; i++) {
        double d = fabs(G_map[i] - G_svec[i]);
        if (d > max_car) max_car = d;
    }

    double max_eq = 0.0;
    for (int m = 0; m < ctx.meq; m++) {
        double d = fabs(eq_v[m] - ctx.b_eq[m]);
        if (d > max_eq) max_eq = d;
    }

    /* Check PSD across all blocks */
    int nD_blk = (mode == CHEM_MODE_A_SINGLET) ? 2 : 3;
    int off = 0;
    double min_ev = 1e30;
    for (int b = 0; b < ctx.nblocks; b++) {
        int dim = ctx.block_dims[b];
        int sz = ctx.block_szs[b];
        const double *ptr = NULL;
        int boff = off;
        if (b < nD_blk) {
            ptr = D_svec + boff;
            off = (b == nD_blk - 1) ? 0 : off + sz;
        } else if (b < 2 * nD_blk) {
            ptr = Q_svec + boff;
            off = (b == 2 * nD_blk - 1) ? 0 : off + sz;
        } else {
            ptr = G_svec + boff;
            off += sz;
        }
        if (dim > 0) {
            accel_syevd(&ctx, dim, ptr, 0, ctx.scr_M0, ctx.ev_val, NULL);
            if (ctx.ev_val[0] < min_ev) min_ev = ctx.ev_val[0];
        }
    }

    double E_from_D = ham->E_core + dot_vec(ctx.nD, ctx.C_D, D_svec);
    if (car_map_err_out) *car_map_err_out = max_car;
    if (eq_err_out) *eq_err_out = max_eq;
    if (min_eig_out) *min_eig_out = min_ev;
    if (energy_err_out) *energy_err_out = E_from_D;

    free(Q_map); free(G_map); free(eq_v);
    mf_ctx_free(&ctx);
    return 0;
}

/* ============================================================================
 * Proof-Carrying JSON Certificate Writer & Independent Verifier
 * ============================================================================ */

static void write_json_array_f64(FILE *fp, const double *arr, int n) {
    fputc('[', fp);
    for (int i = 0; i < n; i++) {
        if (i > 0) fputc(',', fp);
        fprintf(fp, "%.17g", (arr[i] == 0.0) ? 0.0 : arr[i]);
    }
    fputc(']', fp);
}

static void write_json_escaped_str(FILE *fp, const char *s) {
    fputc('"', fp);
    for (const char *p = s; p && *p; p++) {
        if (*p == '"' || *p == '\\') fputc('\\', fp);
        fputc(*p, fp);
    }
    fputc('"', fp);
}

static void write_json_array_i32(FILE *fp, const int *arr, int n) {
    fputc('[', fp);
    for (int i = 0; i < n; i++) {
        if (i > 0) fputc(',', fp);
        fprintf(fp, "%d", arr[i]);
    }
    fputc(']', fp);
}

int chem_cert_write_json(const char *path, const ChemHam *ham, const char *fcidump_path,
                         const ChemSolveResult *res, const struct ChemVarResult *var_opt,
                         const ChemSolveOpts *opts_in, char *err, size_t err_sz) {
    if (!path || !ham || !res) {
        set_err(err, err_sz, "null argument to chem_cert_write_json");
        return -1;
    }
    FILE *fp = fopen(path, "w");
    if (!fp) {
        set_err(err, err_sz, "cannot open certificate output path: %s", path);
        return -1;
    }

    ChemSolveOpts opts = opts_in ? *opts_in : chem_solve_opts_default(ham);
    const char *mode_str = (res->mode_used == CHEM_MODE_A_SINGLET) ? "A_SINGLET" : "B_GENERAL";

    fprintf(fp, "{\n");
    fprintf(fp, "  \"schema_version\": \"%s\",\n", res->witness.t2_enabled ? "chem-cert-v2" : "chem-cert-v1");
    fprintf(fp, "  \"fcidump_path\": ");
    write_json_escaped_str(fp, fcidump_path ? fcidump_path : "");
    fprintf(fp, ",\n");
    fprintf(fp, "  \"hamiltonian_sha256\": \"%s\",\n", ham->sha256_hex);
    fprintf(fp, "  \"K\": %d,\n", ham->K);
    fprintf(fp, "  \"nelec\": %d,\n", ham->nelec);
    fprintf(fp, "  \"ms2\": %d,\n", ham->ms2);
    fprintf(fp, "  \"na\": %d,\n", ham->na);
    fprintf(fp, "  \"nb\": %d,\n", ham->nb);
    fprintf(fp, "  \"two_s\": %d,\n", res->witness.two_s);
    fprintf(fp, "  \"constrain_s2\": %d,\n", res->witness.constrain_s2);
    fprintf(fp, "  \"t2_enabled\": %d,\n", res->witness.t2_enabled);
    fprintf(fp, "  \"t2_norbs\": %d,\n", res->witness.t2_norbs);
    fprintf(fp, "  \"t2_orbs\": ");
    write_json_array_i32(fp, res->witness.t2_orbs, res->witness.t2_norbs);
    fprintf(fp, ",\n");
    fprintf(fp, "  \"mode\": \"%s\",\n", mode_str);
    fprintf(fp, "  \"E_core\": %.17g,\n", ham->E_core);

    fprintf(fp, "  \"solver\": {\n");
    fprintf(fp, "    \"iters\": %d,\n", res->iters);
    fprintf(fp, "    \"converged\": %d,\n", res->converged);
    fprintf(fp, "    \"vetoed\": %d,\n", res->vetoed);
    fprintf(fp, "    \"iter_stopped\": %d,\n", res->iter_stopped);
    fprintf(fp, "    \"E_primal\": %.17g,\n", res->E_primal);
    fprintf(fp, "    \"E_dual_raw\": %.17g,\n", res->E_dual_raw);
    fprintf(fp, "    \"E_lb_cert\": %.17g,\n", res->E_lb_cert);
    fprintf(fp, "    \"min_eig_fp\": %.17g,\n", res->min_eig_fp);
    fprintf(fp, "    \"min_eig_cert\": %.17g,\n", res->min_eig_cert);
    fprintf(fp, "    \"delta_round\": %.17g,\n", res->delta_round);
    fprintf(fp, "    \"prim_res\": %.17g,\n", res->prim_res);
    fprintf(fp, "    \"dual_res\": %.17g,\n", res->dual_res);
    fprintf(fp, "    \"elapsed_sec\": %.6f\n", res->elapsed_sec);
    fprintf(fp, "  },\n");

    double e_ub_v = var_opt ? var_opt->E_ub : res->E_primal;
    fprintf(fp, "  \"veto\": {\n");
    fprintf(fp, "    \"enabled\": %d,\n", opts.veto_enabled);
    fprintf(fp, "    \"E_ai\": %.17g,\n", opts.E_ai);
    fprintf(fp, "    \"veto_tol\": %.17g,\n", opts.veto_tol);
    fprintf(fp, "    \"vetoed\": %d,\n", res->vetoed);
    fprintf(fp, "    \"iter_stopped\": %d,\n", res->iter_stopped);
    fprintf(fp, "    \"E_ub_variational\": %.17g,\n", e_ub_v);
    fprintf(fp, "    \"interval_width\": %.17g\n", e_ub_v - res->E_lb_cert);
    fprintf(fp, "  },\n");

    long long lb_scaled = (long long)floor(res->E_lb_cert * 1e9);
    long long ub_scaled = (long long)ceil(e_ub_v * 1e9);
    long long ai_scaled = (long long)floor((opts.E_ai - opts.veto_tol) * 1e9);
    fprintf(fp, "  \"lean_scaled_witness\": {\n");
    fprintf(fp, "    \"scale_nha\": 1000000000,\n");
    fprintf(fp, "    \"E_lb_floor\": %lld,\n", lb_scaled);
    fprintf(fp, "    \"E_ub_ceil\": %lld,\n", ub_scaled);
    fprintf(fp, "    \"E_ai_minus_tol_floor\": %lld\n", ai_scaled);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"variational\": {\n");
    if (var_opt) {
        int emit_w = (var_opt->ndet_var > 0 && var_opt->ndet_var <= 2048 && var_opt->dets &&
                      var_opt->psi0)
                         ? 1
                         : 0;
        fprintf(fp, "    \"E_hf0\": %.17g,\n", var_opt->E_hf0);
        fprintf(fp, "    \"E_scf\": %.17g,\n", var_opt->E_scf);
        fprintf(fp, "    \"E_searu_rot\": %.17g,\n", var_opt->E_searu_rot);
        fprintf(fp, "    \"E_ci\": %.17g,\n", var_opt->E_ci);
        fprintf(fp, "    \"E_ub\": %.17g,\n", var_opt->E_ub);
        fprintf(fp, "    \"is_exact_fci\": %d,\n", var_opt->is_exact_fci);
        fprintf(fp, "    \"ndet_space\": %llu,\n", (unsigned long long)var_opt->ndet_space);
        fprintf(fp, "    \"ndet_var\": %d,\n", var_opt->ndet_var);
        fprintf(fp, "    \"has_witness\": %d", emit_w);
        if (emit_w) {
            fprintf(fp, ",\n    \"dets\": [");
            for (int i = 0; i < var_opt->ndet_var; i++) {
                if (i > 0) fputc(',', fp);
                fprintf(fp, "\"0x%016llx\"", (unsigned long long)var_opt->dets[i]);
            }
            fprintf(fp, "],\n    \"psi0\": ");
            write_json_array_f64(fp, var_opt->psi0, var_opt->ndet_var);
            fputc('\n', fp);
        } else {
            fputc('\n', fp);
        }
    } else {
        fprintf(fp, "    \"has_witness\": 0,\n");
        fprintf(fp, "    \"E_ub\": %.17g\n", e_ub_v);
    }
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"dual_witness\": {\n");
    fprintf(fp, "    \"meq\": %d,\n", res->witness.meq);
    fprintf(fp, "    \"nD\": %d,\n", res->witness.nD);
    fprintf(fp, "    \"nQ\": %d,\n", res->witness.nQ);
    fprintf(fp, "    \"nG\": %d,\n", res->witness.nG);
    fprintf(fp, "    \"nT2\": %d,\n", res->witness.nT2);
    fprintf(fp, "    \"y_eq\": ");
    write_json_array_f64(fp, res->witness.y_eq, res->witness.meq);
    fprintf(fp, ",\n    \"Z_Q\": ");
    write_json_array_f64(fp, res->witness.Z_Q, res->witness.nQ);
    fprintf(fp, ",\n    \"Z_G\": ");
    write_json_array_f64(fp, res->witness.Z_G, res->witness.nG);
    if (res->witness.t2_enabled && res->witness.nT2 > 0 && res->witness.Z_T2) {
        fprintf(fp, ",\n    \"Z_T2\": ");
        write_json_array_f64(fp, res->witness.Z_T2, res->witness.nT2);
    }
    fprintf(fp, "\n  }\n}\n");

    fclose(fp);
    return 0;
}

typedef struct {
    char magic[8]; /* "FLCH0002" */
    char sha256_hex[65];
    char pad[7];
    uint32_t fcidump_len;
    int32_t K, na, nb, two_s, constrain_s2, mode;
    int32_t t2_enabled, t2_norbs;
    int32_t t2_orbs[CHEM_MAX_ORBITALS];
    int32_t veto_enabled, vetoed, iter_stopped, converged, iters;
    int32_t has_witness, ndet_var;
    int32_t meq, nD, nQ, nG, nT2;
    double E_lb_cert, E_ub, E_ci, E_ai, veto_tol;
} ChemBinCertHeader;

int chem_cert_write_bin(const char *path, const ChemHam *ham, const char *fcidump_path,
                        const ChemSolveResult *res, const struct ChemVarResult *var_opt,
                        const ChemSolveOpts *opts_in, char *err, size_t err_sz) {
    if (!path || !ham || !res) {
        set_err(err, err_sz, "null argument to chem_cert_write_bin");
        return -1;
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        set_err(err, err_sz, "cannot open binary cert output path: %s", path);
        return -1;
    }
    ChemSolveOpts opts = opts_in ? *opts_in : chem_solve_opts_default(ham);
    const char *fstr = fcidump_path ? fcidump_path : "";
    uint32_t flen = (uint32_t)strlen(fstr);
    int emit_w = (var_opt && var_opt->ndet_var > 0 && var_opt->ndet_var <= 2048 &&
                  var_opt->dets && var_opt->psi0)
                     ? 1
                     : 0;

    ChemBinCertHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "FLCH0002", 8);
    memcpy(hdr.sha256_hex, ham->sha256_hex, 65);
    hdr.fcidump_len = flen;
    hdr.K = ham->K;
    hdr.na = ham->na;
    hdr.nb = ham->nb;
    hdr.two_s = res->witness.two_s;
    hdr.constrain_s2 = res->witness.constrain_s2;
    hdr.mode = (int32_t)res->mode_used;
    hdr.t2_enabled = res->witness.t2_enabled;
    hdr.t2_norbs = res->witness.t2_norbs;
    for (int i = 0; i < res->witness.t2_norbs && i < CHEM_MAX_ORBITALS; i++) {
        hdr.t2_orbs[i] = res->witness.t2_orbs[i];
    }
    hdr.veto_enabled = opts.veto_enabled;
    hdr.vetoed = res->vetoed;
    hdr.iter_stopped = res->iter_stopped;
    hdr.converged = res->converged;
    hdr.iters = res->iters;
    hdr.has_witness = emit_w;
    hdr.ndet_var = emit_w ? var_opt->ndet_var : 0;
    hdr.meq = res->witness.meq;
    hdr.nD = res->witness.nD;
    hdr.nQ = res->witness.nQ;
    hdr.nG = res->witness.nG;
    hdr.nT2 = res->witness.nT2;
    hdr.E_lb_cert = res->E_lb_cert;
    hdr.E_ub = var_opt ? var_opt->E_ub : res->E_primal;
    hdr.E_ci = var_opt ? var_opt->E_ci : res->E_primal;
    hdr.E_ai = opts.E_ai;
    hdr.veto_tol = opts.veto_tol;

    int ok = 1;
    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) ok = 0;
    if (ok && flen > 0 && fwrite(fstr, 1, (size_t)flen, fp) != (size_t)flen) ok = 0;
    if (ok && hdr.meq > 0 && fwrite(res->witness.y_eq, sizeof(double), (size_t)hdr.meq, fp) != (size_t)hdr.meq) ok = 0;
    if (ok && hdr.nQ > 0 && fwrite(res->witness.Z_Q, sizeof(double), (size_t)hdr.nQ, fp) != (size_t)hdr.nQ) ok = 0;
    if (ok && hdr.nG > 0 && fwrite(res->witness.Z_G, sizeof(double), (size_t)hdr.nG, fp) != (size_t)hdr.nG) ok = 0;
    if (ok && hdr.nT2 > 0 && res->witness.Z_T2 &&
        fwrite(res->witness.Z_T2, sizeof(double), (size_t)hdr.nT2, fp) != (size_t)hdr.nT2) ok = 0;
    if (ok && emit_w) {
        if (fwrite(var_opt->dets, sizeof(uint64_t), (size_t)hdr.ndet_var, fp) != (size_t)hdr.ndet_var ||
            fwrite(var_opt->psi0, sizeof(double), (size_t)hdr.ndet_var, fp) != (size_t)hdr.ndet_var) {
            ok = 0;
        }
    }
    fclose(fp);
    if (!ok) {
        set_err(err, err_sz, "short write to binary cert: %s", path);
        return -1;
    }
    return 0;
}

/* Minimal, strict JSON helper for chem_cert_verify_file */
static char *read_entire_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    if (sz < 0) {
        fclose(fp);
        return NULL;
    }
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1U);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t nr = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[nr] = '\0';
    if (len_out) *len_out = nr;
    return buf;
}

static const char *json_find_key_in_range(const char *start, const char *end, const char *key) {
    size_t klen = strlen(key);
    const char *p = start;
    while (p < end) {
        const char *q = strchr(p, '"');
        if (!q || q >= end) return NULL;
        const char *q2 = q + 1;
        while (q2 < end && *q2 != '"') {
            if (*q2 == '\\' && q2 + 1 < end) q2 += 2;
            else q2++;
        }
        if (q2 >= end) return NULL;
        if ((size_t)(q2 - (q + 1)) == klen && memcmp(q + 1, key, klen) == 0) {
            const char *c = q2 + 1;
            while (c < end && isspace((unsigned char)*c)) c++;
            if (c < end && *c == ':') {
                c++;
                while (c < end && isspace((unsigned char)*c)) c++;
                return c;
            }
        }
        p = q2 + 1;
    }
    return NULL;
}

static int json_get_string(const char *start, const char *end, const char *key, char *out,
                           size_t out_sz) {
    const char *v = json_find_key_in_range(start, end, key);
    if (!v || *v != '"') return -1;
    v++;
    size_t i = 0;
    while (v < end && *v != '"') {
        if (*v == '\\' && v + 1 < end) v++;
        if (i + 1 < out_sz) out[i++] = *v;
        v++;
    }
    if (v >= end) return -1;
    out[i] = '\0';
    return 0;
}

static int json_get_int(const char *start, const char *end, const char *key, int *out) {
    const char *v = json_find_key_in_range(start, end, key);
    if (!v) return -1;
    char *ep = NULL;
    long val = strtol(v, &ep, 10);
    if (ep == v) return -1;
    *out = (int)val;
    return 0;
}

static int json_get_double(const char *start, const char *end, const char *key, double *out) {
    const char *v = json_find_key_in_range(start, end, key);
    if (!v) return -1;
    char *ep = NULL;
    double val = strtod(v, &ep);
    if (ep == v) return -1;
    *out = val;
    return 0;
}

static int json_get_object_range(const char *start, const char *end, const char *key,
                                 const char **obj_s, const char **obj_e) {
    const char *v = json_find_key_in_range(start, end, key);
    if (!v || *v != '{') return -1;
    int depth = 0;
    int in_str = 0;
    for (const char *p = v; p < end; p++) {
        if (in_str) {
            if (*p == '\\' && p + 1 < end) p++;
            else if (*p == '"') in_str = 0;
            continue;
        }
        if (*p == '"') in_str = 1;
        else if (*p == '{') depth++;
        else if (*p == '}') {
            depth--;
            if (depth == 0) {
                *obj_s = v;
                *obj_e = p + 1;
                return 0;
            }
        }
    }
    return -1;
}

static int json_get_int_array(const char *start, const char *end, const char *key,
                              int expected_n, int *arr_out) {
    const char *v = json_find_key_in_range(start, end, key);
    if (!v || *v != '[') return -1;
    v++;
    int cnt = 0;
    const char *p = v;
    while (p < end) {
        while (p < end && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (p < end && *p == ']') break;
        if (cnt >= expected_n) return -1;
        char *ep = NULL;
        arr_out[cnt++] = (int)strtol(p, &ep, 10);
        if (ep == p) return -1;
        p = ep;
    }
    return (cnt == expected_n) ? 0 : -1;
}

static int json_get_double_array(const char *start, const char *end, const char *key,
                                 int expected_n, double **arr_out) {
    const char *v = json_find_key_in_range(start, end, key);
    if (!v || *v != '[') return -1;
    v++;
    double *arr = (double *)malloc((size_t)(expected_n > 0 ? expected_n : 1) * sizeof(double));
    if (!arr) return -1;
    int cnt = 0;
    const char *p = v;
    while (p < end) {
        while (p < end && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (p < end && *p == ']') break;
        if (cnt >= expected_n) {
            free(arr);
            return -1;
        }
        char *ep = NULL;
        arr[cnt++] = strtod(p, &ep);
        if (ep == p) {
            free(arr);
            return -1;
        }
        p = ep;
    }
    if (cnt != expected_n) {
        free(arr);
        return -1;
    }
    *arr_out = arr;
    return 0;
}

static int json_get_u64_hex_array(const char *start, const char *end, const char *key,
                                  int expected_n, uint64_t **arr_out) {
    const char *v = json_find_key_in_range(start, end, key);
    if (!v || *v != '[') return -1;
    v++;
    uint64_t *arr = (uint64_t *)malloc((size_t)expected_n * sizeof(uint64_t));
    if (!arr) return -1;
    int cnt = 0;
    const char *p = v;
    while (p < end) {
        while (p < end && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (p < end && *p == ']') break;
        if (cnt >= expected_n || *p != '"') {
            free(arr);
            return -1;
        }
        p++;
        char *ep = NULL;
        arr[cnt++] = (uint64_t)strtoull(p, &ep, 16);
        if (ep == p || *ep != '"') {
            free(arr);
            return -1;
        }
        p = ep + 1;
    }
    if (cnt != expected_n) {
        free(arr);
        return -1;
    }
    *arr_out = arr;
    return 0;
}

static int verify_cert_bin(const char *fcidump_override, const char *buf, size_t len,
                           ChemCertVerifyResult *out) {
    out->is_binary = 1;
    if (len < sizeof(ChemBinCertHeader)) {
        snprintf(out->msg, sizeof(out->msg), "truncated binary certificate");
        return -1;
    }
    ChemBinCertHeader hdr;
    memcpy(&hdr, buf, sizeof(hdr));
    if (hdr.meq < 0 || hdr.nQ < 0 || hdr.nG < 0 || hdr.nT2 < 0 || hdr.ndet_var < 0 ||
        hdr.fcidump_len > 4096U || hdr.t2_norbs < 0 || hdr.t2_norbs > CHEM_MAX_ORBITALS) {
        snprintf(out->msg, sizeof(out->msg), "corrupt binary certificate header dimensions");
        return -1;
    }
    size_t expect = sizeof(hdr) + (size_t)hdr.fcidump_len +
                    sizeof(double) * ((size_t)hdr.meq + (size_t)hdr.nQ + (size_t)hdr.nG + (size_t)hdr.nT2) +
                    (hdr.has_witness ? ((sizeof(uint64_t) + sizeof(double)) * (size_t)hdr.ndet_var) : 0U);
    if (len != expect) {
        snprintf(out->msg, sizeof(out->msg), "binary certificate length mismatch (%zu != %zu)",
                 len, expect);
        return -1;
    }

    char fcidump_file[4098] = {0};
    const char *ptr = buf + sizeof(hdr);
    if (fcidump_override && fcidump_override[0]) {
        snprintf(fcidump_file, sizeof(fcidump_file), "%s", fcidump_override);
    } else if (hdr.fcidump_len > 0) {
        memcpy(fcidump_file, ptr, (size_t)hdr.fcidump_len);
        fcidump_file[hdr.fcidump_len] = '\0';
    } else {
        snprintf(out->msg, sizeof(out->msg), "binary certificate missing fcidump_path");
        return -1;
    }
    ptr += hdr.fcidump_len;

    ChemHam ham;
    char err[256] = {0};
    if (chem_fcidump_load(fcidump_file, &ham, err, sizeof(err)) != 0) {
        snprintf(out->msg, sizeof(out->msg), "FCIDUMP load failed: %s", err);
        return -1;
    }
    hdr.sha256_hex[64] = '\0';
    if (strcmp(ham.sha256_hex, hdr.sha256_hex) != 0) {
        out->sha_ok = 0;
        snprintf(out->msg, sizeof(out->msg), "SHA-256 mismatch in binary cert");
        chem_ham_free(&ham);
        return -1;
    }
    out->sha_ok = 1;
    out->E_lb_claimed = hdr.E_lb_cert;

    ChemDualWitness dw;
    memset(&dw, 0, sizeof(dw));
    dw.mode = (ChemMode)hdr.mode;
    dw.K = ham.K;
    dw.na = ham.na;
    dw.nb = ham.nb;
    dw.two_s = hdr.two_s;
    dw.constrain_s2 = hdr.constrain_s2;
    dw.t2_enabled = hdr.t2_enabled;
    dw.t2_norbs = hdr.t2_norbs;
    for (int i = 0; i < hdr.t2_norbs; i++) dw.t2_orbs[i] = hdr.t2_orbs[i];
    dw.meq = hdr.meq;
    dw.nD = hdr.nD;
    dw.nQ = hdr.nQ;
    dw.nG = hdr.nG;
    dw.nT2 = hdr.nT2;

    if (dw.meq > 0) {
        dw.y_eq = (double *)malloc((size_t)dw.meq * sizeof(double));
        memcpy(dw.y_eq, ptr, (size_t)dw.meq * sizeof(double));
        ptr += (size_t)dw.meq * sizeof(double);
    }
    if (dw.nQ > 0) {
        dw.Z_Q = (double *)malloc((size_t)dw.nQ * sizeof(double));
        memcpy(dw.Z_Q, ptr, (size_t)dw.nQ * sizeof(double));
        ptr += (size_t)dw.nQ * sizeof(double);
    }
    if (dw.nG > 0) {
        dw.Z_G = (double *)malloc((size_t)dw.nG * sizeof(double));
        memcpy(dw.Z_G, ptr, (size_t)dw.nG * sizeof(double));
        ptr += (size_t)dw.nG * sizeof(double);
    }
    if (dw.nT2 > 0) {
        dw.Z_T2 = (double *)malloc((size_t)dw.nT2 * sizeof(double));
        memcpy(dw.Z_T2, ptr, (size_t)dw.nT2 * sizeof(double));
        ptr += (size_t)dw.nT2 * sizeof(double);
    }

    ChemSolveResult recomp;
    memset(&recomp, 0, sizeof(recomp));
    if (dw.meq == 0 && dw.nD == 0) {
        ChemSolveOpts topts = chem_solve_opts_default(&ham);
        topts.mode = dw.mode;
        topts.two_s = dw.two_s;
        topts.constrain_s2 = dw.constrain_s2;
        if (solve_trivial_space_cost(&ham, ham.E_core, NULL, dw.mode, dw.two_s, dw.constrain_s2,
                                     &topts, &recomp, err, sizeof(err)) != 0) {
            chem_dual_witness_free(&dw);
            chem_ham_free(&ham);
            snprintf(out->msg, sizeof(out->msg), "trivial space verification failed");
            return -1;
        }
    } else if (chem_certify_dual_witness(&ham, &dw, &recomp, err, sizeof(err)) != 0) {
        chem_dual_witness_free(&dw);
        chem_ham_free(&ham);
        snprintf(out->msg, sizeof(out->msg), "binary dual witness failed: %s", err);
        return -1;
    }

    out->E_lb_recomputed = recomp.E_lb_cert;
    out->min_eig_cert = recomp.min_eig_cert;
    out->lb_ok = (fabs(recomp.E_lb_cert - hdr.E_lb_cert) <= 1e-8) ? 1 : 0;
    out->veto_ok = (!hdr.vetoed || (recomp.E_lb_cert > hdr.E_ai - hdr.veto_tol)) ? 1 : 0;
    out->var_ok = 1;
    out->E_var_recomputed = NAN;
    if (hdr.has_witness && hdr.ndet_var > 0) {
        uint64_t *dets = (uint64_t *)malloc((size_t)hdr.ndet_var * sizeof(uint64_t));
        double *psi0 = (double *)malloc((size_t)hdr.ndet_var * sizeof(double));
        memcpy(dets, ptr, (size_t)hdr.ndet_var * sizeof(uint64_t));
        ptr += (size_t)hdr.ndet_var * sizeof(uint64_t);
        memcpy(psi0, ptr, (size_t)hdr.ndet_var * sizeof(double));
        double e_re = 0.0;
        if (chem_var_eval_witness(&ham, hdr.ndet_var, dets, psi0, &e_re, err, sizeof(err)) != 0) {
            out->var_ok = 0;
        } else {
            out->E_var_recomputed = e_re;
            if (fabs(e_re - hdr.E_ci) > 1e-6 || recomp.E_lb_cert > e_re + 1e-9 ||
                recomp.E_lb_cert > hdr.E_ub + 1e-9) {
                out->var_ok = 0;
            }
        }
        free(dets);
        free(psi0);
    }

    out->ok = (out->sha_ok && out->lb_ok && out->veto_ok && out->var_ok) ? 1 : 0;
    if (out->ok) {
        snprintf(out->msg, sizeof(out->msg),
                 "BINARY CERT VERIFIED: sha=OK E_lb=%.8f (diff=%.2e)",
                 recomp.E_lb_cert, fabs(recomp.E_lb_cert - hdr.E_lb_cert));
    } else {
        snprintf(out->msg, sizeof(out->msg),
                 "BINARY CERT FAILED: sha=%d lb=%d veto=%d var=%d",
                 out->sha_ok, out->lb_ok, out->veto_ok, out->var_ok);
    }

    chem_solve_result_free(&recomp);
    chem_dual_witness_free(&dw);
    chem_ham_free(&ham);
    return out->ok ? 0 : -1;
}

int chem_cert_verify_file(const char *fcidump_override, const char *cert_path,
                          ChemCertVerifyResult *out) {
    memset(out, 0, sizeof(*out));
    size_t len = 0;
    char *txt = read_entire_file(cert_path, &len);
    if (!txt) {
        snprintf(out->msg, sizeof(out->msg), "cannot read certificate: %s", cert_path);
        return -1;
    }
    if (len >= 8U && memcmp(txt, "FLCH0002", 8) == 0) {
        int rc = verify_cert_bin(fcidump_override, txt, len, out);
        free(txt);
        return rc;
    }

    const char *s = txt, *e = txt + len;

    char fcidump_file[1024] = {0};
    char sha_claimed[128] = {0};
    char mode_str[64] = {0};
    int two_s = 0, cs2 = 1;
    if (json_get_string(s, e, "hamiltonian_sha256", sha_claimed, sizeof(sha_claimed)) != 0 ||
        json_get_string(s, e, "mode", mode_str, sizeof(mode_str)) != 0 ||
        json_get_int(s, e, "two_s", &two_s) != 0 ||
        json_get_int(s, e, "constrain_s2", &cs2) != 0) {
        free(txt);
        snprintf(out->msg, sizeof(out->msg), "malformed top-level certificate fields");
        return -1;
    }
    if (fcidump_override && fcidump_override[0]) {
        snprintf(fcidump_file, sizeof(fcidump_file), "%s", fcidump_override);
    } else if (json_get_string(s, e, "fcidump_path", fcidump_file, sizeof(fcidump_file)) != 0 ||
               fcidump_file[0] == '\0') {
        free(txt);
        snprintf(out->msg, sizeof(out->msg), "certificate missing fcidump_path");
        return -1;
    }

    ChemHam ham;
    char err[256] = {0};
    if (chem_fcidump_load(fcidump_file, &ham, err, sizeof(err)) != 0) {
        free(txt);
        snprintf(out->msg, sizeof(out->msg), "FCIDUMP load failed: %s", err);
        return -1;
    }
    if (strcmp(ham.sha256_hex, sha_claimed) != 0) {
        out->sha_ok = 0;
        snprintf(out->msg, sizeof(out->msg), "SHA-256 mismatch (expected %.16s..., got %.16s...)",
                 sha_claimed, ham.sha256_hex);
        chem_ham_free(&ham);
        free(txt);
        return -1;
    }
    out->sha_ok = 1;

    int t2_enabled = 0, t2_norbs = 0;
    int t2_orbs[CHEM_MAX_ORBITALS] = {0};
    if (json_get_int(s, e, "t2_enabled", &t2_enabled) == 0 && t2_enabled) {
        if (json_get_int(s, e, "t2_norbs", &t2_norbs) != 0 || t2_norbs <= 0 ||
            t2_norbs > CHEM_MAX_ORBITALS ||
            json_get_int_array(s, e, "t2_orbs", t2_norbs, t2_orbs) != 0) {
            chem_ham_free(&ham);
            free(txt);
            snprintf(out->msg, sizeof(out->msg), "malformed t2 metadata in certificate");
            return -1;
        }
    }

    const char *sol_s = NULL, *sol_e = NULL;
    const char *vet_s = NULL, *vet_e = NULL;
    const char *var_s = NULL, *var_e = NULL;
    const char *dw_s  = NULL, *dw_e  = NULL;
    if (json_get_object_range(s, e, "solver", &sol_s, &sol_e) != 0 ||
        json_get_object_range(s, e, "dual_witness", &dw_s, &dw_e) != 0) {
        chem_ham_free(&ham);
        free(txt);
        snprintf(out->msg, sizeof(out->msg), "missing solver or dual_witness JSON section");
        return -1;
    }

    double E_lb_claimed = 0.0;
    if (json_get_double(sol_s, sol_e, "E_lb_cert", &E_lb_claimed) != 0) {
        chem_ham_free(&ham);
        free(txt);
        snprintf(out->msg, sizeof(out->msg), "missing solver.E_lb_cert");
        return -1;
    }
    out->E_lb_claimed = E_lb_claimed;

    int meq = 0, nD = 0, nQ = 0, nG = 0, nT2 = 0;
    if (json_get_int(dw_s, dw_e, "meq", &meq) != 0 ||
        json_get_int(dw_s, dw_e, "nD", &nD) != 0 ||
        json_get_int(dw_s, dw_e, "nQ", &nQ) != 0 ||
        json_get_int(dw_s, dw_e, "nG", &nG) != 0) {
        chem_ham_free(&ham);
        free(txt);
        snprintf(out->msg, sizeof(out->msg), "missing dual_witness dimensions");
        return -1;
    }
    if (t2_enabled && json_get_int(dw_s, dw_e, "nT2", &nT2) != 0) {
        chem_ham_free(&ham);
        free(txt);
        snprintf(out->msg, sizeof(out->msg), "missing dual_witness.nT2");
        return -1;
    }

    ChemMode mode = (strcmp(mode_str, "A_SINGLET") == 0) ? CHEM_MODE_A_SINGLET
                                                         : CHEM_MODE_B_GENERAL;
    ChemDualWitness dw;
    memset(&dw, 0, sizeof(dw));
    dw.mode = mode;
    dw.K = ham.K;
    dw.na = ham.na;
    dw.nb = ham.nb;
    dw.two_s = two_s;
    dw.constrain_s2 = cs2;
    dw.t2_enabled = t2_enabled;
    dw.t2_norbs = t2_norbs;
    if (t2_enabled) memcpy(dw.t2_orbs, t2_orbs, (size_t)t2_norbs * sizeof(int));
    dw.meq = meq;
    dw.nD = nD;
    dw.nQ = nQ;
    dw.nG = nG;
    dw.nT2 = nT2;

    if (json_get_double_array(dw_s, dw_e, "y_eq", meq, &dw.y_eq) != 0 ||
        json_get_double_array(dw_s, dw_e, "Z_Q", nQ, &dw.Z_Q) != 0 ||
        json_get_double_array(dw_s, dw_e, "Z_G", nG, &dw.Z_G) != 0 ||
        (t2_enabled && nT2 > 0 && json_get_double_array(dw_s, dw_e, "Z_T2", nT2, &dw.Z_T2) != 0)) {
        chem_dual_witness_free(&dw);
        chem_ham_free(&ham);
        free(txt);
        snprintf(out->msg, sizeof(out->msg), "failed parsing dual_witness arrays");
        return -1;
    }

    ChemSolveResult recomp;
    memset(&recomp, 0, sizeof(recomp));
    if (meq == 0 && nD == 0) {
        ChemSolveOpts topts = chem_solve_opts_default(&ham);
        topts.mode = mode;
        topts.two_s = two_s;
        topts.constrain_s2 = cs2;
        if (solve_trivial_space_cost(&ham, ham.E_core, NULL, mode, two_s, cs2, &topts, &recomp,
                                     err, sizeof(err)) != 0) {
            chem_dual_witness_free(&dw);
            chem_ham_free(&ham);
            free(txt);
            snprintf(out->msg, sizeof(out->msg), "trivial space certification failed");
            return -1;
        }
    } else if (chem_certify_dual_witness(&ham, &dw, &recomp, err, sizeof(err)) != 0) {
        chem_dual_witness_free(&dw);
        chem_ham_free(&ham);
        free(txt);
        snprintf(out->msg, sizeof(out->msg), "dual witness certification failed: %s", err);
        return -1;
    }

    out->E_lb_recomputed = recomp.E_lb_cert;
    out->min_eig_cert = recomp.min_eig_cert;
    out->lb_ok = (fabs(recomp.E_lb_cert - E_lb_claimed) <= 1e-8) ? 1 : 0;

    /* Check veto consistency */
    out->veto_ok = 1;
    if (json_get_object_range(s, e, "veto", &vet_s, &vet_e) == 0) {
        int vetoed = 0;
        double e_ai = 0.0, vtol = 1e-4;
        if (json_get_int(vet_s, vet_e, "vetoed", &vetoed) == 0 && vetoed) {
            if (json_get_double(vet_s, vet_e, "E_ai", &e_ai) == 0 &&
                json_get_double(vet_s, vet_e, "veto_tol", &vtol) == 0) {
                if (!(recomp.E_lb_cert > e_ai - vtol)) out->veto_ok = 0;
            } else {
                out->veto_ok = 0;
            }
        }
    }

    /* Check variational witness if present */
    out->var_ok = 1;
    out->E_var_recomputed = NAN;
    if (json_get_object_range(s, e, "variational", &var_s, &var_e) == 0) {
        int has_w = 0, ndet_var = 0;
        double e_ub_claimed = 0.0, e_ci_claimed = 0.0;
        if (json_get_int(var_s, var_e, "has_witness", &has_w) == 0 && has_w &&
            json_get_int(var_s, var_e, "ndet_var", &ndet_var) == 0 &&
            json_get_double(var_s, var_e, "E_ub", &e_ub_claimed) == 0 &&
            json_get_double(var_s, var_e, "E_ci", &e_ci_claimed) == 0) {
            uint64_t *dets = NULL;
            double *psi0 = NULL;
            if (json_get_u64_hex_array(var_s, var_e, "dets", ndet_var, &dets) != 0 ||
                json_get_double_array(var_s, var_e, "psi0", ndet_var, &psi0) != 0) {
                out->var_ok = 0;
            } else {
                double e_re = 0.0;
                if (chem_var_eval_witness(&ham, ndet_var, dets, psi0, &e_re, err, sizeof(err)) != 0) {
                    out->var_ok = 0;
                } else {
                    out->E_var_recomputed = e_re;
                    if (fabs(e_re - e_ci_claimed) > 1e-6) out->var_ok = 0;
                    if (recomp.E_lb_cert > e_re + 1e-9) out->var_ok = 0;
                    if (recomp.E_lb_cert > e_ub_claimed + 1e-9) out->var_ok = 0;
                }
            }
            free(dets);
            free(psi0);
        }
    }

    out->ok = (out->sha_ok && out->lb_ok && out->veto_ok && out->var_ok) ? 1 : 0;
    if (out->ok) {
        snprintf(out->msg, sizeof(out->msg),
                 "CERT VERIFIED: sha=OK E_lb=%.8f (diff=%.2e)",
                 recomp.E_lb_cert, fabs(recomp.E_lb_cert - E_lb_claimed));
    } else {
        snprintf(out->msg, sizeof(out->msg),
                 "CERT FAILED: sha=%d lb=%d (diff=%.2e) veto=%d var=%d",
                 out->sha_ok, out->lb_ok, fabs(recomp.E_lb_cert - E_lb_claimed),
                 out->veto_ok, out->var_ok);
    }

    chem_solve_result_free(&recomp);
    chem_dual_witness_free(&dw);
    chem_ham_free(&ham);
    free(txt);
    return out->ok ? 0 : -1;
}
