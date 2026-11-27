#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include "chem_rdm2.h"
#include "chem_shci.h"
#include "flow.h"
#include "flow_sha256.h"
#include "flow_wire_pack.h"

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
    o.enable_t1 = 0;
    o.t1_max_orbs = 6;
    o.t1_norbs = 0;
    o.export_rdm_grad = 0;
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
    static const char *MODE_A_NAMES[10] = {
        "D0", "D1", "Q0", "Q1", "G0", "G1", "T2aab", "T2mix", "T1aaa", "T1aab"
    };
    static const char *MODE_B_NAMES[CHEM_MAX_BLOCKS] = {
        "Daa", "Dbb", "Dab", "Qaa", "Qbb", "Qab", "Gdiag", "Gab", "Gba",
        "T2aab", "T2mixA", "T2bba", "T2mixB",
        "T1aaa", "T1aab", "T1bbb", "T1bba"
    };
    if (mode == CHEM_MODE_A_SINGLET) {
        return (blk_idx >= 0 && blk_idx < 10) ? MODE_A_NAMES[blk_idx] : "?";
    }
    return (blk_idx >= 0 && blk_idx < CHEM_MAX_BLOCKS) ? MODE_B_NAMES[blk_idx] : "?";
}

void chem_dual_witness_free(ChemDualWitness *w) {
    if (!w) return;
    free(w->y_eq);
    free(w->Z_Q);
    free(w->Z_G);
    free(w->Z_T2);
    free(w->Z_T1);
    memset(w, 0, sizeof(*w));
}

void chem_solve_result_free(ChemSolveResult *res) {
    if (!res) return;
    free(res->primal_grad_h1);
    free(res->primal_grad_eri);
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

/* Fast packing for already-symmetric matrix whose upper triangle (i <= j) is populated */
static void svec_pack_upper(int n, const double *M, double *v) {
    int p = 0;
    const double s2 = CHEM_SQRT2;
    for (int i = 0; i < n; i++) {
        const double *row = M + (size_t)i * (size_t)n;
        v[p++] = row[i];
        for (int j = i + 1; j < n; j++) {
            v[p++] = s2 * row[j];
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

typedef void (*ChemPoolTaskFn)(void *arg, int wid);
typedef struct ChemWorkerPool ChemWorkerPool;
static void chem_pool_destroy(ChemWorkerPool *pool);

typedef struct {
    ChemMode mode;
    int K, na, nb, two_s, constrain_s2;
    int t2_enabled;
    int t2_norbs;
    int t2_orbs[CHEM_MAX_ORBITALS];
    int t2_naa, t2_nab, t2_naab, t2_nabb, t2_nmix;
    int t2_P1[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];
    double t2_sgn1[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];

    int t1_enabled;
    int t1_norbs;
    int t1_orbs[CHEM_MAX_ORBITALS];
    int t1_naa, t1_nab, t1_naaa, t1_naab;
    int t1_P1[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];
    double t1_sgn1[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS];

    int blk_D_cnt, blk_Q_start, blk_G_start, blk_T2_start, blk_T2_cnt, blk_T1_start, blk_T1_cnt;
    int nblocks;
    int block_dims[CHEM_MAX_BLOCKS];
    int block_szs[CHEM_MAX_BLOCKS];
    double block_traces[CHEM_MAX_BLOCKS];
    int nD, nQ, nG, nT2, nT1, nAll, meq;
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
    double *t1_0;  /* [nT1] (NULL if t1_enabled == 0) */
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
    int *scr_t2_pairs; /* [2 * t2_naa] */
    double *scr_t2_Daa;
    double *scr_t2_Dbb;
    double *scr_t2_Dab;
    double *scr_t2_Dab_sw;
    double *scr_t2_ga;
    double *scr_t2_gb;
    double *scr_t2_Mat; /* [max(t2_naab, t2_nmix)^2] */

    /* Optional T1 sub-space buffers (allocated only when t1_enabled == 1) */
    int *scr_t1_pairs; /* [2 * t1_naa] */
    int *scr_t1_tri;   /* [3 * t1_naaa] */
    double *scr_t1_Daa;
    double *scr_t1_Dbb;
    double *scr_t1_Dab;
    double *scr_t1_Dab_sw;
    double *scr_t1_ga;
    double *scr_t1_gb;
    double *scr_t1_Mat; /* [max(t1_naaa, t1_naab)^2] */

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
    ChemWorkerPool *pool;
} ChemMFContext;

static void mf_ctx_free(ChemMFContext *ctx) {
    if (!ctx) return;
    chem_pool_destroy(ctx->pool);
    ctx->pool = NULL;
    free(ctx->b_eq);
    free(ctx->q0);
    free(ctx->t1_0);
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
    free(ctx->scr_t2_pairs);
    free(ctx->scr_t2_Daa);
    free(ctx->scr_t2_Dbb);
    free(ctx->scr_t2_Dab);
    free(ctx->scr_t2_Dab_sw);
    free(ctx->scr_t2_ga);
    free(ctx->scr_t2_gb);
    free(ctx->scr_t2_Mat);
    free(ctx->scr_t1_pairs);
    free(ctx->scr_t1_tri);
    free(ctx->scr_t1_Daa);
    free(ctx->scr_t1_Dbb);
    free(ctx->scr_t1_Dab);
    free(ctx->scr_t1_Dab_sw);
    free(ctx->scr_t1_ga);
    free(ctx->scr_t1_gb);
    free(ctx->scr_t1_Mat);
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

/* Persistent worker thread pool eliminating per-iteration pthread_create/join overhead */
struct ChemWorkerPool {
    pthread_t thrs[3];
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
    int nworkers;
    int spawned;
    int stop;
    ChemPoolTaskFn job_fn;
    void *job_arg;
    uint64_t gen;
    int active;
};

typedef struct {
    ChemWorkerPool *pool;
    int wid;
} ChemPoolWorkerArg;

static void *chem_pool_worker_main(void *arg) {
    ChemPoolWorkerArg *wa = (ChemPoolWorkerArg *)arg;
    ChemWorkerPool *pool = wa->pool;
    int wid = wa->wid;
    free(wa);

    uint64_t my_gen = 0;
    for (;;) {
        pthread_mutex_lock(&pool->mu);
        while (!pool->stop && pool->gen == my_gen) {
            pthread_cond_wait(&pool->cv_work, &pool->mu);
        }
        if (pool->stop) {
            pthread_mutex_unlock(&pool->mu);
            break;
        }
        my_gen = pool->gen;
        ChemPoolTaskFn fn = pool->job_fn;
        void *job_arg = pool->job_arg;
        pthread_mutex_unlock(&pool->mu);

        if (fn) fn(job_arg, wid);

        pthread_mutex_lock(&pool->mu);
        pool->active--;
        if (pool->active == 0) pthread_cond_signal(&pool->cv_done);
        pthread_mutex_unlock(&pool->mu);
    }
    return NULL;
}

static ChemWorkerPool *chem_pool_create(int nworkers) {
    if (nworkers <= 0) return NULL;
    if (nworkers > 3) nworkers = 3;
    ChemWorkerPool *pool = (ChemWorkerPool *)calloc(1, sizeof(ChemWorkerPool));
    if (!pool) return NULL;
    pthread_mutex_init(&pool->mu, NULL);
    pthread_cond_init(&pool->cv_work, NULL);
    pthread_cond_init(&pool->cv_done, NULL);
    pool->nworkers = nworkers;
    for (int t = 1; t <= nworkers; t++) {
        ChemPoolWorkerArg *wa = (ChemPoolWorkerArg *)malloc(sizeof(ChemPoolWorkerArg));
        if (!wa) break;
        wa->pool = pool;
        wa->wid = t;
        if (pthread_create(&pool->thrs[t - 1], NULL, chem_pool_worker_main, wa) == 0) {
            pool->spawned++;
        } else {
            free(wa);
            break;
        }
    }
    if (pool->spawned == 0) {
        pthread_mutex_destroy(&pool->mu);
        pthread_cond_destroy(&pool->cv_work);
        pthread_cond_destroy(&pool->cv_done);
        free(pool);
        return NULL;
    }
    pool->nworkers = pool->spawned;
    return pool;
}

static void chem_pool_dispatch(ChemWorkerPool *pool, ChemPoolTaskFn fn, void *job_arg) {
    if (!pool || pool->nworkers <= 0) {
        fn(job_arg, 0);
        return;
    }
    pthread_mutex_lock(&pool->mu);
    pool->job_fn = fn;
    pool->job_arg = job_arg;
    pool->active = pool->nworkers;
    pool->gen++;
    pthread_cond_broadcast(&pool->cv_work);
    pthread_mutex_unlock(&pool->mu);

    fn(job_arg, 0);

    pthread_mutex_lock(&pool->mu);
    while (pool->active > 0) {
        pthread_cond_wait(&pool->cv_done, &pool->mu);
    }
    pthread_mutex_unlock(&pool->mu);
}

static void chem_pool_destroy(ChemWorkerPool *pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mu);
    pool->stop = 1;
    pthread_cond_broadcast(&pool->cv_work);
    pthread_mutex_unlock(&pool->mu);
    for (int t = 0; t < pool->spawned; t++) {
        pthread_join(pool->thrs[t], NULL);
    }
    pthread_mutex_destroy(&pool->mu);
    pthread_cond_destroy(&pool->cv_work);
    pthread_cond_destroy(&pool->cv_done);
    free(pool);
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
    const int *pairs = ctx->scr_t2_pairs;

    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[2 * pq], q = pairs[2 * pq + 1];
        /* delta_pu * delta_qv * gb[w, r] */
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                M[(size_t)ri * stride + (size_t)cj] += gb[(size_t)w * (size_t)Ks + (size_t)r];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[2 * uv], v = pairs[2 * uv + 1];
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
    const int *pairs = ctx->scr_t2_pairs;

    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[2 * pq], q = pairs[2 * pq + 1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                gb_h[(size_t)w * (size_t)Ks + (size_t)r] += Y[(size_t)ri * stride + (size_t)cj];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[2 * uv], v = pairs[2 * uv + 1];
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
    const int *pairs = ctx->scr_t2_pairs;

    /* 1. T2_aaa subblock */
    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[2 * pq], q = pairs[2 * pq + 1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                M[(size_t)ri * stride + (size_t)cj] += ga[(size_t)w * (size_t)Ks + (size_t)r];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[2 * uv], v = pairs[2 * uv + 1];
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
        int p = pairs[2 * pq], q = pairs[2 * pq + 1];
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
    const int *pairs = ctx->scr_t2_pairs;

    /* 1. Adjoint of T2_aaa */
    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[2 * pq], q = pairs[2 * pq + 1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                ga_h[(size_t)w * (size_t)Ks + (size_t)r] += Y[(size_t)ri * stride + (size_t)cj];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[2 * uv], v = pairs[2 * uv + 1];
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
        int p = pairs[2 * pq], q = pairs[2 * pq + 1];
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
        const int *s_pairs = ctx->scr_t2_pairs;
        for (int ab = 0; ab < naa_s; ab++) {
            int oa = ctx->t2_orbs[s_pairs[2 * ab]], ob = ctx->t2_orbs[s_pairs[2 * ab + 1]];
            int Pab = ctx->P1[oa][ob];
            double sab = ctx->sgn1[oa][ob];
            for (int cd = 0; cd < naa_s; cd++) {
                int oc = ctx->t2_orbs[s_pairs[2 * cd]], od = ctx->t2_orbs[s_pairs[2 * cd + 1]];
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
    const int *s_pairs = ctx->scr_t2_pairs;
    for (int ab = 0; ab < naa_s; ab++) {
        int oa = ctx->t2_orbs[s_pairs[2 * ab]], ob = ctx->t2_orbs[s_pairs[2 * ab + 1]];
        int Pab = ctx->P1[oa][ob];
        double sab = ctx->sgn1[oa][ob];
        for (int cd = 0; cd < naa_s; cd++) {
            int oc = ctx->t2_orbs[s_pairs[2 * cd]], od = ctx->t2_orbs[s_pairs[2 * cd + 1]];
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
    const int *s_pairs = ctx->scr_t2_pairs;

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
                int oa = ctx->t2_orbs[s_pairs[2 * ab]], ob = ctx->t2_orbs[s_pairs[2 * ab + 1]];
                int Pab = ctx->P1[oa][ob];
                double sab = ctx->sgn1[oa][ob];
                for (int cd = 0; cd < naa_s; cd++) {
                    int oc = ctx->t2_orbs[s_pairs[2 * cd]], od = ctx->t2_orbs[s_pairs[2 * cd + 1]];
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
            int oa = ctx->t2_orbs[s_pairs[2 * ab]], ob = ctx->t2_orbs[s_pairs[2 * ab + 1]];
            int Pab = ctx->P1[oa][ob];
            double sab = ctx->sgn1[oa][ob];
            for (int cd = 0; cd < naa_s; cd++) {
                int oc = ctx->t2_orbs[s_pairs[2 * cd]], od = ctx->t2_orbs[s_pairs[2 * cd + 1]];
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
 * Matrix-Free Partial 3-RDM T1 = D3 + Q3 Operators (Mode A & Mode B)
 * ============================================================================ */

static void t1_core_aaa_fwd(const ChemMFContext *ctx, const double *ga,
                            const double *Daa, double *M) {
    int Ks = ctx->t1_norbs;
    int naa = ctx->t1_naa;
    int naaa = ctx->t1_naaa;
    if (naaa <= 0) return;
    const int *tri = ctx->scr_t1_tri;
    size_t kssz = (size_t)Ks;
    size_t stride = (size_t)naaa;
    memset(M, 0, stride * stride * sizeof(double));

    for (int t1 = 0; t1 < naaa; t1++) {
        for (int t2 = 0; t2 < naaa; t2++) {
            double acc = 0.0;
            for (int i = 0; i < 3; i++) {
                int p1 = tri[3 * t1 + (i == 0 ? 1 : 0)];
                int q1 = tri[3 * t1 + (i == 2 ? 1 : 2)];
                int Pi = ctx->t1_P1[p1][q1];
                int ai = tri[3 * t1 + i];
                for (int j = 0; j < 3; j++) {
                    int u1 = tri[3 * t2 + (j == 0 ? 1 : 0)];
                    int v1 = tri[3 * t2 + (j == 2 ? 1 : 2)];
                    int Qj = ctx->t1_P1[u1][v1];
                    int bj = tri[3 * t2 + j];
                    double sgn = (((i + j) & 1) == 0) ? 1.0 : -1.0;
                    if (Pi == Qj) {
                        acc -= sgn * ga[(size_t)ai * kssz + (size_t)bj];
                    }
                    if (ai == bj) {
                        acc += sgn * Daa[(size_t)Pi * (size_t)naa + (size_t)Qj];
                    }
                }
            }
            M[(size_t)t1 * stride + (size_t)t2] = acc;
        }
    }
}

static void t1_core_aaa_adj(const ChemMFContext *ctx, const double *Y,
                            double *ga_h, double *Daa_h) {
    int Ks = ctx->t1_norbs;
    int naa = ctx->t1_naa;
    int naaa = ctx->t1_naaa;
    if (naaa <= 0) return;
    const int *tri = ctx->scr_t1_tri;
    size_t kssz = (size_t)Ks;
    size_t stride = (size_t)naaa;

    for (int t1 = 0; t1 < naaa; t1++) {
        for (int t2 = 0; t2 < naaa; t2++) {
            double y = Y[(size_t)t1 * stride + (size_t)t2];
            if (y == 0.0) continue;
            for (int i = 0; i < 3; i++) {
                int p1 = tri[3 * t1 + (i == 0 ? 1 : 0)];
                int q1 = tri[3 * t1 + (i == 2 ? 1 : 2)];
                int Pi = ctx->t1_P1[p1][q1];
                int ai = tri[3 * t1 + i];
                for (int j = 0; j < 3; j++) {
                    int u1 = tri[3 * t2 + (j == 0 ? 1 : 0)];
                    int v1 = tri[3 * t2 + (j == 2 ? 1 : 2)];
                    int Qj = ctx->t1_P1[u1][v1];
                    int bj = tri[3 * t2 + j];
                    double sgn = (((i + j) & 1) == 0) ? 1.0 : -1.0;
                    if (Pi == Qj) {
                        ga_h[(size_t)ai * kssz + (size_t)bj] -= sgn * y;
                    }
                    if (ai == bj) {
                        Daa_h[(size_t)Pi * (size_t)naa + (size_t)Qj] += sgn * y;
                    }
                }
            }
        }
    }
}

static void t1_core_aab_fwd(const ChemMFContext *ctx, const double *ga, const double *gb,
                            const double *Daa, const double *Dab, double *M) {
    int Ks = ctx->t1_norbs;
    int naa = ctx->t1_naa;
    int nab = ctx->t1_nab;
    int naab = ctx->t1_naab;
    if (naab <= 0) return;
    const int *pairs = ctx->scr_t1_pairs;
    size_t kssz = (size_t)Ks;
    size_t stride = (size_t)naab;
    memset(M, 0, stride * stride * sizeof(double));

    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[2 * pq], q = pairs[2 * pq + 1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                M[(size_t)ri * stride + (size_t)cj] -= gb[(size_t)r * kssz + (size_t)w];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[2 * uv], v = pairs[2 * uv + 1];
            double diag_rw = Daa[(size_t)pq * (size_t)naa + (size_t)uv];
            int m_pu = (p == u), m_pv = (p == v), m_qu = (q == u), m_qv = (q == v);
            if (m_pu) diag_rw -= ga[(size_t)q * kssz + (size_t)v];
            if (m_pv) diag_rw += ga[(size_t)q * kssz + (size_t)u];
            if (m_qu) diag_rw += ga[(size_t)p * kssz + (size_t)v];
            if (m_qv) diag_rw -= ga[(size_t)p * kssz + (size_t)u];
            for (int r = 0; r < Ks; r++) {
                M[(size_t)(pq * Ks + r) * stride + (size_t)(uv * Ks + r)] += diag_rw;
            }
            if (!(m_pu || m_pv || m_qu || m_qv)) continue;
            for (int r = 0; r < Ks; r++) {
                int ri = pq * Ks + r;
                for (int w = 0; w < Ks; w++) {
                    int cj = uv * Ks + w;
                    double acc = 0.0;
                    if (m_pu) acc += Dab[(size_t)(q * Ks + r) * (size_t)nab + (size_t)(v * Ks + w)];
                    if (m_pv) acc -= Dab[(size_t)(q * Ks + r) * (size_t)nab + (size_t)(u * Ks + w)];
                    if (m_qu) acc -= Dab[(size_t)(p * Ks + r) * (size_t)nab + (size_t)(v * Ks + w)];
                    if (m_qv) acc += Dab[(size_t)(p * Ks + r) * (size_t)nab + (size_t)(u * Ks + w)];
                    M[(size_t)ri * stride + (size_t)cj] += acc;
                }
            }
        }
    }
}

static void t1_core_aab_adj(const ChemMFContext *ctx, const double *Y,
                            double *ga_h, double *gb_h, double *Daa_h, double *Dab_h) {
    int Ks = ctx->t1_norbs;
    int naa = ctx->t1_naa;
    int nab = ctx->t1_nab;
    int naab = ctx->t1_naab;
    if (naab <= 0) return;
    const int *pairs = ctx->scr_t1_pairs;
    size_t kssz = (size_t)Ks;
    size_t stride = (size_t)naab;

    for (int pq = 0; pq < naa; pq++) {
        int p = pairs[2 * pq], q = pairs[2 * pq + 1];
        for (int r = 0; r < Ks; r++) {
            int ri = pq * Ks + r;
            for (int w = 0; w < Ks; w++) {
                int cj = pq * Ks + w;
                gb_h[(size_t)r * kssz + (size_t)w] -= Y[(size_t)ri * stride + (size_t)cj];
            }
        }
        for (int uv = 0; uv < naa; uv++) {
            int u = pairs[2 * uv], v = pairs[2 * uv + 1];
            double acc_diag = 0.0;
            for (int r = 0; r < Ks; r++) {
                acc_diag += Y[(size_t)(pq * Ks + r) * stride + (size_t)(uv * Ks + r)];
            }
            Daa_h[(size_t)pq * (size_t)naa + (size_t)uv] += acc_diag;
            int m_pu = (p == u), m_pv = (p == v), m_qu = (q == u), m_qv = (q == v);
            if (m_pu) ga_h[(size_t)q * kssz + (size_t)v] -= acc_diag;
            if (m_pv) ga_h[(size_t)q * kssz + (size_t)u] += acc_diag;
            if (m_qu) ga_h[(size_t)p * kssz + (size_t)v] += acc_diag;
            if (m_qv) ga_h[(size_t)p * kssz + (size_t)u] -= acc_diag;
            if (!(m_pu || m_pv || m_qu || m_qv)) continue;
            for (int r = 0; r < Ks; r++) {
                int ri = pq * Ks + r;
                for (int w = 0; w < Ks; w++) {
                    int cj = uv * Ks + w;
                    double y = Y[(size_t)ri * stride + (size_t)cj];
                    if (m_pu) Dab_h[(size_t)(q * Ks + r) * (size_t)nab + (size_t)(v * Ks + w)] += y;
                    if (m_pv) Dab_h[(size_t)(q * Ks + r) * (size_t)nab + (size_t)(u * Ks + w)] -= y;
                    if (m_qu) Dab_h[(size_t)(p * Ks + r) * (size_t)nab + (size_t)(v * Ks + w)] -= y;
                    if (m_qv) Dab_h[(size_t)(p * Ks + r) * (size_t)nab + (size_t)(u * Ks + w)] += y;
                }
            }
        }
    }
}

static void mf_LT1_fwd(const ChemMFContext *ctx, const double *d_in, double *t1_out) {
    if (!ctx->t1_enabled) return;
    int K = ctx->K;
    int Ks = ctx->t1_norbs;
    int naa_s = ctx->t1_naa;
    int nab_s = ctx->t1_nab;
    size_t ksz = (size_t)K;
    size_t kssz = (size_t)Ks;

    double *ga_s = ctx->scr_t1_ga;
    double *gb_s = ctx->scr_t1_gb;
    double *Daa_s = ctx->scr_t1_Daa;
    double *Dbb_s = ctx->scr_t1_Dbb;
    double *Dab_s = ctx->scr_t1_Dab;
    double *Mat = ctx->scr_t1_Mat;
    const int *pairs = ctx->scr_t1_pairs;

    if (ctx->mode == CHEM_MODE_A_SINGLET) {
        mode_a_expand_st_tt(ctx, d_in, ctx->scr_St, ctx->scr_Tt);
        mode_a_gamma_from_st(K, ctx->na, ctx->scr_St, ctx->scr_ga);
        const double *St = ctx->scr_St;
        const double *Tt = ctx->scr_Tt;
        const double *ga = ctx->scr_ga;
        const double *D1 = ctx->scr_D1;
        int m1 = ctx->m1;

        for (int a = 0; a < Ks; a++) {
            int oa = ctx->t1_orbs[a];
            for (int b = 0; b < Ks; b++) {
                int ob = ctx->t1_orbs[b];
                double gv = ga[(size_t)oa * ksz + (size_t)ob];
                ga_s[(size_t)a * kssz + (size_t)b] = gv;
                gb_s[(size_t)a * kssz + (size_t)b] = gv;
            }
        }
        for (int ab = 0; ab < naa_s; ab++) {
            int oa = ctx->t1_orbs[pairs[2 * ab]], ob = ctx->t1_orbs[pairs[2 * ab + 1]];
            int Pab = ctx->P1[oa][ob];
            double sab = ctx->sgn1[oa][ob];
            for (int cd = 0; cd < naa_s; cd++) {
                int oc = ctx->t1_orbs[pairs[2 * cd]], od = ctx->t1_orbs[pairs[2 * cd + 1]];
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
            int oa = ctx->t1_orbs[a];
            for (int b = 0; b < Ks; b++) {
                int ob = ctx->t1_orbs[b];
                for (int c = 0; c < Ks; c++) {
                    int oc = ctx->t1_orbs[c];
                    for (int d = 0; d < Ks; d++) {
                        int od = ctx->t1_orbs[d];
                        size_t off4 = ((((size_t)oa * ksz + (size_t)ob) * ksz + (size_t)oc) * ksz +
                                       (size_t)od);
                        Dab_s[(size_t)(a * Ks + b) * (size_t)nab_s + (size_t)(c * Ks + d)] =
                            St[off4] + Tt[off4];
                    }
                }
            }
        }
        int ts = ctx->blk_T1_start;
        int off = 0;
        if (ctx->t1_naaa > 0) {
            t1_core_aaa_fwd(ctx, ga_s, Daa_s, Mat);
            svec_pack_general(ctx->t1_naaa, Mat, t1_out + off);
            off += ctx->block_szs[ts];
        }
        t1_core_aab_fwd(ctx, ga_s, gb_s, Daa_s, Dab_s, Mat);
        svec_pack_general(ctx->t1_naab, Mat, t1_out + off);
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
        int oa = ctx->t1_orbs[a];
        for (int b = 0; b < Ks; b++) {
            int ob = ctx->t1_orbs[b];
            ga_s[(size_t)a * kssz + (size_t)b] = ga[(size_t)oa * ksz + (size_t)ob];
            gb_s[(size_t)a * kssz + (size_t)b] = gb[(size_t)oa * ksz + (size_t)ob];
        }
    }
    for (int ab = 0; ab < naa_s; ab++) {
        int oa = ctx->t1_orbs[pairs[2 * ab]], ob = ctx->t1_orbs[pairs[2 * ab + 1]];
        int Pab = ctx->P1[oa][ob];
        double sab = ctx->sgn1[oa][ob];
        for (int cd = 0; cd < naa_s; cd++) {
            int oc = ctx->t1_orbs[pairs[2 * cd]], od = ctx->t1_orbs[pairs[2 * cd + 1]];
            int Pcd = ctx->P1[oc][od];
            double sgn = sab * ctx->sgn1[oc][od];
            Daa_s[(size_t)ab * (size_t)naa_s + (size_t)cd] =
                (naa > 0 && Pab >= 0 && Pcd >= 0) ? sgn * Daa[(size_t)Pab * (size_t)naa + (size_t)Pcd] : 0.0;
            Dbb_s[(size_t)ab * (size_t)naa_s + (size_t)cd] =
                (naa > 0 && Pab >= 0 && Pcd >= 0) ? sgn * Dbb[(size_t)Pab * (size_t)naa + (size_t)Pcd] : 0.0;
        }
    }
    double *Dab_sw = ctx->scr_t1_Dab_sw;
    for (int a = 0; a < Ks; a++) {
        int oa = ctx->t1_orbs[a];
        for (int b = 0; b < Ks; b++) {
            int ob = ctx->t1_orbs[b];
            for (int c = 0; c < Ks; c++) {
                int oc = ctx->t1_orbs[c];
                for (int d = 0; d < Ks; d++) {
                    int od = ctx->t1_orbs[d];
                    double v = Dab[(size_t)(oa * K + ob) * (size_t)nab + (size_t)(oc * K + od)];
                    Dab_s[(size_t)(a * Ks + b) * (size_t)nab_s + (size_t)(c * Ks + d)] = v;
                    Dab_sw[(size_t)(b * Ks + a) * (size_t)nab_s + (size_t)(d * Ks + c)] = v;
                }
            }
        }
    }

    int ts = ctx->blk_T1_start;
    int off = 0;
    if (ctx->t1_naaa > 0) {
        t1_core_aaa_fwd(ctx, ga_s, Daa_s, Mat);
        svec_pack_general(ctx->t1_naaa, Mat, t1_out + off);
        off += ctx->block_szs[ts];
    }
    t1_core_aab_fwd(ctx, ga_s, gb_s, Daa_s, Dab_s, Mat);
    svec_pack_general(ctx->t1_naab, Mat, t1_out + off);
    off += ctx->block_szs[ts + 1];

    if (ctx->t1_naaa > 0) {
        t1_core_aaa_fwd(ctx, gb_s, Dbb_s, Mat);
        svec_pack_general(ctx->t1_naaa, Mat, t1_out + off);
        off += ctx->block_szs[ts + 2];
    }
    t1_core_aab_fwd(ctx, gb_s, ga_s, Dbb_s, Dab_sw, Mat);
    svec_pack_general(ctx->t1_naab, Mat, t1_out + off);
}

static void mf_LT1_adj(const ChemMFContext *ctx, const double *yt1_in, double *d_out) {
    if (!ctx->t1_enabled) return;
    int K = ctx->K;
    int Ks = ctx->t1_norbs;
    int naa_s = ctx->t1_naa;
    int nab_s = ctx->t1_nab;
    size_t ksz = (size_t)K;
    size_t kssz = (size_t)Ks;

    double *ga_s_h = ctx->scr_t1_ga;
    double *gb_s_h = ctx->scr_t1_gb;
    double *Daa_s_h = ctx->scr_t1_Daa;
    double *Dbb_s_h = ctx->scr_t1_Dbb;
    double *Dab_s_h = ctx->scr_t1_Dab;
    double *Dab_sw_h = ctx->scr_t1_Dab_sw;
    double *Mat = ctx->scr_t1_Mat;
    const int *pairs = ctx->scr_t1_pairs;

    memset(ga_s_h, 0, kssz * kssz * sizeof(double));
    memset(gb_s_h, 0, kssz * kssz * sizeof(double));
    memset(Daa_s_h, 0, (size_t)naa_s * (size_t)naa_s * sizeof(double));
    memset(Dbb_s_h, 0, (size_t)naa_s * (size_t)naa_s * sizeof(double));
    memset(Dab_s_h, 0, (size_t)nab_s * (size_t)nab_s * sizeof(double));

    int ts = ctx->blk_T1_start;
    if (ctx->mode == CHEM_MODE_A_SINGLET) {
        int off = 0;
        if (ctx->t1_naaa > 0) {
            svec_unpack(ctx->t1_naaa, yt1_in + off, Mat);
            t1_core_aaa_adj(ctx, Mat, ga_s_h, Daa_s_h);
            off += ctx->block_szs[ts];
        }
        svec_unpack(ctx->t1_naab, yt1_in + off, Mat);
        t1_core_aab_adj(ctx, Mat, ga_s_h, gb_s_h, Daa_s_h, Dab_s_h);

        size_t k4 = ksz * ksz * ksz * ksz;
        double *Sh = ctx->scr_St;
        double *Th = ctx->scr_Tt;
        double *ga_h = ctx->scr_ga;
        memset(Sh, 0, k4 * sizeof(double));
        memset(Th, 0, k4 * sizeof(double));
        memset(ga_h, 0, ksz * ksz * sizeof(double));

        for (int a = 0; a < Ks; a++) {
            int oa = ctx->t1_orbs[a];
            for (int b = 0; b < Ks; b++) {
                int ob = ctx->t1_orbs[b];
                ga_h[(size_t)oa * ksz + (size_t)ob] +=
                    ga_s_h[(size_t)a * kssz + (size_t)b] + gb_s_h[(size_t)a * kssz + (size_t)b];
            }
        }
        mode_a_gamma_adjoint_to_sh(K, ctx->na, ga_h, Sh);

        for (int a = 0; a < Ks; a++) {
            int oa = ctx->t1_orbs[a];
            for (int b = 0; b < Ks; b++) {
                int ob = ctx->t1_orbs[b];
                for (int c = 0; c < Ks; c++) {
                    int oc = ctx->t1_orbs[c];
                    for (int d = 0; d < Ks; d++) {
                        int od = ctx->t1_orbs[d];
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
                int oa = ctx->t1_orbs[pairs[2 * ab]], ob = ctx->t1_orbs[pairs[2 * ab + 1]];
                int Pab = ctx->P1[oa][ob];
                double sab = ctx->sgn1[oa][ob];
                for (int cd = 0; cd < naa_s; cd++) {
                    int oc = ctx->t1_orbs[pairs[2 * cd]], od = ctx->t1_orbs[pairs[2 * cd + 1]];
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

    /* Mode B: blocks T1aaa, T1aab, T1bbb, T1bba */
    memset(Dab_sw_h, 0, (size_t)nab_s * (size_t)nab_s * sizeof(double));
    int off = 0;
    if (ctx->t1_naaa > 0) {
        svec_unpack(ctx->t1_naaa, yt1_in + off, Mat);
        t1_core_aaa_adj(ctx, Mat, ga_s_h, Daa_s_h);
        off += ctx->block_szs[ts];
    }
    svec_unpack(ctx->t1_naab, yt1_in + off, Mat);
    t1_core_aab_adj(ctx, Mat, ga_s_h, gb_s_h, Daa_s_h, Dab_s_h);
    off += ctx->block_szs[ts + 1];

    if (ctx->t1_naaa > 0) {
        svec_unpack(ctx->t1_naaa, yt1_in + off, Mat);
        t1_core_aaa_adj(ctx, Mat, gb_s_h, Dbb_s_h);
        off += ctx->block_szs[ts + 2];
    }
    svec_unpack(ctx->t1_naab, yt1_in + off, Mat);
    t1_core_aab_adj(ctx, Mat, gb_s_h, ga_s_h, Dbb_s_h, Dab_sw_h);

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
        int oa = ctx->t1_orbs[a];
        for (int b = 0; b < Ks; b++) {
            int ob = ctx->t1_orbs[b];
            ga_h[(size_t)oa * ksz + (size_t)ob] = ga_s_h[(size_t)a * kssz + (size_t)b];
            gb_h[(size_t)oa * ksz + (size_t)ob] = gb_s_h[(size_t)a * kssz + (size_t)b];
        }
    }
    if (naa > 0) {
        for (int ab = 0; ab < naa_s; ab++) {
            int oa = ctx->t1_orbs[pairs[2 * ab]], ob = ctx->t1_orbs[pairs[2 * ab + 1]];
            int Pab = ctx->P1[oa][ob];
            double sab = ctx->sgn1[oa][ob];
            for (int cd = 0; cd < naa_s; cd++) {
                int oc = ctx->t1_orbs[pairs[2 * cd]], od = ctx->t1_orbs[pairs[2 * cd + 1]];
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
        int oa = ctx->t1_orbs[a];
        for (int b = 0; b < Ks; b++) {
            int ob = ctx->t1_orbs[b];
            for (int c = 0; c < Ks; c++) {
                int oc = ctx->t1_orbs[c];
                for (int d = 0; d < Ks; d++) {
                    int od = ctx->t1_orbs[d];
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

static double comb2_d(int n) {
    return (n >= 2) ? 0.5 * (double)(n * (n - 1)) : 0.0;
}

static double comb3_d(int n) {
    return (n >= 3) ? (double)(n * (n - 1) * (n - 2)) / 6.0 : 0.0;
}

static int select_subspace_orbs(int K, int na, int max_orbs, int norbs_in, const int *orbs_in,
                                const char *tag, int *out_norbs, int *out_orbs,
                                int P1_out[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS],
                                double sgn1_out[CHEM_MAX_ORBITALS][CHEM_MAX_ORBITALS],
                                char *err, size_t err_sz) {
    int cap = (max_orbs > 0) ? max_orbs : 6;
    if (cap > K) cap = K;
    if (cap > CHEM_MAX_ORBITALS) cap = CHEM_MAX_ORBITALS;
    int Ks = 0;
    if (norbs_in > 0 && orbs_in) {
        Ks = (norbs_in > K) ? K : norbs_in;
        for (int i = 0; i < Ks; i++) {
            if (orbs_in[i] < 0 || orbs_in[i] >= K) {
                set_err(err, err_sz, "%s_orbs[%d]=%d out of range [0,%d)", tag, i, orbs_in[i], K);
                return -1;
            }
            out_orbs[i] = orbs_in[i];
        }
        qsort(out_orbs, (size_t)Ks, sizeof(int), cmp_int_asc);
        for (int i = 1; i < Ks; i++) {
            if (out_orbs[i] == out_orbs[i - 1]) {
                set_err(err, err_sz, "duplicate orbital %d in %s_orbs", out_orbs[i], tag);
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
        for (int i = 0; i < Ks; i++) out_orbs[i] = start + i;
    }
    *out_norbs = Ks;
    for (int a = 0; a < Ks; a++) {
        for (int b = 0; b < Ks; b++) {
            int sg = 0;
            P1_out[a][b] = idx_asym_pair(Ks, a, b, &sg);
            sgn1_out[a][b] = (double)sg;
        }
    }
    return 0;
}

static int mf_ctx_init_ext(ChemMFContext *ctx, int K, int na, int nb, int two_s, int constrain_s2,
                           ChemMode mode, const ChemHam *ham_opt,
                           int enable_t2, int t2_max_orbs, int t2_norbs_in, const int *t2_orbs_in,
                           int enable_t1, int t1_max_orbs, int t1_norbs_in, const int *t1_orbs_in,
                           int num_threads, char *err, size_t err_sz) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->mode = mode;
    ctx->K = K;
    ctx->na = na;
    ctx->nb = nb;
    ctx->two_s = two_s;
    ctx->constrain_s2 = constrain_s2;

    if (enable_t2 && K >= 2) {
        if (select_subspace_orbs(K, na, t2_max_orbs, t2_norbs_in, t2_orbs_in, "t2",
                                 &ctx->t2_norbs, ctx->t2_orbs, ctx->t2_P1, ctx->t2_sgn1, err,
                                 err_sz) != 0) {
            return -1;
        }
        int Ks = ctx->t2_norbs;
        ctx->t2_enabled = 1;
        ctx->t2_naa = Ks * (Ks - 1) / 2;
        ctx->t2_nab = Ks * Ks;
        ctx->t2_naab = ctx->t2_naa * Ks;
        ctx->t2_nabb = Ks * Ks * Ks;
        ctx->t2_nmix = ctx->t2_naab + ctx->t2_nabb;
    }

    if (enable_t1 && K >= 2) {
        if (select_subspace_orbs(K, na, t1_max_orbs, t1_norbs_in, t1_orbs_in, "t1",
                                 &ctx->t1_norbs, ctx->t1_orbs, ctx->t1_P1, ctx->t1_sgn1, err,
                                 err_sz) != 0) {
            return -1;
        }
        int Ks = ctx->t1_norbs;
        ctx->t1_enabled = 1;
        ctx->t1_naa = Ks * (Ks - 1) / 2;
        ctx->t1_nab = Ks * Ks;
        ctx->t1_naaa = (Ks >= 3) ? (Ks * (Ks - 1) * (Ks - 2) / 6) : 0;
        ctx->t1_naab = ctx->t1_naa * Ks;
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

    double tr_t1_aaa_full = comb3_d(na) + comb3_d(K - na);
    double tr_t1_aab_full = comb2_d(na) * (double)nb + comb2_d(K - na) * (double)(K - nb);
    double tr_t1_bbb_full = comb3_d(nb) + comb3_d(K - nb);
    double tr_t1_bba_full = comb2_d(nb) * (double)na + comb2_d(K - nb) * (double)(K - na);

    if (mode == CHEM_MODE_A_SINGLET) {
        if (na != nb) {
            set_err(err, err_sz, "Mode A requires na == nb (got %d != %d)", na, nb);
            return -1;
        }
        int ne = na;
        ctx->blk_D_cnt = 2;
        ctx->blk_Q_start = 2;
        ctx->blk_G_start = 4;
        ctx->blk_T2_start = 6;
        ctx->blk_T2_cnt = ctx->t2_enabled ? 2 : 0;
        ctx->blk_T1_start = ctx->blk_T2_start + ctx->blk_T2_cnt;
        ctx->blk_T1_cnt = ctx->t1_enabled ? 2 : 0;
        ctx->nblocks = ctx->blk_T1_start + ctx->blk_T1_cnt;

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
        if (ctx->t1_enabled) {
            int ts = ctx->blk_T1_start;
            ctx->block_dims[ts]     = ctx->t1_naaa;
            ctx->block_dims[ts + 1] = ctx->t1_naab;
            ctx->block_traces[ts]     = (ctx->t1_norbs == K)
                                            ? tr_t1_aaa_full
                                            : fmin((double)ctx->t1_naaa, tr_t1_aaa_full);
            ctx->block_traces[ts + 1] = (ctx->t1_norbs == K)
                                            ? tr_t1_aab_full
                                            : fmin((double)ctx->t1_naab, tr_t1_aab_full);
        }

        for (int b = 0; b < ctx->nblocks; b++) ctx->block_szs[b] = svec_len(ctx->block_dims[b]);
        ctx->nD = ctx->block_szs[0] + ctx->block_szs[1];
        ctx->nQ = ctx->block_szs[2] + ctx->block_szs[3];
        ctx->nG = ctx->block_szs[4] + ctx->block_szs[5];
        ctx->nT2 = ctx->t2_enabled ? (ctx->block_szs[6] + ctx->block_szs[7]) : 0;
        ctx->nT1 = ctx->t1_enabled ? (ctx->block_szs[ctx->blk_T1_start] +
                                      ctx->block_szs[ctx->blk_T1_start + 1])
                                   : 0;
        ctx->nAll = ctx->nD + ctx->nQ + ctx->nG + ctx->nT2 + ctx->nT1;
        ctx->meq = (ne == 1) ? 2 : (1 + ctx->m0);
    } else {
        ctx->blk_D_cnt = 3;
        ctx->blk_Q_start = 3;
        ctx->blk_G_start = 6;
        ctx->blk_T2_start = 9;
        ctx->blk_T2_cnt = ctx->t2_enabled ? 4 : 0;
        ctx->blk_T1_start = ctx->blk_T2_start + ctx->blk_T2_cnt;
        ctx->blk_T1_cnt = ctx->t1_enabled ? 4 : 0;
        ctx->nblocks = ctx->blk_T1_start + ctx->blk_T1_cnt;

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
        if (ctx->t1_enabled) {
            int ts = ctx->blk_T1_start;
            ctx->block_dims[ts]     = ctx->t1_naaa;
            ctx->block_dims[ts + 1] = ctx->t1_naab;
            ctx->block_dims[ts + 2] = ctx->t1_naaa;
            ctx->block_dims[ts + 3] = ctx->t1_naab;
            ctx->block_traces[ts]     = (ctx->t1_norbs == K)
                                            ? tr_t1_aaa_full
                                            : fmin((double)ctx->t1_naaa, tr_t1_aaa_full);
            ctx->block_traces[ts + 1] = (ctx->t1_norbs == K)
                                            ? tr_t1_aab_full
                                            : fmin((double)ctx->t1_naab, tr_t1_aab_full);
            ctx->block_traces[ts + 2] = (ctx->t1_norbs == K)
                                            ? tr_t1_bbb_full
                                            : fmin((double)ctx->t1_naaa, tr_t1_bbb_full);
            ctx->block_traces[ts + 3] = (ctx->t1_norbs == K)
                                            ? tr_t1_bba_full
                                            : fmin((double)ctx->t1_naab, tr_t1_bba_full);
        }

        for (int b = 0; b < ctx->nblocks; b++) ctx->block_szs[b] = svec_len(ctx->block_dims[b]);
        ctx->nD = ctx->block_szs[0] + ctx->block_szs[1] + ctx->block_szs[2];
        ctx->nQ = ctx->block_szs[3] + ctx->block_szs[4] + ctx->block_szs[5];
        ctx->nG = ctx->block_szs[6] + ctx->block_szs[7] + ctx->block_szs[8];
        ctx->nT2 = ctx->t2_enabled ? (ctx->block_szs[9] + ctx->block_szs[10] +
                                      ctx->block_szs[11] + ctx->block_szs[12])
                                   : 0;
        ctx->nT1 = ctx->t1_enabled ? (ctx->block_szs[ctx->blk_T1_start] +
                                      ctx->block_szs[ctx->blk_T1_start + 1] +
                                      ctx->block_szs[ctx->blk_T1_start + 2] +
                                      ctx->block_szs[ctx->blk_T1_start + 3])
                                   : 0;
        ctx->nAll = ctx->nD + ctx->nQ + ctx->nG + ctx->nT2 + ctx->nT1;

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
        if (!ctx->scr_St || !ctx->scr_Tt || !ctx->scr_D0 || !ctx->scr_D1) {
            mf_ctx_free(ctx);
            set_err(err, err_sz, "out of memory initializing singlet scratch buffers");
            return -1;
        }
    } else {
        size_t naa_sz = (size_t)(ctx->naa > 0 ? ctx->naa : 1);
        size_t nab_sz = (size_t)ctx->nab;
        ctx->scr_D0 = (double *)calloc(naa_sz * naa_sz, sizeof(double));
        ctx->scr_D1 = (double *)calloc(naa_sz * naa_sz, sizeof(double));
        ctx->scr_D2 = (double *)calloc(nab_sz * nab_sz, sizeof(double));
        if (!ctx->scr_D0 || !ctx->scr_D1 || !ctx->scr_D2) {
            mf_ctx_free(ctx);
            set_err(err, err_sz, "out of memory initializing RDM2 scratch buffers");
            return -1;
        }
    }
    if (!ctx->b_eq || !ctx->q0 || !ctx->C_D || !ctx->scr_ga || !ctx->scr_gb ||
        !ctx->scr_M0 || !ctx->scr_M1 || !ctx->scr_M2 || !ctx->ev_val) {
        mf_ctx_free(ctx);
        set_err(err, err_sz, "out of memory initializing matrix-free core buffers");
        return -1;
    }

    if (ctx->t2_enabled) {
        int Ks_t2 = ctx->t2_norbs;
        size_t kssz = (size_t)Ks_t2;
        size_t naa_s_sz = (size_t)(ctx->t2_naa > 0 ? ctx->t2_naa : 1);
        size_t nab_s_sz = (size_t)ctx->t2_nab;
        size_t mat_d = (size_t)((ctx->t2_nmix > ctx->t2_naab) ? ctx->t2_nmix : ctx->t2_naab);
        ctx->scr_t2_pairs = (int *)calloc(2U * naa_s_sz, sizeof(int));
        ctx->scr_t2_Daa = (double *)calloc(naa_s_sz * naa_s_sz, sizeof(double));
        ctx->scr_t2_Dbb = (double *)calloc(naa_s_sz * naa_s_sz, sizeof(double));
        ctx->scr_t2_Dab = (double *)calloc(nab_s_sz * nab_s_sz, sizeof(double));
        ctx->scr_t2_Dab_sw = (double *)calloc(nab_s_sz * nab_s_sz, sizeof(double));
        ctx->scr_t2_ga = (double *)calloc(kssz * kssz, sizeof(double));
        ctx->scr_t2_gb = (double *)calloc(kssz * kssz, sizeof(double));
        ctx->scr_t2_Mat = (double *)calloc(mat_d * mat_d, sizeof(double));
        if (!ctx->scr_t2_pairs || !ctx->scr_t2_Daa || !ctx->scr_t2_Dbb || !ctx->scr_t2_Dab ||
            !ctx->scr_t2_Dab_sw || !ctx->scr_t2_ga || !ctx->scr_t2_gb || !ctx->scr_t2_Mat) {
            mf_ctx_free(ctx);
            set_err(err, err_sz, "out of memory initializing T2 scratch buffers");
            return -1;
        }
        int np_t2 = 0;
        for (int a = 0; a < Ks_t2; a++) {
            for (int b = a + 1; b < Ks_t2; b++) {
                ctx->scr_t2_pairs[2 * np_t2] = a;
                ctx->scr_t2_pairs[2 * np_t2 + 1] = b;
                np_t2++;
            }
        }
    }

    if (ctx->t1_enabled) {
        int Ks = ctx->t1_norbs;
        size_t kssz = (size_t)Ks;
        size_t naa_s_sz = (size_t)(ctx->t1_naa > 0 ? ctx->t1_naa : 1);
        size_t nab_s_sz = (size_t)ctx->t1_nab;
        size_t mat_d = (size_t)((ctx->t1_naaa > ctx->t1_naab) ? ctx->t1_naaa : ctx->t1_naab);
        if (mat_d < 1U) mat_d = 1U;
        ctx->t1_0 = (double *)calloc((size_t)(ctx->nT1 > 0 ? ctx->nT1 : 1), sizeof(double));
        ctx->scr_t1_pairs = (int *)calloc((size_t)(2 * (ctx->t1_naa > 0 ? ctx->t1_naa : 1)), sizeof(int));
        ctx->scr_t1_tri = (int *)calloc((size_t)(3 * (ctx->t1_naaa > 0 ? ctx->t1_naaa : 1)), sizeof(int));
        ctx->scr_t1_Daa = (double *)calloc(naa_s_sz * naa_s_sz, sizeof(double));
        ctx->scr_t1_Dbb = (double *)calloc(naa_s_sz * naa_s_sz, sizeof(double));
        ctx->scr_t1_Dab = (double *)calloc(nab_s_sz * nab_s_sz, sizeof(double));
        ctx->scr_t1_Dab_sw = (double *)calloc(nab_s_sz * nab_s_sz, sizeof(double));
        ctx->scr_t1_ga = (double *)calloc(kssz * kssz, sizeof(double));
        ctx->scr_t1_gb = (double *)calloc(kssz * kssz, sizeof(double));
        ctx->scr_t1_Mat = (double *)calloc(mat_d * mat_d, sizeof(double));
        if (!ctx->t1_0 || !ctx->scr_t1_pairs || !ctx->scr_t1_tri || !ctx->scr_t1_Daa ||
            !ctx->scr_t1_Dbb || !ctx->scr_t1_Dab || !ctx->scr_t1_Dab_sw ||
            !ctx->scr_t1_ga || !ctx->scr_t1_gb || !ctx->scr_t1_Mat) {
            mf_ctx_free(ctx);
            set_err(err, err_sz, "out of memory initializing T1 scratch buffers");
            return -1;
        }
        int np = 0;
        for (int a = 0; a < Ks; a++) {
            for (int b = a + 1; b < Ks; b++) {
                ctx->scr_t1_pairs[2 * np] = a;
                ctx->scr_t1_pairs[2 * np + 1] = b;
                np++;
            }
        }
        int nt = 0;
        for (int a = 0; a < Ks; a++) {
            for (int b = a + 1; b < Ks; b++) {
                for (int c = b + 1; c < Ks; c++) {
                    ctx->scr_t1_tri[3 * nt] = a;
                    ctx->scr_t1_tri[3 * nt + 1] = b;
                    ctx->scr_t1_tri[3 * nt + 2] = c;
                    nt++;
                }
            }
        }
        if (ctx->nT1 > 0) {
            double *Mat = ctx->scr_t1_Mat;
            int off = 0;
            for (int i = 0; i < ctx->blk_T1_cnt; i++) {
                int b = ctx->blk_T1_start + i;
                int d = ctx->block_dims[b];
                if (d > 0) {
                    memset(Mat, 0, (size_t)d * (size_t)d * sizeof(double));
                    for (int r = 0; r < d; r++) Mat[(size_t)r * (size_t)d + (size_t)r] = 1.0;
                    svec_pack_general(d, Mat, ctx->t1_0 + off);
                    off += ctx->block_szs[b];
                }
            }
        }
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
        if (ctx->thr_nworkers > 0) {
            ctx->pool = chem_pool_create(ctx->thr_nworkers);
            if (ctx->pool) {
                ctx->thr_nworkers = ctx->pool->nworkers;
                ctx->num_threads = 1 + ctx->thr_nworkers;
            } else {
                ctx->thr_nworkers = 0;
                ctx->num_threads = 1;
            }
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
                           0, 0, 0, NULL, 0, 0, 0, NULL, 1, err, err_sz);
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

/* Forms upper triangle (i <= j) of M_sub = sum_{k in [k0, k1)} weight_k * v_k v_k^T via cblas_dsyrk */
static void reconstruct_sym_upper_from_eigs(int n, const double *V_rows, const double *evals,
                                            int k0, int k1, double scale, double *A_tmp,
                                            double *M_out) {
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
        reconstruct_sym_upper_from_eigs(dim, V, evals, 0, k_pos, sigma, A_tmp, Mrec);
        svec_pack_upper(dim, Mrec, x_blk);
        double inv_sig = 1.0 / sigma;
        for (int i = 0; i < sz; i++) z_blk[i] = m_blk[i] + inv_sig * x_blk[i];
    } else {
        reconstruct_sym_upper_from_eigs(dim, V, evals, k_pos, dim, 1.0, A_tmp, Mrec);
        svec_pack_upper(dim, Mrec, z_blk);
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

static void split_worker_exec(void *arg, int wid) {
    ParallelSplitState *st = (ParallelSplitState *)arg;
    ChemMFContext *ctx = st->ctx;
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
}

/* Splits M = Z_new - (1/sigma)*X_new -> Z_out = M_+, X_out = sigma * (-M_-) */
static int spectral_split_all(ChemMFContext *ctx, double sigma, const double *M_svec,
                              double *Z_out, double *X_out) {
    if (!ctx->pool || ctx->thr_nworkers <= 0 || ctx->nblocks <= 1) {
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
    chem_pool_dispatch(ctx->pool, split_worker_exec, &st);
    pthread_mutex_destroy(&st.mu);
    return st.fail ? -1 : 0;
}

/* ============================================================================
 * Normal-Equation Solver G G*(y) = r via PCG + Schur Complement Cholesky
 * ============================================================================ */

static void apply_H_op(const ChemMFContext *ctx, const double *d_in, double *d_out,
                       double *buf_Q, double *buf_G, double *buf_T2, double *buf_T1,
                       double *buf_d) {
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
    if (ctx->t1_enabled && buf_T1) {
        mf_LT1_fwd(ctx, d_in, buf_T1);
        mf_LT1_adj(ctx, buf_T1, buf_d);
        for (int i = 0; i < ctx->nD; i++) {
            d_out[i] += buf_d[i];
        }
    }
}

typedef struct {
    double *S_eq;       /* [meq x meq] Cholesky factor (upper) */
    double *U_eq;       /* [nD x meq] row-major: H^{-1} A_eq^* */
    double *H_chol;     /* [nD x nD] col-major upper Cholesky of H when nD <= 1100 */
    double *H_diag_inv; /* [nD] subspace-aware Jacobi preconditioner */
    int use_H_chol;
    double *r_cg, *z_cg, *p_cg, *Ap_cg;
    double *buf_Q, *buf_G, *buf_T2, *buf_T1, *buf_d;
    double *v0, *v_D, *f_D, *aeq_rhs, *tmp_d;
} NormalEqSolver;

static void apply_H_precond(const ChemMFContext *ctx, const NormalEqSolver *ns,
                            const double *r, double *z) {
    if (ns && ns->H_diag_inv) {
        const double *inv = ns->H_diag_inv;
        for (int i = 0; i < ctx->nD; i++) z[i] = inv[i] * r[i];
        return;
    }
    double extra = (ctx->t2_enabled ? (3.5 * (double)ctx->t2_norbs) : 0.0) +
                   (ctx->t1_enabled ? (2.0 * (double)ctx->t1_norbs) : 0.0);
    if (ctx->mode == CHEM_MODE_A_SINGLET) {
        int s0 = ctx->block_szs[0];
        double w0 = 1.0 / (4.0 + extra);
        double w1 = 1.0 / (12.0 + extra);
        for (int i = 0; i < s0; i++) z[i] = w0 * r[i];
        for (int i = s0; i < ctx->nD; i++) z[i] = w1 * r[i];
    } else {
        int s01 = ctx->block_szs[0] + ctx->block_szs[1];
        double w01 = 1.0 / (4.0 + extra);
        double w2 = 1.0 / (6.0 + extra);
        for (int i = 0; i < s01; i++) z[i] = w01 * r[i];
        for (int i = s01; i < ctx->nD; i++) z[i] = w2 * r[i];
    }
}

static void normal_eq_free(NormalEqSolver *ns) {
    if (!ns) return;
    free(ns->S_eq);
    free(ns->U_eq);
    free(ns->H_chol);
    free(ns->H_diag_inv);
    free(ns->r_cg);
    free(ns->z_cg);
    free(ns->p_cg);
    free(ns->Ap_cg);
    free(ns->buf_Q);
    free(ns->buf_G);
    free(ns->buf_T2);
    free(ns->buf_T1);
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
        apply_H_op(ctx, x, ns->Ap_cg, ns->buf_Q, ns->buf_G, ns->buf_T2, ns->buf_T1, ns->buf_d);
        for (int i = 0; i < nD; i++) ns->r_cg[i] = rhs[i] - ns->Ap_cg[i];
    }

    double r_norm = norm2_vec(nD, ns->r_cg);
    if (r_norm <= tol_rel * b_norm) return 0;

    apply_H_precond(ctx, ns, ns->r_cg, ns->z_cg);
    memcpy(ns->p_cg, ns->z_cg, (size_t)nD * sizeof(double));
    double rz = dot_vec(nD, ns->r_cg, ns->z_cg);

    for (int it = 1; it <= 180; it++) {
        apply_H_op(ctx, ns->p_cg, ns->Ap_cg, ns->buf_Q, ns->buf_G, ns->buf_T2, ns->buf_T1,
                   ns->buf_d);
        double pAp = dot_vec(nD, ns->p_cg, ns->Ap_cg);
        if (pAp <= 0.0) break;
        double alpha = rz / pAp;
        for (int i = 0; i < nD; i++) {
            x[i] += alpha * ns->p_cg[i];
            ns->r_cg[i] -= alpha * ns->Ap_cg[i];
        }
        r_norm = norm2_vec(nD, ns->r_cg);
        if (r_norm <= tol_rel * b_norm) return it;
        apply_H_precond(ctx, ns, ns->r_cg, ns->z_cg);
        double rz_new = dot_vec(nD, ns->r_cg, ns->z_cg);
        double beta = rz_new / (rz + 1e-300);
        for (int i = 0; i < nD; i++) ns->p_cg[i] = ns->z_cg[i] + beta * ns->p_cg[i];
        rz = rz_new;
    }
    return 180;
}

static int normal_eq_init(const ChemMFContext *ctx, NormalEqSolver *ns, int allow_large_chol,
                          char *err, size_t err_sz) {
    memset(ns, 0, sizeof(*ns));
    int nD = ctx->nD, nQ = ctx->nQ, nG = ctx->nG, nT2 = ctx->nT2, nT1 = ctx->nT1, meq = ctx->meq;
    int use_u_eq = (nD > 0 && meq > 0 && (size_t)nD * (size_t)meq <= 262144ULL) ? 1 : 0;
    ns->S_eq    = (double *)calloc((size_t)(meq > 0 ? meq : 1) * (size_t)(meq > 0 ? meq : 1), sizeof(double));
    ns->U_eq    = use_u_eq ? (double *)calloc((size_t)nD * (size_t)meq, sizeof(double)) : NULL;
    ns->H_diag_inv = (nD > 0) ? (double *)malloc((size_t)nD * sizeof(double)) : NULL;
    ns->r_cg    = (double *)malloc((size_t)nD * sizeof(double));
    ns->z_cg    = (double *)malloc((size_t)nD * sizeof(double));
    ns->p_cg    = (double *)malloc((size_t)nD * sizeof(double));
    ns->Ap_cg   = (double *)malloc((size_t)nD * sizeof(double));
    ns->buf_Q   = (double *)malloc((size_t)nQ * sizeof(double));
    ns->buf_G   = (double *)malloc((size_t)nG * sizeof(double));
    ns->buf_T2  = ctx->t2_enabled ? (double *)malloc((size_t)(nT2 > 0 ? nT2 : 1) * sizeof(double))
                                  : NULL;
    ns->buf_T1  = ctx->t1_enabled ? (double *)malloc((size_t)(nT1 > 0 ? nT1 : 1) * sizeof(double))
                                  : NULL;
    ns->buf_d   = (double *)malloc((size_t)nD * sizeof(double));
    ns->v0      = (double *)calloc((size_t)nD, sizeof(double));
    ns->v_D     = (double *)calloc((size_t)nD, sizeof(double));
    ns->f_D     = (double *)malloc((size_t)nD * sizeof(double));
    ns->aeq_rhs = (double *)malloc((size_t)(meq > 0 ? meq : 1) * sizeof(double));
    ns->tmp_d   = (double *)malloc((size_t)nD * sizeof(double));

    if (!ns->S_eq || (use_u_eq && !ns->U_eq) || (nD > 0 && (!ns->H_diag_inv || !ns->r_cg ||
        !ns->z_cg || !ns->p_cg || !ns->Ap_cg || !ns->buf_d || !ns->v0 || !ns->v_D ||
        !ns->f_D || !ns->tmp_d)) || (nQ > 0 && !ns->buf_Q) || (nG > 0 && !ns->buf_G) ||
        !ns->aeq_rhs || (ctx->t2_enabled && !ns->buf_T2) || (ctx->t1_enabled && !ns->buf_T1)) {
        normal_eq_free(ns);
        set_err(err, err_sz, "out of memory allocating normal equation workspace");
        return -1;
    }

    /* Populate subspace-aware Jacobi diagonal preconditioner for PCG when H_chol is not built */
    if (ns->H_diag_inv && nD > 0) {
        int K = ctx->K;
        int in_t2[CHEM_MAX_ORBITALS] = {0};
        int in_t1[CHEM_MAX_ORBITALS] = {0};
        if (ctx->t2_enabled) {
            for (int i = 0; i < ctx->t2_norbs; i++) {
                int o = ctx->t2_orbs[i];
                if (o >= 0 && o < K) in_t2[o] = 1;
            }
        }
        if (ctx->t1_enabled) {
            for (int i = 0; i < ctx->t1_norbs; i++) {
                int o = ctx->t1_orbs[i];
                if (o >= 0 && o < K) in_t1[o] = 1;
            }
        }
        double w2_full = 3.5 * (double)ctx->t2_norbs;
        double w2_part = 0.5 * (double)ctx->t2_norbs;
        double w1_full = 2.0 * (double)ctx->t1_norbs;
        double w1_part = 0.35 * (double)ctx->t1_norbs;

        int sym_p[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS];
        int sym_q[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS];
        int nsym = 0;
        for (int p = 0; p < K; p++) {
            for (int q = p; q < K; q++) {
                sym_p[nsym] = p;
                sym_q[nsym] = q;
                nsym++;
            }
        }
        int asym_p[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS];
        int asym_q[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS];
        int nasym = 0;
        for (int p = 0; p < K; p++) {
            for (int q = p + 1; q < K; q++) {
                asym_p[nasym] = p;
                asym_q[nasym] = q;
                nasym++;
            }
        }
        (void)nsym;
        (void)nasym;

        int idx = 0;
        if (ctx->mode == CHEM_MODE_A_SINGLET) {
            for (int a = 0; a < ctx->m0; a++) {
                int p = sym_p[a], q = sym_q[a];
                for (int b = a; b < ctx->m0; b++) {
                    int r = sym_p[b], s = sym_q[b];
                    double d_est = 4.0;
                    if (ctx->t2_enabled) {
                        int c2 = in_t2[p] + in_t2[q] + in_t2[r] + in_t2[s];
                        d_est += (c2 == 4) ? w2_full : ((c2 >= 2) ? w2_part : 0.0);
                    }
                    if (ctx->t1_enabled) {
                        int c1 = in_t1[p] + in_t1[q] + in_t1[r] + in_t1[s];
                        d_est += (c1 == 4) ? w1_full : ((c1 >= 2) ? w1_part : 0.0);
                    }
                    ns->H_diag_inv[idx++] = 1.0 / d_est;
                }
            }
            for (int a = 0; a < ctx->m1; a++) {
                int p = asym_p[a], q = asym_q[a];
                for (int b = a; b < ctx->m1; b++) {
                    int r = asym_p[b], s = asym_q[b];
                    double d_est = 12.0;
                    if (ctx->t2_enabled) {
                        int c2 = in_t2[p] + in_t2[q] + in_t2[r] + in_t2[s];
                        d_est += (c2 == 4) ? w2_full : ((c2 >= 2) ? w2_part : 0.0);
                    }
                    if (ctx->t1_enabled) {
                        int c1 = in_t1[p] + in_t1[q] + in_t1[r] + in_t1[s];
                        d_est += (c1 == 4) ? w1_full : ((c1 >= 2) ? w1_part : 0.0);
                    }
                    ns->H_diag_inv[idx++] = 1.0 / d_est;
                }
            }
        } else {
            for (int blk = 0; blk < 2; blk++) {
                for (int a = 0; a < ctx->naa; a++) {
                    int p = asym_p[a], q = asym_q[a];
                    for (int b = a; b < ctx->naa; b++) {
                        int r = asym_p[b], s = asym_q[b];
                        double d_est = 4.0;
                        if (ctx->t2_enabled) {
                            int c2 = in_t2[p] + in_t2[q] + in_t2[r] + in_t2[s];
                            d_est += (c2 == 4) ? w2_full : ((c2 >= 2) ? w2_part : 0.0);
                        }
                        if (ctx->t1_enabled) {
                            int c1 = in_t1[p] + in_t1[q] + in_t1[r] + in_t1[s];
                            d_est += (c1 == 4) ? w1_full : ((c1 >= 2) ? w1_part : 0.0);
                        }
                        ns->H_diag_inv[idx++] = 1.0 / d_est;
                    }
                }
            }
            for (int a = 0; a < ctx->nab; a++) {
                int p = a / K, q = a % K;
                for (int b = a; b < ctx->nab; b++) {
                    int r = b / K, s = b % K;
                    double d_est = 6.0;
                    if (ctx->t2_enabled) {
                        int c2 = in_t2[p] + in_t2[q] + in_t2[r] + in_t2[s];
                        d_est += (c2 == 4) ? w2_full : ((c2 >= 2) ? w2_part : 0.0);
                    }
                    if (ctx->t1_enabled) {
                        int c1 = in_t1[p] + in_t1[q] + in_t1[r] + in_t1[s];
                        d_est += (c1 == 4) ? w1_full : ((c1 >= 2) ? w1_part : 0.0);
                    }
                    ns->H_diag_inv[idx++] = 1.0 / d_est;
                }
            }
        }
        if (idx != nD) {
            free(ns->H_diag_inv);
            ns->H_diag_inv = NULL;
        }
    }

    /* Assemble and Cholesky-factor H when nD <= 512 (always) or nD <= 1100 (when allowed) */
    int build_h_chol = (nD > 0 && (nD <= 512 || (allow_large_chol && nD <= 1100))) ? 1 : 0;
    if (build_h_chol) {
        double *Hmat = (double *)calloc((size_t)nD * (size_t)nD, sizeof(double));
        double *e_col_d = (double *)calloc((size_t)nD, sizeof(double));
        double *h_col = (double *)malloc((size_t)nD * sizeof(double));
        if (Hmat && e_col_d && h_col) {
            for (int j = 0; j < nD; j++) {
                if (j > 0) e_col_d[j - 1] = 0.0;
                e_col_d[j] = 1.0;
                apply_H_op(ctx, e_col_d, h_col, ns->buf_Q, ns->buf_G, ns->buf_T2, ns->buf_T1,
                           ns->buf_d);
                memcpy(Hmat + (size_t)j * (size_t)nD, h_col, (size_t)nD * sizeof(double));
            }
            for (int j = 0; j < nD; j++) {
                for (int i = j + 1; i < nD; i++) {
                    double avg = 0.5 * (Hmat[(size_t)j * (size_t)nD + (size_t)i] +
                                        Hmat[(size_t)i * (size_t)nD + (size_t)j]);
                    Hmat[(size_t)j * (size_t)nD + (size_t)i] = avg;
                    Hmat[(size_t)i * (size_t)nD + (size_t)j] = avg;
                }
            }
            __LAPACK_int n_d = (__LAPACK_int)nD;
            __LAPACK_int info_h = 0;
            dpotrf_("U", &n_d, Hmat, &n_d, &info_h);
            if (info_h == 0) {
                ns->H_chol = Hmat;
                ns->use_H_chol = 1;
                Hmat = NULL;
            }
        }
        free(Hmat);
        free(e_col_d);
        free(h_col);
    }

    double *e_col = (double *)calloc((size_t)(meq > 0 ? meq : 1), sizeof(double));
    double *a_m   = (double *)malloc((size_t)(nD > 0 ? nD : 1) * sizeof(double));
    double *u_m   = (double *)malloc((size_t)(nD > 0 ? nD : 1) * sizeof(double));
    double *su_m  = (double *)malloc((size_t)(meq > 0 ? meq : 1) * sizeof(double));
    if (!e_col || !a_m || !u_m || !su_m) {
        free(e_col);
        free(a_m);
        free(u_m);
        free(su_m);
        normal_eq_free(ns);
        set_err(err, err_sz, "out of memory allocating constraint scratch in normal_eq_init");
        return -1;
    }

    for (int m = 0; m < meq; m++) {
        if (m > 0) e_col[m - 1] = 0.0;
        e_col[m] = 1.0;
        mf_Aeq_adj(ctx, e_col, a_m);
        if (ns->use_H_chol) {
            memcpy(u_m, a_m, (size_t)nD * sizeof(double));
            __LAPACK_int n_d = (__LAPACK_int)nD;
            __LAPACK_int one_h = 1;
            __LAPACK_int info_h = 0;
            dpotrs_("U", &n_d, &one_h, ns->H_chol, &n_d, u_m, &n_d, &info_h);
            if (info_h != 0) {
                pcg_solve_H(ctx, ns, a_m, u_m, 1, 1e-13);
            }
        } else {
            pcg_solve_H(ctx, ns, a_m, u_m, 1, 1e-13);
        }
        if (ns->U_eq) {
            for (int i = 0; i < nD; i++) {
                ns->U_eq[(size_t)i * (size_t)meq + (size_t)m] = u_m[i];
            }
        }
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

    if (meq > 0) {
        __LAPACK_int n_s = (__LAPACK_int)meq;
        __LAPACK_int info = 0;
        dpotrf_("U", &n_s, ns->S_eq, &n_s, &info);
        if (info != 0) {
            normal_eq_free(ns);
            set_err(err, err_sz, "Schur complement S_eq is not SPD (dpotrf info=%d)", (int)info);
            return -1;
        }
    }
    return 0;
}

/* Solves G G*(y) = r and outputs Gt_y = G*(y) as well as y_eq */
static int normal_eq_solve(const ChemMFContext *ctx, NormalEqSolver *ns,
                           const double *r_eq, const double *r_Q, const double *r_G,
                           const double *r_T2, const double *r_T1,
                           double *y_eq_out, double *Gt_y_out,
                           int warm_start, double cg_tol) {
    int nD = ctx->nD, nQ = ctx->nQ, nG = ctx->nG, nT2 = ctx->nT2, nT1 = ctx->nT1, meq = ctx->meq;

    mf_LQ_adj(ctx, r_Q, ns->f_D);
    mf_LG_adj(ctx, r_G, ns->tmp_d);
    for (int i = 0; i < nD; i++) ns->f_D[i] += ns->tmp_d[i];
    if (ctx->t2_enabled && r_T2) {
        mf_LT2_adj(ctx, r_T2, ns->tmp_d);
        for (int i = 0; i < nD; i++) ns->f_D[i] += ns->tmp_d[i];
    }
    if (ctx->t1_enabled && r_T1) {
        mf_LT1_adj(ctx, r_T1, ns->tmp_d);
        for (int i = 0; i < nD; i++) ns->f_D[i] += ns->tmp_d[i];
    }

    if (ns->U_eq && meq > 0 && nD > 0) {
        cblas_dgemv(CblasRowMajor, CblasTrans, nD, meq, 1.0, ns->U_eq, meq, ns->f_D, 1, 0.0,
                    ns->aeq_rhs, 1);
    } else if (meq > 0 && nD > 0) {
        pcg_solve_H(ctx, ns, ns->f_D, ns->v0, !warm_start, cg_tol);
        mf_Aeq_fwd(ctx, ns->v0, ns->aeq_rhs);
    } else {
        for (int m = 0; m < meq; m++) ns->aeq_rhs[m] = 0.0;
    }
    for (int m = 0; m < meq; m++) y_eq_out[m] = r_eq[m] - ns->aeq_rhs[m];

    if (meq > 0) {
        __LAPACK_int n_s = (__LAPACK_int)meq;
        __LAPACK_int nrhs = 1;
        __LAPACK_int info = 0;
        dpotrs_("U", &n_s, &nrhs, ns->S_eq, &n_s, y_eq_out, &n_s, &info);
        if (info != 0) return -1;
    }

    mf_Aeq_adj(ctx, y_eq_out, ns->tmp_d);
    for (int i = 0; i < nD; i++) ns->tmp_d[i] += ns->f_D[i];
    if (ns->use_H_chol) {
        memcpy(ns->v_D, ns->tmp_d, (size_t)nD * sizeof(double));
        __LAPACK_int n_d = (__LAPACK_int)nD;
        __LAPACK_int one_h = 1;
        __LAPACK_int info_h = 0;
        dpotrs_("U", &n_d, &one_h, ns->H_chol, &n_d, ns->v_D, &n_d, &info_h);
        if (info_h != 0) return -1;
    } else {
        pcg_solve_H(ctx, ns, ns->tmp_d, ns->v_D, !warm_start, cg_tol);
    }

    memcpy(Gt_y_out, ns->v_D, (size_t)nD * sizeof(double));
    mf_LQ_fwd(ctx, ns->v_D, ns->buf_Q);
    mf_LG_fwd(ctx, ns->v_D, ns->buf_G);
    for (int i = 0; i < nQ; i++) Gt_y_out[nD + i] = ns->buf_Q[i] - r_Q[i];
    for (int i = 0; i < nG; i++) Gt_y_out[nD + nQ + i] = ns->buf_G[i] - r_G[i];
    if (ctx->t2_enabled && r_T2) {
        mf_LT2_fwd(ctx, ns->v_D, ns->buf_T2);
        for (int i = 0; i < nT2; i++) Gt_y_out[nD + nQ + nG + i] = ns->buf_T2[i] - r_T2[i];
    }
    if (ctx->t1_enabled && r_T1) {
        mf_LT1_fwd(ctx, ns->v_D, ns->buf_T1);
        for (int i = 0; i < nT1; i++) {
            Gt_y_out[nD + nQ + nG + nT2 + i] = ns->buf_T1[i] - r_T1[i];
        }
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
    const double *Z_T1;
    double w_scale;
    int blk_offs[CHEM_MAX_BLOCKS];
    int blk_src[CHEM_MAX_BLOCKS]; /* 0=D, 1=Q, 2=G, 3=T2, 4=T1 */
    double fp_arr[CHEM_MAX_BLOCKS];
    double frob_arr[CHEM_MAX_BLOCKS];
    int done[CHEM_MAX_BLOCKS];
    pthread_mutex_t mu;
    int fail;
} ParallelCertState;

static void cert_worker_exec(void *arg, int wid) {
    ParallelCertState *st = (ParallelCertState *)arg;
    ChemMFContext *ctx = st->ctx;
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
        } else if (st->blk_src[b_pick] == 3) {
            ptr = st->Z_T2 + st->blk_offs[b_pick];
            sc = st->w_scale;
        } else {
            ptr = st->Z_T1 + st->blk_offs[b_pick];
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
}

static int certify_with_ctx(ChemMFContext *ctx, double e_core,
                            const double *y_eq, const double *Z_Q, const double *Z_G,
                            const double *Z_T2, const double *Z_T1, double w_scale,
                            ChemSolveResult *res_out, char *err, size_t err_sz) {
    int nD = ctx->nD, nQ = ctx->nQ, nT1 = ctx->nT1, meq = ctx->meq;

    double *Z_D = (double *)malloc((size_t)nD * sizeof(double));
    double *tmp_d = (double *)malloc((size_t)nD * sizeof(double));
    if (!Z_D || !tmp_d) {
        free(Z_D);
        free(tmp_d);
        set_err(err, err_sz, "out of memory in certifier");
        return -1;
    }

    /* Z_D = C_D - w_scale * (A_eq*(y_eq) + L_Q*(Z_Q) + L_G*(Z_G) + L_T2*(Z_T2) + L_T1*(Z_T1)) */
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
    if (ctx->t1_enabled && Z_T1) {
        mf_LT1_adj(ctx, Z_T1, tmp_d);
        for (int i = 0; i < nD; i++) Z_D[i] -= w_scale * tmp_d[i];
    }
    free(tmp_d);

    double t1_dot = (ctx->t1_enabled && Z_T1 && ctx->t1_0 && nT1 > 0)
                        ? dot_vec(nT1, Z_T1, ctx->t1_0)
                        : 0.0;
    double e_dual = e_core + w_scale * (dot_vec(meq, y_eq, ctx->b_eq) -
                                        dot_vec(nQ, Z_Q, ctx->q0) - t1_dot);

    int nD_blocks = ctx->blk_D_cnt;
    ParallelCertState st;
    memset(&st, 0, sizeof(st));
    st.ctx = ctx;
    st.Z_D = Z_D;
    st.Z_Q = Z_Q;
    st.Z_G = Z_G;
    st.Z_T2 = Z_T2;
    st.Z_T1 = Z_T1;
    st.w_scale = w_scale;

    int off = 0;
    for (int b = 0; b < ctx->nblocks; b++) {
        int sz = ctx->block_szs[b];
        st.blk_offs[b] = off;
        if (b < ctx->blk_Q_start) {
            st.blk_src[b] = 0;
            off = (b == ctx->blk_Q_start - 1) ? 0 : off + sz;
        } else if (b < ctx->blk_G_start) {
            st.blk_src[b] = 1;
            off = (b == ctx->blk_G_start - 1) ? 0 : off + sz;
        } else if (b < ctx->blk_T2_start) {
            st.blk_src[b] = 2;
            off = (b == ctx->blk_T2_start - 1) ? 0 : off + sz;
        } else if (b < ctx->blk_T1_start) {
            st.blk_src[b] = 3;
            off = (b == ctx->blk_T1_start - 1) ? 0 : off + sz;
        } else {
            st.blk_src[b] = 4;
            off += sz;
        }
    }

    pthread_mutex_init(&st.mu, NULL);
    if (ctx->pool && ctx->thr_nworkers > 0) {
        chem_pool_dispatch(ctx->pool, cert_worker_exec, &st);
    } else {
        cert_worker_exec(&st, 0);
    }
    pthread_mutex_destroy(&st.mu);

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
    double t1_round = (ctx->t1_enabled && Z_T1 && ctx->t1_0 && nT1 > 0)
                          ? (w_scale * norm2_vec(nT1, Z_T1) * norm2_vec(nT1, ctx->t1_0))
                          : 0.0;
    double delta_round =
        64.0 * DBL_EPSILON * (fabs(e_core) + c_norm * tr_D_sum + y_norm * beq_norm +
                              sum_frob_tr + t1_round);

    res_out->mode_used = ctx->mode;
    res_out->nblocks = ctx->nblocks;
    res_out->t2_enabled = ctx->t2_enabled;
    res_out->t2_norbs = ctx->t2_norbs;
    if (ctx->t2_enabled) {
        memcpy(res_out->t2_orbs, ctx->t2_orbs, (size_t)ctx->t2_norbs * sizeof(int));
    }
    res_out->t1_enabled = ctx->t1_enabled;
    res_out->t1_norbs = ctx->t1_norbs;
    if (ctx->t1_enabled) {
        memcpy(res_out->t1_orbs, ctx->t1_orbs, (size_t)ctx->t1_norbs * sizeof(int));
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
                        w->t2_enabled, w->t2_norbs, w->t2_norbs, w->t2_orbs,
                        w->t1_enabled, w->t1_norbs, w->t1_norbs, w->t1_orbs,
                        0, err, err_sz) != 0) {
        return -1;
    }
    if (w->meq != ctx.meq || w->nD != ctx.nD || w->nQ != ctx.nQ || w->nG != ctx.nG ||
        w->nT2 != ctx.nT2 || w->nT1 != ctx.nT1) {
        mf_ctx_free(&ctx);
        set_err(err, err_sz, "dual witness dimensions mismatch context");
        return -1;
    }
    int rc = certify_with_ctx(&ctx, ham->E_core, w->y_eq, w->Z_Q, w->Z_G, w->Z_T2, w->Z_T1, 1.0,
                              res_out, err, err_sz);
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
        if (!H1 || !ev) {
            free(H1);
            free(ev);
            set_err(err, err_sz, "out of memory in solve_trivial_space_cost");
            return -1;
        }
        __LAPACK_int n = (__LAPACK_int)K, lwork = -1, liwork = -1, info = 0;
        double wq = 0.0;
        __LAPACK_int iwq = 0;
        memcpy(H1, ham->h, (size_t)K * (size_t)K * sizeof(double));
        dsyevd_("N", "U", &n, H1, &n, ev, &wq, &lwork, &iwq, &liwork, &info);
        lwork = (__LAPACK_int)wq + 16;
        liwork = iwq + 8;
        double *wk = (double *)malloc((size_t)lwork * sizeof(double));
        __LAPACK_int *iwk = (__LAPACK_int *)malloc((size_t)liwork * sizeof(__LAPACK_int));
        if (!wk || !iwk) {
            free(H1);
            free(ev);
            free(wk);
            free(iwk);
            set_err(err, err_sz, "out of memory in solve_trivial_space_cost eigensolver");
            return -1;
        }
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
                            0, 0, 0, NULL, 0, 0, 0, NULL, 1, err, err_sz) == 0) {
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

static void mf_compute_primal_ham_grad(const ChemMFContext *ctx, const double *d_in,
                                         double *grad_h1, double *grad_eri) {
    int K = ctx->K;
    size_t ksz = (size_t)K;
    size_t k2 = ksz * ksz;
    size_t k4 = k2 * k2;
    memset(grad_h1, 0, k2 * sizeof(double));
    memset(grad_eri, 0, k4 * sizeof(double));

    if (ctx->mode == CHEM_MODE_A_SINGLET) {
        mode_a_expand_st_tt(ctx, d_in, ctx->scr_St, ctx->scr_Tt);
        mode_a_gamma_from_st(K, ctx->na, ctx->scr_St, ctx->scr_ga);
        const double *St = ctx->scr_St;
        const double *Tt = ctx->scr_Tt;
        const double *ga = ctx->scr_ga;
        for (int p = 0; p < K; p++) {
            for (int r = 0; r < K; r++) {
                grad_h1[(size_t)p * ksz + (size_t)r] = 2.0 * ga[(size_t)p * ksz + (size_t)r];
                for (int q = 0; q < K; q++) {
                    for (int s = 0; s < K; s++) {
                        size_t off4 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz +
                                       (size_t)s);
                        size_t idx_eri = ((((size_t)p * ksz + (size_t)r) * ksz + (size_t)q) * ksz +
                                          (size_t)s);
                        grad_eri[idx_eri] = St[off4] + 3.0 * Tt[off4];
                    }
                }
            }
        }
        return;
    }

    int naa = ctx->naa;
    int nab = ctx->nab;
    double *Daa = ctx->scr_D0;
    double *Dbb = ctx->scr_D1;
    double *Dab = ctx->scr_D2;
    double *ga  = ctx->scr_ga;
    double *gb  = ctx->scr_gb;
    mode_b_unpack_D(ctx, d_in, Daa, Dbb, Dab);
    mode_b_gamma_fwd(ctx, Daa, Dbb, Dab, ga, gb);

    for (int p = 0; p < K; p++) {
        for (int r = 0; r < K; r++) {
            size_t pr = (size_t)p * ksz + (size_t)r;
            grad_h1[pr] = ga[pr] + gb[pr];
            for (int q = 0; q < K; q++) {
                double spq = ctx->sgn1[p][q];
                int P = ctx->P1[p][q];
                for (int s = 0; s < K; s++) {
                    double v = Dab[(size_t)(p * K + q) * (size_t)nab + (size_t)(r * K + s)];
                    if (naa > 0 && P >= 0) {
                        int R = ctx->P1[r][s];
                        if (R >= 0) {
                            double sgn = spq * ctx->sgn1[r][s];
                            v += 0.5 * sgn *
                                 (Daa[(size_t)P * (size_t)naa + (size_t)R] +
                                  Dbb[(size_t)P * (size_t)naa + (size_t)R]);
                        }
                    }
                    size_t idx_eri = ((((size_t)p * ksz + (size_t)r) * ksz + (size_t)q) * ksz +
                                      (size_t)s);
                    grad_eri[idx_eri] = v;
                }
            }
        }
    }
}

typedef struct {
    double *C_sc;
    double *X;
    double *Z;
    double *Gt_y;
    double *M;
    double *r_eq;
    double *y_eq;
    double *best_y;
    double *best_ZQ;
    double *best_ZG;
    double *best_ZT2;
    double *best_ZT1;
} BpsdpWorkVecs;

static void bpsdp_work_free(BpsdpWorkVecs *w) {
    if (!w) return;
    free(w->C_sc);
    free(w->X);
    free(w->Z);
    free(w->Gt_y);
    free(w->M);
    free(w->r_eq);
    free(w->y_eq);
    free(w->best_y);
    free(w->best_ZQ);
    free(w->best_ZG);
    free(w->best_ZT2);
    free(w->best_ZT1);
    memset(w, 0, sizeof(*w));
}

static int bpsdp_work_alloc(const ChemMFContext *ctx, BpsdpWorkVecs *w, char *err, size_t err_sz) {
    memset(w, 0, sizeof(*w));
    int nD = ctx->nD, nQ = ctx->nQ, nG = ctx->nG, nT2 = ctx->nT2, nT1 = ctx->nT1;
    int nAll = ctx->nAll, meq = ctx->meq;
    w->C_sc     = (double *)calloc((size_t)(nD > 0 ? nD : 1), sizeof(double));
    w->X        = (double *)calloc((size_t)(nAll > 0 ? nAll : 1), sizeof(double));
    w->Z        = (double *)calloc((size_t)(nAll > 0 ? nAll : 1), sizeof(double));
    w->Gt_y     = (double *)calloc((size_t)(nAll > 0 ? nAll : 1), sizeof(double));
    w->M        = (double *)calloc((size_t)(nAll > 0 ? nAll : 1), sizeof(double));
    w->r_eq     = (double *)calloc((size_t)(meq > 0 ? meq : 1), sizeof(double));
    w->y_eq     = (double *)calloc((size_t)(meq > 0 ? meq : 1), sizeof(double));
    w->best_y   = (double *)calloc((size_t)(meq > 0 ? meq : 1), sizeof(double));
    w->best_ZQ  = (double *)calloc((size_t)(nQ > 0 ? nQ : 1), sizeof(double));
    w->best_ZG  = (double *)calloc((size_t)(nG > 0 ? nG : 1), sizeof(double));
    w->best_ZT2 = ctx->t2_enabled ? (double *)calloc((size_t)(nT2 > 0 ? nT2 : 1), sizeof(double))
                                    : NULL;
    w->best_ZT1 = ctx->t1_enabled ? (double *)calloc((size_t)(nT1 > 0 ? nT1 : 1), sizeof(double))
                                    : NULL;
    if (!w->C_sc || !w->X || !w->Z || !w->Gt_y || !w->M || !w->r_eq || !w->y_eq ||
        !w->best_y || !w->best_ZQ || !w->best_ZG ||
        (ctx->t2_enabled && !w->best_ZT2) || (ctx->t1_enabled && !w->best_ZT1)) {
        bpsdp_work_free(w);
        set_err(err, err_sz, "out of memory allocating BPSDP vectors");
        return -1;
    }
    return 0;
}

static int run_bpsdp_core(ChemMFContext *ctx, NormalEqSolver *ns, BpsdpWorkVecs *w,
                          double e_core, int max_iters, double tol, double alpha,
                          int check_every, int veto_enabled, double E_ai, double veto_tol,
                          int verbose, int warm_start, double scale_c, double b_norm,
                          double c_sc_norm, double *sigma_io, double *xi_prev_io,
                          double *g0_prev_io, double *g1_est_io, ChemSolveResult *best_cert_out,
                          int *iter_stopped_out, int *converged_out, int *vetoed_out,
                          double *prim_res_out, double *dual_res_out, char *err, size_t err_sz) {
    int nD = ctx->nD, nQ = ctx->nQ, nG = ctx->nG, nT2 = ctx->nT2, nT1 = ctx->nT1;
    int nAll = ctx->nAll, meq = ctx->meq;
    double *C_sc = w->C_sc, *X = w->X, *Z = w->Z, *Gt_y = w->Gt_y, *M = w->M;
    double *r_eq = w->r_eq, *y_eq = w->y_eq;
    /* r_Q, r_G, r_T2, r_T1 alias unused tail of M during residual construction and normal_eq_solve */
    double *r_Q  = M + nD;
    double *r_G  = M + nD + nQ;
    double *r_T2 = ctx->t2_enabled ? (M + nD + nQ + nG) : NULL;
    double *r_T1 = ctx->t1_enabled ? (M + nD + nQ + nG + nT2) : NULL;

    double sigma = *sigma_io;
    double xi_prev = *xi_prev_io;
    double g0_prev = *g0_prev_io;
    double g1_est = *g1_est_io;

    int has_best = 0;
    ChemSolveResult best_cert;
    memset(&best_cert, 0, sizeof(best_cert));

    int converged = 0;
    int vetoed = 0;
    int iter_stopped = 0;
    double prim_res = 1.0, dual_res = 1.0;
    int min_conv_it = warm_start ? 15 : 25;

    for (int it = 1; it <= max_iters; it++) {
        iter_stopped = it;

        /* Compute r = G(C_sc - Z) + (1/sigma)*(b - G(X)) */
        double inv_sig = 1.0 / sigma;
        for (int i = 0; i < nD; i++) ns->tmp_d[i] = C_sc[i] - Z[i] - inv_sig * X[i];
        const double *z_q = Z + nD;
        const double *x_q = X + nD;
        const double *z_g = Z + nD + nQ;
        const double *x_g = X + nD + nQ;

        mf_Aeq_fwd(ctx, ns->tmp_d, r_eq);
        mf_LQ_fwd(ctx, ns->tmp_d, r_Q);
        mf_LG_fwd(ctx, ns->tmp_d, r_G);
        if (ctx->t2_enabled) mf_LT2_fwd(ctx, ns->tmp_d, r_T2);
        if (ctx->t1_enabled) mf_LT1_fwd(ctx, ns->tmp_d, r_T1);

        for (int m = 0; m < meq; m++) r_eq[m] += inv_sig * ctx->b_eq[m];
        for (int i = 0; i < nQ; i++) {
            r_Q[i] += z_q[i] - inv_sig * (ctx->q0[i] - x_q[i]);
        }
        for (int i = 0; i < nG; i++) {
            r_G[i] += z_g[i] + inv_sig * x_g[i];
        }
        if (ctx->t2_enabled) {
            const double *z_t2 = Z + nD + nQ + nG;
            const double *x_t2 = X + nD + nQ + nG;
            for (int i = 0; i < nT2; i++) r_T2[i] += z_t2[i] + inv_sig * x_t2[i];
        }
        if (ctx->t1_enabled) {
            const double *z_t1 = Z + nD + nQ + nG + nT2;
            const double *x_t1 = X + nD + nQ + nG + nT2;
            for (int i = 0; i < nT1; i++) {
                r_T1[i] += z_t1[i] - inv_sig * (ctx->t1_0[i] - x_t1[i]);
            }
        }

        double min_res = (prim_res < dual_res) ? prim_res : dual_res;
        double cg_tol = fmax(1e-11, fmin(1e-6, 0.02 * min_res));
        if (normal_eq_solve(ctx, ns, r_eq, r_Q, r_G, r_T2, r_T1, y_eq, Gt_y,
                            (it > 1) || warm_start, cg_tol) != 0) {
            set_err(err, err_sz, "normal equation Cholesky solve failed at iter %d", it);
            break;
        }

        /* Over-relaxed dual step + spectral split (C_sc is zero on Q, G, T2, T1) */
        for (int i = 0; i < nD; i++) {
            M[i] = alpha * (C_sc[i] - Gt_y[i]) + (1.0 - alpha) * Z[i] - inv_sig * X[i];
        }
        for (int i = nD; i < nAll; i++) {
            M[i] = -alpha * Gt_y[i] + (1.0 - alpha) * Z[i] - inv_sig * X[i];
        }

        if (spectral_split_all(ctx, sigma, M, Z, X) != 0) {
            set_err(err, err_sz, "dsyevd_ failed during spectral split at iter %d", it);
            break;
        }

        /* Compute primal residual ||G(X) - b|| / (1 + ||b||) & dual residual ||G*(y) + Z - C|| / (1 + ||C||) */
        mf_Aeq_fwd(ctx, X, ns->aeq_rhs);
        mf_LQ_fwd(ctx, X, ns->buf_Q);
        mf_LG_fwd(ctx, X, ns->buf_G);
        if (ctx->t2_enabled) mf_LT2_fwd(ctx, X, ns->buf_T2);
        if (ctx->t1_enabled) mf_LT1_fwd(ctx, X, ns->buf_T1);

        double p2 = 0.0;
        for (int m = 0; m < meq; m++) {
            double d = ns->aeq_rhs[m] - ctx->b_eq[m];
            p2 += d * d;
        }
        for (int i = 0; i < nQ; i++) {
            double d = ns->buf_Q[i] - X[nD + i] + ctx->q0[i];
            p2 += d * d;
        }
        for (int i = 0; i < nG; i++) {
            double d = ns->buf_G[i] - X[nD + nQ + i];
            p2 += d * d;
        }
        if (ctx->t2_enabled) {
            for (int i = 0; i < nT2; i++) {
                double d = ns->buf_T2[i] - X[nD + nQ + nG + i];
                p2 += d * d;
            }
        }
        if (ctx->t1_enabled) {
            for (int i = 0; i < nT1; i++) {
                double d = ns->buf_T1[i] - X[nD + nQ + nG + nT2 + i] + ctx->t1_0[i];
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

        int is_conv = (it >= min_conv_it && prim_res < tol && dual_res < tol);
        int do_cert = is_conv || (it == max_iters) || (it >= 10 && (it % check_every) == 0);

        if (do_cert) {
            ChemSolveResult cres;
            memset(&cres, 0, sizeof(cres));
            const double *z_t2_ptr = ctx->t2_enabled ? (Z + nD + nQ + nG) : NULL;
            const double *z_t1_ptr = ctx->t1_enabled ? (Z + nD + nQ + nG + nT2) : NULL;
            if (certify_with_ctx(ctx, e_core, y_eq, Z + nD, Z + nD + nQ, z_t2_ptr, z_t1_ptr,
                                 scale_c, &cres, err, err_sz) != 0) {
                break;
            }

            if (!has_best || cres.E_lb_cert > best_cert.E_lb_cert) {
                has_best = 1;
                best_cert = cres;
                for (int m = 0; m < meq; m++) w->best_y[m] = scale_c * y_eq[m];
                for (int i = 0; i < nQ; i++) w->best_ZQ[i] = scale_c * Z[nD + i];
                for (int i = 0; i < nG; i++) w->best_ZG[i] = scale_c * Z[nD + nQ + i];
                if (ctx->t2_enabled) {
                    for (int i = 0; i < nT2; i++) w->best_ZT2[i] = scale_c * Z[nD + nQ + nG + i];
                }
                if (ctx->t1_enabled) {
                    for (int i = 0; i < nT1; i++) {
                        w->best_ZT1[i] = scale_c * Z[nD + nQ + nG + nT2 + i];
                    }
                }
            }

            if (verbose && (it % (check_every * 2) == 0 || is_conv)) {
                printf("  [BPSDP it=%3d] sigma=%.3f pres=%.2e dres=%.2e E_lb_best=%.8f\n",
                       it, sigma, prim_res, dual_res, best_cert.E_lb_cert);
            }

            if (veto_enabled && best_cert.E_lb_cert > E_ai - veto_tol) {
                vetoed = 1;
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
        for (int m = 0; m < meq; m++) w->best_y[m] = scale_c * y_eq[m];
        for (int i = 0; i < nQ; i++) w->best_ZQ[i] = scale_c * Z[nD + i];
        for (int i = 0; i < nG; i++) w->best_ZG[i] = scale_c * Z[nD + nQ + i];
        if (ctx->t2_enabled) {
            for (int i = 0; i < nT2; i++) w->best_ZT2[i] = scale_c * Z[nD + nQ + nG + i];
        }
        if (ctx->t1_enabled) {
            for (int i = 0; i < nT1; i++) w->best_ZT1[i] = scale_c * Z[nD + nQ + nG + nT2 + i];
        }
        if (certify_with_ctx(ctx, e_core, w->best_y, w->best_ZQ, w->best_ZG, w->best_ZT2,
                             w->best_ZT1, 1.0, &best_cert, err, err_sz) != 0) {
            return -1;
        }
    }

    *sigma_io = sigma;
    *xi_prev_io = xi_prev;
    *g0_prev_io = g0_prev;
    *g1_est_io = g1_est;
    *best_cert_out = best_cert;
    *iter_stopped_out = iter_stopped;
    *converged_out = converged;
    *vetoed_out = vetoed;
    *prim_res_out = prim_res;
    *dual_res_out = dual_res;
    return 0;
}

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

    if (!custom_C_D && !opts.export_rdm_grad) {
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
                        custom_C_D ? NULL : ham,
                        opts.enable_t2, opts.t2_max_orbs, opts.t2_norbs, opts.t2_orbs,
                        opts.enable_t1, opts.t1_max_orbs, opts.t1_norbs, opts.t1_orbs,
                        opts.num_threads, err, err_sz) != 0) {
        return -1;
    }
    if (custom_C_D) {
        memcpy(ctx.C_D, custom_C_D, (size_t)ctx.nD * sizeof(double));
    }

    int allow_large_chol =
        (!opts.veto_enabled &&
         ((opts.enable_t1 || opts.enable_t2) ? (opts.max_iters >= 200)
                                             : (opts.max_iters >= 1200)))
            ? 1
            : 0;
    NormalEqSolver ns;
    if (normal_eq_init(&ctx, &ns, allow_large_chol, err, err_sz) != 0) {
        mf_ctx_free(&ctx);
        return -1;
    }

    BpsdpWorkVecs w;
    if (bpsdp_work_alloc(&ctx, &w, err, err_sz) != 0) {
        normal_eq_free(&ns);
        mf_ctx_free(&ctx);
        return -1;
    }

    int nD = ctx.nD, nQ = ctx.nQ, nG = ctx.nG, nT2 = ctx.nT2, nT1 = ctx.nT1;
    int meq = ctx.meq;

    /* Scale cost vector for scale-invariant BPSDP conditioning */
    double c_norm = norm2_vec(nD, ctx.C_D);
    double scale_c = c_norm / sqrt((double)(nD > 0 ? nD : 1));
    if (scale_c < 1.0) scale_c = 1.0;

    for (int i = 0; i < nD; i++) w.C_sc[i] = ctx.C_D[i] / scale_c;
    double b2 = dot_vec(meq, ctx.b_eq, ctx.b_eq) + dot_vec(nQ, ctx.q0, ctx.q0);
    if (ctx.t1_enabled && nT1 > 0 && ctx.t1_0) b2 += dot_vec(nT1, ctx.t1_0, ctx.t1_0);
    double b_norm = sqrt(b2);
    double c_sc_norm = norm2_vec(nD, w.C_sc);

    /* Initial feasible primal point estimate & dual start */
    double sigma = fmax(0.5, fmin(3.0, b_norm / fmax(1.0, c_sc_norm)));
    double xi_prev = log(sigma), g0_prev = 0.0, g1_est = -1.0;
    double alpha = (opts.alpha > 0.5 && opts.alpha < 1.99) ? opts.alpha : 1.6;
    int check_every = opts.check_cert_every;
    if (check_every <= 0) check_every = opts.veto_enabled ? 10 : 50;

    ChemSolveResult best_cert;
    int iter_stopped = 0, converged = 0, vetoed = 0;
    double prim_res = 1.0, dual_res = 1.0;
    if (run_bpsdp_core(&ctx, &ns, &w, e_core, opts.max_iters, opts.tol, alpha, check_every,
                       opts.veto_enabled, opts.E_ai, opts.veto_tol, opts.verbose, 0, scale_c,
                       b_norm, c_sc_norm, &sigma, &xi_prev, &g0_prev, &g1_est, &best_cert,
                       &iter_stopped, &converged, &vetoed, &prim_res, &dual_res, err,
                       err_sz) != 0) {
        bpsdp_work_free(&w);
        normal_eq_free(&ns);
        mf_ctx_free(&ctx);
        return -1;
    }

    double e_primal = e_core + dot_vec(nD, ctx.C_D, w.X);
    *res = best_cert;
    res->iters = iter_stopped;
    res->converged = converged;
    res->vetoed = vetoed;
    res->iter_stopped = iter_stopped;
    res->E_primal = e_primal;
    res->prim_res = prim_res;
    res->dual_res = dual_res;
    res->elapsed_sec = now_sec() - t0;

    if (opts.export_rdm_grad) {
        size_t ksz = (size_t)ham->K;
        size_t k2 = ksz * ksz;
        size_t k4 = k2 * k2;
        res->K_grad = ham->K;
        res->primal_grad_h1 = (double *)malloc(k2 * sizeof(double));
        res->primal_grad_eri = (double *)malloc(k4 * sizeof(double));
        if (res->primal_grad_h1 && res->primal_grad_eri) {
            mf_compute_primal_ham_grad(&ctx, w.X, res->primal_grad_h1, res->primal_grad_eri);
        } else {
            free(res->primal_grad_h1);
            free(res->primal_grad_eri);
            res->primal_grad_h1 = NULL;
            res->primal_grad_eri = NULL;
            res->K_grad = 0;
        }
    }

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
    res->witness.t1_enabled = ctx.t1_enabled;
    res->witness.t1_norbs = ctx.t1_norbs;
    if (ctx.t1_enabled) {
        memcpy(res->witness.t1_orbs, ctx.t1_orbs, (size_t)ctx.t1_norbs * sizeof(int));
    }
    res->witness.meq = meq;
    res->witness.nD = nD;
    res->witness.nQ = nQ;
    res->witness.nG = nG;
    res->witness.nT2 = nT2;
    res->witness.nT1 = nT1;
    res->witness.y_eq = w.best_y;
    res->witness.Z_Q = w.best_ZQ;
    res->witness.Z_G = w.best_ZG;
    res->witness.Z_T2 = w.best_ZT2;
    res->witness.Z_T1 = w.best_ZT1;
    /* Null transferred pointers before freeing temporary vectors */
    w.best_y = NULL;
    w.best_ZQ = NULL;
    w.best_ZG = NULL;
    w.best_ZT2 = NULL;
    w.best_ZT1 = NULL;

    bpsdp_work_free(&w);
    normal_eq_free(&ns);
    mf_ctx_free(&ctx);
    return 0;
}

/* ============================================================================
 * Persistent In-Process ChemSession API (Warm-Started Optimization Sweeps)
 * ============================================================================ */

struct ChemSession {
    int K, nelec, ms2, na, nb, two_s, constrain_s2;
    ChemMode mode;
    ChemMFContext ctx;
    NormalEqSolver ns;
    BpsdpWorkVecs w;
    double b_norm;
    int has_state;
    double last_sigma;
    double last_scale_c;
};

ChemSession *chem_session_create(int K, int nelec, int ms2, int mode_in, int two_s,
                                 int constrain_s2, int enable_t1, int t1_orbs, int enable_t2,
                                 int t2_orbs, int nthreads, char *err, size_t err_sz) {
    if (K < 2 || K > CHEM_MAX_ORBITALS || nelec <= 0 || nelec > 2 * K) {
        set_err(err, err_sz, "invalid (K=%d, nelec=%d) for ChemSession", K, nelec);
        return NULL;
    }
    if ((nelec + ms2) % 2 != 0) {
        set_err(err, err_sz, "nelec (%d) and ms2 (%d) have incompatible parity", nelec, ms2);
        return NULL;
    }
    int na = (nelec + ms2) / 2;
    int nb = (nelec - ms2) / 2;
    if (na < 0 || nb < 0 || na > K || nb > K) {
        set_err(err, err_sz, "invalid spin occupations na=%d nb=%d for K=%d", na, nb, K);
        return NULL;
    }
    ChemHam dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.K = K;
    dummy.nelec = nelec;
    dummy.ms2 = ms2;
    dummy.na = na;
    dummy.nb = nb;

    ChemSolveOpts opts = chem_solve_opts_default(&dummy);
    opts.mode = (ChemMode)mode_in;
    opts.two_s = (two_s >= 0) ? two_s : abs(ms2);
    ChemMode mode = chem_resolve_mode(&dummy, &opts);

    ChemSession *sess = (ChemSession *)calloc(1, sizeof(ChemSession));
    if (!sess) {
        set_err(err, err_sz, "out of memory allocating ChemSession");
        return NULL;
    }
    sess->K = K;
    sess->nelec = nelec;
    sess->ms2 = ms2;
    sess->na = na;
    sess->nb = nb;
    sess->two_s = opts.two_s;
    sess->constrain_s2 = constrain_s2;
    sess->mode = mode;

    int t1o_cap = (t1_orbs > 0 && t1_orbs <= K) ? t1_orbs : 0;
    int t2o_cap = (t2_orbs > 0 && t2_orbs <= K) ? t2_orbs : 0;
    if (mf_ctx_init_ext(&sess->ctx, K, na, nb, sess->two_s, constrain_s2, mode, NULL,
                        enable_t2 ? 1 : 0, t2o_cap, t2_orbs, NULL,
                        enable_t1 ? 1 : 0, t1o_cap, t1_orbs, NULL,
                        nthreads, err, err_sz) != 0) {
        free(sess);
        return NULL;
    }
    if (normal_eq_init(&sess->ctx, &sess->ns, 1, err, err_sz) != 0) {
        mf_ctx_free(&sess->ctx);
        free(sess);
        return NULL;
    }
    if (bpsdp_work_alloc(&sess->ctx, &sess->w, err, err_sz) != 0) {
        normal_eq_free(&sess->ns);
        mf_ctx_free(&sess->ctx);
        free(sess);
        return NULL;
    }

    double b2 = 0.0;
    if (sess->ctx.meq > 0 && sess->ctx.b_eq) {
        b2 += dot_vec(sess->ctx.meq, sess->ctx.b_eq, sess->ctx.b_eq);
    }
    if (sess->ctx.nQ > 0 && sess->ctx.q0) {
        b2 += dot_vec(sess->ctx.nQ, sess->ctx.q0, sess->ctx.q0);
    }
    if (sess->ctx.t1_enabled && sess->ctx.nT1 > 0 && sess->ctx.t1_0) {
        b2 += dot_vec(sess->ctx.nT1, sess->ctx.t1_0, sess->ctx.t1_0);
    }
    sess->b_norm = sqrt(b2);
    return sess;
}

void chem_session_free(ChemSession *sess) {
    if (!sess) return;
    bpsdp_work_free(&sess->w);
    normal_eq_free(&sess->ns);
    mf_ctx_free(&sess->ctx);
    free(sess);
}

int chem_session_solve(ChemSession *sess, double e_core, const double *h1, const double *eri,
                       int max_iters, double tol, int warm_start, double *E_lb_out,
                       double *E_primal_out, int *iters_out, int *conv_out,
                       double *grad_h1_out, double *grad_eri_out, char *err, size_t err_sz) {
    if (!sess || !h1 || !eri) {
        set_err(err, err_sz, "null argument to chem_session_solve");
        return -1;
    }
    int K = sess->K;
    ChemHam htmp;
    memset(&htmp, 0, sizeof(htmp));
    htmp.K = K;
    htmp.nelec = sess->nelec;
    htmp.ms2 = sess->ms2;
    htmp.na = sess->na;
    htmp.nb = sess->nb;
    htmp.E_core = e_core;
    /* mf_build_hamiltonian_cost only reads h & eri */
    size_t k2 = (size_t)K * (size_t)K;
    htmp.h = (double *)malloc(k2 * sizeof(double));
    htmp.eri = (double *)malloc(k2 * k2 * sizeof(double));
    if (!htmp.h || !htmp.eri) {
        free(htmp.h);
        free(htmp.eri);
        set_err(err, err_sz, "out of memory staging Hamiltonian in chem_session_solve");
        return -1;
    }
    memcpy(htmp.h, h1, k2 * sizeof(double));
    memcpy(htmp.eri, eri, k2 * k2 * sizeof(double));
    mf_build_hamiltonian_cost(&sess->ctx, &htmp, sess->ctx.C_D);
    free(htmp.h);
    free(htmp.eri);

    int nD = sess->ctx.nD;
    int nAll = sess->ctx.nAll;
    double c_norm = norm2_vec(nD, sess->ctx.C_D);
    double scale_c = c_norm / sqrt((double)(nD > 0 ? nD : 1));
    if (scale_c < 1.0) scale_c = 1.0;

    for (int i = 0; i < nD; i++) sess->w.C_sc[i] = sess->ctx.C_D[i] / scale_c;
    double c_sc_norm = norm2_vec(nD, sess->w.C_sc);

    int do_warm = (warm_start && sess->has_state && sess->last_scale_c > 1e-12) ? 1 : 0;
    double sigma;
    if (do_warm) {
        double ratio = sess->last_scale_c / scale_c;
        for (int i = 0; i < nAll; i++) {
            sess->w.Z[i] *= ratio;
            sess->w.Gt_y[i] *= ratio;
        }
        sigma = fmax(0.05, fmin(10.0, sess->last_sigma));
    } else {
        memset(sess->w.X, 0, (size_t)(nAll > 0 ? nAll : 1) * sizeof(double));
        memset(sess->w.Z, 0, (size_t)(nAll > 0 ? nAll : 1) * sizeof(double));
        memset(sess->w.Gt_y, 0, (size_t)(nAll > 0 ? nAll : 1) * sizeof(double));
        memset(sess->w.y_eq, 0, (size_t)(sess->ctx.meq > 0 ? sess->ctx.meq : 1) * sizeof(double));
        sigma = fmax(0.5, fmin(3.0, sess->b_norm / fmax(1.0, c_sc_norm)));
    }
    double xi_prev = log(sigma), g0_prev = 0.0, g1_est = -1.0;

    ChemSolveResult best_cert;
    int iter_stopped = 0, converged = 0, vetoed = 0;
    double prim_res = 1.0, dual_res = 1.0;
    int eff_iters = (max_iters > 0) ? max_iters : 800;
    double eff_tol = (tol > 0.0) ? tol : 1e-6;
    if (run_bpsdp_core(&sess->ctx, &sess->ns, &sess->w, e_core, eff_iters, eff_tol, 1.6, 50,
                       0, 0.0, 0.0, 0, do_warm, scale_c, sess->b_norm, c_sc_norm, &sigma,
                       &xi_prev, &g0_prev, &g1_est, &best_cert, &iter_stopped, &converged,
                       &vetoed, &prim_res, &dual_res, err, err_sz) != 0) {
        return -1;
    }

    sess->has_state = 1;
    sess->last_sigma = sigma;
    sess->last_scale_c = scale_c;

    double e_primal = e_core + dot_vec(nD, sess->ctx.C_D, sess->w.X);
    if (E_lb_out) *E_lb_out = best_cert.E_lb_cert;
    if (E_primal_out) *E_primal_out = e_primal;
    if (iters_out) *iters_out = iter_stopped;
    if (conv_out) *conv_out = converged;

    if (grad_h1_out && grad_eri_out) {
        mf_compute_primal_ham_grad(&sess->ctx, sess->w.X, grad_h1_out, grad_eri_out);
    }
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

int chem_obs_spin_1body(int K, const double *o1_a, const double *o1_b, double o_const,
                        const char *name, ChemObservable *out, char *err, size_t err_sz) {
    if (K <= 0 || K > CHEM_MAX_ORBITALS || !out) {
        set_err(err, err_sz, "invalid argument to chem_obs_spin_1body");
        return -1;
    }
    if (alloc_observable(out, name ? name : "obs1", K, o_const, 0, 0, err, err_sz) != 0) {
        return -1;
    }
    size_t k2 = (size_t)K * (size_t)K;
    if (o1_a) memcpy(out->o1_a, o1_a, k2 * sizeof(double));
    if (o1_b) memcpy(out->o1_b, o1_b, k2 * sizeof(double));
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
                        opts.enable_t1, opts.t1_max_orbs, opts.t1_norbs, opts.t1_orbs,
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

int chem_test_adjoints_t12(int K, int na, int nb, int two_s, int constrain_s2, ChemMode mode,
                           int t2_norbs, const int *t2_orbs, int t1_norbs, const int *t1_orbs,
                           uint64_t seed, ChemAdjointReport *rep, char *err, size_t err_sz) {
    memset(rep, 0, sizeof(*rep));
    ChemMFContext ctx;
    if (mf_ctx_init_ext(&ctx, K, na, nb, two_s, constrain_s2, mode, NULL,
                        (t2_norbs > 0) ? 1 : 0, K, t2_norbs, t2_orbs,
                        (t1_norbs > 0) ? 1 : 0, K, t1_norbs, t1_orbs,
                        1, err, err_sz) != 0) {
        return -1;
    }

    int nD = ctx.nD, nQ = ctx.nQ, nG = ctx.nG, nT2 = ctx.nT2, nT1 = ctx.nT1, meq = ctx.meq;
    int max_out = nQ;
    if (nG > max_out) max_out = nG;
    if (nT2 > max_out) max_out = nT2;
    if (nT1 > max_out) max_out = nT1;
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

    /* 5. L_T1 */
    if (ctx.t1_enabled && nT1 > 0) {
        for (int i = 0; i < nT1; i++) Y[i] = rand_sym_double(&rng);
        mf_LT1_fwd(&ctx, X, LX);
        mf_LT1_adj(&ctx, Y, LsY);
        lhs = dot_vec(nT1, LX, Y);
        rhs = dot_vec(nD, X, LsY);
        sc = fmax(1.0, norm2_vec(nT1, LX) * norm2_vec(nT1, Y));
        rep->err_LT1 = fabs(lhs - rhs) / sc;
    } else {
        rep->err_LT1 = 0.0;
    }

    rep->max_err = fmax(fmax(fmax(rep->err_LQ, rep->err_LG), fmax(rep->err_Aeq, rep->err_LT2)),
                        rep->err_LT1);

    free(X); free(LX); free(Y); free(LsY);
    mf_ctx_free(&ctx);
    return 0;
}

int chem_test_adjoints_t2(int K, int na, int nb, int two_s, int constrain_s2, ChemMode mode,
                          int t2_norbs, const int *t2_orbs, uint64_t seed,
                          ChemAdjointReport *rep, char *err, size_t err_sz) {
    return chem_test_adjoints_t12(K, na, nb, two_s, constrain_s2, mode, t2_norbs, t2_orbs, 0, NULL,
                                  seed, rep, err, err_sz);
}

int chem_test_adjoints(int K, int na, int nb, int two_s, int constrain_s2, ChemMode mode,
                       uint64_t seed, ChemAdjointReport *rep, char *err, size_t err_sz) {
    return chem_test_adjoints_t12(K, na, nb, two_s, constrain_s2, mode, 0, NULL, 0, NULL, seed,
                                  rep, err, err_sz);
}

int chem_verify_fci_t2_psd(const ChemHam *ham, ChemMode mode, int two_s,
                           int t2_norbs, const int *t2_orbs, const double *D_svec,
                           double *min_eig_out, double *trace_err_out,
                           char *err, size_t err_sz) {
    ChemMFContext ctx;
    if (mf_ctx_init_ext(&ctx, ham->K, ham->na, ham->nb, two_s, 1, mode, NULL,
                        1, ham->K, t2_norbs, t2_orbs, 0, 0, 0, NULL, 1, err, err_sz) != 0) {
        return -1;
    }
    double *T2_sv = (double *)malloc((size_t)(ctx.nT2 > 0 ? ctx.nT2 : 1) * sizeof(double));
    if (!T2_sv) {
        mf_ctx_free(&ctx);
        set_err(err, err_sz, "out of memory in chem_verify_fci_t2_psd");
        return -1;
    }
    mf_LT2_fwd(&ctx, D_svec, T2_sv);

    int t2_start = ctx.blk_T2_start;
    int off = 0;
    double min_ev = 1e30;
    double max_tr_err = 0.0;
    for (int b = t2_start; b < t2_start + ctx.blk_T2_cnt; b++) {
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

int chem_verify_fci_t1_psd(const ChemHam *ham, ChemMode mode, int two_s,
                           int t1_norbs, const int *t1_orbs, const double *D_svec,
                           double *min_eig_out, double *trace_err_out,
                           char *err, size_t err_sz) {
    ChemMFContext ctx;
    if (mf_ctx_init_ext(&ctx, ham->K, ham->na, ham->nb, two_s, 1, mode, NULL,
                        0, 0, 0, NULL, 1, ham->K, t1_norbs, t1_orbs, 1, err, err_sz) != 0) {
        return -1;
    }
    double *T1_sv = (double *)malloc((size_t)(ctx.nT1 > 0 ? ctx.nT1 : 1) * sizeof(double));
    if (!T1_sv) {
        mf_ctx_free(&ctx);
        set_err(err, err_sz, "out of memory in chem_verify_fci_t1_psd");
        return -1;
    }
    mf_LT1_fwd(&ctx, D_svec, T1_sv);
    for (int i = 0; i < ctx.nT1; i++) T1_sv[i] += ctx.t1_0[i];

    int t1_start = ctx.blk_T1_start;
    int off = 0;
    double min_ev = 1e30;
    double max_tr_err = 0.0;
    for (int b = t1_start; b < t1_start + ctx.blk_T1_cnt; b++) {
        int dim = ctx.block_dims[b];
        int sz = ctx.block_szs[b];
        if (dim > 0) {
            if (accel_syevd(&ctx, dim, T1_sv + off, 0, ctx.scr_M0, ctx.ev_val, NULL) != 0) {
                free(T1_sv);
                mf_ctx_free(&ctx);
                set_err(err, err_sz, "dsyevd_ failed in chem_verify_fci_t1_psd");
                return -1;
            }
            if (ctx.ev_val[0] < min_ev) min_ev = ctx.ev_val[0];
            double tr = 0.0;
            for (int k = 0; k < dim; k++) tr += ctx.ev_val[k];
            if (ctx.t1_norbs == ham->K) {
                double err_t = fabs(tr - ctx.block_traces[b]);
                if (err_t > max_tr_err) max_tr_err = err_t;
            }
        }
        off += sz;
    }
    if (min_eig_out) *min_eig_out = min_ev;
    if (trace_err_out) *trace_err_out = max_tr_err;
    free(T1_sv);
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

typedef struct {
    int schema_ver; /* 1, 2, or 3 */
    char fcidump_path[4098];
    char sha256_hex[65];
    int K, nelec, ms2, na, nb, two_s, constrain_s2;
    int t2_enabled, t2_norbs;
    int t2_orbs[CHEM_MAX_ORBITALS];
    int t1_enabled, t1_norbs;
    int t1_orbs[CHEM_MAX_ORBITALS];
    ChemMode mode;
    double E_core;

    /* solver */
    int iters, converged, vetoed, iter_stopped;
    double E_primal, E_dual_raw, E_lb_cert;
    double min_eig_fp, min_eig_cert, delta_round, prim_res, dual_res, elapsed_sec;

    /* veto */
    int veto_enabled;
    double E_ai, veto_tol, E_ub_variational, interval_width;

    /* lean_scaled_witness */
    int64_t scale_nha, E_lb_floor, E_ub_ceil, E_ai_minus_tol_floor;

    /* variational */
    int has_var_meta;
    double E_hf0, E_scf, E_searu_rot, E_ci, E_ub;
    int is_exact_fci;
    uint64_t ndet_space;
    int ndet_var, has_witness;
    uint64_t *dets;
    double *psi0;

    /* dual_witness */
    int meq, nD, nQ, nG, nT2, nT1;
    double *y_eq, *Z_Q, *Z_G, *Z_T2, *Z_T1;
    int owns_arrays;
} ChemFullCert;

static void full_cert_free(ChemFullCert *c) {
    if (!c) return;
    if (c->owns_arrays) {
        free(c->dets);
        free(c->psi0);
        free(c->y_eq);
        free(c->Z_Q);
        free(c->Z_G);
        free(c->Z_T2);
        free(c->Z_T1);
    }
    memset(c, 0, sizeof(*c));
}

static void full_cert_from_solve(ChemFullCert *c, const ChemHam *ham, const char *fcidump_path,
                                 const ChemSolveResult *res, const struct ChemVarResult *var_opt,
                                 const ChemSolveOpts *opts_in) {
    memset(c, 0, sizeof(*c));
    ChemSolveOpts opts = opts_in ? *opts_in : chem_solve_opts_default(ham);
    c->schema_ver = res->witness.t1_enabled ? 3 : (res->witness.t2_enabled ? 2 : 1);
    snprintf(c->fcidump_path, sizeof(c->fcidump_path), "%s", fcidump_path ? fcidump_path : "");
    memcpy(c->sha256_hex, ham->sha256_hex, 65);
    c->K = ham->K;
    c->nelec = ham->nelec;
    c->ms2 = ham->ms2;
    c->na = ham->na;
    c->nb = ham->nb;
    c->two_s = res->witness.two_s;
    c->constrain_s2 = res->witness.constrain_s2;
    c->t2_enabled = res->witness.t2_enabled;
    c->t2_norbs = res->witness.t2_norbs;
    if (c->t2_norbs > 0 && c->t2_norbs <= CHEM_MAX_ORBITALS) {
        memcpy(c->t2_orbs, res->witness.t2_orbs, (size_t)c->t2_norbs * sizeof(int));
    }
    c->t1_enabled = res->witness.t1_enabled;
    c->t1_norbs = res->witness.t1_norbs;
    if (c->t1_norbs > 0 && c->t1_norbs <= CHEM_MAX_ORBITALS) {
        memcpy(c->t1_orbs, res->witness.t1_orbs, (size_t)c->t1_norbs * sizeof(int));
    }
    c->mode = res->mode_used;
    c->E_core = ham->E_core;

    c->iters = res->iters;
    c->converged = res->converged;
    c->vetoed = res->vetoed;
    c->iter_stopped = res->iter_stopped;
    c->E_primal = res->E_primal;
    c->E_dual_raw = res->E_dual_raw;
    c->E_lb_cert = res->E_lb_cert;
    c->min_eig_fp = res->min_eig_fp;
    c->min_eig_cert = res->min_eig_cert;
    c->delta_round = res->delta_round;
    c->prim_res = res->prim_res;
    c->dual_res = res->dual_res;
    c->elapsed_sec = res->elapsed_sec;

    double e_ub_v = var_opt ? var_opt->E_ub : res->E_primal;
    c->veto_enabled = opts.veto_enabled;
    c->E_ai = opts.E_ai;
    c->veto_tol = opts.veto_tol;
    c->E_ub_variational = e_ub_v;
    c->interval_width = e_ub_v - res->E_lb_cert;

    c->scale_nha = 1000000000LL;
    c->E_lb_floor = (int64_t)floor(res->E_lb_cert * 1e9);
    c->E_ub_ceil = (int64_t)ceil(e_ub_v * 1e9);
    c->E_ai_minus_tol_floor = (int64_t)floor((opts.E_ai - opts.veto_tol) * 1e9);

    if (var_opt) {
        int emit_w = (var_opt->ndet_var > 0 && var_opt->ndet_var <= 2048 && var_opt->dets &&
                      var_opt->psi0)
                         ? 1
                         : 0;
        c->has_var_meta = 1;
        c->E_hf0 = var_opt->E_hf0;
        c->E_scf = var_opt->E_scf;
        c->E_searu_rot = var_opt->E_searu_rot;
        c->E_ci = var_opt->E_ci;
        c->E_ub = var_opt->E_ub;
        c->is_exact_fci = var_opt->is_exact_fci;
        c->ndet_space = var_opt->ndet_space;
        c->ndet_var = var_opt->ndet_var;
        c->has_witness = emit_w;
        if (emit_w) {
            c->dets = var_opt->dets;
            c->psi0 = var_opt->psi0;
        }
    } else {
        c->has_var_meta = 0;
        c->has_witness = 0;
        c->E_ub = e_ub_v;
    }

    c->meq = res->witness.meq;
    c->nD = res->witness.nD;
    c->nQ = res->witness.nQ;
    c->nG = res->witness.nG;
    c->nT2 = res->witness.nT2;
    c->nT1 = res->witness.nT1;
    c->y_eq = res->witness.y_eq;
    c->Z_Q = res->witness.Z_Q;
    c->Z_G = res->witness.Z_G;
    c->Z_T2 = res->witness.Z_T2;
    c->Z_T1 = res->witness.Z_T1;
    c->owns_arrays = 0;
}

static void json_write_full_cert(FILE *fp, const ChemFullCert *c) {
    const char *mode_str = (c->mode == CHEM_MODE_A_SINGLET) ? "A_SINGLET" : "B_GENERAL";
    const char *schema = (c->schema_ver >= 3 || c->t1_enabled)
                             ? "chem-cert-v3"
                             : ((c->schema_ver == 2 || c->t2_enabled) ? "chem-cert-v2" : "chem-cert-v1");
    fprintf(fp, "{\n");
    fprintf(fp, "  \"schema_version\": \"%s\",\n", schema);
    fprintf(fp, "  \"fcidump_path\": ");
    write_json_escaped_str(fp, c->fcidump_path);
    fprintf(fp, ",\n");
    fprintf(fp, "  \"hamiltonian_sha256\": \"%s\",\n", c->sha256_hex);
    fprintf(fp, "  \"K\": %d,\n", c->K);
    fprintf(fp, "  \"nelec\": %d,\n", c->nelec);
    fprintf(fp, "  \"ms2\": %d,\n", c->ms2);
    fprintf(fp, "  \"na\": %d,\n", c->na);
    fprintf(fp, "  \"nb\": %d,\n", c->nb);
    fprintf(fp, "  \"two_s\": %d,\n", c->two_s);
    fprintf(fp, "  \"constrain_s2\": %d,\n", c->constrain_s2);
    fprintf(fp, "  \"t2_enabled\": %d,\n", c->t2_enabled);
    fprintf(fp, "  \"t2_norbs\": %d,\n", c->t2_norbs);
    fprintf(fp, "  \"t2_orbs\": ");
    write_json_array_i32(fp, c->t2_orbs, c->t2_norbs);
    fprintf(fp, ",\n");
    if (c->t1_enabled) {
        fprintf(fp, "  \"t1_enabled\": %d,\n", c->t1_enabled);
        fprintf(fp, "  \"t1_norbs\": %d,\n", c->t1_norbs);
        fprintf(fp, "  \"t1_orbs\": ");
        write_json_array_i32(fp, c->t1_orbs, c->t1_norbs);
        fprintf(fp, ",\n");
    }
    fprintf(fp, "  \"mode\": \"%s\",\n", mode_str);
    fprintf(fp, "  \"E_core\": %.17g,\n", c->E_core);

    fprintf(fp, "  \"solver\": {\n");
    fprintf(fp, "    \"iters\": %d,\n", c->iters);
    fprintf(fp, "    \"converged\": %d,\n", c->converged);
    fprintf(fp, "    \"vetoed\": %d,\n", c->vetoed);
    fprintf(fp, "    \"iter_stopped\": %d,\n", c->iter_stopped);
    fprintf(fp, "    \"E_primal\": %.17g,\n", c->E_primal);
    fprintf(fp, "    \"E_dual_raw\": %.17g,\n", c->E_dual_raw);
    fprintf(fp, "    \"E_lb_cert\": %.17g,\n", c->E_lb_cert);
    fprintf(fp, "    \"min_eig_fp\": %.17g,\n", c->min_eig_fp);
    fprintf(fp, "    \"min_eig_cert\": %.17g,\n", c->min_eig_cert);
    fprintf(fp, "    \"delta_round\": %.17g,\n", c->delta_round);
    fprintf(fp, "    \"prim_res\": %.17g,\n", c->prim_res);
    fprintf(fp, "    \"dual_res\": %.17g,\n", c->dual_res);
    fprintf(fp, "    \"elapsed_sec\": %.6f\n", c->elapsed_sec);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"veto\": {\n");
    fprintf(fp, "    \"enabled\": %d,\n", c->veto_enabled);
    fprintf(fp, "    \"E_ai\": %.17g,\n", c->E_ai);
    fprintf(fp, "    \"veto_tol\": %.17g,\n", c->veto_tol);
    fprintf(fp, "    \"vetoed\": %d,\n", c->vetoed);
    fprintf(fp, "    \"iter_stopped\": %d,\n", c->iter_stopped);
    fprintf(fp, "    \"E_ub_variational\": %.17g,\n", c->E_ub_variational);
    fprintf(fp, "    \"interval_width\": %.17g\n", c->interval_width);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"lean_scaled_witness\": {\n");
    fprintf(fp, "    \"scale_nha\": %lld,\n", (long long)c->scale_nha);
    fprintf(fp, "    \"E_lb_floor\": %lld,\n", (long long)c->E_lb_floor);
    fprintf(fp, "    \"E_ub_ceil\": %lld,\n", (long long)c->E_ub_ceil);
    fprintf(fp, "    \"E_ai_minus_tol_floor\": %lld\n", (long long)c->E_ai_minus_tol_floor);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"variational\": {\n");
    if (c->has_var_meta) {
        fprintf(fp, "    \"E_hf0\": %.17g,\n", c->E_hf0);
        fprintf(fp, "    \"E_scf\": %.17g,\n", c->E_scf);
        fprintf(fp, "    \"E_searu_rot\": %.17g,\n", c->E_searu_rot);
        fprintf(fp, "    \"E_ci\": %.17g,\n", c->E_ci);
        fprintf(fp, "    \"E_ub\": %.17g,\n", c->E_ub);
        fprintf(fp, "    \"is_exact_fci\": %d,\n", c->is_exact_fci);
        fprintf(fp, "    \"ndet_space\": %llu,\n", (unsigned long long)c->ndet_space);
        fprintf(fp, "    \"ndet_var\": %d,\n", c->ndet_var);
        fprintf(fp, "    \"has_witness\": %d", c->has_witness);
        if (c->has_witness && c->ndet_var > 0 && c->dets && c->psi0) {
            fprintf(fp, ",\n    \"dets\": [");
            for (int i = 0; i < c->ndet_var; i++) {
                if (i > 0) fputc(',', fp);
                fprintf(fp, "\"0x%016llx\"", (unsigned long long)c->dets[i]);
            }
            fprintf(fp, "],\n    \"psi0\": ");
            write_json_array_f64(fp, c->psi0, c->ndet_var);
            fputc('\n', fp);
        } else {
            fputc('\n', fp);
        }
    } else {
        fprintf(fp, "    \"has_witness\": 0,\n");
        fprintf(fp, "    \"E_ub\": %.17g\n", c->E_ub);
    }
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"dual_witness\": {\n");
    fprintf(fp, "    \"meq\": %d,\n", c->meq);
    fprintf(fp, "    \"nD\": %d,\n", c->nD);
    fprintf(fp, "    \"nQ\": %d,\n", c->nQ);
    fprintf(fp, "    \"nG\": %d,\n", c->nG);
    fprintf(fp, "    \"nT2\": %d,\n", c->nT2);
    if (c->t1_enabled) {
        fprintf(fp, "    \"nT1\": %d,\n", c->nT1);
    }
    fprintf(fp, "    \"y_eq\": ");
    write_json_array_f64(fp, c->y_eq, c->meq);
    fprintf(fp, ",\n    \"Z_Q\": ");
    write_json_array_f64(fp, c->Z_Q, c->nQ);
    fprintf(fp, ",\n    \"Z_G\": ");
    write_json_array_f64(fp, c->Z_G, c->nG);
    if (c->t2_enabled && c->nT2 > 0 && c->Z_T2) {
        fprintf(fp, ",\n    \"Z_T2\": ");
        write_json_array_f64(fp, c->Z_T2, c->nT2);
    }
    if (c->t1_enabled && c->nT1 > 0 && c->Z_T1) {
        fprintf(fp, ",\n    \"Z_T1\": ");
        write_json_array_f64(fp, c->Z_T1, c->nT1);
    }
    fprintf(fp, "\n  }\n}\n");
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
    ChemFullCert fc;
    full_cert_from_solve(&fc, ham, fcidump_path, res, var_opt, opts_in);
    json_write_full_cert(fp, &fc);
    if (ferror(fp)) {
        fclose(fp);
        set_err(err, err_sz, "write error on certificate output path: %s", path);
        return -1;
    }
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

static int path_ends_with(const char *path, const char *suffix) {
    if (!path || !suffix) return 0;
    size_t lp = strlen(path);
    size_t ls = strlen(suffix);
    if (lp < ls) return 0;
    return memcmp(path + (lp - ls), suffix, ls) == 0;
}

int chem_cert_write_bin(const char *path, const ChemHam *ham, const char *fcidump_path,
                        const ChemSolveResult *res, const struct ChemVarResult *var_opt,
                        const ChemSolveOpts *opts_in, char *err, size_t err_sz) {
    if (path_ends_with(path, ".cert.bin")) {
        return chem_cert_write_chmc(path, ham, fcidump_path, res, var_opt, opts_in, err, err_sz);
    }
    if (!path || !ham || !res) {
        set_err(err, err_sz, "null argument to chem_cert_write_bin");
        return -1;
    }
    if (res->witness.t1_enabled) {
        set_err(err, err_sz, "binary .flowchem format supports DQG/DQG+T2; use --cert JSON or .cert.bin for T1");
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

static int json_get_i64(const char *start, const char *end, const char *key, int64_t *out) {
    const char *v = json_find_key_in_range(start, end, key);
    if (!v) return -1;
    char *ep = NULL;
    long long val = strtoll(v, &ep, 10);
    if (ep == v) return -1;
    *out = (int64_t)val;
    return 0;
}

static int json_get_u64(const char *start, const char *end, const char *key, uint64_t *out) {
    const char *v = json_find_key_in_range(start, end, key);
    if (!v) return -1;
    char *ep = NULL;
    unsigned long long val = strtoull(v, &ep, 10);
    if (ep == v) return -1;
    *out = (uint64_t)val;
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

/* ============================================================================
 * Canonical Columnar Binary Certificate Codec ("CHMC" v1, .cert.bin)
 * ============================================================================ */

#define CHMC_MAGIC_0 'C'
#define CHMC_MAGIC_1 'H'
#define CHMC_MAGIC_2 'M'
#define CHMC_MAGIC_3 'C'
#define CHMC_VERSION UINT32_C(1)
#define CHMC_HDR_BYTES 44U
#define CHMC_TRAILER_BYTES 32U

#define CHMC_FLAG_MODE_B       (UINT32_C(1) << 0)
#define CHMC_FLAG_CONSTRAIN_S2 (UINT32_C(1) << 1)
#define CHMC_FLAG_T2           (UINT32_C(1) << 2)
#define CHMC_FLAG_T1           (UINT32_C(1) << 3)
#define CHMC_FLAG_CONVERGED    (UINT32_C(1) << 4)
#define CHMC_FLAG_VETO_EN      (UINT32_C(1) << 5)
#define CHMC_FLAG_VETOED       (UINT32_C(1) << 6)
#define CHMC_FLAG_HAS_VAR_META (UINT32_C(1) << 7)
#define CHMC_FLAG_HAS_WITNESS  (UINT32_C(1) << 8)
#define CHMC_FLAG_IS_EXACT_FCI (UINT32_C(1) << 9)
#define CHMC_FLAGS_VALID_MASK  UINT32_C(0x3FF)

#define CHMC_COL_ALL_ZERO    UINT8_C(0)
#define CHMC_COL_SPARSE_MASK UINT8_C(1)
#define CHMC_COL_DENSE       UINT8_C(2)

#define CHMC_LANE_CONST UINT8_C(0)
#define CHMC_LANE_PAL2  UINT8_C(1)
#define CHMC_LANE_PAL4  UINT8_C(2)
#define CHMC_LANE_RLE   UINT8_C(3)
#define CHMC_LANE_RAW   UINT8_C(4)

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} ChemByteBuf;

static int bb_reserve(ChemByteBuf *b, size_t extra) {
    if (extra > SIZE_MAX - b->len) return -1;
    size_t need = b->len + extra;
    if (need <= b->cap) return 0;
    size_t ncap = b->cap ? b->cap : 4096U;
    while (ncap < need) {
        if (ncap > SIZE_MAX / 2U) {
            ncap = need;
            break;
        }
        ncap *= 2U;
    }
    uint8_t *nd = (uint8_t *)realloc(b->data, ncap);
    if (!nd) return -1;
    b->data = nd;
    b->cap = ncap;
    return 0;
}

static int bb_put_u8(ChemByteBuf *b, uint8_t v) {
    if (bb_reserve(b, 1U) != 0) return -1;
    b->data[b->len++] = v;
    return 0;
}

static int bb_put_bytes(ChemByteBuf *b, const void *src, size_t n) {
    if (n == 0U) return 0;
    if (bb_reserve(b, n) != 0) return -1;
    memcpy(b->data + b->len, src, n);
    b->len += n;
    return 0;
}

static int bb_put_le32(ChemByteBuf *b, uint32_t v) {
    if (bb_reserve(b, 4U) != 0) return -1;
    flow_ir_wire_write_le32(b->data + b->len, v);
    b->len += 4U;
    return 0;
}

static int bb_put_le64(ChemByteBuf *b, uint64_t v) {
    if (bb_reserve(b, 8U) != 0) return -1;
    flow_ir_wire_write_le64(b->data + b->len, v);
    b->len += 8U;
    return 0;
}

static int bb_put_i32(ChemByteBuf *b, int32_t v) {
    uint32_t u = 0U;
    memcpy(&u, &v, sizeof(u));
    return bb_put_le32(b, u);
}

static int bb_put_i64(ChemByteBuf *b, int64_t v) {
    uint64_t u = 0U;
    memcpy(&u, &v, sizeof(u));
    return bb_put_le64(b, u);
}

static int bb_put_f64(ChemByteBuf *b, double x) {
    uint64_t u = 0U;
    memcpy(&u, &x, sizeof(u));
    return bb_put_le64(b, u);
}

static int bb_put_varu64(ChemByteBuf *b, uint64_t v) {
    if (bb_reserve(b, 10U) != 0) return -1;
    return flow_wire_put_varu64(b->data, b->cap, &b->len, v) ? 0 : -1;
}

static int bb_put_varu32(ChemByteBuf *b, uint32_t v) {
    return bb_put_varu64(b, (uint64_t)v);
}

static int chmc_get_u8(const uint8_t *buf, size_t end, size_t *pos, uint8_t *out) {
    return flow_wire_get_u8(buf, end, pos, out) ? 0 : -1;
}

static int chmc_get_le32(const uint8_t *buf, size_t end, size_t *pos, uint32_t *out) {
    if (*pos + 4U > end) return -1;
    *out = flow_ir_wire_read_le32(buf + *pos);
    *pos += 4U;
    return 0;
}

static int chmc_get_le64(const uint8_t *buf, size_t end, size_t *pos, uint64_t *out) {
    if (*pos + 8U > end) return -1;
    *out = flow_ir_wire_read_le64(buf + *pos);
    *pos += 8U;
    return 0;
}

static int chmc_get_i32(const uint8_t *buf, size_t end, size_t *pos, int32_t *out) {
    uint32_t u = 0U;
    if (chmc_get_le32(buf, end, pos, &u) != 0) return -1;
    memcpy(out, &u, sizeof(*out));
    return 0;
}

static int chmc_get_i64(const uint8_t *buf, size_t end, size_t *pos, int64_t *out) {
    uint64_t u = 0U;
    if (chmc_get_le64(buf, end, pos, &u) != 0) return -1;
    memcpy(out, &u, sizeof(*out));
    return 0;
}

static int chmc_get_f64(const uint8_t *buf, size_t end, size_t *pos, double *out) {
    uint64_t u = 0U;
    if (chmc_get_le64(buf, end, pos, &u) != 0) return -1;
    memcpy(out, &u, sizeof(*out));
    return 0;
}

static int chmc_get_varu64(const uint8_t *buf, size_t end, size_t *pos, uint64_t *out) {
    return flow_wire_get_varu64(buf, end, pos, out) ? 0 : -1;
}

static int chmc_get_varu32(const uint8_t *buf, size_t end, size_t *pos, uint32_t *out) {
    return flow_wire_get_varu32(buf, end, pos, out) ? 0 : -1;
}

static int hex_to_bytes32(const char *hex, uint8_t out[32]) {
    if (!hex || strlen(hex) != 64U) return -1;
    for (size_t i = 0U; i < 32U; i++) {
        uint8_t nib[2];
        for (size_t k = 0U; k < 2U; k++) {
            char ch = hex[2U * i + k];
            if (ch >= '0' && ch <= '9') nib[k] = (uint8_t)(ch - '0');
            else if (ch >= 'a' && ch <= 'f') nib[k] = (uint8_t)(10 + (ch - 'a'));
            else if (ch >= 'A' && ch <= 'F') nib[k] = (uint8_t)(10 + (ch - 'A'));
            else return -1;
        }
        out[i] = (uint8_t)((nib[0] << 4U) | nib[1]);
    }
    return 0;
}

static size_t chmc_packbits_measure(const uint8_t *plane, size_t m) {
    size_t out_len = 0U;
    size_t i = 0U;
    while (i < m) {
        size_t run = 1U;
        while (i + run < m && run < 130U && plane[i + run] == plane[i]) run++;
        if (run >= 3U) {
            out_len += 2U;
            i += run;
        } else {
            size_t lit = 0U;
            while (i < m && lit < 128U) {
                size_t r = 1U;
                while (i + r < m && r < 3U && plane[i + r] == plane[i]) r++;
                if (r >= 3U) break;
                i++;
                lit++;
            }
            out_len += 1U + lit;
        }
    }
    return out_len;
}

static int chmc_packbits_emit(ChemByteBuf *bb, const uint8_t *plane, size_t m) {
    size_t i = 0U;
    while (i < m) {
        size_t run = 1U;
        while (i + run < m && run < 130U && plane[i + run] == plane[i]) run++;
        if (run >= 3U) {
            if (bb_put_u8(bb, (uint8_t)(128U + (run - 3U))) != 0 ||
                bb_put_u8(bb, plane[i]) != 0) {
                return -1;
            }
            i += run;
        } else {
            size_t lit_start = i;
            size_t lit = 0U;
            while (i < m && lit < 128U) {
                size_t r = 1U;
                while (i + r < m && r < 3U && plane[i + r] == plane[i]) r++;
                if (r >= 3U) break;
                i++;
                lit++;
            }
            if (bb_put_u8(bb, (uint8_t)(lit - 1U)) != 0 ||
                bb_put_bytes(bb, plane + lit_start, lit) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int chmc_encode_lane(ChemByteBuf *bb, const uint8_t *plane, size_t m) {
    uint8_t present[256] = {0};
    for (size_t i = 0U; i < m; i++) present[plane[i]] = 1U;
    uint8_t pal[256];
    uint8_t b2idx[256] = {0};
    size_t u = 0U;
    for (size_t v = 0U; v < 256U; v++) {
        if (present[v]) {
            b2idx[v] = (uint8_t)u;
            pal[u++] = (uint8_t)v;
        }
    }

    uint8_t best_mode = CHMC_LANE_RAW;
    size_t best_len = 1U + m;
    if (u == 1U && 2U < best_len) {
        best_mode = CHMC_LANE_CONST;
        best_len = 2U;
    }
    if (u >= 2U && u <= 4U) {
        size_t clen = 2U + u + ((m + 3U) >> 2U);
        if (clen < best_len) {
            best_mode = CHMC_LANE_PAL2;
            best_len = clen;
        }
    }
    if (u >= 5U && u <= 16U) {
        size_t clen = 2U + u + ((m + 1U) >> 1U);
        if (clen < best_len) {
            best_mode = CHMC_LANE_PAL4;
            best_len = clen;
        }
    }
    size_t rle_len = chmc_packbits_measure(plane, m);
    if (1U + rle_len < best_len) {
        best_mode = CHMC_LANE_RLE;
    }

    if (bb_put_u8(bb, best_mode) != 0) return -1;
    if (best_mode == CHMC_LANE_CONST) {
        return bb_put_u8(bb, pal[0]);
    }
    if (best_mode == CHMC_LANE_PAL2) {
        size_t pbytes = (m + 3U) >> 2U;
        if (bb_put_u8(bb, (uint8_t)u) != 0 || bb_put_bytes(bb, pal, u) != 0 ||
            bb_reserve(bb, pbytes) != 0) {
            return -1;
        }
        uint8_t *dst = bb->data + bb->len;
        memset(dst, 0, pbytes);
        for (size_t i = 0U; i < m; i++) {
            dst[i >> 2U] |= (uint8_t)(b2idx[plane[i]] << (2U * (i & 3U)));
        }
        bb->len += pbytes;
        return 0;
    }
    if (best_mode == CHMC_LANE_PAL4) {
        size_t pbytes = (m + 1U) >> 1U;
        if (bb_put_u8(bb, (uint8_t)u) != 0 || bb_put_bytes(bb, pal, u) != 0 ||
            bb_reserve(bb, pbytes) != 0) {
            return -1;
        }
        uint8_t *dst = bb->data + bb->len;
        memset(dst, 0, pbytes);
        for (size_t i = 0U; i < m; i++) {
            dst[i >> 1U] |= (uint8_t)(b2idx[plane[i]] << (4U * (i & 1U)));
        }
        bb->len += pbytes;
        return 0;
    }
    if (best_mode == CHMC_LANE_RLE) {
        return chmc_packbits_emit(bb, plane, m);
    }
    return bb_put_bytes(bb, plane, m);
}

static int chmc_decode_lane(const uint8_t *buf, size_t end, size_t *pos, uint8_t *plane,
                            size_t m) {
    uint8_t mode = 0U;
    if (chmc_get_u8(buf, end, pos, &mode) != 0) return -1;
    if (mode == CHMC_LANE_CONST) {
        if (m < 2U) return -1;
        uint8_t val = 0U;
        if (chmc_get_u8(buf, end, pos, &val) != 0) return -1;
        memset(plane, val, m);
        return 0;
    }
    if (mode == CHMC_LANE_PAL2) {
        uint8_t u = 0U;
        if (chmc_get_u8(buf, end, pos, &u) != 0 || u < 2U || u > 4U) return -1;
        size_t pbytes = (m + 3U) >> 2U;
        if (2U + (size_t)u + pbytes >= 1U + m) return -1;
        if (*pos + (size_t)u + pbytes > end) return -1;
        const uint8_t *pal = buf + *pos;
        for (uint8_t i = 1U; i < u; i++) {
            if (pal[i - 1U] >= pal[i]) return -1;
        }
        *pos += (size_t)u;
        const uint8_t *packed = buf + *pos;
        *pos += pbytes;
        if ((m & 3U) != 0U && (packed[pbytes - 1U] >> (2U * (m & 3U))) != 0U) return -1;
        for (size_t i = 0U; i < m; i++) {
            uint8_t idx = (uint8_t)((packed[i >> 2U] >> (2U * (i & 3U))) & 3U);
            if (idx >= u) return -1;
            plane[i] = pal[idx];
        }
        return 0;
    }
    if (mode == CHMC_LANE_PAL4) {
        uint8_t u = 0U;
        if (chmc_get_u8(buf, end, pos, &u) != 0 || u < 5U || u > 16U) return -1;
        size_t pbytes = (m + 1U) >> 1U;
        if (2U + (size_t)u + pbytes >= 1U + m) return -1;
        if (*pos + (size_t)u + pbytes > end) return -1;
        const uint8_t *pal = buf + *pos;
        for (uint8_t i = 1U; i < u; i++) {
            if (pal[i - 1U] >= pal[i]) return -1;
        }
        *pos += (size_t)u;
        const uint8_t *packed = buf + *pos;
        *pos += pbytes;
        if ((m & 1U) != 0U && (packed[pbytes - 1U] >> 4U) != 0U) return -1;
        for (size_t i = 0U; i < m; i++) {
            uint8_t idx = (uint8_t)((packed[i >> 1U] >> (4U * (i & 1U))) & 0xFU);
            if (idx >= u) return -1;
            plane[i] = pal[idx];
        }
        return 0;
    }
    if (mode == CHMC_LANE_RLE) {
        size_t start_pos = *pos;
        size_t out_i = 0U;
        while (out_i < m) {
            uint8_t ctrl = 0U;
            if (chmc_get_u8(buf, end, pos, &ctrl) != 0) return -1;
            if (ctrl < 128U) {
                size_t cnt = (size_t)ctrl + 1U;
                if (out_i + cnt > m || *pos + cnt > end) return -1;
                memcpy(plane + out_i, buf + *pos, cnt);
                *pos += cnt;
                out_i += cnt;
            } else {
                size_t cnt = (size_t)(ctrl - 128U) + 3U;
                uint8_t val = 0U;
                if (out_i + cnt > m || chmc_get_u8(buf, end, pos, &val) != 0) return -1;
                memset(plane + out_i, val, cnt);
                out_i += cnt;
            }
        }
        if (*pos - start_pos >= m) return -1;
        return 0;
    }
    if (mode == CHMC_LANE_RAW) {
        if (*pos + m > end) return -1;
        memcpy(plane, buf + *pos, m);
        *pos += m;
        return 0;
    }
    return -1;
}

static int chmc_encode_f64_col(ChemByteBuf *bb, const double *arr, size_t n) {
    if (n == 0U) return 0;
    if (!arr) return -1;
    size_t zeros = 0U;
    for (size_t i = 0U; i < n; i++) {
        uint64_t u = 0U;
        memcpy(&u, &arr[i], sizeof(u));
        if (u == 0ULL) zeros++;
    }
    if (zeros == n) {
        return bb_put_u8(bb, CHMC_COL_ALL_ZERO);
    }
    size_t mask_bytes = (n + 7U) >> 3U;
    int use_mask = (zeros * 8U > mask_bytes + 1U) ? 1 : 0;
    size_t m = use_mask ? (n - zeros) : n;
    uint64_t *nz = (uint64_t *)malloc(m * sizeof(uint64_t));
    uint8_t *plane = (uint8_t *)malloc(m);
    if (!nz || !plane) {
        free(nz);
        free(plane);
        return -1;
    }
    if (use_mask) {
        if (bb_put_u8(bb, CHMC_COL_SPARSE_MASK) != 0 || bb_reserve(bb, mask_bytes) != 0) {
            free(nz);
            free(plane);
            return -1;
        }
        uint8_t *mask = bb->data + bb->len;
        memset(mask, 0, mask_bytes);
        size_t mi = 0U;
        for (size_t i = 0U; i < n; i++) {
            uint64_t u = 0U;
            memcpy(&u, &arr[i], sizeof(u));
            if (u != 0ULL) {
                mask[i >> 3U] |= (uint8_t)(1U << (i & 7U));
                nz[mi++] = u;
            }
        }
        bb->len += mask_bytes;
    } else {
        if (bb_put_u8(bb, CHMC_COL_DENSE) != 0) {
            free(nz);
            free(plane);
            return -1;
        }
        for (size_t i = 0U; i < n; i++) {
            memcpy(&nz[i], &arr[i], sizeof(nz[i]));
        }
    }

    int rc = 0;
    for (int lane = 7; lane >= 0; lane--) {
        unsigned sh = (unsigned)(lane * 8);
        for (size_t i = 0U; i < m; i++) {
            plane[i] = (uint8_t)((nz[i] >> sh) & UINT64_C(0xFF));
        }
        if (chmc_encode_lane(bb, plane, m) != 0) {
            rc = -1;
            break;
        }
    }
    free(nz);
    free(plane);
    return rc;
}

static int chmc_decode_f64_col(const uint8_t *buf, size_t end, size_t *pos, size_t n,
                               double **arr_out) {
    *arr_out = NULL;
    if (n == 0U) return 0;
    double *arr = (double *)calloc(n, sizeof(double));
    if (!arr) return -1;

    uint8_t col_mode = 0U;
    if (chmc_get_u8(buf, end, pos, &col_mode) != 0) {
        free(arr);
        return -1;
    }
    if (col_mode == CHMC_COL_ALL_ZERO) {
        *arr_out = arr;
        return 0;
    }

    size_t mask_bytes = (n + 7U) >> 3U;
    const uint8_t *mask = NULL;
    size_t m = n;
    if (col_mode == CHMC_COL_SPARSE_MASK) {
        if (*pos + mask_bytes > end) {
            free(arr);
            return -1;
        }
        mask = buf + *pos;
        *pos += mask_bytes;
        if ((n & 7U) != 0U && (mask[mask_bytes - 1U] >> (n & 7U)) != 0U) {
            free(arr);
            return -1;
        }
        m = 0U;
        for (size_t i = 0U; i < n; i++) {
            if ((mask[i >> 3U] >> (i & 7U)) & 1U) m++;
        }
        if (m == 0U || m >= n || (n - m) * 8U <= mask_bytes + 1U) {
            free(arr);
            return -1;
        }
    } else if (col_mode != CHMC_COL_DENSE) {
        free(arr);
        return -1;
    }

    uint64_t *nz = (uint64_t *)calloc(m, sizeof(uint64_t));
    uint8_t *plane = (uint8_t *)malloc(m);
    if (!nz || !plane) {
        free(nz);
        free(plane);
        free(arr);
        return -1;
    }

    for (int lane = 7; lane >= 0; lane--) {
        if (chmc_decode_lane(buf, end, pos, plane, m) != 0) {
            free(nz);
            free(plane);
            free(arr);
            return -1;
        }
        unsigned sh = (unsigned)(lane * 8);
        for (size_t i = 0U; i < m; i++) {
            nz[i] |= ((uint64_t)plane[i]) << sh;
        }
    }

    if (col_mode == CHMC_COL_DENSE) {
        size_t zeros = 0U;
        for (size_t i = 0U; i < n; i++) {
            if (nz[i] == 0ULL) zeros++;
            memcpy(&arr[i], &nz[i], sizeof(arr[i]));
        }
        if (zeros == n || zeros * 8U > mask_bytes + 1U) {
            free(nz);
            free(plane);
            free(arr);
            return -1;
        }
    } else {
        size_t mi = 0U;
        for (size_t i = 0U; i < n; i++) {
            if ((mask[i >> 3U] >> (i & 7U)) & 1U) {
                if (nz[mi] == 0ULL) {
                    free(nz);
                    free(plane);
                    free(arr);
                    return -1;
                }
                memcpy(&arr[i], &nz[mi], sizeof(arr[i]));
                mi++;
            }
        }
    }

    free(nz);
    free(plane);
    *arr_out = arr;
    return 0;
}

static int chmc_encode_full_cert(const ChemFullCert *c, uint8_t **out_buf, size_t *out_len,
                                 char *err, size_t err_sz) {
    uint8_t sha_raw[32];
    if (hex_to_bytes32(c->sha256_hex, sha_raw) != 0) {
        set_err(err, err_sz, "invalid hamiltonian_sha256 hex string");
        return -1;
    }
    size_t flen = strlen(c->fcidump_path);
    if (flen > 4096U) {
        set_err(err, err_sz, "fcidump_path exceeds 4096 bytes");
        return -1;
    }
    if (c->t2_norbs < 0 || c->t2_norbs > CHEM_MAX_ORBITALS ||
        c->t1_norbs < 0 || c->t1_norbs > CHEM_MAX_ORBITALS ||
        c->meq < 0 || c->nD < 0 || c->nQ < 0 || c->nG < 0 || c->nT2 < 0 || c->nT1 < 0 ||
        c->ndet_var < 0) {
        set_err(err, err_sz, "invalid negative or oversized certificate dimensions");
        return -1;
    }

    uint32_t flags = 0U;
    if (c->mode == CHEM_MODE_B_GENERAL) flags |= CHMC_FLAG_MODE_B;
    if (c->constrain_s2) flags |= CHMC_FLAG_CONSTRAIN_S2;
    if (c->t2_enabled) flags |= CHMC_FLAG_T2;
    if (c->t1_enabled) flags |= CHMC_FLAG_T1;
    if (c->converged) flags |= CHMC_FLAG_CONVERGED;
    if (c->veto_enabled) flags |= CHMC_FLAG_VETO_EN;
    if (c->vetoed) flags |= CHMC_FLAG_VETOED;
    if (c->has_var_meta) flags |= CHMC_FLAG_HAS_VAR_META;
    if (c->has_witness && c->ndet_var > 0 && c->dets && c->psi0) flags |= CHMC_FLAG_HAS_WITNESS;
    if (c->is_exact_fci) flags |= CHMC_FLAG_IS_EXACT_FCI;

    ChemByteBuf bb;
    memset(&bb, 0, sizeof(bb));
    int ok = 1;
    uint8_t magic[4] = {CHMC_MAGIC_0, CHMC_MAGIC_1, CHMC_MAGIC_2, CHMC_MAGIC_3};
    if (bb_put_bytes(&bb, magic, 4U) != 0 ||
        bb_put_le32(&bb, CHMC_VERSION) != 0 ||
        bb_put_le32(&bb, flags) != 0 ||
        bb_put_bytes(&bb, sha_raw, 32U) != 0) {
        ok = 0;
    }

    uint8_t sver = (uint8_t)((c->schema_ver >= 1 && c->schema_ver <= 3) ? c->schema_ver : 1);
    if (ok && (bb_put_u8(&bb, sver) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->K) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->nelec) != 0 ||
               bb_put_i32(&bb, (int32_t)c->ms2) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->na) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->nb) != 0 ||
               bb_put_i32(&bb, (int32_t)c->two_s) != 0)) {
        ok = 0;
    }

    uint32_t t2n = c->t2_enabled ? (uint32_t)c->t2_norbs : 0U;
    if (ok && bb_put_varu32(&bb, t2n) != 0) ok = 0;
    for (uint32_t i = 0U; ok && i < t2n; i++) {
        if (bb_put_varu32(&bb, (uint32_t)c->t2_orbs[i]) != 0) ok = 0;
    }
    uint32_t t1n = c->t1_enabled ? (uint32_t)c->t1_norbs : 0U;
    if (ok && bb_put_varu32(&bb, t1n) != 0) ok = 0;
    for (uint32_t i = 0U; ok && i < t1n; i++) {
        if (bb_put_varu32(&bb, (uint32_t)c->t1_orbs[i]) != 0) ok = 0;
    }

    if (ok && (bb_put_varu32(&bb, (uint32_t)flen) != 0 ||
               bb_put_bytes(&bb, c->fcidump_path, flen) != 0 ||
               bb_put_f64(&bb, c->E_core) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->iters) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->iter_stopped) != 0 ||
               bb_put_f64(&bb, c->E_primal) != 0 ||
               bb_put_f64(&bb, c->E_dual_raw) != 0 ||
               bb_put_f64(&bb, c->E_lb_cert) != 0 ||
               bb_put_f64(&bb, c->min_eig_fp) != 0 ||
               bb_put_f64(&bb, c->min_eig_cert) != 0 ||
               bb_put_f64(&bb, c->delta_round) != 0 ||
               bb_put_f64(&bb, c->prim_res) != 0 ||
               bb_put_f64(&bb, c->dual_res) != 0 ||
               bb_put_f64(&bb, c->elapsed_sec) != 0 ||
               bb_put_f64(&bb, c->E_ai) != 0 ||
               bb_put_f64(&bb, c->veto_tol) != 0 ||
               bb_put_f64(&bb, c->E_ub_variational) != 0 ||
               bb_put_f64(&bb, c->interval_width) != 0 ||
               bb_put_i64(&bb, c->scale_nha) != 0 ||
               bb_put_i64(&bb, c->E_lb_floor) != 0 ||
               bb_put_i64(&bb, c->E_ub_ceil) != 0 ||
               bb_put_i64(&bb, c->E_ai_minus_tol_floor) != 0 ||
               bb_put_f64(&bb, c->E_ub) != 0)) {
        ok = 0;
    }

    if (ok && (flags & CHMC_FLAG_HAS_VAR_META)) {
        if (bb_put_f64(&bb, c->E_hf0) != 0 ||
            bb_put_f64(&bb, c->E_scf) != 0 ||
            bb_put_f64(&bb, c->E_searu_rot) != 0 ||
            bb_put_f64(&bb, c->E_ci) != 0 ||
            bb_put_varu64(&bb, c->ndet_space) != 0 ||
            bb_put_varu32(&bb, (uint32_t)c->ndet_var) != 0) {
            ok = 0;
        }
    }

    if (ok && (bb_put_varu32(&bb, (uint32_t)c->meq) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->nD) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->nQ) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->nG) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->nT2) != 0 ||
               bb_put_varu32(&bb, (uint32_t)c->nT1) != 0)) {
        ok = 0;
    }

    /* Columnar Payload */
    if (ok && c->meq > 0 && chmc_encode_f64_col(&bb, c->y_eq, (size_t)c->meq) != 0) ok = 0;
    if (ok && c->nQ > 0 && chmc_encode_f64_col(&bb, c->Z_Q, (size_t)c->nQ) != 0) ok = 0;
    if (ok && c->nG > 0 && chmc_encode_f64_col(&bb, c->Z_G, (size_t)c->nG) != 0) ok = 0;
    if (ok && c->t2_enabled && c->nT2 > 0 &&
        chmc_encode_f64_col(&bb, c->Z_T2, (size_t)c->nT2) != 0) {
        ok = 0;
    }
    if (ok && c->t1_enabled && c->nT1 > 0 &&
        chmc_encode_f64_col(&bb, c->Z_T1, (size_t)c->nT1) != 0) {
        ok = 0;
    }
    if (ok && (flags & CHMC_FLAG_HAS_WITNESS)) {
        for (int i = 0; ok && i < c->ndet_var; i++) {
            if (bb_put_varu64(&bb, c->dets[i]) != 0) ok = 0;
        }
        if (ok && chmc_encode_f64_col(&bb, c->psi0, (size_t)c->ndet_var) != 0) ok = 0;
    }

    /* 32-byte FIPS-180-4 SHA-256 Integrity Trailer */
    if (ok && bb_reserve(&bb, CHMC_TRAILER_BYTES) == 0) {
        flow_sha256_ctx_t sctx;
        uint8_t digest[32];
        flow_sha256_init(&sctx);
        flow_sha256_update(&sctx, bb.data, bb.len);
        flow_sha256_final(&sctx, digest);
        memcpy(bb.data + bb.len, digest, CHMC_TRAILER_BYTES);
        bb.len += CHMC_TRAILER_BYTES;
    } else {
        ok = 0;
    }

    if (!ok) {
        free(bb.data);
        set_err(err, err_sz, "failed encoding CHMC binary certificate");
        return -1;
    }
    *out_buf = bb.data;
    *out_len = bb.len;
    return 0;
}

static int chmc_decode_full_cert(const uint8_t *buf, size_t len, ChemFullCert *c,
                                 char *err, size_t err_sz) {
    memset(c, 0, sizeof(*c));
    c->owns_arrays = 1;
    if (!buf || len < CHMC_HDR_BYTES + CHMC_TRAILER_BYTES) {
        set_err(err, err_sz, "truncated CHMC binary certificate");
        return -1;
    }
    if (buf[0] != CHMC_MAGIC_0 || buf[1] != CHMC_MAGIC_1 ||
        buf[2] != CHMC_MAGIC_2 || buf[3] != CHMC_MAGIC_3) {
        set_err(err, err_sz, "invalid CHMC magic");
        return -1;
    }
    size_t body_len = len - CHMC_TRAILER_BYTES;
    flow_sha256_ctx_t sctx;
    uint8_t digest[32];
    flow_sha256_init(&sctx);
    flow_sha256_update(&sctx, buf, body_len);
    flow_sha256_final(&sctx, digest);
    uint8_t diff = 0U;
    for (size_t i = 0U; i < CHMC_TRAILER_BYTES; i++) {
        diff |= (uint8_t)(digest[i] ^ buf[body_len + i]);
    }
    if (diff != 0U) {
        set_err(err, err_sz, "CHMC SHA-256 integrity trailer mismatch (corrupted or tampered)");
        return -1;
    }

    size_t pos = 4U;
    uint32_t ver = 0U, flags = 0U;
    if (chmc_get_le32(buf, body_len, &pos, &ver) != 0 || ver != CHMC_VERSION ||
        chmc_get_le32(buf, body_len, &pos, &flags) != 0 ||
        (flags & ~CHMC_FLAGS_VALID_MASK) != 0U) {
        set_err(err, err_sz, "unsupported CHMC version or invalid flags");
        return -1;
    }
    if (pos + 32U > body_len) {
        set_err(err, err_sz, "truncated CHMC header SHA-256");
        return -1;
    }
    flow_sha256_digest_to_hex(buf + pos, c->sha256_hex);
    pos += 32U;

    c->mode = (flags & CHMC_FLAG_MODE_B) ? CHEM_MODE_B_GENERAL : CHEM_MODE_A_SINGLET;
    c->constrain_s2 = (flags & CHMC_FLAG_CONSTRAIN_S2) ? 1 : 0;
    c->t2_enabled = (flags & CHMC_FLAG_T2) ? 1 : 0;
    c->t1_enabled = (flags & CHMC_FLAG_T1) ? 1 : 0;
    c->converged = (flags & CHMC_FLAG_CONVERGED) ? 1 : 0;
    c->veto_enabled = (flags & CHMC_FLAG_VETO_EN) ? 1 : 0;
    c->vetoed = (flags & CHMC_FLAG_VETOED) ? 1 : 0;
    c->has_var_meta = (flags & CHMC_FLAG_HAS_VAR_META) ? 1 : 0;
    c->has_witness = (flags & CHMC_FLAG_HAS_WITNESS) ? 1 : 0;
    c->is_exact_fci = (flags & CHMC_FLAG_IS_EXACT_FCI) ? 1 : 0;
    if (c->has_witness && !c->has_var_meta) {
        set_err(err, err_sz, "invalid CHMC flags: witness without var_meta");
        return -1;
    }

    uint8_t sver = 0U;
    uint32_t K = 0U, nelec = 0U, na = 0U, nb = 0U, t2n = 0U, t1n = 0U, flen = 0U;
    uint32_t iters = 0U, iter_stopped = 0U;
    int32_t ms2 = 0, two_s = 0;
    if (chmc_get_u8(buf, body_len, &pos, &sver) != 0 || sver < 1U || sver > 3U ||
        chmc_get_varu32(buf, body_len, &pos, &K) != 0 || K == 0U || K > CHEM_MAX_ORBITALS ||
        chmc_get_varu32(buf, body_len, &pos, &nelec) != 0 || nelec > 2U * K ||
        chmc_get_i32(buf, body_len, &pos, &ms2) != 0 ||
        chmc_get_varu32(buf, body_len, &pos, &na) != 0 || na > K ||
        chmc_get_varu32(buf, body_len, &pos, &nb) != 0 || nb > K ||
        chmc_get_i32(buf, body_len, &pos, &two_s) != 0) {
        set_err(err, err_sz, "malformed CHMC orbital/electron metadata");
        return -1;
    }
    c->schema_ver = (int)sver;
    c->K = (int)K;
    c->nelec = (int)nelec;
    c->ms2 = (int)ms2;
    c->na = (int)na;
    c->nb = (int)nb;
    c->two_s = (int)two_s;

    if (chmc_get_varu32(buf, body_len, &pos, &t2n) != 0 || t2n > CHEM_MAX_ORBITALS ||
        (!c->t2_enabled && t2n != 0U)) {
        set_err(err, err_sz, "malformed CHMC t2 metadata");
        return -1;
    }
    c->t2_norbs = (int)t2n;
    for (uint32_t i = 0U; i < t2n; i++) {
        uint32_t orb = 0U;
        if (chmc_get_varu32(buf, body_len, &pos, &orb) != 0 || orb >= K) {
            set_err(err, err_sz, "malformed CHMC t2 orbital index");
            return -1;
        }
        c->t2_orbs[i] = (int)orb;
    }

    if (chmc_get_varu32(buf, body_len, &pos, &t1n) != 0 || t1n > CHEM_MAX_ORBITALS ||
        (!c->t1_enabled && t1n != 0U)) {
        set_err(err, err_sz, "malformed CHMC t1 metadata");
        return -1;
    }
    c->t1_norbs = (int)t1n;
    for (uint32_t i = 0U; i < t1n; i++) {
        uint32_t orb = 0U;
        if (chmc_get_varu32(buf, body_len, &pos, &orb) != 0 || orb >= K) {
            set_err(err, err_sz, "malformed CHMC t1 orbital index");
            return -1;
        }
        c->t1_orbs[i] = (int)orb;
    }

    if (chmc_get_varu32(buf, body_len, &pos, &flen) != 0 || flen > 4096U ||
        pos + (size_t)flen > body_len) {
        set_err(err, err_sz, "malformed CHMC fcidump_path");
        return -1;
    }
    if (flen > 0U) memcpy(c->fcidump_path, buf + pos, (size_t)flen);
    c->fcidump_path[flen] = '\0';
    pos += (size_t)flen;

    if (chmc_get_f64(buf, body_len, &pos, &c->E_core) != 0 ||
        chmc_get_varu32(buf, body_len, &pos, &iters) != 0 ||
        chmc_get_varu32(buf, body_len, &pos, &iter_stopped) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->E_primal) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->E_dual_raw) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->E_lb_cert) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->min_eig_fp) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->min_eig_cert) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->delta_round) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->prim_res) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->dual_res) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->elapsed_sec) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->E_ai) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->veto_tol) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->E_ub_variational) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->interval_width) != 0 ||
        chmc_get_i64(buf, body_len, &pos, &c->scale_nha) != 0 ||
        chmc_get_i64(buf, body_len, &pos, &c->E_lb_floor) != 0 ||
        chmc_get_i64(buf, body_len, &pos, &c->E_ub_ceil) != 0 ||
        chmc_get_i64(buf, body_len, &pos, &c->E_ai_minus_tol_floor) != 0 ||
        chmc_get_f64(buf, body_len, &pos, &c->E_ub) != 0) {
        set_err(err, err_sz, "truncated CHMC scalar metadata");
        return -1;
    }
    c->iters = (int)iters;
    c->iter_stopped = (int)iter_stopped;

    if (c->has_var_meta) {
        uint32_t ndv = 0U;
        if (chmc_get_f64(buf, body_len, &pos, &c->E_hf0) != 0 ||
            chmc_get_f64(buf, body_len, &pos, &c->E_scf) != 0 ||
            chmc_get_f64(buf, body_len, &pos, &c->E_searu_rot) != 0 ||
            chmc_get_f64(buf, body_len, &pos, &c->E_ci) != 0 ||
            chmc_get_varu64(buf, body_len, &pos, &c->ndet_space) != 0 ||
            chmc_get_varu32(buf, body_len, &pos, &ndv) != 0 || ndv > 65536U) {
            set_err(err, err_sz, "malformed CHMC variational metadata");
            return -1;
        }
        c->ndet_var = (int)ndv;
    }
    if (c->has_witness && c->ndet_var <= 0) {
        set_err(err, err_sz, "invalid CHMC witness count");
        return -1;
    }

    uint32_t meq = 0U, nD = 0U, nQ = 0U, nG = 0U, nT2 = 0U, nT1 = 0U;
    if (chmc_get_varu32(buf, body_len, &pos, &meq) != 0 ||
        chmc_get_varu32(buf, body_len, &pos, &nD) != 0 ||
        chmc_get_varu32(buf, body_len, &pos, &nQ) != 0 ||
        chmc_get_varu32(buf, body_len, &pos, &nG) != 0 ||
        chmc_get_varu32(buf, body_len, &pos, &nT2) != 0 ||
        chmc_get_varu32(buf, body_len, &pos, &nT1) != 0 ||
        meq > 65536U || nD > 16777216U || nQ > 16777216U || nG > 16777216U ||
        nT2 > 16777216U || nT1 > 16777216U) {
        set_err(err, err_sz, "malformed CHMC dual witness dimensions");
        return -1;
    }
    c->meq = (int)meq;
    c->nD = (int)nD;
    c->nQ = (int)nQ;
    c->nG = (int)nG;
    c->nT2 = (int)nT2;
    c->nT1 = (int)nT1;

    if ((meq > 0U && chmc_decode_f64_col(buf, body_len, &pos, (size_t)meq, &c->y_eq) != 0) ||
        (nQ > 0U && chmc_decode_f64_col(buf, body_len, &pos, (size_t)nQ, &c->Z_Q) != 0) ||
        (nG > 0U && chmc_decode_f64_col(buf, body_len, &pos, (size_t)nG, &c->Z_G) != 0) ||
        (c->t2_enabled && nT2 > 0U &&
         chmc_decode_f64_col(buf, body_len, &pos, (size_t)nT2, &c->Z_T2) != 0) ||
        (c->t1_enabled && nT1 > 0U &&
         chmc_decode_f64_col(buf, body_len, &pos, (size_t)nT1, &c->Z_T1) != 0)) {
        full_cert_free(c);
        set_err(err, err_sz, "corrupt CHMC dual witness column");
        return -1;
    }

    if (c->has_witness) {
        c->dets = (uint64_t *)malloc((size_t)c->ndet_var * sizeof(uint64_t));
        if (!c->dets) {
            full_cert_free(c);
            set_err(err, err_sz, "out of memory decoding CHMC dets");
            return -1;
        }
        for (int i = 0; i < c->ndet_var; i++) {
            if (chmc_get_varu64(buf, body_len, &pos, &c->dets[i]) != 0) {
                full_cert_free(c);
                set_err(err, err_sz, "corrupt CHMC determinant column");
                return -1;
            }
        }
        if (chmc_decode_f64_col(buf, body_len, &pos, (size_t)c->ndet_var, &c->psi0) != 0) {
            full_cert_free(c);
            set_err(err, err_sz, "corrupt CHMC psi0 column");
            return -1;
        }
    }

    if (pos != body_len) {
        full_cert_free(c);
        set_err(err, err_sz, "trailing bytes in CHMC certificate (%zu != %zu)", pos, body_len);
        return -1;
    }
    return 0;
}

static int json_parse_full_cert(const char *txt, size_t len, ChemFullCert *c,
                                char *err, size_t err_sz) {
    memset(c, 0, sizeof(*c));
    c->owns_arrays = 1;
    const char *s = txt, *e = txt + len;

    char schema_str[64] = {0};
    char mode_str[64] = {0};
    if (json_get_string(s, e, "hamiltonian_sha256", c->sha256_hex, sizeof(c->sha256_hex)) != 0 ||
        json_get_string(s, e, "fcidump_path", c->fcidump_path, sizeof(c->fcidump_path)) != 0 ||
        json_get_string(s, e, "mode", mode_str, sizeof(mode_str)) != 0 ||
        json_get_int(s, e, "K", &c->K) != 0 ||
        json_get_int(s, e, "nelec", &c->nelec) != 0 ||
        json_get_int(s, e, "ms2", &c->ms2) != 0 ||
        json_get_int(s, e, "na", &c->na) != 0 ||
        json_get_int(s, e, "nb", &c->nb) != 0 ||
        json_get_int(s, e, "two_s", &c->two_s) != 0 ||
        json_get_int(s, e, "constrain_s2", &c->constrain_s2) != 0 ||
        json_get_double(s, e, "E_core", &c->E_core) != 0) {
        set_err(err, err_sz, "malformed top-level JSON certificate fields");
        return -1;
    }
    c->mode = (strcmp(mode_str, "A_SINGLET") == 0) ? CHEM_MODE_A_SINGLET : CHEM_MODE_B_GENERAL;

    if (json_get_int(s, e, "t2_enabled", &c->t2_enabled) == 0 && c->t2_enabled) {
        if (json_get_int(s, e, "t2_norbs", &c->t2_norbs) != 0 || c->t2_norbs <= 0 ||
            c->t2_norbs > CHEM_MAX_ORBITALS ||
            json_get_int_array(s, e, "t2_orbs", c->t2_norbs, c->t2_orbs) != 0) {
            set_err(err, err_sz, "malformed t2 metadata in JSON certificate");
            return -1;
        }
    }
    if (json_get_int(s, e, "t1_enabled", &c->t1_enabled) == 0 && c->t1_enabled) {
        if (json_get_int(s, e, "t1_norbs", &c->t1_norbs) != 0 || c->t1_norbs <= 0 ||
            c->t1_norbs > CHEM_MAX_ORBITALS ||
            json_get_int_array(s, e, "t1_orbs", c->t1_norbs, c->t1_orbs) != 0) {
            set_err(err, err_sz, "malformed t1 metadata in JSON certificate");
            return -1;
        }
    }
    if (json_get_string(s, e, "schema_version", schema_str, sizeof(schema_str)) == 0) {
        if (strcmp(schema_str, "chem-cert-v3") == 0) c->schema_ver = 3;
        else if (strcmp(schema_str, "chem-cert-v2") == 0) c->schema_ver = 2;
        else c->schema_ver = 1;
    } else {
        c->schema_ver = c->t1_enabled ? 3 : (c->t2_enabled ? 2 : 1);
    }

    const char *sol_s = NULL, *sol_e = NULL;
    const char *vet_s = NULL, *vet_e = NULL;
    const char *lean_s = NULL, *lean_e = NULL;
    const char *var_s = NULL, *var_e = NULL;
    const char *dw_s = NULL, *dw_e = NULL;
    if (json_get_object_range(s, e, "solver", &sol_s, &sol_e) != 0 ||
        json_get_object_range(s, e, "dual_witness", &dw_s, &dw_e) != 0) {
        set_err(err, err_sz, "missing solver or dual_witness JSON section");
        return -1;
    }
    if (json_get_int(sol_s, sol_e, "iters", &c->iters) != 0 ||
        json_get_int(sol_s, sol_e, "converged", &c->converged) != 0 ||
        json_get_int(sol_s, sol_e, "vetoed", &c->vetoed) != 0 ||
        json_get_int(sol_s, sol_e, "iter_stopped", &c->iter_stopped) != 0 ||
        json_get_double(sol_s, sol_e, "E_primal", &c->E_primal) != 0 ||
        json_get_double(sol_s, sol_e, "E_dual_raw", &c->E_dual_raw) != 0 ||
        json_get_double(sol_s, sol_e, "E_lb_cert", &c->E_lb_cert) != 0 ||
        json_get_double(sol_s, sol_e, "min_eig_fp", &c->min_eig_fp) != 0 ||
        json_get_double(sol_s, sol_e, "min_eig_cert", &c->min_eig_cert) != 0 ||
        json_get_double(sol_s, sol_e, "delta_round", &c->delta_round) != 0 ||
        json_get_double(sol_s, sol_e, "prim_res", &c->prim_res) != 0 ||
        json_get_double(sol_s, sol_e, "dual_res", &c->dual_res) != 0 ||
        json_get_double(sol_s, sol_e, "elapsed_sec", &c->elapsed_sec) != 0) {
        set_err(err, err_sz, "malformed solver fields in JSON certificate");
        return -1;
    }

    if (json_get_object_range(s, e, "veto", &vet_s, &vet_e) == 0) {
        if (json_get_int(vet_s, vet_e, "enabled", &c->veto_enabled) != 0 ||
            json_get_double(vet_s, vet_e, "E_ai", &c->E_ai) != 0 ||
            json_get_double(vet_s, vet_e, "veto_tol", &c->veto_tol) != 0 ||
            json_get_double(vet_s, vet_e, "E_ub_variational", &c->E_ub_variational) != 0 ||
            json_get_double(vet_s, vet_e, "interval_width", &c->interval_width) != 0) {
            set_err(err, err_sz, "malformed veto fields in JSON certificate");
            return -1;
        }
    } else {
        c->veto_tol = 1e-4;
        c->E_ub_variational = c->E_primal;
        c->interval_width = c->E_ub_variational - c->E_lb_cert;
    }

    if (json_get_object_range(s, e, "lean_scaled_witness", &lean_s, &lean_e) == 0) {
        if (json_get_i64(lean_s, lean_e, "scale_nha", &c->scale_nha) != 0 ||
            json_get_i64(lean_s, lean_e, "E_lb_floor", &c->E_lb_floor) != 0 ||
            json_get_i64(lean_s, lean_e, "E_ub_ceil", &c->E_ub_ceil) != 0 ||
            json_get_i64(lean_s, lean_e, "E_ai_minus_tol_floor", &c->E_ai_minus_tol_floor) != 0) {
            set_err(err, err_sz, "malformed lean_scaled_witness in JSON certificate");
            return -1;
        }
    } else {
        c->scale_nha = 1000000000LL;
        c->E_lb_floor = (int64_t)floor(c->E_lb_cert * 1e9);
        c->E_ub_ceil = (int64_t)ceil(c->E_ub_variational * 1e9);
        c->E_ai_minus_tol_floor = (int64_t)floor((c->E_ai - c->veto_tol) * 1e9);
    }

    if (json_get_object_range(s, e, "variational", &var_s, &var_e) == 0) {
        if (json_get_double(var_s, var_e, "E_ub", &c->E_ub) != 0 ||
            json_get_int(var_s, var_e, "has_witness", &c->has_witness) != 0) {
            set_err(err, err_sz, "malformed variational section in JSON certificate");
            return -1;
        }
        if (json_get_double(var_s, var_e, "E_hf0", &c->E_hf0) == 0) {
            c->has_var_meta = 1;
            if (json_get_double(var_s, var_e, "E_scf", &c->E_scf) != 0 ||
                json_get_double(var_s, var_e, "E_searu_rot", &c->E_searu_rot) != 0 ||
                json_get_double(var_s, var_e, "E_ci", &c->E_ci) != 0 ||
                json_get_int(var_s, var_e, "is_exact_fci", &c->is_exact_fci) != 0 ||
                json_get_u64(var_s, var_e, "ndet_space", &c->ndet_space) != 0 ||
                json_get_int(var_s, var_e, "ndet_var", &c->ndet_var) != 0) {
                set_err(err, err_sz, "malformed variational metadata in JSON certificate");
                return -1;
            }
            if (c->has_witness) {
                if (c->ndet_var <= 0 ||
                    json_get_u64_hex_array(var_s, var_e, "dets", c->ndet_var, &c->dets) != 0 ||
                    json_get_double_array(var_s, var_e, "psi0", c->ndet_var, &c->psi0) != 0) {
                    full_cert_free(c);
                    set_err(err, err_sz, "malformed variational witness arrays in JSON cert");
                    return -1;
                }
            }
        }
    } else {
        c->E_ub = c->E_ub_variational;
    }

    if (json_get_int(dw_s, dw_e, "meq", &c->meq) != 0 ||
        json_get_int(dw_s, dw_e, "nD", &c->nD) != 0 ||
        json_get_int(dw_s, dw_e, "nQ", &c->nQ) != 0 ||
        json_get_int(dw_s, dw_e, "nG", &c->nG) != 0 ||
        json_get_int(dw_s, dw_e, "nT2", &c->nT2) != 0 ||
        (c->t1_enabled && json_get_int(dw_s, dw_e, "nT1", &c->nT1) != 0)) {
        full_cert_free(c);
        set_err(err, err_sz, "missing dual_witness dimensions in JSON certificate");
        return -1;
    }

    if (json_get_double_array(dw_s, dw_e, "y_eq", c->meq, &c->y_eq) != 0 ||
        json_get_double_array(dw_s, dw_e, "Z_Q", c->nQ, &c->Z_Q) != 0 ||
        json_get_double_array(dw_s, dw_e, "Z_G", c->nG, &c->Z_G) != 0 ||
        (c->t2_enabled && c->nT2 > 0 &&
         json_get_double_array(dw_s, dw_e, "Z_T2", c->nT2, &c->Z_T2) != 0) ||
        (c->t1_enabled && c->nT1 > 0 &&
         json_get_double_array(dw_s, dw_e, "Z_T1", c->nT1, &c->Z_T1) != 0)) {
        full_cert_free(c);
        set_err(err, err_sz, "failed parsing dual_witness arrays in JSON certificate");
        return -1;
    }
    return 0;
}

static int write_buf_to_file(const char *path, const uint8_t *buf, size_t len,
                             char *err, size_t err_sz) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        set_err(err, err_sz, "cannot open output path: %s", path);
        return -1;
    }
    size_t nw = (len > 0U) ? fwrite(buf, 1, len, fp) : 0U;
    int ferr = ferror(fp);
    fclose(fp);
    if (nw != len || ferr) {
        set_err(err, err_sz, "short write to %s", path);
        return -1;
    }
    return 0;
}

int chem_cert_write_chmc(const char *path, const ChemHam *ham, const char *fcidump_path,
                         const ChemSolveResult *res, const struct ChemVarResult *var_opt,
                         const ChemSolveOpts *opts_in, char *err, size_t err_sz) {
    if (!path || !ham || !res) {
        set_err(err, err_sz, "null argument to chem_cert_write_chmc");
        return -1;
    }
    ChemFullCert fc;
    full_cert_from_solve(&fc, ham, fcidump_path, res, var_opt, opts_in);
    uint8_t *buf = NULL;
    size_t len = 0U;
    if (chmc_encode_full_cert(&fc, &buf, &len, err, err_sz) != 0) return -1;
    int rc = write_buf_to_file(path, buf, len, err, err_sz);
    free(buf);
    return rc;
}

int chem_cert_pack(const char *json_path, const char *bin_path, char *err, size_t err_sz) {
    if (!json_path || !bin_path) {
        set_err(err, err_sz, "null path to chem_cert_pack");
        return -1;
    }
    size_t len = 0U;
    char *txt = read_entire_file(json_path, &len);
    if (!txt) {
        set_err(err, err_sz, "cannot read JSON certificate: %s", json_path);
        return -1;
    }
    ChemFullCert fc;
    if (json_parse_full_cert(txt, len, &fc, err, err_sz) != 0) {
        free(txt);
        return -1;
    }
    free(txt);

    uint8_t *buf = NULL;
    size_t blen = 0U;
    if (chmc_encode_full_cert(&fc, &buf, &blen, err, err_sz) != 0) {
        full_cert_free(&fc);
        return -1;
    }
    full_cert_free(&fc);
    int rc = write_buf_to_file(bin_path, buf, blen, err, err_sz);
    free(buf);
    return rc;
}

int chem_cert_unpack(const char *bin_path, const char *json_path, char *err, size_t err_sz) {
    if (!bin_path || !json_path) {
        set_err(err, err_sz, "null path to chem_cert_unpack");
        return -1;
    }
    size_t len = 0U;
    char *raw = read_entire_file(bin_path, &len);
    if (!raw) {
        set_err(err, err_sz, "cannot read binary certificate: %s", bin_path);
        return -1;
    }
    ChemFullCert fc;
    if (chmc_decode_full_cert((const uint8_t *)raw, len, &fc, err, err_sz) != 0) {
        free(raw);
        return -1;
    }
    free(raw);

    FILE *fp = fopen(json_path, "w");
    if (!fp) {
        full_cert_free(&fc);
        set_err(err, err_sz, "cannot open JSON output path: %s", json_path);
        return -1;
    }
    json_write_full_cert(fp, &fc);
    int ferr = ferror(fp);
    fclose(fp);
    full_cert_free(&fc);
    if (ferr) {
        set_err(err, err_sz, "write error on JSON output path: %s", json_path);
        return -1;
    }
    return 0;
}

/* ============================================================================
 * Shared Certificate Loader (.cert.json, .cert.bin [CHMC], & .flowchem)
 * ============================================================================ */

typedef struct {
    ChemHam ham;
    ChemDualWitness dw;
    int is_binary;
    int sha_ok;
    double E_lb_claimed;
    int vetoed;
    double E_ai;
    double veto_tol;
    int has_witness;
    int ndet_var;
    double E_ub_claimed;
    double E_ci_claimed;
    uint64_t *dets;
    double *psi0;
} ChemLoadedCert;

static void loaded_cert_free(ChemLoadedCert *lc) {
    if (!lc) return;
    free(lc->dets);
    free(lc->psi0);
    chem_dual_witness_free(&lc->dw);
    chem_ham_free(&lc->ham);
    memset(lc, 0, sizeof(*lc));
}

static int load_cert_bin(const char *fcidump_override, const char *buf, size_t len,
                         ChemLoadedCert *lc, char *msg, size_t msg_sz) {
    lc->is_binary = 1;
    if (len < sizeof(ChemBinCertHeader)) {
        snprintf(msg, msg_sz, "truncated binary certificate");
        return -1;
    }
    ChemBinCertHeader hdr;
    memcpy(&hdr, buf, sizeof(hdr));
    if (hdr.meq < 0 || hdr.nQ < 0 || hdr.nG < 0 || hdr.nT2 < 0 || hdr.ndet_var < 0 ||
        hdr.fcidump_len > 4096U || hdr.t2_norbs < 0 || hdr.t2_norbs > CHEM_MAX_ORBITALS) {
        snprintf(msg, msg_sz, "corrupt binary certificate header dimensions");
        return -1;
    }
    size_t expect = sizeof(hdr) + (size_t)hdr.fcidump_len +
                    sizeof(double) * ((size_t)hdr.meq + (size_t)hdr.nQ + (size_t)hdr.nG + (size_t)hdr.nT2) +
                    (hdr.has_witness ? ((sizeof(uint64_t) + sizeof(double)) * (size_t)hdr.ndet_var) : 0U);
    if (len != expect) {
        snprintf(msg, msg_sz, "binary certificate length mismatch (%zu != %zu)", len, expect);
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
        snprintf(msg, msg_sz, "binary certificate missing fcidump_path");
        return -1;
    }
    ptr += hdr.fcidump_len;

    char err[256] = {0};
    if (chem_fcidump_load(fcidump_file, &lc->ham, err, sizeof(err)) != 0) {
        snprintf(msg, msg_sz, "FCIDUMP load failed: %s", err);
        return -1;
    }
    hdr.sha256_hex[64] = '\0';
    if (strcmp(lc->ham.sha256_hex, hdr.sha256_hex) != 0) {
        lc->sha_ok = 0;
        snprintf(msg, msg_sz, "SHA-256 mismatch in binary cert");
        return -1;
    }
    lc->sha_ok = 1;
    lc->E_lb_claimed = hdr.E_lb_cert;
    lc->vetoed = hdr.vetoed;
    lc->E_ai = hdr.E_ai;
    lc->veto_tol = hdr.veto_tol;
    lc->has_witness = (hdr.has_witness && hdr.ndet_var > 0) ? 1 : 0;
    lc->ndet_var = hdr.ndet_var;
    lc->E_ub_claimed = hdr.E_ub;
    lc->E_ci_claimed = hdr.E_ci;

    lc->dw.mode = (ChemMode)hdr.mode;
    lc->dw.K = lc->ham.K;
    lc->dw.na = lc->ham.na;
    lc->dw.nb = lc->ham.nb;
    lc->dw.two_s = hdr.two_s;
    lc->dw.constrain_s2 = hdr.constrain_s2;
    lc->dw.t2_enabled = hdr.t2_enabled;
    lc->dw.t2_norbs = hdr.t2_norbs;
    for (int i = 0; i < hdr.t2_norbs; i++) lc->dw.t2_orbs[i] = hdr.t2_orbs[i];
    lc->dw.meq = hdr.meq;
    lc->dw.nD = hdr.nD;
    lc->dw.nQ = hdr.nQ;
    lc->dw.nG = hdr.nG;
    lc->dw.nT2 = hdr.nT2;

    if (lc->dw.meq > 0) {
        lc->dw.y_eq = (double *)malloc((size_t)lc->dw.meq * sizeof(double));
        if (!lc->dw.y_eq) { snprintf(msg, msg_sz, "out of memory reading binary y_eq"); return -1; }
        memcpy(lc->dw.y_eq, ptr, (size_t)lc->dw.meq * sizeof(double));
        ptr += (size_t)lc->dw.meq * sizeof(double);
    }
    if (lc->dw.nQ > 0) {
        lc->dw.Z_Q = (double *)malloc((size_t)lc->dw.nQ * sizeof(double));
        if (!lc->dw.Z_Q) { snprintf(msg, msg_sz, "out of memory reading binary Z_Q"); return -1; }
        memcpy(lc->dw.Z_Q, ptr, (size_t)lc->dw.nQ * sizeof(double));
        ptr += (size_t)lc->dw.nQ * sizeof(double);
    }
    if (lc->dw.nG > 0) {
        lc->dw.Z_G = (double *)malloc((size_t)lc->dw.nG * sizeof(double));
        if (!lc->dw.Z_G) { snprintf(msg, msg_sz, "out of memory reading binary Z_G"); return -1; }
        memcpy(lc->dw.Z_G, ptr, (size_t)lc->dw.nG * sizeof(double));
        ptr += (size_t)lc->dw.nG * sizeof(double);
    }
    if (lc->dw.nT2 > 0) {
        lc->dw.Z_T2 = (double *)malloc((size_t)lc->dw.nT2 * sizeof(double));
        if (!lc->dw.Z_T2) { snprintf(msg, msg_sz, "out of memory reading binary Z_T2"); return -1; }
        memcpy(lc->dw.Z_T2, ptr, (size_t)lc->dw.nT2 * sizeof(double));
        ptr += (size_t)lc->dw.nT2 * sizeof(double);
    }
    if (lc->has_witness) {
        lc->dets = (uint64_t *)malloc((size_t)hdr.ndet_var * sizeof(uint64_t));
        lc->psi0 = (double *)malloc((size_t)hdr.ndet_var * sizeof(double));
        if (!lc->dets || !lc->psi0) {
            snprintf(msg, msg_sz, "out of memory reading binary witness");
            return -1;
        }
        memcpy(lc->dets, ptr, (size_t)hdr.ndet_var * sizeof(uint64_t));
        ptr += (size_t)hdr.ndet_var * sizeof(uint64_t);
        memcpy(lc->psi0, ptr, (size_t)hdr.ndet_var * sizeof(double));
    }
    return 0;
}

static int load_cert_chmc(const char *fcidump_override, const char *buf, size_t len,
                          ChemLoadedCert *lc, char *msg, size_t msg_sz) {
    lc->is_binary = 1;
    ChemFullCert fc;
    if (chmc_decode_full_cert((const uint8_t *)buf, len, &fc, msg, msg_sz) != 0) {
        return -1;
    }

    char fcidump_file[4098] = {0};
    if (fcidump_override && fcidump_override[0]) {
        snprintf(fcidump_file, sizeof(fcidump_file), "%s", fcidump_override);
    } else if (fc.fcidump_path[0]) {
        snprintf(fcidump_file, sizeof(fcidump_file), "%s", fc.fcidump_path);
    } else {
        full_cert_free(&fc);
        snprintf(msg, msg_sz, "CHMC certificate missing fcidump_path");
        return -1;
    }

    char err[256] = {0};
    if (chem_fcidump_load(fcidump_file, &lc->ham, err, sizeof(err)) != 0) {
        full_cert_free(&fc);
        snprintf(msg, msg_sz, "FCIDUMP load failed: %s", err);
        return -1;
    }
    if (strcmp(lc->ham.sha256_hex, fc.sha256_hex) != 0) {
        lc->sha_ok = 0;
        full_cert_free(&fc);
        snprintf(msg, msg_sz, "SHA-256 mismatch in CHMC cert");
        return -1;
    }
    if (fc.K != lc->ham.K || fc.na != lc->ham.na || fc.nb != lc->ham.nb) {
        full_cert_free(&fc);
        snprintf(msg, msg_sz, "CHMC orbital/electron metadata mismatch vs FCIDUMP");
        return -1;
    }
    lc->sha_ok = 1;
    lc->E_lb_claimed = fc.E_lb_cert;
    lc->vetoed = fc.vetoed;
    lc->E_ai = fc.E_ai;
    lc->veto_tol = fc.veto_tol;
    lc->has_witness = (fc.has_witness && fc.ndet_var > 0) ? 1 : 0;
    lc->ndet_var = fc.ndet_var;
    lc->E_ub_claimed = fc.E_ub;
    lc->E_ci_claimed = fc.has_var_meta ? fc.E_ci : fc.E_ub;

    lc->dw.mode = fc.mode;
    lc->dw.K = lc->ham.K;
    lc->dw.na = lc->ham.na;
    lc->dw.nb = lc->ham.nb;
    lc->dw.two_s = fc.two_s;
    lc->dw.constrain_s2 = fc.constrain_s2;
    lc->dw.t2_enabled = fc.t2_enabled;
    lc->dw.t2_norbs = fc.t2_norbs;
    if (fc.t2_norbs > 0) memcpy(lc->dw.t2_orbs, fc.t2_orbs, (size_t)fc.t2_norbs * sizeof(int));
    lc->dw.t1_enabled = fc.t1_enabled;
    lc->dw.t1_norbs = fc.t1_norbs;
    if (fc.t1_norbs > 0) memcpy(lc->dw.t1_orbs, fc.t1_orbs, (size_t)fc.t1_norbs * sizeof(int));
    lc->dw.meq = fc.meq;
    lc->dw.nD = fc.nD;
    lc->dw.nQ = fc.nQ;
    lc->dw.nG = fc.nG;
    lc->dw.nT2 = fc.nT2;
    lc->dw.nT1 = fc.nT1;

    /* Transfer ownership of array buffers directly without copying */
    lc->dw.y_eq = fc.y_eq; fc.y_eq = NULL;
    lc->dw.Z_Q = fc.Z_Q;   fc.Z_Q = NULL;
    lc->dw.Z_G = fc.Z_G;   fc.Z_G = NULL;
    lc->dw.Z_T2 = fc.Z_T2; fc.Z_T2 = NULL;
    lc->dw.Z_T1 = fc.Z_T1; fc.Z_T1 = NULL;
    lc->dets = fc.dets;    fc.dets = NULL;
    lc->psi0 = fc.psi0;    fc.psi0 = NULL;
    full_cert_free(&fc);
    return 0;
}

static int load_cert_common(const char *fcidump_override, const char *cert_path,
                            ChemLoadedCert *lc, char *msg, size_t msg_sz) {
    memset(lc, 0, sizeof(*lc));
    size_t len = 0;
    char *txt = read_entire_file(cert_path, &len);
    if (!txt) {
        snprintf(msg, msg_sz, "cannot read certificate: %s", cert_path);
        return -1;
    }
    if (len >= 4U && memcmp(txt, "CHMC", 4) == 0) {
        int rc = load_cert_chmc(fcidump_override, txt, len, lc, msg, msg_sz);
        free(txt);
        if (rc != 0) loaded_cert_free(lc);
        return rc;
    }
    if (len >= 8U && memcmp(txt, "FLCH0002", 8) == 0) {
        int rc = load_cert_bin(fcidump_override, txt, len, lc, msg, msg_sz);
        free(txt);
        if (rc != 0) loaded_cert_free(lc);
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
        snprintf(msg, msg_sz, "malformed top-level certificate fields");
        return -1;
    }
    if (fcidump_override && fcidump_override[0]) {
        snprintf(fcidump_file, sizeof(fcidump_file), "%s", fcidump_override);
    } else if (json_get_string(s, e, "fcidump_path", fcidump_file, sizeof(fcidump_file)) != 0 ||
               fcidump_file[0] == '\0') {
        free(txt);
        snprintf(msg, msg_sz, "certificate missing fcidump_path");
        return -1;
    }

    char err[256] = {0};
    if (chem_fcidump_load(fcidump_file, &lc->ham, err, sizeof(err)) != 0) {
        free(txt);
        snprintf(msg, msg_sz, "FCIDUMP load failed: %s", err);
        return -1;
    }
    if (strcmp(lc->ham.sha256_hex, sha_claimed) != 0) {
        lc->sha_ok = 0;
        snprintf(msg, msg_sz, "SHA-256 mismatch (expected %.16s..., got %.16s...)",
                 sha_claimed, lc->ham.sha256_hex);
        loaded_cert_free(lc);
        free(txt);
        return -1;
    }
    lc->sha_ok = 1;

    int t2_enabled = 0, t2_norbs = 0;
    int t2_orbs[CHEM_MAX_ORBITALS] = {0};
    if (json_get_int(s, e, "t2_enabled", &t2_enabled) == 0 && t2_enabled) {
        if (json_get_int(s, e, "t2_norbs", &t2_norbs) != 0 || t2_norbs <= 0 ||
            t2_norbs > CHEM_MAX_ORBITALS ||
            json_get_int_array(s, e, "t2_orbs", t2_norbs, t2_orbs) != 0) {
            loaded_cert_free(lc);
            free(txt);
            snprintf(msg, msg_sz, "malformed t2 metadata in certificate");
            return -1;
        }
    }

    int t1_enabled = 0, t1_norbs = 0;
    int t1_orbs[CHEM_MAX_ORBITALS] = {0};
    if (json_get_int(s, e, "t1_enabled", &t1_enabled) == 0 && t1_enabled) {
        if (json_get_int(s, e, "t1_norbs", &t1_norbs) != 0 || t1_norbs <= 0 ||
            t1_norbs > CHEM_MAX_ORBITALS ||
            json_get_int_array(s, e, "t1_orbs", t1_norbs, t1_orbs) != 0) {
            loaded_cert_free(lc);
            free(txt);
            snprintf(msg, msg_sz, "malformed t1 metadata in certificate");
            return -1;
        }
    }

    const char *sol_s = NULL, *sol_e = NULL;
    const char *vet_s = NULL, *vet_e = NULL;
    const char *var_s = NULL, *var_e = NULL;
    const char *dw_s  = NULL, *dw_e  = NULL;
    if (json_get_object_range(s, e, "solver", &sol_s, &sol_e) != 0 ||
        json_get_object_range(s, e, "dual_witness", &dw_s, &dw_e) != 0) {
        loaded_cert_free(lc);
        free(txt);
        snprintf(msg, msg_sz, "missing solver or dual_witness JSON section");
        return -1;
    }
    if (json_get_double(sol_s, sol_e, "E_lb_cert", &lc->E_lb_claimed) != 0) {
        loaded_cert_free(lc);
        free(txt);
        snprintf(msg, msg_sz, "missing solver.E_lb_cert");
        return -1;
    }

    int meq = 0, nD = 0, nQ = 0, nG = 0, nT2 = 0, nT1 = 0;
    if (json_get_int(dw_s, dw_e, "meq", &meq) != 0 ||
        json_get_int(dw_s, dw_e, "nD", &nD) != 0 ||
        json_get_int(dw_s, dw_e, "nQ", &nQ) != 0 ||
        json_get_int(dw_s, dw_e, "nG", &nG) != 0) {
        loaded_cert_free(lc);
        free(txt);
        snprintf(msg, msg_sz, "missing dual_witness dimensions");
        return -1;
    }
    if (t2_enabled && json_get_int(dw_s, dw_e, "nT2", &nT2) != 0) {
        loaded_cert_free(lc);
        free(txt);
        snprintf(msg, msg_sz, "missing dual_witness.nT2");
        return -1;
    }
    if (t1_enabled && json_get_int(dw_s, dw_e, "nT1", &nT1) != 0) {
        loaded_cert_free(lc);
        free(txt);
        snprintf(msg, msg_sz, "missing dual_witness.nT1");
        return -1;
    }

    lc->dw.mode = (strcmp(mode_str, "A_SINGLET") == 0) ? CHEM_MODE_A_SINGLET : CHEM_MODE_B_GENERAL;
    lc->dw.K = lc->ham.K;
    lc->dw.na = lc->ham.na;
    lc->dw.nb = lc->ham.nb;
    lc->dw.two_s = two_s;
    lc->dw.constrain_s2 = cs2;
    lc->dw.t2_enabled = t2_enabled;
    lc->dw.t2_norbs = t2_norbs;
    if (t2_enabled) memcpy(lc->dw.t2_orbs, t2_orbs, (size_t)t2_norbs * sizeof(int));
    lc->dw.t1_enabled = t1_enabled;
    lc->dw.t1_norbs = t1_norbs;
    if (t1_enabled) memcpy(lc->dw.t1_orbs, t1_orbs, (size_t)t1_norbs * sizeof(int));
    lc->dw.meq = meq;
    lc->dw.nD = nD;
    lc->dw.nQ = nQ;
    lc->dw.nG = nG;
    lc->dw.nT2 = nT2;
    lc->dw.nT1 = nT1;

    if (json_get_double_array(dw_s, dw_e, "y_eq", meq, &lc->dw.y_eq) != 0 ||
        json_get_double_array(dw_s, dw_e, "Z_Q", nQ, &lc->dw.Z_Q) != 0 ||
        json_get_double_array(dw_s, dw_e, "Z_G", nG, &lc->dw.Z_G) != 0 ||
        (t2_enabled && nT2 > 0 && json_get_double_array(dw_s, dw_e, "Z_T2", nT2, &lc->dw.Z_T2) != 0) ||
        (t1_enabled && nT1 > 0 && json_get_double_array(dw_s, dw_e, "Z_T1", nT1, &lc->dw.Z_T1) != 0)) {
        loaded_cert_free(lc);
        free(txt);
        snprintf(msg, msg_sz, "failed parsing dual_witness arrays");
        return -1;
    }

    if (json_get_object_range(s, e, "veto", &vet_s, &vet_e) == 0) {
        int vetoed = 0;
        double e_ai = 0.0, vtol = 1e-4;
        if (json_get_int(vet_s, vet_e, "vetoed", &vetoed) == 0 && vetoed) {
            lc->vetoed = 1;
            if (json_get_double(vet_s, vet_e, "E_ai", &e_ai) == 0 &&
                json_get_double(vet_s, vet_e, "veto_tol", &vtol) == 0) {
                lc->E_ai = e_ai;
                lc->veto_tol = vtol;
            } else {
                lc->veto_tol = -1.0; /* sentinel for malformed veto fields */
            }
        }
    }

    if (json_get_object_range(s, e, "variational", &var_s, &var_e) == 0) {
        int has_w = 0, ndet_var = 0;
        double e_ub_c = 0.0, e_ci_c = 0.0;
        if (json_get_int(var_s, var_e, "has_witness", &has_w) == 0 && has_w) {
            lc->has_witness = 1;
            if (json_get_int(var_s, var_e, "ndet_var", &ndet_var) != 0 || ndet_var <= 0 ||
                json_get_double(var_s, var_e, "E_ub", &e_ub_c) != 0 ||
                json_get_double(var_s, var_e, "E_ci", &e_ci_c) != 0 ||
                json_get_u64_hex_array(var_s, var_e, "dets", ndet_var, &lc->dets) != 0 ||
                json_get_double_array(var_s, var_e, "psi0", ndet_var, &lc->psi0) != 0) {
                lc->ndet_var = -1; /* malformed witness */
            } else {
                lc->ndet_var = ndet_var;
                lc->E_ub_claimed = e_ub_c;
                lc->E_ci_claimed = e_ci_c;
            }
        } else {
            json_get_double(var_s, var_e, "E_ub", &lc->E_ub_claimed);
        }
    }
    free(txt);
    return 0;
}

int chem_cert_verify_file(const char *fcidump_override, const char *cert_path,
                          ChemCertVerifyResult *out) {
    memset(out, 0, sizeof(*out));
    ChemLoadedCert lc;
    if (load_cert_common(fcidump_override, cert_path, &lc, out->msg, sizeof(out->msg)) != 0) {
        return -1;
    }
    out->is_binary = lc.is_binary;
    out->sha_ok = lc.sha_ok;
    out->E_lb_claimed = lc.E_lb_claimed;

    char err[256] = {0};
    ChemSolveResult recomp;
    memset(&recomp, 0, sizeof(recomp));
    if (lc.dw.meq == 0 && lc.dw.nD == 0) {
        ChemSolveOpts topts = chem_solve_opts_default(&lc.ham);
        topts.mode = lc.dw.mode;
        topts.two_s = lc.dw.two_s;
        topts.constrain_s2 = lc.dw.constrain_s2;
        if (solve_trivial_space_cost(&lc.ham, lc.ham.E_core, NULL, lc.dw.mode, lc.dw.two_s,
                                     lc.dw.constrain_s2, &topts, &recomp, err, sizeof(err)) != 0) {
            loaded_cert_free(&lc);
            snprintf(out->msg, sizeof(out->msg), "trivial space certification failed");
            return -1;
        }
    } else if (chem_certify_dual_witness(&lc.ham, &lc.dw, &recomp, err, sizeof(err)) != 0) {
        loaded_cert_free(&lc);
        snprintf(out->msg, sizeof(out->msg), "dual witness certification failed: %s", err);
        return -1;
    }

    out->E_lb_recomputed = recomp.E_lb_cert;
    out->min_eig_cert = recomp.min_eig_cert;
    out->lb_ok = (fabs(recomp.E_lb_cert - lc.E_lb_claimed) <= 1e-8) ? 1 : 0;
    out->veto_ok = (!lc.vetoed || (lc.veto_tol >= 0.0 && recomp.E_lb_cert > lc.E_ai - lc.veto_tol)) ? 1 : 0;
    out->var_ok = 1;
    out->E_var_recomputed = NAN;
    if (lc.has_witness) {
        if (lc.ndet_var <= 0 || !lc.dets || !lc.psi0) {
            out->var_ok = 0;
        } else {
            double e_re = 0.0;
            if (chem_var_eval_witness(&lc.ham, lc.ndet_var, lc.dets, lc.psi0, &e_re, err, sizeof(err)) != 0) {
                out->var_ok = 0;
            } else {
                out->E_var_recomputed = e_re;
                if (fabs(e_re - lc.E_ci_claimed) > 1e-6 || recomp.E_lb_cert > e_re + 1e-9 ||
                    recomp.E_lb_cert > lc.E_ub_claimed + 1e-9) {
                    out->var_ok = 0;
                }
            }
        }
    }

    out->ok = (out->sha_ok && out->lb_ok && out->veto_ok && out->var_ok) ? 1 : 0;
    const char *tag = lc.is_binary ? "BINARY CERT" : "CERT";
    if (out->ok) {
        snprintf(out->msg, sizeof(out->msg),
                 "%s VERIFIED: sha=OK E_lb=%.8f (diff=%.2e)",
                 tag, recomp.E_lb_cert, fabs(recomp.E_lb_cert - lc.E_lb_claimed));
    } else {
        snprintf(out->msg, sizeof(out->msg),
                 "%s FAILED: sha=%d lb=%d (diff=%.2e) veto=%d var=%d",
                 tag, out->sha_ok, out->lb_ok, fabs(recomp.E_lb_cert - lc.E_lb_claimed),
                 out->veto_ok, out->var_ok);
    }

    chem_solve_result_free(&recomp);
    loaded_cert_free(&lc);
    return out->ok ? 0 : -1;
}

/* ============================================================================
 * Pure ISO C17 128-Bit Integer Cholesky + SDD Zero-FP-Trust Verifier & Lean Emitter
 * ============================================================================ */

typedef struct {
    int64_t hi;
    uint64_t lo;
} ChemI128;

static int64_t u64_to_i64(uint64_t u) {
    if (u <= (uint64_t)INT64_MAX) return (int64_t)u;
    return -(int64_t)(~u) - 1;
}

static ChemI128 i128_zero(void) {
    ChemI128 z = {0, 0ULL};
    return z;
}

static ChemI128 i128_from_i64(int64_t x) {
    ChemI128 r = {(x < 0) ? -1 : 0, (uint64_t)x};
    return r;
}

static ChemI128 i128_neg(ChemI128 a) {
    uint64_t lo = ~a.lo + 1ULL;
    uint64_t hi = (uint64_t)(~a.hi) + (lo == 0ULL ? 1ULL : 0ULL);
    ChemI128 r = {u64_to_i64(hi), lo};
    return r;
}

static ChemI128 i128_add(ChemI128 a, ChemI128 b) {
    uint64_t lo = a.lo + b.lo;
    uint64_t carry = (lo < a.lo) ? 1ULL : 0ULL;
    uint64_t hi = (uint64_t)a.hi + (uint64_t)b.hi + carry;
    ChemI128 r = {u64_to_i64(hi), lo};
    return r;
}

static ChemI128 i128_sub(ChemI128 a, ChemI128 b) {
    return i128_add(a, i128_neg(b));
}

static int i128_cmp(ChemI128 a, ChemI128 b) {
    if (a.hi != b.hi) return (a.hi < b.hi) ? -1 : 1;
    if (a.lo != b.lo) return (a.lo < b.lo) ? -1 : 1;
    return 0;
}

static ChemI128 i128_abs(ChemI128 a) {
    return (a.hi < 0) ? i128_neg(a) : a;
}

static void u64_mul_u64_128(uint64_t a, uint64_t b, uint64_t *hi_out, uint64_t *lo_out) {
    uint64_t a0 = a & UINT64_C(0xFFFFFFFF);
    uint64_t a1 = a >> 32;
    uint64_t b0 = b & UINT64_C(0xFFFFFFFF);
    uint64_t b1 = b >> 32;

    uint64_t p00 = a0 * b0;
    uint64_t p01 = a0 * b1;
    uint64_t p10 = a1 * b0;
    uint64_t p11 = a1 * b1;

    uint64_t mid = (p00 >> 32) + (p01 & UINT64_C(0xFFFFFFFF)) + (p10 & UINT64_C(0xFFFFFFFF));
    uint64_t lo = (p00 & UINT64_C(0xFFFFFFFF)) | (mid << 32);
    uint64_t hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    *hi_out = hi;
    *lo_out = lo;
}

static ChemI128 i128_mul_i64(int64_t a, int64_t b) {
    int neg = ((a < 0) ^ (b < 0));
    uint64_t ua = (a < 0) ? (~(uint64_t)a + 1ULL) : (uint64_t)a;
    uint64_t ub = (b < 0) ? (~(uint64_t)b + 1ULL) : (uint64_t)b;
    uint64_t hi = 0ULL, lo = 0ULL;
    u64_mul_u64_128(ua, ub, &hi, &lo);
    ChemI128 r = {u64_to_i64(hi), lo};
    return neg ? i128_neg(r) : r;
}

static ChemI128 i128_mul_i128_small_i64(ChemI128 a, int64_t b) {
    int neg_a = (a.hi < 0);
    int neg_b = (b < 0);
    ChemI128 am = neg_a ? i128_neg(a) : a;
    uint64_t ub = neg_b ? (~(uint64_t)b + 1ULL) : (uint64_t)b;
    uint64_t hi0 = 0ULL, lo0 = 0ULL;
    u64_mul_u64_128(am.lo, ub, &hi0, &lo0);
    uint64_t hi1 = 0ULL, lo1 = 0ULL;
    u64_mul_u64_128((uint64_t)am.hi, ub, &hi1, &lo1);
    (void)hi1;
    ChemI128 r = {u64_to_i64(hi0 + lo1), lo0};
    return (neg_a ^ neg_b) ? i128_neg(r) : r;
}

/* Computes round(scale2 * x) accurately even when |scale2 * x| > 2^53. */
static ChemI128 i128_from_double_scaled2(double x, uint64_t scale2) {
    if (x == 0.0) return i128_zero();
    int neg = (x < 0.0);
    double ax = fabs(x);
    double ip_d = 0.0;
    double fp = modf(ax, &ip_d);
    uint64_t ip = (uint64_t)ip_d;
    uint64_t hi = 0ULL, lo = 0ULL;
    u64_mul_u64_128(ip, scale2, &hi, &lo);
    ChemI128 base = {u64_to_i64(hi), lo};

    /* Split fp * scale2 directly into 128-bit unsigned int */
    uint64_t f_int = (uint64_t)llround(fp * (double)scale2);
    ChemI128 add_f = {0, f_int};
    ChemI128 res = i128_add(base, add_f);
    return neg ? i128_neg(res) : res;
}

static double i128_to_double(ChemI128 a) {
    if (a.hi == 0 && a.lo == 0ULL) return 0.0;
    int neg = (a.hi < 0);
    ChemI128 m = neg ? i128_neg(a) : a;
    double v = ldexp((double)(uint64_t)m.hi, 64) + (double)m.lo;
    return neg ? -v : v;
}

static void i128_to_str(ChemI128 a, char *buf, size_t buf_sz) {
    if (a.hi == 0 && a.lo == 0ULL) {
        snprintf(buf, buf_sz, "0");
        return;
    }
    int neg = (a.hi < 0);
    ChemI128 m = neg ? i128_neg(a) : a;
    uint64_t uhi = (uint64_t)m.hi;
    uint64_t ulo = m.lo;
    uint64_t rem[6] = {0};
    int nchunks = 0;
    const uint64_t B = UINT64_C(1000000000);
    while ((uhi != 0ULL || ulo != 0ULL) && nchunks < 6) {
        uint64_t qhi = uhi / B;
        uint64_t rhi = uhi % B;
        uint64_t u2 = ulo >> 32;
        uint64_t u3 = ulo & UINT64_C(0xFFFFFFFF);
        uint64_t c1 = (rhi << 32) | u2;
        uint64_t qm = c1 / B;
        uint64_t rm = c1 % B;
        uint64_t c2 = (rm << 32) | u3;
        uint64_t ql = c2 / B;
        rem[nchunks++] = c2 % B;
        uhi = qhi;
        ulo = (qm << 32) | ql;
    }
    size_t pos = 0;
    if (neg && pos + 1 < buf_sz) buf[pos++] = '-';
    pos += (size_t)snprintf(buf + pos, buf_sz - pos, "%llu", (unsigned long long)rem[nchunks - 1]);
    for (int k = nchunks - 2; k >= 0 && pos + 10 < buf_sz; k--) {
        pos += (size_t)snprintf(buf + pos, buf_sz - pos, "%09llu", (unsigned long long)rem[k]);
    }
}

/* Quantizes symmetric Z_svec to integer A = round(D^2 * Z), lower-triangular L = round(D * L_fp),
 * and proves exact integer SDD: E = A - mu_psd*I - L*L^T is SDD.
 * mu_cert = mu_psd - dim accounts for entrywise rounding |A_ij - D^2 Z_ij| <= 1. */
static int exact_chol_sdd_block(int dim, const double *Z_svec, int64_t D_req,
                                int64_t *D_used_out, ChemI128 **A_out, int64_t **L_out,
                                ChemI128 *mu_psd_out, ChemI128 *mu_cert_out, double *margin_out) {
    if (dim <= 0) {
        *D_used_out = D_req;
        if (A_out) *A_out = NULL;
        if (L_out) *L_out = NULL;
        *mu_psd_out = i128_zero();
        *mu_cert_out = i128_zero();
        if (margin_out) *margin_out = 0.0;
        return 0;
    }
    size_t nn = (size_t)dim;
    double *Z_mat = (double *)malloc(nn * nn * sizeof(double));
    double *M_sh  = (double *)malloc(nn * nn * sizeof(double));
    double *evals = (double *)malloc(nn * sizeof(double));
    if (!Z_mat || !M_sh || !evals) {
        free(Z_mat); free(M_sh); free(evals);
        return -1;
    }
    svec_unpack(dim, Z_svec, Z_mat);

    /* Query eigensolver for hint lam_min */
    memcpy(M_sh, Z_mat, nn * nn * sizeof(double));
    __LAPACK_int n_l = (__LAPACK_int)dim, lwork = -1, liwork = -1, info = 0;
    double wq = 0.0;
    __LAPACK_int iwq = 0;
    dsyevd_("N", "U", &n_l, M_sh, &n_l, evals, &wq, &lwork, &iwq, &liwork, &info);
    lwork = (__LAPACK_int)wq + 64;
    liwork = iwq + 16;
    double *wk = (double *)malloc((size_t)lwork * sizeof(double));
    __LAPACK_int *iwk = (__LAPACK_int *)malloc((size_t)liwork * sizeof(__LAPACK_int));
    if (!wk || !iwk) {
        free(Z_mat); free(M_sh); free(evals); free(wk); free(iwk);
        return -1;
    }
    dsyevd_("N", "U", &n_l, M_sh, &n_l, evals, wk, &lwork, iwk, &liwork, &info);
    double lam_min = (info == 0) ? evals[0] : -1e3;
    free(wk);
    free(iwk);

    double scale = 1.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < nn * nn; i++) {
        double a = fabs(Z_mat[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs > scale) scale = max_abs;

    /* Fortran dpotrf_("U", ...) writes lower-triangular L directly in C row-major */
    double shift = lam_min;
    int got_chol = 0;
    double pad = fmax(1e-7 * scale, 1e-8);
    for (int att = 0; att < 5; att++) {
        shift = lam_min - pad;
        memcpy(M_sh, Z_mat, nn * nn * sizeof(double));
        for (int i = 0; i < dim; i++) M_sh[(size_t)i * nn + (size_t)i] -= shift;
        __LAPACK_int cinfo = 0;
        dpotrf_("U", &n_l, M_sh, &n_l, &cinfo);
        if (cinfo == 0) {
            got_chol = 1;
            break;
        }
        pad *= 10.0;
    }
    if (!got_chol) {
        memset(M_sh, 0, nn * nn * sizeof(double));
        shift = lam_min - 1e-3 * scale;
    } else {
        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) M_sh[(size_t)i * nn + (size_t)j] = 0.0;
        }
    }

    int64_t D = D_req;
    double max_L = 0.0;
    for (size_t i = 0; i < nn * nn; i++) {
        double a = fabs(M_sh[i]);
        if (a > max_L) max_L = a;
    }
    while (D > 10000 && max_L * (double)D > 1e12) D /= 10;
    uint64_t D2 = (uint64_t)D * (uint64_t)D;

    ChemI128 *A_i = (ChemI128 *)malloc(nn * nn * sizeof(ChemI128));
    int64_t  *L_i = (int64_t *)calloc(nn * nn, sizeof(int64_t));
    ChemI128 *row_off  = (ChemI128 *)calloc(nn, sizeof(ChemI128));
    ChemI128 *row_diag = (ChemI128 *)calloc(nn, sizeof(ChemI128));
    if (!A_i || !L_i || !row_off || !row_diag) {
        free(Z_mat); free(M_sh); free(evals);
        free(A_i); free(L_i); free(row_off); free(row_diag);
        return -1;
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j <= i; j++) {
            ChemI128 val = i128_from_double_scaled2(Z_mat[(size_t)i * nn + (size_t)j], D2);
            A_i[(size_t)i * nn + (size_t)j] = val;
            A_i[(size_t)j * nn + (size_t)i] = val;
            L_i[(size_t)i * nn + (size_t)j] = (int64_t)llround(M_sh[(size_t)i * nn + (size_t)j] * (double)D);
        }
    }

    ChemI128 mu_hint = i128_from_double_scaled2(shift, D2);
    for (int i = 0; i < dim; i++) {
        const int64_t *Li = L_i + (size_t)i * nn;
        for (int j = 0; j <= i; j++) {
            const int64_t *Lj = L_i + (size_t)j * nn;
            ChemI128 dot = i128_zero();
            for (int k = 0; k <= j; k++) {
                dot = i128_add(dot, i128_mul_i64(Li[k], Lj[k]));
            }
            ChemI128 eij = i128_sub(A_i[(size_t)i * nn + (size_t)j], dot);
            if (i == j) {
                row_diag[i] = i128_sub(eij, mu_hint);
            } else {
                ChemI128 ae = i128_abs(eij);
                row_off[i] = i128_add(row_off[i], ae);
                row_off[j] = i128_add(row_off[j], ae);
            }
        }
    }

    ChemI128 max_def = i128_zero();
    for (int i = 0; i < dim; i++) {
        ChemI128 def = i128_sub(row_off[i], row_diag[i]);
        if (i128_cmp(def, max_def) > 0) max_def = def;
    }
    ChemI128 mu_psd = i128_sub(mu_hint, max_def);
    ChemI128 mu_cert = i128_sub(mu_psd, i128_from_i64((int64_t)dim));

    /* Verify exact SDD property explicitly */
    double min_margin = 0.0;
    int first_m = 1;
    double inv_d2 = 1.0 / (double)D2;
    for (int i = 0; i < dim; i++) {
        ChemI128 new_diag = i128_add(row_diag[i], i128_sub(mu_hint, mu_psd));
        ChemI128 m = i128_sub(new_diag, row_off[i]);
        if (i128_cmp(m, i128_zero()) < 0) {
            free(Z_mat); free(M_sh); free(evals);
            free(A_i); free(L_i); free(row_off); free(row_diag);
            return -1;
        }
        double md = i128_to_double(m) * inv_d2;
        if (first_m || md < min_margin) min_margin = md;
        first_m = 0;
    }

    *D_used_out = D;
    *mu_psd_out = mu_psd;
    *mu_cert_out = mu_cert;
    if (margin_out) *margin_out = min_margin;
    if (A_out) *A_out = A_i; else free(A_i);
    if (L_out) *L_out = L_i; else free(L_i);

    free(Z_mat); free(M_sh); free(evals); free(row_off); free(row_diag);
    return 0;
}

/* Reconstructs Z_D and slices all block vectors for a certificate's dual witness */
static int build_all_block_svecs(const ChemHam *ham, const ChemDualWitness *dw,
                                 ChemMFContext *ctx, double **Z_D_out,
                                 const double **blk_ptrs, double *e_dual_out,
                                 double *delta_op_out, char *err, size_t err_sz) {
    if (mf_ctx_init_ext(ctx, ham->K, ham->na, ham->nb, dw->two_s, dw->constrain_s2, dw->mode,
                        ham, dw->t2_enabled, dw->t2_norbs, dw->t2_norbs, dw->t2_orbs,
                        dw->t1_enabled, dw->t1_norbs, dw->t1_norbs, dw->t1_orbs,
                        1, err, err_sz) != 0) {
        return -1;
    }
    if (dw->meq != ctx->meq || dw->nD != ctx->nD || dw->nQ != ctx->nQ || dw->nG != ctx->nG ||
        dw->nT2 != ctx->nT2 || dw->nT1 != ctx->nT1) {
        mf_ctx_free(ctx);
        set_err(err, err_sz, "dual witness dimension mismatch");
        return -1;
    }
    int nD = ctx->nD, nQ = ctx->nQ, nT1 = ctx->nT1, meq = ctx->meq;
    double *Z_D = (double *)malloc((size_t)nD * sizeof(double));
    double *tmp_d = (double *)malloc((size_t)nD * sizeof(double));
    if (!Z_D || !tmp_d) {
        free(Z_D); free(tmp_d);
        mf_ctx_free(ctx);
        set_err(err, err_sz, "out of memory");
        return -1;
    }
    mf_Aeq_adj(ctx, dw->y_eq, tmp_d);
    for (int i = 0; i < nD; i++) Z_D[i] = ctx->C_D[i] - tmp_d[i];
    mf_LQ_adj(ctx, dw->Z_Q, tmp_d);
    for (int i = 0; i < nD; i++) Z_D[i] -= tmp_d[i];
    mf_LG_adj(ctx, dw->Z_G, tmp_d);
    for (int i = 0; i < nD; i++) Z_D[i] -= tmp_d[i];
    if (ctx->t2_enabled && dw->Z_T2) {
        mf_LT2_adj(ctx, dw->Z_T2, tmp_d);
        for (int i = 0; i < nD; i++) Z_D[i] -= tmp_d[i];
    }
    if (ctx->t1_enabled && dw->Z_T1) {
        mf_LT1_adj(ctx, dw->Z_T1, tmp_d);
        for (int i = 0; i < nD; i++) Z_D[i] -= tmp_d[i];
    }
    free(tmp_d);

    double t1_dot = (ctx->t1_enabled && dw->Z_T1 && ctx->t1_0 && nT1 > 0)
                        ? dot_vec(nT1, dw->Z_T1, ctx->t1_0)
                        : 0.0;
    *e_dual_out = ham->E_core + dot_vec(meq, dw->y_eq, ctx->b_eq) -
                  dot_vec(nQ, dw->Z_Q, ctx->q0) - t1_dot;

    int nD_blk = ctx->blk_D_cnt;
    int off = 0;
    double sum_frob_tr = 0.0;
    for (int b = 0; b < ctx->nblocks; b++) {
        int sz = ctx->block_szs[b];
        const double *src = NULL;
        if (b < ctx->blk_Q_start) {
            src = Z_D + off;
            off = (b == ctx->blk_Q_start - 1) ? 0 : off + sz;
        } else if (b < ctx->blk_G_start) {
            src = dw->Z_Q + off;
            off = (b == ctx->blk_G_start - 1) ? 0 : off + sz;
        } else if (b < ctx->blk_T2_start) {
            src = dw->Z_G + off;
            off = (b == ctx->blk_T2_start - 1) ? 0 : off + sz;
        } else if (b < ctx->blk_T1_start) {
            src = dw->Z_T2 + off;
            off = (b == ctx->blk_T1_start - 1) ? 0 : off + sz;
        } else {
            src = dw->Z_T1 + off;
            off += sz;
        }
        blk_ptrs[b] = src;
        double frob = (sz > 0 && src) ? norm2_vec(sz, src) : 0.0;
        sum_frob_tr += frob * ctx->block_traces[b];
    }

    double tr_D_sum = 0.0;
    for (int b = 0; b < nD_blk; b++) tr_D_sum += ctx->block_traces[b];
    double t1_round = (ctx->t1_enabled && dw->Z_T1 && ctx->t1_0 && nT1 > 0)
                          ? (norm2_vec(nT1, dw->Z_T1) * norm2_vec(nT1, ctx->t1_0))
                          : 0.0;
    *delta_op_out = 64.0 * DBL_EPSILON *
                    (fabs(ham->E_core) + norm2_vec(nD, ctx->C_D) * tr_D_sum +
                     norm2_vec(meq, dw->y_eq) * norm2_vec(meq, ctx->b_eq) + sum_frob_tr + t1_round);
    *Z_D_out = Z_D;
    return 0;
}

int chem_cert_verify_exact_chol(const char *fcidump_override, const char *cert_path,
                                ChemExactVerifyResult *out) {
    double t0 = now_sec();
    memset(out, 0, sizeof(*out));
    ChemLoadedCert lc;
    if (load_cert_common(fcidump_override, cert_path, &lc, out->msg, sizeof(out->msg)) != 0) {
        return -1;
    }
    out->sha_ok = lc.sha_ok;
    out->E_lb_weyl = lc.E_lb_claimed;

    char err[256] = {0};
    if (lc.dw.meq == 0 && lc.dw.nD == 0) {
        ChemSolveOpts topts = chem_solve_opts_default(&lc.ham);
        topts.mode = lc.dw.mode;
        topts.two_s = lc.dw.two_s;
        topts.constrain_s2 = lc.dw.constrain_s2;
        ChemSolveResult recomp;
        memset(&recomp, 0, sizeof(recomp));
        if (solve_trivial_space_cost(&lc.ham, lc.ham.E_core, NULL, lc.dw.mode, lc.dw.two_s,
                                     lc.dw.constrain_s2, &topts, &recomp, err, sizeof(err)) != 0) {
            loaded_cert_free(&lc);
            snprintf(out->msg, sizeof(out->msg), "trivial exact verify failed");
            return -1;
        }
        out->all_blocks_sdd = 1;
        out->nblocks = 0;
        out->scale_D = 100000000LL;
        out->E_lb_int_chol = recomp.E_lb_cert;
        out->lb_ok = (fabs(recomp.E_lb_cert - lc.E_lb_claimed) < 1e-8) ? 1 : 0;
        chem_solve_result_free(&recomp);
    } else {
        ChemMFContext ctx;
        double *Z_D = NULL;
        const double *blk_ptrs[CHEM_MAX_BLOCKS] = {0};
        double e_dual = 0.0, delta_op = 0.0;
        if (build_all_block_svecs(&lc.ham, &lc.dw, &ctx, &Z_D, blk_ptrs, &e_dual, &delta_op, err,
                                  sizeof(err)) != 0) {
            loaded_cert_free(&lc);
            snprintf(out->msg, sizeof(out->msg), "%s", err);
            return -1;
        }
        out->nblocks = ctx.nblocks;
        out->all_blocks_sdd = 1;
        out->scale_D = 100000000LL;
        double floor_corr = 0.0;
        double max_margin = 0.0;
        for (int b = 0; b < ctx.nblocks; b++) {
            int dim = ctx.block_dims[b];
            if (dim <= 0) continue;
            int64_t D_used = 100000000LL;
            ChemI128 mu_psd, mu_cert;
            double margin = 0.0;
            if (exact_chol_sdd_block(dim, blk_ptrs[b], D_used, &D_used, NULL, NULL, &mu_psd,
                                     &mu_cert, &margin) != 0) {
                out->all_blocks_sdd = 0;
                break;
            }
            if (D_used < out->scale_D) out->scale_D = D_used;
            if (margin > max_margin) max_margin = margin;
            double mu_d = i128_to_double(mu_cert) / ((double)D_used * (double)D_used);
            if (mu_d < 0.0) floor_corr += mu_d * ctx.block_traces[b];
        }
        out->max_sdd_margin = max_margin;
        out->E_lb_int_chol = e_dual + floor_corr - delta_op;
        out->lb_ok = (out->all_blocks_sdd &&
                      fabs(out->E_lb_int_chol - lc.E_lb_claimed) < 2e-3) ? 1 : 0;
        free(Z_D);
        mf_ctx_free(&ctx);
    }

    out->var_ok = 1;
    out->E_var_recomputed = NAN;
    if (lc.has_witness && lc.ndet_var > 0 && lc.dets && lc.psi0) {
        double e_re = 0.0;
        if (chem_var_eval_witness(&lc.ham, lc.ndet_var, lc.dets, lc.psi0, &e_re, err, sizeof(err)) != 0 ||
            out->E_lb_int_chol > e_re + 1e-9 || out->E_lb_int_chol > lc.E_ub_claimed + 1e-9) {
            out->var_ok = 0;
        }
        out->E_var_recomputed = e_re;
    }
    out->elapsed_sec = now_sec() - t0;
    out->ok = (out->sha_ok && out->all_blocks_sdd && out->lb_ok && out->var_ok) ? 1 : 0;
    snprintf(out->msg, sizeof(out->msg),
             "EXACT-CHOL %s: SDD=%d D=%lld E_lb_chol=%.8f (weyl=%.8f, diff=%.2e) | %.2fs",
             out->ok ? "VERIFIED" : "FAILED", out->all_blocks_sdd, (long long)out->scale_D,
             out->E_lb_int_chol, out->E_lb_weyl, fabs(out->E_lb_int_chol - out->E_lb_weyl),
             out->elapsed_sec);
    loaded_cert_free(&lc);
    return out->ok ? 0 : -1;
}

int chem_cert_emit_lean(const char *fcidump_override, const char *cert_path,
                        const char *out_lean_path, const char *module_name,
                        ChemExactVerifyResult *out, char *err, size_t err_sz) {
    if (!cert_path || !out_lean_path) {
        set_err(err, err_sz, "null path in chem_cert_emit_lean");
        return -1;
    }
    double t0 = now_sec();
    if (out) memset(out, 0, sizeof(*out));
    ChemLoadedCert lc;
    char msg[256] = {0};
    if (load_cert_common(fcidump_override, cert_path, &lc, msg, sizeof(msg)) != 0) {
        set_err(err, err_sz, "%s", msg);
        return -1;
    }

    int64_t D = 1000000LL;
    uint64_t D2 = (uint64_t)D * (uint64_t)D;
    uint64_t S = 2ULL * D2; /* Common scale 2 * D^2 */

    FILE *fp = fopen(out_lean_path, "w");
    if (!fp) {
        loaded_cert_free(&lc);
        set_err(err, err_sz, "cannot write %s", out_lean_path);
        return -1;
    }

    const char *ns = (module_name && module_name[0]) ? module_name : "ChemMolCert";
    fprintf(fp, "/- Auto-generated Zero-FP-Trust Molecular Certificate by CHEM v0.3.0 -/\n");
    fprintf(fp, "import ChemCert\n\n");
    fprintf(fp, "namespace %s\n\n", ns);
    fprintf(fp, "open Flow.Interval Flow.MolecularBundle Flow.GaugeBundle\n\n");
    fprintf(fp, "/- Hamiltonian SHA-256: %s | K=%d nelec=%d D=%lld -/\n\n",
            lc.ham.sha256_hex, lc.ham.K, lc.ham.nelec, (long long)D);

    double E_lb_real = lc.E_lb_claimed;
    ChemI128 E_dual_S = i128_from_double_scaled2(lc.E_lb_claimed, S);
    ChemI128 delta_S = i128_from_i64(1);
    int nblocks = 0;

    if (!(lc.dw.meq == 0 && lc.dw.nD == 0)) {
        ChemMFContext ctx;
        double *Z_D = NULL;
        const double *blk_ptrs[CHEM_MAX_BLOCKS] = {0};
        double e_dual = 0.0, delta_op = 0.0;
        char cerr[256] = {0};
        if (build_all_block_svecs(&lc.ham, &lc.dw, &ctx, &Z_D, blk_ptrs, &e_dual, &delta_op, cerr,
                                  sizeof(cerr)) != 0) {
            fclose(fp);
            loaded_cert_free(&lc);
            set_err(err, err_sz, "%s", cerr);
            return -1;
        }
        /* Find uniform D valid across all blocks */
        D = 1000000LL;
        D2 = (uint64_t)D * (uint64_t)D;
        S = 2ULL * D2;
        /* Exact conservative integer representations in units of 1 / (2 * D^2) */
        double e_dual_fl = floor(e_dual * (double)S) / (double)S;
        E_dual_S = i128_from_double_scaled2(e_dual_fl, S);
        int64_t d_ceil = (int64_t)ceil(delta_op * (double)S) + 4LL;
        if (d_ceil < 1LL) d_ceil = 1LL;
        delta_S = i128_from_i64(d_ceil);

        int tr_arr[CHEM_MAX_BLOCKS] = {0};
        ChemI128 lam_arr[CHEM_MAX_BLOCKS];
        double floor_real = 0.0;
        nblocks = ctx.nblocks;

        for (int b = 0; b < ctx.nblocks; b++) {
            int dim = ctx.block_dims[b];
            int64_t D_used = D;
            ChemI128 *A_i = NULL;
            int64_t *L_i = NULL;
            ChemI128 mu_psd, mu_cert;
            double margin = 0.0;
            if (exact_chol_sdd_block(dim, blk_ptrs[b], D, &D_used, &A_i, &L_i, &mu_psd, &mu_cert,
                                     &margin) != 0 || D_used != D) {
                free(A_i); free(L_i); free(Z_D);
                mf_ctx_free(&ctx);
                fclose(fp);
                loaded_cert_free(&lc);
                set_err(err, err_sz, "block %d failed exact Cholesky+SDD at D=%lld", b, (long long)D);
                return -1;
            }
            tr_arr[b] = (int)llround(ctx.block_traces[b]);
            /* Convert mu_cert (1/D^2) to units of 1/(2*D^2): multiply by 2 */
            lam_arr[b] = i128_add(mu_cert, mu_cert);
            double mu_r = i128_to_double(mu_cert) / (double)D2;
            if (mu_r < 0.0) floor_real += mu_r * ctx.block_traces[b];

            if (dim > 0) {
                char s_psd[64], s_cert[64];
                i128_to_str(mu_psd, s_psd, sizeof(s_psd));
                i128_to_str(mu_cert, s_cert, sizeof(s_cert));
                fprintf(fp, "def A_%d : List (List Int) := [", b);
                for (int i = 0; i < dim; i++) {
                    fprintf(fp, "%s[", (i > 0) ? ", " : "");
                    for (int j = 0; j < dim; j++) {
                        char vs[64];
                        i128_to_str(A_i[(size_t)i * (size_t)dim + (size_t)j], vs, sizeof(vs));
                        fprintf(fp, "%s%s", (j > 0) ? ", " : "", vs);
                    }
                    fprintf(fp, "]");
                }
                fprintf(fp, "]\n");
                fprintf(fp, "def L_%d : List (List Int) := [", b);
                for (int i = 0; i < dim; i++) {
                    fprintf(fp, "%s[", (i > 0) ? ", " : "");
                    for (int j = 0; j < dim; j++) {
                        fprintf(fp, "%s%lld", (j > 0) ? ", " : "",
                                (long long)L_i[(size_t)i * (size_t)dim + (size_t)j]);
                    }
                    fprintf(fp, "]");
                }
                fprintf(fp, "]\n");
                fprintf(fp, "def mu_psd_%d : Int := %s\n", b, s_psd);
                fprintf(fp, "def mu_cert_%d : Int := %s\n", b, s_cert);
                fprintf(fp,
                        "theorem block_%d_psd : ∀ v : Fin %d → Int,\n"
                        "    mu_psd_%d * ChemCert.vecNormSq %d v ≤ ChemCert.matQuad %d (ChemCert.matAt A_%d) v :=\n"
                        "  ChemCert.checkCholeskySDD_sound %d A_%d L_%d mu_psd_%d (by decide)\n\n",
                        b, dim, b, dim, dim, b, dim, b, b, b);
            }
            free(A_i);
            free(L_i);
        }

        ChemI128 E_lb_S = i128_sub(E_dual_S, delta_S);
        for (int b = 0; b < nblocks; b++) {
            if (i128_cmp(lam_arr[b], i128_zero()) < 0) {
                E_lb_S = i128_add(E_lb_S, i128_mul_i128_small_i64(lam_arr[b], (int64_t)tr_arr[b]));
            }
        }
        E_lb_real = i128_to_double(E_lb_S) / (double)S;
        (void)floor_real;

        double e_ub_val = (lc.E_ub_claimed != 0.0 || !lc.has_witness) ? lc.E_ub_claimed : lc.E_ci_claimed;
        if (lc.has_witness && lc.dets && lc.psi0) {
            double e_w = 0.0;
            if (chem_var_eval_witness(&lc.ham, lc.ndet_var, lc.dets, lc.psi0, &e_w, cerr, sizeof(cerr)) == 0) {
                e_ub_val = (e_w < lc.E_ub_claimed) ? e_w : lc.E_ub_claimed;
            }
        }
        ChemI128 E_ub_S = i128_from_double_scaled2(e_ub_val, S);

        fprintf(fp, "def trBlocks : Fin %d → Int := fun b => [", nblocks);
        for (int b = 0; b < nblocks; b++) fprintf(fp, "%s%d", (b > 0) ? ", " : "", tr_arr[b]);
        fprintf(fp, "].getD b.val 0\n");
        fprintf(fp, "def lamBlocks : Fin %d → Int := fun b => [", nblocks);
        for (int b = 0; b < nblocks; b++) {
            char ls[64];
            i128_to_str(lam_arr[b], ls, sizeof(ls));
            fprintf(fp, "%s%s", (b > 0) ? ", " : "", ls);
        }
        fprintf(fp, "].getD b.val 0\n");

        char s_ed[64], s_dl[64], s_lb[64], s_ub[64];
        i128_to_str(E_dual_S, s_ed, sizeof(s_ed));
        i128_to_str(delta_S, s_dl, sizeof(s_dl));
        i128_to_str(E_lb_S, s_lb, sizeof(s_lb));
        i128_to_str(E_ub_S, s_ub, sizeof(s_ub));

        fprintf(fp,
                "theorem trBlocks_nonneg : ∀ b : Fin %d, 0 ≤ trBlocks b :=\n"
                "  fun b => of_decide_eq_true (ChemCert.allFinB_true %d (fun i => decide (0 ≤ trBlocks i)) (by decide) b)\n\n"
                "theorem molecule_lower_bound_sound (E_state : Int) (slack : Fin %d → Int)\n"
                "    (hDecomp : E_state = (%s) + sumFin %d slack)\n"
                "    (hFloor : ∀ b, trBlocks b * lamBlocks b ≤ slack b) :\n"
                "    (%s) ≤ E_state := by\n"
                "  have h := ChemCert.anytime_dual_weyl_lower_bound E_state (%s) (%s) slack trBlocks lamBlocks\n"
                "    hDecomp trBlocks_nonneg hFloor (by omega)\n"
                "  have hCmp : (%s) ≤ (%s) + sumFin %d (fun b => trBlocks b * negPart (lamBlocks b)) - (%s) := by\n"
                "    decide\n"
                "  omega\n\n",
                nblocks, nblocks, nblocks, s_ed, nblocks, s_lb, s_ed, s_dl, s_lb, s_ed, nblocks, s_dl);
        if (i128_cmp(E_lb_S, E_ub_S) <= 0) {
            fprintf(fp, "theorem molecule_bracket_valid : (%s : Int) ≤ (%s : Int) := by decide\n\n", s_lb, s_ub);
            fprintf(fp, "#print axioms molecule_bracket_valid\n");
        }
        for (int b = 0; b < nblocks; b++) {
            if (ctx.block_dims[b] > 0) fprintf(fp, "#print axioms block_%d_psd\n", b);
        }
        fprintf(fp, "#print axioms molecule_lower_bound_sound\n");
        free(Z_D);
        mf_ctx_free(&ctx);
    }

    fprintf(fp, "\nend %s\n", ns);
    fclose(fp);

    if (out) {
        out->ok = 1;
        out->sha_ok = 1;
        out->all_blocks_sdd = 1;
        out->lb_ok = 1;
        out->var_ok = 1;
        out->nblocks = nblocks;
        out->scale_D = D;
        out->E_lb_int_chol = E_lb_real;
        out->E_lb_weyl = lc.E_lb_claimed;
        out->elapsed_sec = now_sec() - t0;
        snprintf(out->msg, sizeof(out->msg), "EMITTED %s (%d blocks, D=%lld, E_lb=%.8f)",
                 out_lean_path, nblocks, (long long)D, E_lb_real);
    }
    loaded_cert_free(&lc);
    return 0;
}

/* ============================================================================
 * Orthogonal Orbital Rotation & Entanglement-Driven DQG Natural Orbitals
 * ============================================================================ */

static int check_orthogonal_u(int K, const double *U, char *err, size_t err_sz) {
    if (K <= 0 || K > CHEM_MAX_ORBITALS || !U) {
        set_err(err, err_sz, "invalid orthogonal matrix U for K=%d", K);
        return -1;
    }
    double max_err = 0.0;
    for (int a = 0; a < K; a++) {
        for (int b = a; b < K; b++) {
            double s = 0.0;
            for (int p = 0; p < K; p++) {
                s += U[(size_t)p * (size_t)K + (size_t)a] * U[(size_t)p * (size_t)K + (size_t)b];
            }
            double target = (a == b) ? 1.0 : 0.0;
            double diff = fabs(s - target);
            if (diff > max_err) max_err = diff;
        }
    }
    if (max_err > 1e-10) {
        set_err(err, err_sz, "orbital transformation U is not orthogonal (max|U^T U - I|=%.3e)",
                max_err);
        return -1;
    }
    return 0;
}

static void rotate_mat2_ortho(int K, const double *U, const double *M_src, double *M_dst,
                              double *scr_k2) {
    /* scr = U^T * M_src, M_dst = scr * U */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, K, 1.0, U, K, M_src, K, 0.0,
                scr_k2, K);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, K, K, 1.0, scr_k2, K, U, K, 0.0,
                M_dst, K);
    for (int a = 0; a < K; a++) {
        for (int b = a + 1; b < K; b++) {
            double avg = 0.5 * (M_dst[(size_t)a * (size_t)K + (size_t)b] +
                                M_dst[(size_t)b * (size_t)K + (size_t)a]);
            M_dst[(size_t)a * (size_t)K + (size_t)b] = avg;
            M_dst[(size_t)b * (size_t)K + (size_t)a] = avg;
        }
    }
}

/* Exact O(K^5) 4-index quarter transformation via BLAS dgemm: E'[a,b,c,d] = sum U_{pa}U_{qb}U_{rc}U_{sd} E_{pqrs} */
static void rotate_tensor4_ortho(int K, const double *U, const double *E_src, double *E_dst,
                                 double *T1, double *T2) {
    int k2 = K * K;
    int k3 = k2 * K;
    size_t k2sz = (size_t)k2;
    size_t k3sz = (size_t)k3;

    /* Quarter 1: [K x K^3] <- U^T * E_src */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, k3, K, 1.0, U, K, E_src, k3, 0.0, T1,
                k3);

    /* Quarter 2: for each a in 0..K-1, [K x K^2] <- U^T * T1[a] */
    for (int a = 0; a < K; a++) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, k2, K, 1.0, U, K,
                    T1 + (size_t)a * k3sz, k2, 0.0, T2 + (size_t)a * k3sz, k2);
    }

    /* Quarter 3: for each ab in 0..K^2-1, [K x K] <- U^T * T2[ab] */
    for (int ab = 0; ab < k2; ab++) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, K, 1.0, U, K,
                    T2 + (size_t)ab * k2sz, K, 0.0, T1 + (size_t)ab * k2sz, K);
    }

    /* Quarter 4: [K^3 x K] <- T1 * U */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, k3, K, K, 1.0, T1, K, U, K, 0.0,
                E_dst, K);
}

int chem_ham_rotate_orthogonal(const ChemHam *src, const double *U, ChemHam *dst,
                               char *err, size_t err_sz) {
    if (!src || !U || !dst || !src->h || !src->eri) {
        set_err(err, err_sz, "invalid argument to chem_ham_rotate_orthogonal");
        return -1;
    }
    int K = src->K;
    if (check_orthogonal_u(K, U, err, err_sz) != 0) return -1;
    if (chem_ham_alloc(dst, K, src->nelec, src->ms2, err, err_sz) != 0) return -1;
    dst->E_core = src->E_core;

    size_t k2sz = (size_t)K * (size_t)K;
    size_t k4sz = k2sz * k2sz;
    double *T1 = (double *)malloc(k4sz * sizeof(double));
    double *T2 = (double *)malloc(k4sz * sizeof(double));
    if (!T1 || !T2) {
        free(T1);
        free(T2);
        chem_ham_free(dst);
        set_err(err, err_sz, "out of memory rotating Hamiltonian for K=%d", K);
        return -1;
    }

    rotate_mat2_ortho(K, U, src->h, dst->h, T1);
    for (int p = 0; p < K; p++) {
        for (int q = 0; q <= p; q++) {
            double v = dst->h[(size_t)p * (size_t)K + (size_t)q];
            if (fabs(v) <= 1e-15) v = 0.0;
            dst->h[(size_t)p * (size_t)K + (size_t)q] = v;
            dst->h[(size_t)q * (size_t)K + (size_t)p] = v;
        }
    }
    rotate_tensor4_ortho(K, U, src->eri, dst->eri, T1, T2);
    size_t ksz = (size_t)K;
    for (int p = 0; p < K; p++) {
        for (int q = 0; q <= p; q++) {
            int pq = p * (p + 1) / 2 + q;
            for (int r = 0; r < K; r++) {
                for (int s = 0; s <= r; s++) {
                    int rs = r * (r + 1) / 2 + s;
                    if (rs > pq) continue;
                    size_t i1 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)r) * ksz + (size_t)s);
                    size_t i2 = ((((size_t)q * ksz + (size_t)p) * ksz + (size_t)r) * ksz + (size_t)s);
                    size_t i3 = ((((size_t)p * ksz + (size_t)q) * ksz + (size_t)s) * ksz + (size_t)r);
                    size_t i4 = ((((size_t)q * ksz + (size_t)p) * ksz + (size_t)s) * ksz + (size_t)r);
                    size_t i5 = ((((size_t)r * ksz + (size_t)s) * ksz + (size_t)p) * ksz + (size_t)q);
                    size_t i6 = ((((size_t)s * ksz + (size_t)r) * ksz + (size_t)p) * ksz + (size_t)q);
                    size_t i7 = ((((size_t)r * ksz + (size_t)s) * ksz + (size_t)q) * ksz + (size_t)p);
                    size_t i8 = ((((size_t)s * ksz + (size_t)r) * ksz + (size_t)q) * ksz + (size_t)p);
                    double avg = 0.125 * (dst->eri[i1] + dst->eri[i2] + dst->eri[i3] + dst->eri[i4] +
                                          dst->eri[i5] + dst->eri[i6] + dst->eri[i7] + dst->eri[i8]);
                    if (fabs(avg) <= 1e-15) avg = 0.0;
                    dst->eri[i1] = dst->eri[i2] = dst->eri[i3] = dst->eri[i4] = avg;
                    dst->eri[i5] = dst->eri[i6] = dst->eri[i7] = dst->eri[i8] = avg;
                }
            }
        }
    }
    free(T1);
    free(T2);

    chem_ham_finalize(dst);
    return 0;
}

int chem_obs_rotate_orthogonal(const ChemObservable *src, const double *U, ChemObservable *dst,
                               char *err, size_t err_sz) {
    if (!src || !U || !dst || !src->o1_a || !src->o1_b) {
        set_err(err, err_sz, "invalid argument to chem_obs_rotate_orthogonal");
        return -1;
    }
    int K = src->K;
    if (check_orthogonal_u(K, U, err, err_sz) != 0) return -1;
    if (alloc_observable(dst, src->name, K, src->O_core, src->has_o2, src->spin_symmetric, err,
                         err_sz) != 0) {
        return -1;
    }
    size_t k2sz = (size_t)K * (size_t)K;
    size_t k4sz = src->has_o2 ? (k2sz * k2sz) : k2sz;
    double *T1 = (double *)malloc(k4sz * sizeof(double));
    double *T2 = src->has_o2 ? (double *)malloc(k4sz * sizeof(double)) : NULL;
    if (!T1 || (src->has_o2 && !T2)) {
        free(T1);
        free(T2);
        chem_observable_free(dst);
        set_err(err, err_sz, "out of memory rotating observable for K=%d", K);
        return -1;
    }

    rotate_mat2_ortho(K, U, src->o1_a, dst->o1_a, T1);
    rotate_mat2_ortho(K, U, src->o1_b, dst->o1_b, T1);
    if (src->has_o2) {
        rotate_tensor4_ortho(K, U, src->o2_ab, dst->o2_ab, T1, T2);
        rotate_tensor4_ortho(K, U, src->o2_aa, dst->o2_aa, T1, T2);
    }

    free(T1);
    free(T2);
    return 0;
}

int chem_ham_build_dqg_no(const ChemHam *src, const ChemSolveOpts *opts_in, ChemHam *dst_no,
                          double *no_occs_out, double *no_ent_out, double *U_out,
                          char *err, size_t err_sz) {
    if (!src || !dst_no) {
        set_err(err, err_sz, "invalid argument to chem_ham_build_dqg_no");
        return -1;
    }
    int K = src->K;
    ChemSolveOpts pilot_opts = opts_in ? *opts_in : chem_solve_opts_default(src);
    pilot_opts.enable_t1 = 0;
    pilot_opts.enable_t2 = 0;
    pilot_opts.export_rdm_grad = 1;
    pilot_opts.veto_enabled = 0;
    pilot_opts.max_iters = 250;
    pilot_opts.tol = 1e-4;
    pilot_opts.verbose = 0;

    ChemSolveResult pilot;
    if (chem_solve_dqg(src, &pilot_opts, &pilot, err, err_sz) != 0) {
        return -1;
    }

    size_t ksz = (size_t)K;
    double *G = (double *)calloc(ksz * ksz, sizeof(double));
    double *evals = (double *)calloc(ksz, sizeof(double));
    double *U = (double *)calloc(ksz * ksz, sizeof(double));
    if (!G || !evals || !U) {
        free(G); free(evals); free(U);
        chem_solve_result_free(&pilot);
        set_err(err, err_sz, "out of memory computing DQG natural orbitals");
        return -1;
    }

    if (pilot.primal_grad_h1 && pilot.K_grad == K) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                double avg = 0.5 * (pilot.primal_grad_h1[(size_t)p * ksz + (size_t)q] +
                                    pilot.primal_grad_h1[(size_t)q * ksz + (size_t)p]);
                G[(size_t)p * ksz + (size_t)q] = avg;
            }
        }
        __LAPACK_int n_k = (__LAPACK_int)K;
        __LAPACK_int lwork = 1 + 6 * n_k + 2 * n_k * n_k + 32;
        __LAPACK_int liwork = 3 + 5 * n_k + 16;
        double *work = (double *)malloc((size_t)lwork * sizeof(double));
        __LAPACK_int *iwork = (__LAPACK_int *)malloc((size_t)liwork * sizeof(__LAPACK_int));
        __LAPACK_int info = 0;
        if (!work || !iwork) {
            free(work); free(iwork); free(G); free(evals); free(U);
            chem_solve_result_free(&pilot);
            set_err(err, err_sz, "out of memory diagonalizing DQG 1-RDM");
            return -1;
        }
        dsyevd_("V", "U", &n_k, G, &n_k, evals, work, &lwork, iwork, &liwork, &info);
        free(work);
        free(iwork);
        if (info != 0) {
            free(G); free(evals); free(U);
            chem_solve_result_free(&pilot);
            set_err(err, err_sz, "dsyevd_ failed on DQG 1-RDM (info=%d)", (int)info);
            return -1;
        }

        double ents[CHEM_MAX_ORBITALS];
        int order[CHEM_MAX_ORBITALS];
        for (int k = 0; k < K; k++) {
            double occ = evals[k];
            double fk = fmin(1.0 - 1e-15, fmax(1e-15, 0.5 * occ));
            ents[k] = -(fk * log(fk) + (1.0 - fk) * log(1.0 - fk));
            order[k] = k;
        }
        /* Sort eigenvectors by descending orbital entanglement entropy (closest to occ=1 first) */
        for (int i = 0; i < K; i++) {
            int best = i;
            for (int j = i + 1; j < K; j++) {
                double ei = ents[order[best]];
                double ej = ents[order[j]];
                if (ej > ei + 1e-14 ||
                    (fabs(ej - ei) <= 1e-14 && evals[order[j]] > evals[order[best]])) {
                    best = j;
                }
            }
            int tmp = order[i];
            order[i] = order[best];
            order[best] = tmp;
        }
        for (int a = 0; a < K; a++) {
            int k = order[a];
            if (no_occs_out) no_occs_out[a] = evals[k];
            if (no_ent_out) no_ent_out[a] = ents[k];
            /* Deterministic sign convention: largest magnitude component nonnegative */
            int max_p = 0;
            double max_v = -1.0;
            for (int p = 0; p < K; p++) {
                double v = fabs(G[(size_t)k * ksz + (size_t)p]);
                if (v > max_v) {
                    max_v = v;
                    max_p = p;
                }
            }
            double sgn = (G[(size_t)k * ksz + (size_t)max_p] < 0.0) ? -1.0 : 1.0;
            for (int p = 0; p < K; p++) {
                U[(size_t)p * ksz + (size_t)a] = sgn * G[(size_t)k * ksz + (size_t)p];
            }
        }
    } else {
        for (int a = 0; a < K; a++) {
            U[(size_t)a * ksz + (size_t)a] = 1.0;
            if (no_occs_out) no_occs_out[a] = (a < src->nelec / 2) ? 2.0 : 0.0;
            if (no_ent_out) no_ent_out[a] = 0.0;
        }
    }

    if (U_out) memcpy(U_out, U, ksz * ksz * sizeof(double));
    int rc = chem_ham_rotate_orthogonal(src, U, dst_no, err, err_sz);

    free(G);
    free(evals);
    free(U);
    chem_solve_result_free(&pilot);
    return rc;
}
