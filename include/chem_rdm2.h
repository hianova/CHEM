#ifndef CHEM_RDM2_H
#define CHEM_RDM2_H

#include "chem_fcidump.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ChemVarResult;

#define CHEM_MAX_BLOCKS 17

typedef enum {
    CHEM_MODE_AUTO = 0,
    CHEM_MODE_A_SINGLET = 1,
    CHEM_MODE_B_GENERAL = 2
} ChemMode;

typedef struct {
    ChemMode mode;
    int two_s;            /* 2*S target (default abs(ham->ms2)) */
    int constrain_s2;     /* 1 to enforce <S^2> = S(S+1) in Mode B (default 1) */
    int max_iters;        /* max BPSDP iterations (default 800) */
    double tol;           /* primal/dual residual stop tol (default 1e-6) */
    double alpha;         /* over-relaxation parameter (default 1.5) */
    int check_cert_every; /* frequency of unconditional LB check (0 = auto) */
    int veto_enabled;     /* 1 to halt early if E_lb_cert > E_ai - veto_tol */
    double E_ai;          /* candidate AI/MLIP/single-ref energy (Ha) */
    double veto_tol;      /* veto margin (Ha, default 1e-4) */
    int enable_t2;        /* 1 to include exact/subspace T2 partial 3-RDM blocks (default 0) */
    int t2_max_orbs;      /* auto cap Ks when t2_norbs==0 (0 = auto min(K, 6)) */
    int t2_norbs;         /* explicit T2 active orbital count (0 = auto select) */
    int t2_orbs[CHEM_MAX_ORBITALS];
    int enable_t1;        /* 1 to include exact/subspace T1 = D3 + Q3 partial 3-RDM blocks (default 0) */
    int t1_max_orbs;      /* auto cap Ks when t1_norbs==0 (0 = auto min(K, 6)) */
    int t1_norbs;         /* explicit T1 active orbital count (0 = auto select) */
    int t1_orbs[CHEM_MAX_ORBITALS];
    int export_rdm_grad;  /* 1 to populate primal_grad_h1 & primal_grad_eri on ChemSolveResult */
    int num_threads;      /* 0 = auto (up to 4 pthreads when max_dim >= 120), 1 = serial */
    int verbose;
} ChemSolveOpts;

typedef struct {
    int vetoed;             /* 1 if E_LB_cert > E_ai - tol early or at convergence */
    int iter_stopped;
    double E_lb_cert;       /* rigorous active-space lower bound (Ha) */
    double E_ub_variational;/* best variational upper bound available (Ha) */
    double min_cert_eig;    /* worst certified slack eigenvalue across blocks */
    double interval_width;  /* E_ub_variational - E_lb_cert */
} ChemVetoResult;

typedef struct {
    ChemMode mode;
    int K;
    int na;
    int nb;
    int two_s;
    int constrain_s2;
    int t2_enabled;
    int t2_norbs;
    int t2_orbs[CHEM_MAX_ORBITALS];
    int t1_enabled;
    int t1_norbs;
    int t1_orbs[CHEM_MAX_ORBITALS];
    int meq;
    int nD;
    int nQ;
    int nG;
    int nT2;
    int nT1;
    double *y_eq; /* [meq] */
    double *Z_Q;  /* [nQ] in isometric svec format */
    double *Z_G;  /* [nG] in isometric svec format */
    double *Z_T2; /* [nT2] in isometric svec format (NULL if t2_enabled == 0) */
    double *Z_T1; /* [nT1] in isometric svec format (NULL if t1_enabled == 0) */
} ChemDualWitness;

typedef struct {
    ChemMode mode_used;
    int nblocks; /* 6/8/10 for Mode A, 9/13/17 for Mode B */
    int t2_enabled;
    int t2_norbs;
    int t2_orbs[CHEM_MAX_ORBITALS];
    int t1_enabled;
    int t1_norbs;
    int t1_orbs[CHEM_MAX_ORBITALS];
    int iters;
    int converged;
    int vetoed;
    int iter_stopped;
    double E_primal;
    double E_dual_raw;
    double E_lb_cert;
    double min_eig_fp;
    double min_eig_cert;
    double delta_round;
    double prim_res;
    double dual_res;
    double elapsed_sec;
    int block_dims[CHEM_MAX_BLOCKS];
    double block_traces[CHEM_MAX_BLOCKS];
    double block_min_fp[CHEM_MAX_BLOCKS];
    double block_min_cert[CHEM_MAX_BLOCKS];
    int K_grad;
    double *primal_grad_h1;  /* [K*K] when export_rdm_grad == 1 */
    double *primal_grad_eri; /* [K^4] when export_rdm_grad == 1 */
    ChemDualWitness witness;
} ChemSolveResult;

typedef struct {
    double err_LQ;
    double err_LG;
    double err_LT2;
    double err_LT1;
    double err_Aeq;
    double max_err;
} ChemAdjointReport;

typedef struct {
    char name[64];
    int K;
    double O_core;
    double *o1_a;  /* [K*K] alpha 1-body symmetric operator */
    double *o1_b;  /* [K*K] beta 1-body symmetric operator */
    double *o2_ab; /* [K^4] chemist-indexed (p,r,q,s) for a^+_p a^+_q b a_s b a_r */
    double *o2_aa; /* [K^4] chemist-indexed (p,r,q,s) */
    int has_o2;
    int spin_symmetric;
} ChemObservable;

typedef struct {
    char name[64];
    double E_ub;
    double O_lb_cert;
    double O_ub_cert;
    double O_var;
    double lam_lo;
    double lam_hi;
    double interval_width;
    double elapsed_sec;
} ChemObservableResult;

typedef struct {
    int ok;
    int sha_ok;
    int lb_ok;
    int veto_ok;
    int var_ok;
    int is_binary;
    double E_lb_recomputed;
    double E_lb_claimed;
    double E_var_recomputed;
    double min_eig_cert;
    char msg[256];
} ChemCertVerifyResult;

typedef struct {
    int ok;
    int sha_ok;
    int all_blocks_sdd;
    int lb_ok;
    int var_ok;
    int nblocks;
    int64_t scale_D;
    double E_lb_int_chol;
    double E_lb_weyl;
    double E_var_recomputed;
    double max_sdd_margin;
    double elapsed_sec;
    char msg[256];
} ChemExactVerifyResult;

ChemSolveOpts chem_solve_opts_default(const ChemHam *ham);
ChemMode chem_resolve_mode(const ChemHam *ham, const ChemSolveOpts *opts);
const char *chem_block_name(ChemMode mode, int blk_idx);

void chem_dual_witness_free(ChemDualWitness *w);
void chem_solve_result_free(ChemSolveResult *res);
void chem_observable_free(ChemObservable *obs);

/* Observable & Hellmann-Feynman force difference constructors */
int chem_obs_orbital_occ(int K, int p, ChemObservable *out, char *err, size_t err_sz);
int chem_obs_double_occ(int K, int p, ChemObservable *out, char *err, size_t err_sz);
int chem_obs_spin_corr(int K, int p, int q, ChemObservable *out, char *err, size_t err_sz);
int chem_obs_spin_1body(int K, const double *o1_a, const double *o1_b, double o_const,
                        const char *name, ChemObservable *out, char *err, size_t err_sz);
int chem_obs_ham_diff(const ChemHam *h_plus, const ChemHam *h_minus, double scale,
                      const char *name, ChemObservable *out, char *err, size_t err_sz);

/* Matrix-free BPSDP solver with FLOW 2-Jet adaptive penalty, optional T2, & Anytime Early-Veto */
int chem_solve_dqg(const ChemHam *ham, const ChemSolveOpts *opts, ChemSolveResult *res,
                   char *err, size_t err_sz);

/* Certified observable/force bounds [O_lb_cert, O_ub_cert] conditioned on E <= E_ub */
int chem_certify_observable(const ChemHam *ham, const ChemObservable *obs, double E_ub,
                            const ChemSolveOpts *opts, ChemObservableResult *out,
                            char *err, size_t err_sz);

/* Evaluates <psi | O | psi> / <psi | psi> on sparse determinant witness */
int chem_eval_observable_witness(const ChemHam *ham, const ChemObservable *obs,
                                 int ndet, const uint64_t *dets, const double *coeffs,
                                 double *val_out, char *err, size_t err_sz);

/* Standalone rigorous dual lower-bound certifier given an untrusted dual witness */
int chem_certify_dual_witness(const ChemHam *ham, const ChemDualWitness *w,
                              ChemSolveResult *res_out, char *err, size_t err_sz);

/* Randomized matrix-free adjoint verification: |<L(X),Y> - <X,L*(Y)>| / scale < 1e-14 */
int chem_test_adjoints(int K, int na, int nb, int two_s, int constrain_s2, ChemMode mode,
                       uint64_t seed, ChemAdjointReport *rep, char *err, size_t err_sz);

int chem_test_adjoints_t2(int K, int na, int nb, int two_s, int constrain_s2, ChemMode mode,
                          int t2_norbs, const int *t2_orbs, uint64_t seed,
                          ChemAdjointReport *rep, char *err, size_t err_sz);

int chem_test_adjoints_t12(int K, int na, int nb, int two_s, int constrain_s2, ChemMode mode,
                           int t2_norbs, const int *t2_orbs, int t1_norbs, const int *t1_orbs,
                           uint64_t seed, ChemAdjointReport *rep, char *err, size_t err_sz);

/* Verifies CAR map Q = q0 + L_Q(D), G = g0 + L_G(D), A_eq(D) = b_eq, PSD, and energy on FCI blocks */
int chem_verify_fci_dqg_consistency(const ChemHam *ham, ChemMode mode, int two_s,
                                    const double *D_svec, const double *Q_svec, const double *G_svec,
                                    double *car_map_err_out, double *eq_err_out,
                                    double *min_eig_out, double *energy_err_out,
                                    char *err, size_t err_sz);

/* Verifies T2(D_FCI) >= 0 and exact T2 block traces when t2_norbs == K */
int chem_verify_fci_t2_psd(const ChemHam *ham, ChemMode mode, int two_s,
                           int t2_norbs, const int *t2_orbs, const double *D_svec,
                           double *min_eig_out, double *trace_err_out,
                           char *err, size_t err_sz);

/* Verifies T1(D_FCI) = (D3 + Q3)(D_FCI) >= 0 and exact T1 block traces when t1_norbs == K */
int chem_verify_fci_t1_psd(const ChemHam *ham, ChemMode mode, int two_s,
                           int t1_norbs, const int *t1_orbs, const double *D_svec,
                           double *min_eig_out, double *trace_err_out,
                           char *err, size_t err_sz);

ChemVetoResult chem_make_veto_result(const ChemSolveResult *solve, double E_ub_variational);

int chem_cert_write_json(const char *path, const ChemHam *ham, const char *fcidump_path,
                         const ChemSolveResult *res, const struct ChemVarResult *var_opt,
                         const ChemSolveOpts *opts, char *err, size_t err_sz);

int chem_cert_write_bin(const char *path, const ChemHam *ham, const char *fcidump_path,
                        const ChemSolveResult *res, const struct ChemVarResult *var_opt,
                        const ChemSolveOpts *opts, char *err, size_t err_sz);

/* Writes canonical Columnar Binary Certificate ("CHMC", .cert.bin) with SHA-256 trailer */
int chem_cert_write_chmc(const char *path, const ChemHam *ham, const char *fcidump_path,
                         const ChemSolveResult *res, const struct ChemVarResult *var_opt,
                         const ChemSolveOpts *opts, char *err, size_t err_sz);

/* Packs an existing .cert.json file losslessly into canonical "CHMC" (.cert.bin) */
int chem_cert_pack(const char *json_path, const char *bin_path, char *err, size_t err_sz);

/* Unpacks a canonical "CHMC" (.cert.bin) file losslessly into .cert.json */
int chem_cert_unpack(const char *bin_path, const char *json_path, char *err, size_t err_sz);

int chem_cert_verify_file(const char *fcidump_override, const char *cert_path,
                          ChemCertVerifyResult *out);

/* Zero-FP-Trust integer Cholesky + SDD exact verifier for JSON, .cert.bin, & .flowchem certs */
int chem_cert_verify_exact_chol(const char *fcidump_override, const char *cert_path,
                                ChemExactVerifyResult *out);

/* Emits a standalone kernel-verified Lean 4 certificate (.lean) with CheckCholeskySDD proofs */
int chem_cert_emit_lean(const char *fcidump_override, const char *cert_path,
                        const char *out_lean_path, const char *module_name,
                        ChemExactVerifyResult *out, char *err, size_t err_sz);

/* Persistent in-process BPSDP session for warm-started multi-Hamiltonian optimization sweeps */
typedef struct ChemSession ChemSession;

ChemSession *chem_session_create(int K, int nelec, int ms2, int mode, int two_s, int constrain_s2,
                                 int enable_t1, int t1_orbs, int enable_t2, int t2_orbs,
                                 int nthreads, char *err, size_t err_sz);

void chem_session_free(ChemSession *sess);

int chem_session_solve(ChemSession *sess, double e_core, const double *h1, const double *eri,
                       int max_iters, double tol, int warm_start, double *E_lb_out,
                       double *E_primal_out, int *iters_out, int *conv_out,
                       double *grad_h1_out, double *grad_eri_out, char *err, size_t err_sz);

/* Rotates Hamiltonian and observables by orthogonal K x K matrix U (row-major, columns are new orbitals) */
int chem_ham_rotate_orthogonal(const ChemHam *src, const double *U, ChemHam *dst,
                               char *err, size_t err_sz);

int chem_obs_rotate_orthogonal(const ChemObservable *src, const double *U, ChemObservable *dst,
                               char *err, size_t err_sz);

/* Pilot DQG solve -> entanglement-sorted natural-orbital Hamiltonian (highest static correlation first) */
int chem_ham_build_dqg_no(const ChemHam *src, const ChemSolveOpts *opts_in, ChemHam *dst_no,
                          double *no_occs_out, double *no_ent_out, double *U_out,
                          char *err, size_t err_sz);

#ifdef __cplusplus
}
#endif

#endif /* CHEM_RDM2_H */
