#ifndef CHEM_SHCI_H
#define CHEM_SHCI_H

#include "chem_fcidump.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int max_dets;
    int run_searu;
    int target_two_s; /* -1 = unconstrained M_S, >=0 = project target spin S = target_two_s/2 */
    double eps1_schedule[8];
    int n_eps1;
    uint64_t seed;
} ChemVarOpts;

typedef struct ChemVarResult {
    double E_hf0;
    double E_scf;
    double E_searu_rot;
    double E_ci;
    double E_ub; /* Strictly variational min(E_hf0, E_scf, E_searu_rot, E_ci) */
    double E_ci_excited;
    int has_excited;
    int is_exact_fci;
    uint64_t ndet_space; /* Exact or saturated determinant count */
    int ndet_var;
    uint64_t *dets; /* [ndet_var] bitmasks: 2*p = alpha, 2*p+1 = beta */
    double *psi0;   /* [ndet_var] ground-state eigenvector */
    double var_H_best;
    double E_weinstein;
    int has_weinstein;
    double var_H_hf;
    double E_temple;
    int has_temple;
    int diis_iters;
    double comm_norm;
    double elapsed_sec;
} ChemVarResult;

typedef struct {
    int K;
    int na;
    int nb;
    int nelec;
    double occ_a[CHEM_MAX_ORBITALS];
    double occ_b[CHEM_MAX_ORBITALS];
    double occ_sorted[2 * CHEM_MAX_ORBITALS];
    /* Column-major eigenvectors U[:,eig_idx] (or row p, col eig_idx = U[p*K + eig_idx] or [eig_idx*K+p]) */
    double U_a[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS];
    double U_b[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS];
    double entanglement_entropy;
    int has_bd_3e3o;
    double bd_pair_err;
    double bd_slack;
    int bd_pinned;
    int has_bd_2e2o;
    double bd24_pair_err;
    /* Indices (spin 0=alpha,1=beta and natural orbital idx) for sorted lambdas l4, l5, l6 */
    int bd_obs_spin[3];
    int bd_obs_idx[3];
} ChemGPCResult;

ChemVarOpts chem_var_opts_default(void);
void chem_var_result_free(ChemVarResult *vr);

/* Tiered variational upper bound: HF + DIIS + SEARU x 2-Jet orbital rotation + FCI/SHCI + Weinstein/Temple */
int chem_var_solve(const ChemHam *ham, const ChemVarOpts *opts, ChemVarResult *out,
                   char *err, size_t err_sz);

/* Independently evaluates Rayleigh quotient <psi|H|psi>/<psi|psi> on sparse determinant witness */
int chem_var_eval_witness(const ChemHam *ham, int ndet, const uint64_t *dets, const double *coeffs,
                          double *energy_out, char *err, size_t err_sz);

/* Exact full-Fock-space Hamiltonian variance sigma^2 = ||(H - E_var)|psi>||_2^2 */
int chem_var_eval_variance(const ChemHam *ham, int ndet, const uint64_t *dets, const double *coeffs,
                           double *e_var_out, double *var_out, char *err, size_t err_sz);

/* Evaluates spin expectation value <S^2> on determinant witness */
int chem_var_eval_spin_sq(int na, int nb, int ndet, const uint64_t *dets, const double *coeffs,
                          double *s2_out);

/* Computes spin-summed orbital diagonal occupation numbers occ[0..K-1] in [0, 2] */
int chem_var_compute_rdm1_diag(const ChemHam *ham, int ndet, const uint64_t *dets,
                               const double *coeffs, double *occ_out);

/* Diagonalizes full 1-RDM (alpha & beta), computes entanglement entropy & Borland-Dennis GPC */
int chem_var_analyze_gpc(const ChemHam *ham, int ndet, const uint64_t *dets, const double *coeffs,
                         ChemGPCResult *out, char *err, size_t err_sz);

/* Loads external variational witness JSON (standalone dets/coeffs or cert.json variational_witness) */
int chem_var_load_witness_json(const char *path, const ChemHam *ham, int target_two_s,
                               ChemVarResult *out, char *err, size_t err_sz);

/* Computes exact FCI 2-RDM D, Q, G blocks in Mode A isometric svec format (caller frees outputs) */
int chem_fci_compute_dqg_mode_a(const ChemHam *ham, int ndet, const uint64_t *dets, const double *psi0,
                                double **D_svec_out, double **Q_svec_out, double **G_svec_out,
                                char *err, size_t err_sz);

/* Computes exact FCI 2-RDM D, Q, G blocks in Mode B isometric svec format (caller frees outputs) */
int chem_fci_compute_dqg_mode_b(const ChemHam *ham, int ndet, const uint64_t *dets, const double *psi0,
                                double **D_svec_out, double **Q_svec_out, double **G_svec_out,
                                char *err, size_t err_sz);

#ifdef __cplusplus
}
#endif

#endif /* CHEM_SHCI_H */
