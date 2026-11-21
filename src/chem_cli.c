#include "chem_fcidump.h"
#include "chem_rdm2.h"
#include "chem_shci.h"
#include "flow.h"
#include "searu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static double peak_rss_mb(void) {
    struct rusage r;
    if (getrusage(RUSAGE_SELF, &r) != 0) return 0.0;
#ifdef __APPLE__
    return (double)r.ru_maxrss / (1024.0 * 1024.0);
#else
    return (double)r.ru_maxrss / 1024.0;
#endif
}

static void print_usage(void) {
    fprintf(stderr,
            "Usage:\n"
            "  chem_cli selftest [--json <out.json>]\n"
            "  chem_cli bracket <fcidump> [--mode auto|a|b] [--two-s <int>] [--no-s2]\n"
            "                   [--max-iters <int>] [--tol <double>] [--max-dets <int>]\n"
            "                   [--t2] [--t2-orbs <Ks>] [--t1] [--t1-orbs <Ks>] [--auto-no]\n"
            "                   [--save-no-fcidump <no.fcidump>] [--threads <n>]\n"
            "                   [--witness <file.json>] [--dump-rdm-grad <grad.json>]\n"
            "                   [--cert <cert.json>] [--cert-bin <cert.flowchem>] [--json <out.json>] [--verbose]\n"
            "  chem_cli veto <fcidump> --e-ai <Ha> [--veto-tol <Ha>] [--mode auto|a|b]\n"
            "                [--two-s <int>] [--no-s2] [--max-iters <int>] [--tol <double>]\n"
            "                [--max-dets <int>] [--t2] [--t2-orbs <Ks>] [--t1] [--t1-orbs <Ks>] [--auto-no]\n"
            "                [--threads <n>] [--witness <file.json>] [--cert <cert.json>] [--cert-bin <cert.flowchem>] [--json <out.json>] [--verbose]\n"
            "  chem_cli observable <fcidump> --type occ:<p>|docc:<p>|sdot:<p>:<q>|hf-force:<+>:<->:<dR>\n"
            "                      [--mode auto|a|b] [--two-s <int>] [--no-s2] [--max-iters <int>]\n"
            "                      [--tol <double>] [--max-dets <int>] [--t2] [--t2-orbs <Ks>]\n"
            "                      [--t1] [--t1-orbs <Ks>] [--auto-no] [--save-no-fcidump <no.fcidump>]\n"
            "                      [--threads <n>] [--witness <file.json>] [--json <out.json>] [--verbose]\n"
            "  chem_cli cert-verify [<fcidump>] <cert.json|cert.flowchem>\n"
            "  chem_cli exact-verify [<fcidump>] <cert.json|cert.flowchem> [--json <out.json>]\n"
            "  chem_cli emit-lean <fcidump> <cert.json|cert.flowchem> --out <out.lean> [--name <ModName>]\n"
            "  chem_cli gpc <fcidump> [--mode auto|a|b] [--two-s <int>] [--witness <file.json>] [--certify-bd] [--json <out.json>]\n"
            "  chem_cli atlas [--dir <benchmarks_dir>] [--out <atlas.json>]\n"
            "  chem_cli bench --dir <benchmarks_dir> --out <bench.json> --cert-dir <certs_dir>\n");
}

static void select_frontier_sub_orbs(const ChemHam *ham, const ChemVarResult *vr, int cap,
                                     int *norbs_out, int *orbs_out) {
    int K = ham->K;
    if (K <= cap) {
        *norbs_out = K;
        for (int i = 0; i < K; i++) orbs_out[i] = i;
        return;
    }
    double occ[CHEM_MAX_ORBITALS];
    double score[CHEM_MAX_ORBITALS];
    int idx[CHEM_MAX_ORBITALS];
    if (vr && vr->ndet_var > 0 && vr->dets && vr->psi0 &&
        chem_var_compute_rdm1_diag(ham, vr->ndet_var, vr->dets, vr->psi0, occ) == 0) {
        for (int p = 0; p < K; p++) {
            score[p] = occ[p] * (2.0 - occ[p]);
            idx[p] = p;
        }
    } else {
        int center = (ham->na > 0) ? (ham->na - 1) : (K / 2 - 1);
        for (int p = 0; p < K; p++) {
            score[p] = 1.0 / (1.0 + fabs((double)p - ((double)center + 0.5)));
            idx[p] = p;
        }
    }
    for (int i = 0; i < cap; i++) {
        int best = i;
        for (int j = i + 1; j < K; j++) {
            if (score[idx[j]] > score[idx[best]]) best = j;
        }
        int tmp = idx[i];
        idx[i] = idx[best];
        idx[best] = tmp;
    }
    for (int i = 0; i < cap; i++) orbs_out[i] = idx[i];
    for (int i = 0; i < cap; i++) {
        for (int j = i + 1; j < cap; j++) {
            if (orbs_out[j] < orbs_out[i]) {
                int tmp = orbs_out[i];
                orbs_out[i] = orbs_out[j];
                orbs_out[j] = tmp;
            }
        }
    }
    *norbs_out = cap;
}

static void select_frontier_t2_orbs(const ChemHam *ham, const ChemVarResult *vr, int cap,
                                    ChemSolveOpts *sopts) {
    select_frontier_sub_orbs(ham, vr, cap, &sopts->t2_norbs, sopts->t2_orbs);
}

static void select_frontier_t1_orbs(const ChemHam *ham, const ChemVarResult *vr, int cap,
                                    ChemSolveOpts *sopts) {
    select_frontier_sub_orbs(ham, vr, cap, &sopts->t1_norbs, sopts->t1_orbs);
}

static int write_rdm_grad_json(const char *path, int K, const ChemSolveResult *sr) {
    if (!path || !sr || !sr->primal_grad_h1 || !sr->primal_grad_eri || sr->K_grad != K) {
        return -1;
    }
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    int k2 = K * K;
    int k4 = k2 * k2;
    fprintf(fp, "{\n  \"K\": %d,\n  \"E_primal\": %.17g,\n  \"E_lb_cert\": %.17g,\n",
            K, sr->E_primal, sr->E_lb_cert);
    fprintf(fp, "  \"grad_h1\": [");
    for (int i = 0; i < k2; i++) {
        if (i > 0) fputc(',', fp);
        fprintf(fp, "%.17g", (sr->primal_grad_h1[i] == 0.0) ? 0.0 : sr->primal_grad_h1[i]);
    }
    fprintf(fp, "],\n  \"grad_eri\": [");
    for (int i = 0; i < k4; i++) {
        if (i > 0) fputc(',', fp);
        fprintf(fp, "%.17g", (sr->primal_grad_eri[i] == 0.0) ? 0.0 : sr->primal_grad_eri[i]);
    }
    fprintf(fp, "]\n}\n");
    fclose(fp);
    return 0;
}

static int build_bd_3e3o_observable(const ChemGPCResult *gpc, ChemObservable *out,
                                    char *err, size_t err_sz) {
    if (!gpc || !gpc->has_bd_3e3o) {
        snprintf(err, err_sz, "GPC result does not have 3e3o Borland-Dennis orbitals");
        return -1;
    }
    int K = gpc->K;
    double o1_a[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS] = {0};
    double o1_b[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS] = {0};
    const double w[3] = {-1.0, 1.0, 1.0}; /* lambda_5 + lambda_6 - lambda_4 */
    for (int m = 0; m < 3; m++) {
        int spin = gpc->bd_obs_spin[m];
        int eidx = gpc->bd_obs_idx[m];
        const double *U = (spin == 0) ? gpc->U_a : gpc->U_b;
        const double *v = U + (size_t)eidx * (size_t)K;
        double *dest = (spin == 0) ? o1_a : o1_b;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                dest[(size_t)p * (size_t)K + (size_t)q] += w[m] * v[p] * v[q];
            }
        }
    }
    return chem_obs_spin_1body(K, o1_a, o1_b, 0.0, "gpc-bd-3e3o", out, err, err_sz);
}

static int parse_and_build_obs(int K, const char *spec, ChemObservable *out, char *err,
                               size_t err_sz) {
    if (!spec) {
        snprintf(err, err_sz, "missing --type specification");
        return -1;
    }
    if (strncmp(spec, "occ:", 4) == 0) {
        return chem_obs_orbital_occ(K, atoi(spec + 4), out, err, err_sz);
    }
    if (strncmp(spec, "docc:", 5) == 0) {
        return chem_obs_double_occ(K, atoi(spec + 5), out, err, err_sz);
    }
    if (strncmp(spec, "sdot:", 5) == 0) {
        const char *c2 = strchr(spec + 5, ':');
        if (!c2) {
            snprintf(err, err_sz, "sdot expects --type sdot:<p>:<q>");
            return -1;
        }
        return chem_obs_spin_corr(K, atoi(spec + 5), atoi(c2 + 1), out, err, err_sz);
    }
    if (strncmp(spec, "hf-force:", 9) == 0) {
        char buf[2048];
        snprintf(buf, sizeof(buf), "%s", spec + 9);
        char *c1 = strchr(buf, ':');
        char *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
        if (!c1 || !c2) {
            snprintf(err, err_sz, "hf-force expects --type hf-force:<plus.fcidump>:<minus.fcidump>:<dR>");
            return -1;
        }
        *c1 = '\0';
        *c2 = '\0';
        double dr = atof(c2 + 1);
        if (dr <= 0.0) {
            snprintf(err, err_sz, "hf-force requires positive <dR>");
            return -1;
        }
        ChemHam hp, hm;
        if (chem_fcidump_load(buf, &hp, err, err_sz) != 0) return -1;
        if (chem_fcidump_load(c1 + 1, &hm, err, err_sz) != 0) {
            chem_ham_free(&hp);
            return -1;
        }
        int rc = chem_obs_ham_diff(&hp, &hm, 1.0 / (2.0 * dr), "hf-force", out, err, err_sz);
        chem_ham_free(&hp);
        chem_ham_free(&hm);
        return rc;
    }
    snprintf(err, err_sz, "unknown observable type '%s'", spec);
    return -1;
}

static int parse_mode_arg(const char *s, ChemMode *out) {
    if (strcmp(s, "auto") == 0) { *out = CHEM_MODE_AUTO; return 0; }
    if (strcmp(s, "a") == 0 || strcmp(s, "A") == 0) { *out = CHEM_MODE_A_SINGLET; return 0; }
    if (strcmp(s, "b") == 0 || strcmp(s, "B") == 0) { *out = CHEM_MODE_B_GENERAL; return 0; }
    return -1;
}

static int maybe_apply_auto_no(ChemHam *ham, ChemObservable *obs_opt, ChemSolveOpts *sopts,
                               int t1_cap, int t2_cap, const char *save_no_fcidump,
                               double *no_occs, double *no_ent, const char **fcidump_for_cert,
                               char *err, size_t err_sz) {
    int K = ham->K;
    double U_no[CHEM_MAX_ORBITALS * CHEM_MAX_ORBITALS];
    ChemHam ham_no;
    if (chem_ham_build_dqg_no(ham, sopts, &ham_no, no_occs, no_ent, U_no, err, err_sz) != 0) {
        return -1;
    }
    if (obs_opt) {
        ChemObservable obs_no;
        if (chem_obs_rotate_orthogonal(obs_opt, U_no, &obs_no, err, err_sz) != 0) {
            chem_ham_free(&ham_no);
            return -1;
        }
        chem_observable_free(obs_opt);
        *obs_opt = obs_no;
    }
    chem_ham_free(ham);
    *ham = ham_no;
    if (save_no_fcidump) {
        if (chem_fcidump_write(save_no_fcidump, ham, err, err_sz) != 0) return -1;
        if (fcidump_for_cert) *fcidump_for_cert = save_no_fcidump;
    }
    if (sopts->enable_t2) {
        int cap = (t2_cap <= 0 || t2_cap > K) ? K : t2_cap;
        sopts->t2_norbs = cap;
        for (int i = 0; i < cap; i++) sopts->t2_orbs[i] = i;
    }
    if (sopts->enable_t1) {
        int cap = (t1_cap <= 0 || t1_cap > K) ? K : t1_cap;
        sopts->t1_norbs = cap;
        for (int i = 0; i < cap; i++) sopts->t1_orbs[i] = i;
    }
    return 0;
}

static void write_json_double_arr(FILE *fp, int n, const double *arr) {
    fprintf(fp, "[");
    for (int i = 0; i < n; i++) fprintf(fp, "%s%.10g", (i > 0) ? ", " : "", arr[i]);
    fprintf(fp, "]");
}

/* Builds a strongly correlated symmetric 4-site Hubbard/molecular active-space test Hamiltonian */
static int build_synth_hubbard_ham(ChemHam *ham, int K, int nelec, int ms2, double t_hop, double u_int,
                                   double e_core) {
    char err[256] = {0};
    if (chem_ham_alloc(ham, K, nelec, ms2, err, sizeof(err)) != 0) return -1;
    ham->E_core = e_core;
    size_t ksz = (size_t)K;
    for (int p = 0; p < K; p++) {
        ham->h[(size_t)p * ksz + (size_t)p] = -0.35 * (double)p / (double)K;
        int q = (p + 1) % K;
        if (q != p) {
            ham->h[(size_t)p * ksz + (size_t)q] -= t_hop;
            ham->h[(size_t)q * ksz + (size_t)p] -= t_hop;
        }
        /* Local Hubbard U plus small off-site Coulomb repulsion */
        size_t idx = ((((size_t)p * ksz + (size_t)p) * ksz + (size_t)p) * ksz + (size_t)p);
        ham->eri[idx] = u_int;
        for (int r = p + 1; r < K; r++) {
            double v_pr = 0.25 * u_int / (double)(1 + (r - p));
            size_t i1 = ((((size_t)p * ksz + (size_t)p) * ksz + (size_t)r) * ksz + (size_t)r);
            size_t i2 = ((((size_t)r * ksz + (size_t)r) * ksz + (size_t)p) * ksz + (size_t)p);
            ham->eri[i1] = v_pr;
            ham->eri[i2] = v_pr;
        }
    }
    chem_ham_finalize(ham);
    return 0;
}

static void write_bracket_summary_json_ext(FILE *fp, const char *name, const ChemHam *ham,
                                           const ChemSolveResult *sr, const ChemVarResult *vr,
                                           const ChemSolveResult *sr_veto, double e_ai_synth,
                                           int auto_no, const double *no_occs,
                                           const double *no_ent) {
    fprintf(fp, "    {\n");
    fprintf(fp, "      \"name\": \"%s\",\n", name);
    fprintf(fp, "      \"K\": %d, \"nelec\": %d, \"ms2\": %d, \"na\": %d, \"nb\": %d,\n",
            ham->K, ham->nelec, ham->ms2, ham->na, ham->nb);
    fprintf(fp, "      \"sha256\": \"%s\",\n", ham->sha256_hex);
    fprintf(fp, "      \"mode\": \"%s\",\n",
            (sr->mode_used == CHEM_MODE_A_SINGLET) ? "A_SINGLET" : "B_GENERAL");
    fprintf(fp, "      \"E_lb_cert\": %.17g,\n", sr->E_lb_cert);
    fprintf(fp, "      \"E_primal\": %.17g,\n", sr->E_primal);
    fprintf(fp, "      \"E_dual_raw\": %.17g,\n", sr->E_dual_raw);
    fprintf(fp, "      \"min_eig_fp\": %.17g,\n", sr->min_eig_fp);
    fprintf(fp, "      \"min_eig_cert\": %.17g,\n", sr->min_eig_cert);
    fprintf(fp, "      \"iters\": %d, \"converged\": %d,\n", sr->iters, sr->converged);
    fprintf(fp, "      \"prim_res\": %.6e, \"dual_res\": %.6e,\n", sr->prim_res, sr->dual_res);
    fprintf(fp, "      \"sdp_sec\": %.6f,\n", sr->elapsed_sec);
    fprintf(fp, "      \"E_hf0\": %.17g, \"E_scf\": %.17g, \"E_searu_rot\": %.17g,\n",
            vr->E_hf0, vr->E_scf, vr->E_searu_rot);
    fprintf(fp, "      \"E_ci\": %.17g, \"E_ub\": %.17g,\n", vr->E_ci, vr->E_ub);
    fprintf(fp, "      \"is_exact_fci\": %d, \"ndet_space\": %llu, \"ndet_var\": %d,\n",
            vr->is_exact_fci, (unsigned long long)vr->ndet_space, vr->ndet_var);
    fprintf(fp, "      \"has_weinstein\": %d, \"E_weinstein\": %.17g,\n",
            vr->has_weinstein, vr->E_weinstein);
    fprintf(fp, "      \"has_temple\": %d, \"E_temple\": %.17g,\n",
            vr->has_temple, vr->E_temple);
    fprintf(fp, "      \"var_sec\": %.6f,\n", vr->elapsed_sec);
    fprintf(fp, "      \"interval_width\": %.17g,\n", vr->E_ub - sr->E_lb_cert);
    if (auto_no && no_occs && no_ent) {
        fprintf(fp, "      \"auto_no\": true,\n");
        fprintf(fp, "      \"no_occs\": ");
        write_json_double_arr(fp, ham->K, no_occs);
        fprintf(fp, ",\n      \"no_entropies\": ");
        write_json_double_arr(fp, ham->K, no_ent);
        fprintf(fp, ",\n");
    }
    if (sr_veto) {
        fprintf(fp, "      \"veto_test\": {\"E_ai\": %.17g, \"vetoed\": %d, \"iter_stopped\": %d, "
                    "\"full_iters\": %d, \"veto_sec\": %.6f}\n",
                e_ai_synth, sr_veto->vetoed, sr_veto->iter_stopped, sr->iters,
                sr_veto->elapsed_sec);
    } else {
        fprintf(fp, "      \"veto_test\": null\n");
    }
    fprintf(fp, "    }");
}

static void write_bracket_summary_json(FILE *fp, const char *name, const ChemHam *ham,
                                       const ChemSolveResult *sr, const ChemVarResult *vr,
                                       const ChemSolveResult *sr_veto, double e_ai_synth) {
    write_bracket_summary_json_ext(fp, name, ham, sr, vr, sr_veto, e_ai_synth, 0, NULL, NULL);
}

/* ============================================================================
 * Subcommand: selftest
 * ============================================================================ */

static int cmd_selftest(int argc, char **argv) {
    const char *json_out = NULL;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_out = argv[++i];
        } else {
            fprintf(stderr, "Unknown selftest option: %s\n", argv[i]);
            return 1;
        }
    }

    char err[256] = {0};
    printf("=== CHEM v0.3.0 Self-Test Suite ===\n");

    /* 1. Adjoint tests across Mode A & Mode B */
    const int adj_cases[6][6] = {
        {4, 2, 2, 0, 1, CHEM_MODE_A_SINGLET},
        {6, 3, 3, 0, 1, CHEM_MODE_A_SINGLET},
        {8, 4, 4, 0, 1, CHEM_MODE_A_SINGLET},
        {4, 2, 2, 0, 1, CHEM_MODE_B_GENERAL},
        {5, 2, 1, 1, 1, CHEM_MODE_B_GENERAL},
        {6, 4, 2, 2, 1, CHEM_MODE_B_GENERAL}
    };
    double max_adj_err = 0.0;
    for (int c = 0; c < 6; c++) {
        ChemAdjointReport rep;
        if (chem_test_adjoints(adj_cases[c][0], adj_cases[c][1], adj_cases[c][2],
                               adj_cases[c][3], adj_cases[c][4], (ChemMode)adj_cases[c][5],
                               UINT64_C(0x123456789ABCDEF) + (uint64_t)c * 97U, &rep, err,
                               sizeof(err)) != 0) {
            fprintf(stderr, "FAIL adjoint case %d: %s\n", c, err);
            return 1;
        }
        if (rep.max_err > max_adj_err) max_adj_err = rep.max_err;
        if (!(rep.max_err < 1e-14)) {
            fprintf(stderr, "FAIL adjoint error too large (case %d): %.3e\n", c, rep.max_err);
            return 1;
        }
    }
    printf("  [PASS] Matrix-Free Adjoints (Mode A & Mode B): max_err = %.3e (< 1e-14)\n",
           max_adj_err);

    /* 2. Synthetic strongly correlated H4 (4e,4o) singlet: FCI CAR check + Mode A vs B */
    ChemHam h4;
    if (build_synth_hubbard_ham(&h4, 4, 4, 0, 1.0, 4.0, 0.5) != 0) return 1;
    ChemVarOpts vopts = chem_var_opts_default();
    vopts.target_two_s = 0;
    ChemVarResult vr4;
    if (chem_var_solve(&h4, &vopts, &vr4, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL H4 FCI: %s\n", err);
        return 1;
    }

    double *Da = NULL, *Qa = NULL, *Ga = NULL;
    double *Db = NULL, *Qb = NULL, *Gb = NULL;
    if (chem_fci_compute_dqg_mode_a(&h4, vr4.ndet_var, vr4.dets, vr4.psi0, &Da, &Qa, &Ga, err,
                                    sizeof(err)) != 0 ||
        chem_fci_compute_dqg_mode_b(&h4, vr4.ndet_var, vr4.dets, vr4.psi0, &Db, &Qb, &Gb, err,
                                    sizeof(err)) != 0) {
        fprintf(stderr, "FAIL FCI 2-RDM computation: %s\n", err);
        return 1;
    }

    double car_a = 0, eq_a = 0, me_a = 0, e_a = 0;
    double car_b = 0, eq_b = 0, me_b = 0, e_b = 0;
    if (chem_verify_fci_dqg_consistency(&h4, CHEM_MODE_A_SINGLET, 0, Da, Qa, Ga, &car_a, &eq_a,
                                        &me_a, &e_a, err, sizeof(err)) != 0 ||
        chem_verify_fci_dqg_consistency(&h4, CHEM_MODE_B_GENERAL, 0, Db, Qb, Gb, &car_b, &eq_b,
                                        &me_b, &e_b, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL FCI consistency: %s\n", err);
        return 1;
    }
    /* Check exact T2(D_FCI) and T1(D_FCI) PSD and traces on H4 Mode A & Mode B before freeing Da/Db */
    double me_t2_a = 0.0, te_t2_a = 0.0, me_t2_b = 0.0, te_t2_b = 0.0;
    double me_t1_a = 0.0, te_t1_a = 0.0, me_t1_b = 0.0, te_t1_b = 0.0;
    const int orbs4[4] = {0, 1, 2, 3};
    if (chem_verify_fci_t2_psd(&h4, CHEM_MODE_A_SINGLET, 0, 4, orbs4, Da, &me_t2_a, &te_t2_a, err,
                               sizeof(err)) != 0 ||
        chem_verify_fci_t2_psd(&h4, CHEM_MODE_B_GENERAL, 0, 4, orbs4, Db, &me_t2_b, &te_t2_b, err,
                               sizeof(err)) != 0 ||
        chem_verify_fci_t1_psd(&h4, CHEM_MODE_A_SINGLET, 0, 4, orbs4, Da, &me_t1_a, &te_t1_a, err,
                               sizeof(err)) != 0 ||
        chem_verify_fci_t1_psd(&h4, CHEM_MODE_B_GENERAL, 0, 4, orbs4, Db, &me_t1_b, &te_t1_b, err,
                               sizeof(err)) != 0) {
        fprintf(stderr, "FAIL FCI T2/T1 PSD: %s\n", err);
        return 1;
    }
    if (me_t2_a < -1e-13 || te_t2_a > 1e-12 || me_t2_b < -1e-13 || te_t2_b > 1e-12 ||
        me_t1_a < -1e-13 || te_t1_a > 1e-12 || me_t1_b < -1e-13 || te_t1_b > 1e-12) {
        fprintf(stderr,
                "FAIL FCI T2/T1 PSD/trace: T2=(%.2e,%.2e)/(%.2e,%.2e) T1=(%.2e,%.2e)/(%.2e,%.2e)\n",
                me_t2_a, te_t2_a, me_t2_b, te_t2_b, me_t1_a, te_t1_a, me_t1_b, te_t1_b);
        return 1;
    }

    free(Da); free(Qa); free(Ga);
    free(Db); free(Qb); free(Gb);

    if (car_a > 1e-12 || eq_a > 1e-12 || me_a < -1e-13 || fabs(e_a - vr4.E_ci) > 1e-11 ||
        car_b > 1e-12 || eq_b > 1e-12 || me_b < -1e-13 || fabs(e_b - vr4.E_ci) > 1e-11) {
        fprintf(stderr, "FAIL FCI CAR/PSD/Energy mismatch: A(%.2e,%.2e,%.2e,%.2e) B(%.2e,%.2e,%.2e,%.2e)\n",
                car_a, eq_a, me_a, fabs(e_a - vr4.E_ci),
                car_b, eq_b, me_b, fabs(e_b - vr4.E_ci));
        return 1;
    }
    printf("  [PASS] H4 FCI Exact CAR & PSD (+T2/+T1): car_A=%.2e car_B=%.2e t2=%.2e t1=%.2e dE=%.2e\n",
           car_a, car_b, fmin(me_t2_a, me_t2_b), fmin(me_t1_a, me_t1_b),
           fmax(fabs(e_a - vr4.E_ci), fabs(e_b - vr4.E_ci)));

    /* 1b. T2 & T1 matrix-free adjoint checks across Mode A & Mode B */
    const int t12_adj_cases[7][6] = {
        {4, 2, 2, CHEM_MODE_A_SINGLET, 4, 0},
        {6, 3, 3, CHEM_MODE_A_SINGLET, 6, 0},
        {4, 2, 2, CHEM_MODE_A_SINGLET, 0, 4},
        {6, 3, 3, CHEM_MODE_A_SINGLET, 4, 5},
        {4, 2, 2, CHEM_MODE_B_GENERAL, 4, 4},
        {5, 3, 1, CHEM_MODE_B_GENERAL, 0, 5},
        {6, 4, 2, CHEM_MODE_B_GENERAL, 4, 6}
    };
    double max_t2_adj = 0.0, max_t1_adj = 0.0;
    for (int c = 0; c < 7; c++) {
        int norb2 = t12_adj_cases[c][4], norb1 = t12_adj_cases[c][5];
        int orbs[6] = {0, 1, 2, 3, 4, 5};
        ChemAdjointReport rep;
        int na_c = t12_adj_cases[c][1], nb_c = t12_adj_cases[c][2];
        if (chem_test_adjoints_t12(t12_adj_cases[c][0], na_c, nb_c, na_c - nb_c, 1,
                                   (ChemMode)t12_adj_cases[c][3], norb2, norb2 > 0 ? orbs : NULL,
                                   norb1, norb1 > 0 ? orbs : NULL,
                                   UINT64_C(0x55551111) + (uint64_t)c, &rep, err, sizeof(err)) !=
            0) {
            fprintf(stderr, "FAIL T1/T2 adjoint case %d: %s\n", c, err);
            return 1;
        }
        if (rep.err_LT2 > max_t2_adj) max_t2_adj = rep.err_LT2;
        if (rep.err_LT1 > max_t1_adj) max_t1_adj = rep.err_LT1;
        if (!(rep.max_err < 1e-14)) {
            fprintf(stderr, "FAIL T1/T2 adjoint error %.3e (case %d)\n", rep.max_err, c);
            return 1;
        }
    }
    printf("  [PASS] Matrix-Free T2 & T1 Adjoints (Mode A & B): T2=%.3e T1=%.3e (< 1e-14)\n",
           max_t2_adj, max_t1_adj);

    /* 2a. 2-electron singlet (2e,4o) Mode A == Mode B exact agreement */
    ChemHam h2_4o;
    if (build_synth_hubbard_ham(&h2_4o, 4, 2, 0, 1.0, 3.0, 0.2) != 0) return 1;
    ChemSolveOpts sopts_2a = chem_solve_opts_default(&h2_4o);
    sopts_2a.mode = CHEM_MODE_A_SINGLET;
    sopts_2a.tol = 1e-7;
    ChemSolveOpts sopts_2b = chem_solve_opts_default(&h2_4o);
    sopts_2b.mode = CHEM_MODE_B_GENERAL;
    sopts_2b.tol = 1e-7;
    ChemSolveResult sr2_a, sr2_b;
    if (chem_solve_dqg(&h2_4o, &sopts_2a, &sr2_a, err, sizeof(err)) != 0 ||
        chem_solve_dqg(&h2_4o, &sopts_2b, &sr2_b, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL 2e4o SDP: %s\n", err);
        return 1;
    }
    if (fabs(sr2_a.E_lb_cert - sr2_b.E_lb_cert) > 1e-5) {
        fprintf(stderr, "FAIL 2e4o Mode A vs B: LB_A=%.8f LB_B=%.8f\n",
                sr2_a.E_lb_cert, sr2_b.E_lb_cert);
        return 1;
    }
    printf("  [PASS] 2e4o Singlet Mode A==B: LB_A=%.8f LB_B=%.8f (|A-B|=%.2e)\n",
           sr2_a.E_lb_cert, sr2_b.E_lb_cert, fabs(sr2_a.E_lb_cert - sr2_b.E_lb_cert));
    chem_solve_result_free(&sr2_a);
    chem_solve_result_free(&sr2_b);
    chem_ham_free(&h2_4o);

    /* 2b. Solve Mode A vs Mode B SDP on 4e4o H4 (hierarchy B <= A <= FCI <= UB) */
    ChemSolveOpts sopts_a = chem_solve_opts_default(&h4);
    sopts_a.mode = CHEM_MODE_A_SINGLET;
    ChemSolveOpts sopts_b = chem_solve_opts_default(&h4);
    sopts_b.mode = CHEM_MODE_B_GENERAL;

    ChemSolveResult sr4_a, sr4_b;
    if (chem_solve_dqg(&h4, &sopts_a, &sr4_a, err, sizeof(err)) != 0 ||
        chem_solve_dqg(&h4, &sopts_b, &sr4_b, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL H4 SDP: %s\n", err);
        return 1;
    }
    if (sr4_a.E_lb_cert > vr4.E_ci + 1e-11 || sr4_b.E_lb_cert > vr4.E_ci + 1e-11 ||
        sr4_b.E_lb_cert > sr4_a.E_lb_cert + 1e-4 || vr4.E_ci > vr4.E_ub + 1e-11) {
        fprintf(stderr, "FAIL H4 bracket hierarchy: LB_B=%.8f LB_A=%.8f FCI=%.8f UB=%.8f\n",
                sr4_b.E_lb_cert, sr4_a.E_lb_cert, vr4.E_ci, vr4.E_ub);
        return 1;
    }
    printf("  [PASS] H4 (4e,4o) Bracket: LB_B=%.8f <= LB_A=%.8f <= FCI=%.8f <= UB=%.8f\n",
           sr4_b.E_lb_cert, sr4_a.E_lb_cert, vr4.E_ci, vr4.E_ub);

    /* 3. Open-shell H3 (3e,3o) doublet test (Mode B) */
    ChemHam h3;
    if (build_synth_hubbard_ham(&h3, 3, 3, 1, 1.0, 3.5, 0.2) != 0) return 1;
    ChemVarOpts vopts3 = chem_var_opts_default();
    vopts3.target_two_s = 1;
    ChemVarResult vr3;
    if (chem_var_solve(&h3, &vopts3, &vr3, err, sizeof(err)) != 0) return 1;
    ChemSolveOpts sopts3 = chem_solve_opts_default(&h3);
    ChemSolveResult sr3;
    if (chem_solve_dqg(&h3, &sopts3, &sr3, err, sizeof(err)) != 0) return 1;
    if (sr3.E_lb_cert > vr3.E_ci || vr3.E_ci > vr3.E_ub + 1e-12) {
        fprintf(stderr, "FAIL H3 doublet bracket: LB=%.8f FCI=%.8f UB=%.8f\n",
                sr3.E_lb_cert, vr3.E_ci, vr3.E_ub);
        return 1;
    }
    printf("  [PASS] H3 (3e,3o) Doublet Mode B: LB=%.8f <= FCI=%.8f <= UB=%.8f\n",
           sr3.E_lb_cert, vr3.E_ci, vr3.E_ub);

    /* 4. Anytime Early-Veto test */
    ChemSolveOpts v_sopts = chem_solve_opts_default(&h4);
    v_sopts.mode = CHEM_MODE_A_SINGLET;
    v_sopts.veto_enabled = 1;
    v_sopts.E_ai = vr4.E_ci - 0.15;
    v_sopts.veto_tol = 1e-4;
    v_sopts.check_cert_every = 5;
    ChemSolveResult sr4_veto;
    if (chem_solve_dqg(&h4, &v_sopts, &sr4_veto, err, sizeof(err)) != 0) return 1;
    if (!sr4_veto.vetoed || sr4_veto.iter_stopped >= sr4_a.iters) {
        fprintf(stderr, "FAIL Early-Veto: vetoed=%d stopped=%d full=%d\n",
                sr4_veto.vetoed, sr4_veto.iter_stopped, sr4_a.iters);
        return 1;
    }
    printf("  [PASS] Anytime Early-Veto: stopped at iter %d (vs full %d iters, speedup %.1fx)\n",
           sr4_veto.iter_stopped, sr4_a.iters,
           (double)sr4_a.iters / (double)(sr4_veto.iter_stopped > 0 ? sr4_veto.iter_stopped : 1));

    /* 5. Certificate round-trip (JSON + binary .flowchem) + tamper rejection + witness load */
    const char *tmp_fci = "build/.selftest_h4.fcidump";
    const char *tmp_cert = "build/.selftest_h4.cert.json";
    const char *tmp_bin = "build/.selftest_h4.flowchem";
    if (chem_fcidump_write(tmp_fci, &h4, err, sizeof(err)) != 0) {
        tmp_fci = ".selftest_h4.fcidump";
        tmp_cert = ".selftest_h4.cert.json";
        tmp_bin = ".selftest_h4.flowchem";
        if (chem_fcidump_write(tmp_fci, &h4, err, sizeof(err)) != 0) return 1;
    }
    ChemHam h4_reloaded;
    if (chem_fcidump_load(tmp_fci, &h4_reloaded, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL FCIDUMP reload: %s\n", err);
        return 1;
    }
    ChemSolveResult sr4_r;
    if (chem_solve_dqg(&h4_reloaded, &sopts_a, &sr4_r, err, sizeof(err)) != 0) return 1;
    if (chem_cert_write_json(tmp_cert, &h4_reloaded, tmp_fci, &sr4_r, &vr4, &sopts_a, err,
                             sizeof(err)) != 0 ||
        chem_cert_write_bin(tmp_bin, &h4_reloaded, tmp_fci, &sr4_r, &vr4, &sopts_a, err,
                            sizeof(err)) != 0) {
        fprintf(stderr, "FAIL write cert: %s\n", err);
        return 1;
    }
    ChemCertVerifyResult cvr, cvr_bin;
    if (chem_cert_verify_file(NULL, tmp_cert, &cvr) != 0 || !cvr.ok) {
        fprintf(stderr, "FAIL cert verify: %s\n", cvr.msg);
        return 1;
    }
    if (chem_cert_verify_file(NULL, tmp_bin, &cvr_bin) != 0 || !cvr_bin.ok || !cvr_bin.is_binary) {
        fprintf(stderr, "FAIL binary cert verify: %s\n", cvr_bin.msg);
        return 1;
    }

    ChemExactVerifyResult ev_json, ev_bin;
    if (chem_cert_verify_exact_chol(NULL, tmp_cert, &ev_json) != 0 || !ev_json.ok ||
        !ev_json.all_blocks_sdd || fabs(ev_json.E_lb_int_chol - sr4_a.E_lb_cert) >= 2e-3 ||
        ev_json.E_lb_int_chol > vr4.E_ub + 1e-9 ||
        chem_cert_verify_exact_chol(NULL, tmp_bin, &ev_bin) != 0 || !ev_bin.ok ||
        !ev_bin.all_blocks_sdd || fabs(ev_bin.E_lb_int_chol - sr4_a.E_lb_cert) >= 2e-3) {
        fprintf(stderr, "FAIL exact Cholesky+SDD verify: %s / %s\n", ev_json.msg, ev_bin.msg);
        return 1;
    }
    printf("  [PASS] Zero-FP-Trust Cholesky+SDD (JSON & .flowchem): diff=%.2e Ha\n",
           fabs(ev_json.E_lb_int_chol - sr4_a.E_lb_cert));

    /* External witness ingestion test reading variational witness from tmp_cert */
    ChemVarResult vr_ext;
    if (chem_var_load_witness_json(tmp_cert, &h4_reloaded, 0, &vr_ext, err, sizeof(err)) != 0 ||
        fabs(vr_ext.E_ub - vr4.E_ci) > 1e-10) {
        fprintf(stderr, "FAIL external witness load: %s\n", err);
        return 1;
    }
    chem_var_result_free(&vr_ext);

    /* Tamper test: corrupt E_lb_cert slightly in JSON + flip byte in binary cert */
    FILE *fp_r = fopen(tmp_cert, "rb");
    fseek(fp_r, 0, SEEK_END);
    long csz = ftell(fp_r);
    fseek(fp_r, 0, SEEK_SET);
    char *cbuf = (char *)malloc((size_t)csz + 1U);
    fread(cbuf, 1, (size_t)csz, fp_r);
    fclose(fp_r);
    cbuf[csz] = '\0';
    char *pos = strstr(cbuf, "\"E_lb_cert\": ");
    if (pos) {
        char *nump = pos + strlen("\"E_lb_cert\": ");
        if (*nump == '-') nump++;
        *nump = (*nump == '9') ? '8' : '9';
    }
    FILE *fp_w = fopen(tmp_cert, "wb");
    fwrite(cbuf, 1, (size_t)csz, fp_w);
    fclose(fp_w);
    free(cbuf);

    FILE *bf_r = fopen(tmp_bin, "r+b");
    if (bf_r) {
        fseek(bf_r, -2, SEEK_END);
        int bch = fgetc(bf_r);
        fseek(bf_r, -1, SEEK_CUR);
        fputc(bch ^ 0x40, bf_r);
        fclose(bf_r);
    }

    ChemCertVerifyResult cvr_tamper, cvr_bin_tamper;
    if (chem_cert_verify_file(NULL, tmp_cert, &cvr_tamper) == 0 || cvr_tamper.ok ||
        chem_cert_verify_file(NULL, tmp_bin, &cvr_bin_tamper) == 0 || cvr_bin_tamper.ok) {
        fprintf(stderr, "FAIL tamper test: corrupted certificate unexpectedly passed!\n");
        return 1;
    }
    remove(tmp_fci);
    remove(tmp_cert);
    remove(tmp_bin);
    printf("  [PASS] Certificate JSON + Binary .flowchem + External Witness & Tamper Rejection\n");

    /* 5b. DQG+T2, DQG+T1, DQG+T1+T2 tightening, primal grad, & Triplet T1 FCI PSD */
    ChemSolveOpts sopts_t2 = sopts_a;
    sopts_t2.enable_t2 = 1;
    sopts_t2.t2_norbs = 4;
    for (int i = 0; i < 4; i++) sopts_t2.t2_orbs[i] = i;
    ChemSolveResult sr4_t2;
    if (chem_solve_dqg(&h4, &sopts_t2, &sr4_t2, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL H4 DQG+T2: %s\n", err);
        return 1;
    }
    if (sr4_t2.E_lb_cert < sr4_a.E_lb_cert - 1e-7 || sr4_t2.E_lb_cert > vr4.E_ci + 1e-11) {
        fprintf(stderr, "FAIL H4 DQG+T2 bracket: DQG=%.8f DQG+T2=%.8f FCI=%.8f\n",
                sr4_a.E_lb_cert, sr4_t2.E_lb_cert, vr4.E_ci);
        return 1;
    }

    ChemSolveOpts sopts_t1 = sopts_a;
    sopts_t1.enable_t1 = 1;
    sopts_t1.t1_norbs = 4;
    sopts_t1.export_rdm_grad = 1;
    for (int i = 0; i < 4; i++) sopts_t1.t1_orbs[i] = i;
    ChemSolveResult sr4_t1;
    if (chem_solve_dqg(&h4, &sopts_t1, &sr4_t1, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL H4 DQG+T1: %s\n", err);
        return 1;
    }
    double e_from_grad = h4.E_core;
    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 4; q++) {
            e_from_grad += chem_ham_h1(&h4, p, q) * sr4_t1.primal_grad_h1[p * 4 + q];
        }
    }
    for (int p = 0; p < 4; p++) {
        for (int r = 0; r < 4; r++) {
            for (int q = 0; q < 4; q++) {
                for (int s = 0; s < 4; s++) {
                    size_t idx = (size_t)(((p * 4 + r) * 4 + q) * 4 + s);
                    e_from_grad += chem_ham_eri(&h4, p, r, q, s) * sr4_t1.primal_grad_eri[idx];
                }
            }
        }
    }
    if (sr4_t1.E_lb_cert > vr4.E_ci + 1e-10 || fabs(e_from_grad - sr4_t1.E_primal) > 1e-11) {
        fprintf(stderr, "FAIL H4 DQG+T1 or grad: LB=%.8f FCI=%.8f dE_grad=%.2e\n",
                sr4_t1.E_lb_cert, vr4.E_ci, fabs(e_from_grad - sr4_t1.E_primal));
        return 1;
    }

    ChemHam h4t;
    if (build_synth_hubbard_ham(&h4t, 4, 4, 2, 1.0, 4.0, 0.5) != 0) return 1;
    ChemVarOpts vopts_t = chem_var_opts_default();
    vopts_t.target_two_s = 2;
    ChemVarResult vr4t;
    if (chem_var_solve(&h4t, &vopts_t, &vr4t, err, sizeof(err)) != 0) return 1;
    double *D4t = NULL, *Q4t = NULL, *G4t = NULL;
    double me_t1_4t = 0.0, te_t1_4t = 0.0;
    if (chem_fci_compute_dqg_mode_b(&h4t, vr4t.ndet_var, vr4t.dets, vr4t.psi0, &D4t, &Q4t, &G4t,
                                    err, sizeof(err)) != 0 ||
        chem_verify_fci_t1_psd(&h4t, CHEM_MODE_B_GENERAL, 2, 4, orbs4, D4t, &me_t1_4t, &te_t1_4t,
                               err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL H4 triplet FCI T1 PSD: %s\n", err);
        return 1;
    }
    free(D4t); free(Q4t); free(G4t);
    if (me_t1_4t < -1e-13 || te_t1_4t > 1e-12) {
        fprintf(stderr, "FAIL H4 triplet T1 PSD/trace: eig=%.2e tr=%.2e\n", me_t1_4t, te_t1_4t);
        return 1;
    }
    ChemSolveOpts sopts_t12 = chem_solve_opts_default(&h4t);
    sopts_t12.mode = CHEM_MODE_B_GENERAL;
    sopts_t12.enable_t2 = 1; sopts_t12.t2_norbs = 4;
    sopts_t12.enable_t1 = 1; sopts_t12.t1_norbs = 4;
    sopts_t12.export_rdm_grad = 1;
    for (int i = 0; i < 4; i++) { sopts_t12.t2_orbs[i] = i; sopts_t12.t1_orbs[i] = i; }
    ChemSolveResult sr4t_t12;
    if (chem_solve_dqg(&h4t, &sopts_t12, &sr4t_t12, err, sizeof(err)) != 0 ||
        sr4t_t12.E_lb_cert > vr4t.E_ci + 1e-10) {
        fprintf(stderr, "FAIL H4 triplet DQG+T1+T2: LB=%.8f FCI=%.8f\n",
                sr4t_t12.E_lb_cert, vr4t.E_ci);
        return 1;
    }
    printf("  [PASS] H4 Tightening & T1: DQG=%.6f T1=%.6f T2=%.6f <= FCI=%.6f | Triplet T1+T2=%.6f <= FCI=%.6f (t1_eig=%.2e grad_dE=%.2e)\n",
           sr4_a.E_lb_cert, sr4_t1.E_lb_cert, sr4_t2.E_lb_cert, vr4.E_ci,
           sr4t_t12.E_lb_cert, vr4t.E_ci, me_t1_4t, fabs(e_from_grad - sr4_t1.E_primal));
    chem_solve_result_free(&sr4_t1);
    chem_solve_result_free(&sr4t_t12);
    chem_var_result_free(&vr4t);
    chem_ham_free(&h4t);

    ChemObservable obs_h4[3];
    ChemObservableResult ores_h4;
    if (chem_obs_orbital_occ(4, 0, &obs_h4[0], err, sizeof(err)) != 0 ||
        chem_obs_double_occ(4, 0, &obs_h4[1], err, sizeof(err)) != 0 ||
        chem_obs_spin_corr(4, 0, 1, &obs_h4[2], err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL obs alloc: %s\n", err);
        return 1;
    }
    for (int k = 0; k < 3; k++) {
        double o_fci = 0.0;
        if (chem_eval_observable_witness(&h4, &obs_h4[k], vr4.ndet_var, vr4.dets, vr4.psi0,
                                         &o_fci, err, sizeof(err)) != 0 ||
            chem_certify_observable(&h4, &obs_h4[k], vr4.E_ub, &sopts_a, &ores_h4, err,
                                    sizeof(err)) != 0) {
            fprintf(stderr, "FAIL obs certify k=%d: %s\n", k, err);
            return 1;
        }
        if (ores_h4.O_lb_cert > o_fci + 1e-10 || o_fci > ores_h4.O_ub_cert + 1e-10) {
            fprintf(stderr, "FAIL H4 obs %s: LB=%.8f FCI=%.8f UB=%.8f\n",
                    obs_h4[k].name, ores_h4.O_lb_cert, o_fci, ores_h4.O_ub_cert);
            return 1;
        }
        chem_observable_free(&obs_h4[k]);
    }
    ChemObservable obs_h3;
    ChemObservableResult ores_h3;
    if (chem_obs_double_occ(3, 0, &obs_h3, err, sizeof(err)) != 0) return 1;
    double h3_docc_fci = 0.0;
    if (chem_eval_observable_witness(&h3, &obs_h3, vr3.ndet_var, vr3.dets, vr3.psi0,
                                     &h3_docc_fci, err, sizeof(err)) != 0 ||
        chem_certify_observable(&h3, &obs_h3, vr3.E_ub, &sopts3, &ores_h3, err, sizeof(err)) != 0 ||
        ores_h3.O_lb_cert > h3_docc_fci + 1e-10 || h3_docc_fci > ores_h3.O_ub_cert + 1e-10) {
        fprintf(stderr, "FAIL H3 docc obs bracket\n");
        return 1;
    }
    chem_observable_free(&obs_h3);
    printf("  [PASS] Certified Observables (H4 Mode A & H3 Mode B): [%.6f, %.6f] contains FCI %.6f\n",
           ores_h3.O_lb_cert, ores_h3.O_ub_cert, h3_docc_fci);
    chem_solve_result_free(&sr4_t2);

    /* 5c. Borland-Dennis B_{3,6} GPC + Exact Full-Space Variance & Weinstein Bound */
    ChemGPCResult gpc3;
    if (chem_var_analyze_gpc(&h3, vr3.ndet_var, vr3.dets, vr3.psi0, &gpc3, err, sizeof(err)) != 0 ||
        !gpc3.has_bd_3e3o || gpc3.bd_pair_err >= 1e-12 || gpc3.bd_slack < -1e-12) {
        fprintf(stderr, "FAIL H3 Borland-Dennis B_{3,6} GPC check: pair_err=%.2e slack=%.6f\n",
                gpc3.bd_pair_err, gpc3.bd_slack);
        return 1;
    }
    printf("  [PASS] Borland-Dennis B_{3,6} GPC (H3 Doublet): pair_err=%.2e slack=%.2e (pinned=%d)\n",
           gpc3.bd_pair_err, gpc3.bd_slack, gpc3.bd_pinned);

    double e_fci_re = 0.0, var_fci = 1.0;
    /* Construct low-energy 4-det truncated trial from largest-amplitude FCI determinants (inside gap) */
    uint64_t tr_dets[4] = {0};
    double tr_c[4] = {0};
    int chosen[64] = {0};
    int k_tr = (vr4.ndet_var >= 4) ? 4 : vr4.ndet_var;
    for (int t = 0; t < k_tr; t++) {
        int best_i = -1;
        for (int i = 0; i < vr4.ndet_var; i++) {
            if (!chosen[i] && (best_i < 0 || fabs(vr4.psi0[i]) > fabs(vr4.psi0[best_i]))) {
                best_i = i;
            }
        }
        chosen[best_i] = 1;
        tr_dets[t] = vr4.dets[best_i];
        tr_c[t] = vr4.psi0[best_i];
    }
    double e_tr_re = 0.0, var_tr = 0.0;
    if (chem_var_eval_variance(&h4, vr4.ndet_var, vr4.dets, vr4.psi0, &e_fci_re, &var_fci, err,
                               sizeof(err)) != 0 ||
        sqrt(fmax(0.0, var_fci)) >= 1e-11 ||
        chem_var_eval_variance(&h4, k_tr, tr_dets, tr_c, &e_tr_re, &var_tr, err, sizeof(err)) != 0 ||
        var_tr <= 1e-4 || (e_tr_re - sqrt(var_tr)) > vr4.E_ci + 1e-10) {
        fprintf(stderr, "FAIL Full-Space Variance check: sig_fci=%.2e sig_tr=%.6f E_tr=%.6f E_ci=%.6f\n",
                sqrt(fmax(0.0, var_fci)), sqrt(fmax(0.0, var_tr)), e_tr_re, vr4.E_ci);
        return 1;
    }
    printf("  [PASS] Exact Full-Space Variance: sig(FCI)=%.2e sig(4det)=%.4f (Weinstein=%.6f <= FCI=%.6f)\n",
           sqrt(fmax(0.0, var_fci)), sqrt(var_tr), e_tr_re - sqrt(var_tr), vr4.E_ci);

    /* 6. FLOW 2-Jet & SEARU primitive sanity check */
    uint64_t rng = UINT64_C(0xCAFE123456789ABC);
    uint64_t swan = 0;
    uint64_t jmp = searu_zipf_jump(&rng, SEARU_CHAOS_DEFAULT_Q8, &swan);
    uint64_t f_st[8] = {(uint64_t)(int64_t)(65536), (uint64_t)(int64_t)(-65536), 0, 0, 0, 0, 0, 0};
    uint64_t f_bd[8] = {0, 0, 0, ((uint64_t)(uint32_t)(-5 * 65536) << 32U) | (uint32_t)(5 * 65536),
                        ((uint64_t)(uint32_t)(65536) << 32U), 0, 0, 0};
    uint64_t f_out[8] = {0};
    flow_step(f_out, f_st, f_bd);
    if (jmp == 0 || (int64_t)f_out[3] == 0) {
        fprintf(stderr, "FAIL FLOW/SEARU sanity check\n");
        return 1;
    }
    printf("  [PASS] FLOW 2-Jet & SEARU Primitives: flow_step x_next=%.4f zipf_jmp=%llu\n",
           (double)(int64_t)f_out[3] / 65536.0, (unsigned long long)jmp);

    /* 7. Orthogonal basis rotation invariance & DQG Natural Orbital (--auto-no) */
    const double U_rot[16] = {
         0.5,  0.5,  0.5,  0.5,
         0.5, -0.5,  0.5, -0.5,
         0.5,  0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,  0.5
    };
    ChemHam h4_orth, h4_no;
    ChemVarResult vr4_orth, vr4_no;
    ChemSolveResult sr4_orth, sr4_no;
    double no_occs[CHEM_MAX_ORBITALS] = {0}, no_ent[CHEM_MAX_ORBITALS] = {0};
    if (chem_ham_rotate_orthogonal(&h4, U_rot, &h4_orth, err, sizeof(err)) != 0 ||
        chem_var_solve(&h4_orth, &vopts, &vr4_orth, err, sizeof(err)) != 0 ||
        fabs(vr4_orth.E_ci - vr4.E_ci) > 1e-12 ||
        chem_solve_dqg(&h4_orth, &sopts_a, &sr4_orth, err, sizeof(err)) != 0 ||
        fabs(sr4_orth.E_lb_cert - sr4_a.E_lb_cert) > 1e-4) {
        fprintf(stderr, "FAIL orthogonal invariance check\n");
        return 1;
    }
    ChemSolveOpts no_opts = sopts_a;
    no_opts.enable_t2 = 1;
    no_opts.t2_norbs = 3;
    no_opts.t2_orbs[0] = 0; no_opts.t2_orbs[1] = 1; no_opts.t2_orbs[2] = 2;
    if (chem_ham_build_dqg_no(&h4_orth, &sopts_a, &h4_no, no_occs, no_ent, NULL, err, sizeof(err)) != 0 ||
        chem_var_solve(&h4_no, &vopts, &vr4_no, err, sizeof(err)) != 0 ||
        chem_solve_dqg(&h4_no, &no_opts, &sr4_no, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL DQG auto-NO check: %s\n", err);
        return 1;
    }
    for (int k = 0; k + 1 < h4_no.K; k++) {
        if (no_ent[k] + 1e-12 < no_ent[k + 1]) {
            fprintf(stderr, "FAIL auto-NO entanglement ordering at k=%d: %.6f < %.6f\n",
                    k, no_ent[k], no_ent[k + 1]);
            return 1;
        }
    }
    if (sr4_no.E_lb_cert > vr4.E_ci + 1e-9 || vr4_no.E_ci < sr4_no.E_lb_cert - 1e-9 ||
        fabs(vr4_no.E_ci - vr4.E_ci) > 1e-12) {
        fprintf(stderr, "FAIL auto-NO bracket check\n");
        return 1;
    }
    printf("  [PASS] Orthogonal Invariance & DQG Natural Orbitals: dFCI=%.2e dDQG=%.2e S_ord=[%.3f,%.3f,%.3f,%.3f] LB_no=%.6f\n",
           fabs(vr4_orth.E_ci - vr4.E_ci), fabs(sr4_orth.E_lb_cert - sr4_a.E_lb_cert),
           no_ent[0], no_ent[1], no_ent[2], no_ent[3], sr4_no.E_lb_cert);
    chem_solve_result_free(&sr4_no);
    chem_solve_result_free(&sr4_orth);
    chem_var_result_free(&vr4_no);
    chem_var_result_free(&vr4_orth);
    chem_ham_free(&h4_no);
    chem_ham_free(&h4_orth);

    if (json_out) {
        FILE *jf = fopen(json_out, "w");
        if (jf) {
            fprintf(jf, "{\n  \"max_adjoint_err\": %.3e,\n  \"h4_car_err_a\": %.3e,\n"
                        "  \"h4_car_err_b\": %.3e,\n  \"h4_lb_a\": %.12f,\n  \"h4_lb_b\": %.12f,\n"
                        "  \"h4_fci\": %.12f,\n  \"veto_iter\": %d,\n  \"full_iter\": %d,\n"
                        "  \"passed\": true\n}\n",
                    max_adj_err, car_a, car_b, sr4_a.E_lb_cert, sr4_b.E_lb_cert, vr4.E_ci,
                    sr4_veto.iter_stopped, sr4_a.iters);
            fclose(jf);
        }
    }

    chem_solve_result_free(&sr4_a);
    chem_solve_result_free(&sr4_b);
    chem_solve_result_free(&sr4_veto);
    chem_solve_result_free(&sr4_r);
    chem_solve_result_free(&sr3);
    chem_var_result_free(&vr4);
    chem_var_result_free(&vr3);
    chem_ham_free(&h4);
    chem_ham_free(&h4_reloaded);
    chem_ham_free(&h3);
    printf("ALL CHEM SELFTESTS PASSED.\n");
    return 0;
}

/* ============================================================================
 * Subcommands: bracket & veto
 * ============================================================================ */

static int cmd_bracket_or_veto(int is_veto, int argc, char **argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }
    const char *fcidump = argv[2];
    const char *cert_path = NULL;
    const char *cert_bin_path = NULL;
    const char *json_path = NULL;
    const char *witness_path = NULL;
    const char *dump_grad_path = NULL;
    const char *save_no_fcidump = NULL;
    int has_e_ai = 0;
    double e_ai = 0.0;
    double veto_tol = 1e-4;
    ChemMode mode = CHEM_MODE_AUTO;
    int two_s_override = -1;
    int constrain_s2 = 1;
    int max_iters = 800;
    double tol = 1e-6;
    int max_dets = 1024;
    int enable_t2 = 0;
    int t2_cap = 6;
    int enable_t1 = 0;
    int t1_cap = 6;
    int auto_no = 0;
    int num_threads = 0;
    int verbose = 0;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (parse_mode_arg(argv[++i], &mode) != 0) {
                fprintf(stderr, "Invalid --mode\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--two-s") == 0 && i + 1 < argc) {
            two_s_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-s2") == 0) {
            constrain_s2 = 0;
        } else if (strcmp(argv[i], "--max-iters") == 0 && i + 1 < argc) {
            max_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-dets") == 0 && i + 1 < argc) {
            max_dets = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--t2") == 0) {
            enable_t2 = 1;
        } else if (strcmp(argv[i], "--t2-orbs") == 0 && i + 1 < argc) {
            t2_cap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--t1") == 0) {
            enable_t1 = 1;
        } else if (strcmp(argv[i], "--t1-orbs") == 0 && i + 1 < argc) {
            t1_cap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--auto-no") == 0) {
            auto_no = 1;
        } else if (strcmp(argv[i], "--save-no-fcidump") == 0 && i + 1 < argc) {
            save_no_fcidump = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--witness") == 0 && i + 1 < argc) {
            witness_path = argv[++i];
        } else if (strcmp(argv[i], "--dump-rdm-grad") == 0 && i + 1 < argc) {
            dump_grad_path = argv[++i];
        } else if (strcmp(argv[i], "--e-ai") == 0 && i + 1 < argc) {
            has_e_ai = 1;
            e_ai = atof(argv[++i]);
        } else if (strcmp(argv[i], "--veto-tol") == 0 && i + 1 < argc) {
            veto_tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "--cert") == 0 && i + 1 < argc) {
            cert_path = argv[++i];
        } else if (strcmp(argv[i], "--cert-bin") == 0 && i + 1 < argc) {
            cert_bin_path = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    if (is_veto && !has_e_ai) {
        fprintf(stderr, "veto subcommand requires --e-ai <Ha>\n");
        return 1;
    }

    ChemHam ham;
    char err[256] = {0};
    if (chem_fcidump_load(fcidump, &ham, err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR loading FCIDUMP: %s\n", err);
        return 1;
    }

    ChemSolveOpts sopts = chem_solve_opts_default(&ham);
    sopts.mode = mode;
    if (two_s_override >= 0) sopts.two_s = two_s_override;
    sopts.constrain_s2 = constrain_s2;
    sopts.max_iters = max_iters;
    sopts.tol = tol;
    sopts.veto_enabled = is_veto;
    sopts.E_ai = e_ai;
    sopts.veto_tol = veto_tol;
    sopts.enable_t2 = enable_t2;
    sopts.t2_max_orbs = t2_cap;
    sopts.enable_t1 = enable_t1;
    sopts.t1_max_orbs = t1_cap;
    sopts.export_rdm_grad = (dump_grad_path != NULL) ? 1 : 0;
    sopts.num_threads = num_threads;
    sopts.verbose = verbose;

    double no_occs[CHEM_MAX_ORBITALS] = {0}, no_ent[CHEM_MAX_ORBITALS] = {0};
    const char *fcidump_for_cert = fcidump;
    if (auto_no) {
        if (maybe_apply_auto_no(&ham, NULL, &sopts, t1_cap, t2_cap, save_no_fcidump,
                                no_occs, no_ent, &fcidump_for_cert, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR computing DQG natural orbitals: %s\n", err);
            chem_ham_free(&ham);
            return 1;
        }
    }

    ChemVarResult vr;
    if (witness_path) {
        int ts = sopts.constrain_s2 ? sopts.two_s : -1;
        if (chem_var_load_witness_json(witness_path, &ham, ts, &vr, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR loading witness %s: %s\n", witness_path, err);
            chem_ham_free(&ham);
            return 1;
        }
    } else {
        ChemVarOpts vopts = chem_var_opts_default();
        vopts.max_dets = max_dets;
        vopts.target_two_s = sopts.constrain_s2 ? sopts.two_s : -1;
        if (chem_var_solve(&ham, &vopts, &vr, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR in variational solver: %s\n", err);
            chem_ham_free(&ham);
            return 1;
        }
    }

    if (!auto_no) {
        if (sopts.enable_t2) select_frontier_t2_orbs(&ham, &vr, t2_cap, &sopts);
        if (sopts.enable_t1) select_frontier_t1_orbs(&ham, &vr, t1_cap, &sopts);
    }

    ChemSolveResult sr;
    if (chem_solve_dqg(&ham, &sopts, &sr, err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR in DQG solver: %s\n", err);
        chem_var_result_free(&vr);
        chem_ham_free(&ham);
        return 1;
    }

    ChemVetoResult vres = chem_make_veto_result(&sr, vr.E_ub);
    printf("CHEM %s (%s): K=%d nelec=%d ms2=%d mode=%s%s%s%s\n",
           is_veto ? "VETO" : "BRACKET", fcidump_for_cert, ham.K, ham.nelec, ham.ms2,
           (sr.mode_used == CHEM_MODE_A_SINGLET) ? "A_SINGLET" : "B_GENERAL",
           sr.t2_enabled ? "+T2" : "", sr.t1_enabled ? "+T1" : "",
           auto_no ? "+NO" : "");
    printf("  E_lb_cert        = %.10f Ha (min_cert_eig=%.3e, iters=%d, conv=%d, %.3fs)\n",
           vres.E_lb_cert, vres.min_cert_eig, sr.iters, sr.converged, sr.elapsed_sec);
    printf("  E_ub_variational = %.10f Ha (%s, ndet_var=%d, HF=%.8f, %.3fs)\n",
           vres.E_ub_variational,
           witness_path ? "external witness" : (vr.is_exact_fci ? "exact FCI" : "SHCI"),
           vr.ndet_var, vr.E_hf0, vr.elapsed_sec);
    printf("  Bracket width    = %.4f mHa | Peak RSS = %.2f MB\n",
           vres.interval_width * 1000.0, peak_rss_mb());
    if (is_veto) {
        printf("  Veto decision    = %s (E_ai=%.8f Ha, tol=%.1e, iter_stopped=%d)\n",
               vres.vetoed ? "VETOED (FALSE-LOW HALLUCINATION)" : "NOT VETOED",
               e_ai, veto_tol, vres.iter_stopped);
    }

    if (cert_path) {
        if (chem_cert_write_json(cert_path, &ham, fcidump_for_cert, &sr, &vr, &sopts, err,
                                 sizeof(err)) != 0) {
            fprintf(stderr, "ERROR writing certificate: %s\n", err);
            chem_solve_result_free(&sr);
            chem_var_result_free(&vr);
            chem_ham_free(&ham);
            return 1;
        }
        printf("  Certificate      = %s\n", cert_path);
    }
    if (cert_bin_path) {
        if (chem_cert_write_bin(cert_bin_path, &ham, fcidump_for_cert, &sr, &vr, &sopts, err,
                                sizeof(err)) != 0) {
            fprintf(stderr, "ERROR writing binary certificate: %s\n", err);
            chem_solve_result_free(&sr);
            chem_var_result_free(&vr);
            chem_ham_free(&ham);
            return 1;
        }
        printf("  Binary Cert      = %s\n", cert_bin_path);
    }

    if (dump_grad_path) {
        if (write_rdm_grad_json(dump_grad_path, ham.K, &sr) != 0) {
            fprintf(stderr, "ERROR writing RDM gradient JSON: %s\n", dump_grad_path);
            chem_solve_result_free(&sr);
            chem_var_result_free(&vr);
            chem_ham_free(&ham);
            return 1;
        }
    }

    if (json_path) {
        FILE *jf = fopen(json_path, "w");
        if (jf) {
            fprintf(jf, "{\n  \"peak_rss_mb\": %.3f,\n  \"results\": [\n", peak_rss_mb());
            write_bracket_summary_json_ext(jf, fcidump_for_cert, &ham, &sr, &vr,
                                           is_veto ? &sr : NULL, e_ai, auto_no,
                                           auto_no ? no_occs : NULL,
                                           auto_no ? no_ent : NULL);
            fprintf(jf, "\n  ]\n}\n");
            fclose(jf);
        }
    }

    chem_solve_result_free(&sr);
    chem_var_result_free(&vr);
    chem_ham_free(&ham);
    return 0;
}

/* ============================================================================
 * Subcommand: observable
 * ============================================================================ */

static int cmd_observable(int argc, char **argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }
    const char *fcidump = argv[2];
    const char *obs_type = NULL;
    const char *witness_path = NULL;
    const char *json_path = NULL;
    const char *save_no_fcidump = NULL;
    ChemMode mode = CHEM_MODE_AUTO;
    int two_s_override = -1;
    int constrain_s2 = 1;
    int max_iters = 400;
    double tol = 1e-6;
    int max_dets = 1024;
    int enable_t2 = 0;
    int t2_cap = 6;
    int enable_t1 = 0;
    int t1_cap = 6;
    int auto_no = 0;
    int num_threads = 0;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) obs_type = argv[++i];
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (parse_mode_arg(argv[++i], &mode) != 0) {
                fprintf(stderr, "Invalid --mode\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--two-s") == 0 && i + 1 < argc) {
            two_s_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-s2") == 0) {
            constrain_s2 = 0;
        } else if (strcmp(argv[i], "--max-iters") == 0 && i + 1 < argc) {
            max_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-dets") == 0 && i + 1 < argc) {
            max_dets = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--t2") == 0) {
            enable_t2 = 1;
        } else if (strcmp(argv[i], "--t2-orbs") == 0 && i + 1 < argc) {
            t2_cap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--t1") == 0) {
            enable_t1 = 1;
        } else if (strcmp(argv[i], "--t1-orbs") == 0 && i + 1 < argc) {
            t1_cap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--auto-no") == 0) {
            auto_no = 1;
        } else if (strcmp(argv[i], "--save-no-fcidump") == 0 && i + 1 < argc) {
            save_no_fcidump = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--witness") == 0 && i + 1 < argc) {
            witness_path = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            /* accepted */
        } else {
            fprintf(stderr, "Unknown observable option: %s\n", argv[i]);
            return 1;
        }
    }

    ChemHam ham;
    char err[256] = {0};
    if (chem_fcidump_load(fcidump, &ham, err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR loading FCIDUMP: %s\n", err);
        return 1;
    }

    ChemObservable obs;
    if (parse_and_build_obs(ham.K, obs_type, &obs, err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR building observable: %s\n", err);
        chem_ham_free(&ham);
        return 1;
    }

    ChemSolveOpts sopts = chem_solve_opts_default(&ham);
    sopts.mode = mode;
    if (two_s_override >= 0) sopts.two_s = two_s_override;
    sopts.constrain_s2 = constrain_s2;
    sopts.max_iters = max_iters;
    sopts.tol = tol;
    sopts.enable_t2 = enable_t2;
    sopts.t2_max_orbs = t2_cap;
    sopts.enable_t1 = enable_t1;
    sopts.t1_max_orbs = t1_cap;
    sopts.num_threads = num_threads;

    double no_occs[CHEM_MAX_ORBITALS] = {0}, no_ent[CHEM_MAX_ORBITALS] = {0};
    const char *fcidump_out_name = fcidump;
    if (auto_no) {
        if (maybe_apply_auto_no(&ham, &obs, &sopts, t1_cap, t2_cap, save_no_fcidump,
                                no_occs, no_ent, &fcidump_out_name, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR computing DQG natural orbitals: %s\n", err);
            chem_observable_free(&obs);
            chem_ham_free(&ham);
            return 1;
        }
    }

    ChemVarResult vr;
    if (witness_path) {
        int ts = sopts.constrain_s2 ? sopts.two_s : -1;
        if (chem_var_load_witness_json(witness_path, &ham, ts, &vr, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR loading witness %s: %s\n", witness_path, err);
            chem_observable_free(&obs);
            chem_ham_free(&ham);
            return 1;
        }
    } else {
        ChemVarOpts vopts = chem_var_opts_default();
        vopts.max_dets = max_dets;
        vopts.target_two_s = sopts.constrain_s2 ? sopts.two_s : -1;
        if (chem_var_solve(&ham, &vopts, &vr, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR variational solver: %s\n", err);
            chem_observable_free(&obs);
            chem_ham_free(&ham);
            return 1;
        }
    }

    if (!auto_no) {
        if (sopts.enable_t2) select_frontier_t2_orbs(&ham, &vr, t2_cap, &sopts);
        if (sopts.enable_t1) select_frontier_t1_orbs(&ham, &vr, t1_cap, &sopts);
    }

    double o_var = 0.0;
    if (chem_eval_observable_witness(&ham, &obs, vr.ndet_var, vr.dets, vr.psi0, &o_var, err,
                                     sizeof(err)) != 0) {
        fprintf(stderr, "ERROR evaluating observable witness: %s\n", err);
        chem_var_result_free(&vr);
        chem_observable_free(&obs);
        chem_ham_free(&ham);
        return 1;
    }

    ChemObservableResult ores;
    if (chem_certify_observable(&ham, &obs, vr.E_ub, &sopts, &ores, err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR certifying observable: %s\n", err);
        chem_var_result_free(&vr);
        chem_observable_free(&obs);
        chem_ham_free(&ham);
        return 1;
    }
    ores.O_var = o_var;

    printf("CHEM OBSERVABLE (%s | %s): O_lb_cert = %.8f <= O_var = %.8f <= O_ub_cert = %.8f "
           "(width=%.6f, lam=[%.2f, %.2f], %.3fs)\n",
           fcidump_out_name, ores.name, ores.O_lb_cert, ores.O_var, ores.O_ub_cert,
           ores.interval_width, ores.lam_lo, ores.lam_hi, ores.elapsed_sec);

    if (json_path) {
        FILE *jf = fopen(json_path, "w");
        if (jf) {
            fprintf(jf,
                    "{\n  \"fcidump\": \"%s\",\n  \"observable\": \"%s\",\n"
                    "  \"E_ub\": %.17g,\n  \"O_lb_cert\": %.17g,\n  \"O_var\": %.17g,\n"
                    "  \"O_ub_cert\": %.17g,\n  \"interval_width\": %.17g,\n"
                    "  \"lam_lo\": %.17g,\n  \"lam_hi\": %.17g,\n",
                    fcidump_out_name, ores.name, ores.E_ub, ores.O_lb_cert, ores.O_var,
                    ores.O_ub_cert, ores.interval_width, ores.lam_lo, ores.lam_hi);
            if (auto_no) {
                fprintf(jf, "  \"auto_no\": true,\n  \"no_occs\": ");
                write_json_double_arr(jf, ham.K, no_occs);
                fprintf(jf, ",\n  \"no_entropies\": ");
                write_json_double_arr(jf, ham.K, no_ent);
                fprintf(jf, ",\n");
            }
            fprintf(jf, "  \"elapsed_sec\": %.6f\n}\n", ores.elapsed_sec);
            fclose(jf);
        }
    }

    chem_var_result_free(&vr);
    chem_observable_free(&obs);
    chem_ham_free(&ham);
    return 0;
}

/* ============================================================================
 * Subcommand: cert-verify
 * ============================================================================ */

static int cmd_cert_verify(int argc, char **argv) {
    const char *fcidump_override = NULL;
    const char *cert_path = NULL;
    if (argc == 3) {
        cert_path = argv[2];
    } else if (argc == 4) {
        fcidump_override = argv[2];
        cert_path = argv[3];
    } else {
        print_usage();
        return 1;
    }

    ChemCertVerifyResult res;
    int rc = chem_cert_verify_file(fcidump_override, cert_path, &res);
    printf("%s -> %s\n", cert_path, res.msg);
    if (rc != 0 || !res.ok) return 1;
    return 0;
}

/* ============================================================================
 * Subcommands: exact-verify, emit-lean, gpc (v0.3.0)
 * ============================================================================ */

static int cmd_exact_verify(int argc, char **argv) {
    const char *fcidump_override = NULL;
    const char *cert_path = NULL;
    const char *json_path = NULL;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (!cert_path) {
            cert_path = argv[i];
        } else if (!fcidump_override) {
            fcidump_override = cert_path;
            cert_path = argv[i];
        } else {
            print_usage();
            return 1;
        }
    }
    if (!cert_path) {
        print_usage();
        return 1;
    }

    ChemExactVerifyResult res;
    int rc = chem_cert_verify_exact_chol(fcidump_override, cert_path, &res);
    printf("%s -> %s\n", cert_path, res.msg);
    if (json_path) {
        FILE *jf = fopen(json_path, "w");
        if (jf) {
            fprintf(jf,
                    "{\n  \"ok\": %s,\n  \"sha_ok\": %d,\n  \"all_blocks_sdd\": %d,\n"
                    "  \"lb_ok\": %d,\n  \"var_ok\": %d,\n  \"nblocks\": %d,\n"
                    "  \"scale_D\": %lld,\n  \"E_lb_int_chol\": %.17g,\n"
                    "  \"E_lb_weyl\": %.17g,\n  \"max_sdd_margin\": %.17g,\n"
                    "  \"elapsed_sec\": %.6f\n}\n",
                    res.ok ? "true" : "false", res.sha_ok, res.all_blocks_sdd, res.lb_ok,
                    res.var_ok, res.nblocks, (long long)res.scale_D, res.E_lb_int_chol,
                    res.E_lb_weyl, res.max_sdd_margin, res.elapsed_sec);
            fclose(jf);
        }
    }
    if (rc != 0 || !res.ok) return 1;
    return 0;
}

static int cmd_emit_lean(int argc, char **argv) {
    if (argc < 4) {
        print_usage();
        return 1;
    }
    const char *fcidump = argv[2];
    const char *cert_path = argv[3];
    const char *out_lean = NULL;
    const char *mod_name = "ChemMolCert";
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_lean = argv[++i];
        else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) mod_name = argv[++i];
        else {
            print_usage();
            return 1;
        }
    }
    if (!out_lean) {
        fprintf(stderr, "emit-lean requires --out <path.lean>\n");
        return 1;
    }
    ChemExactVerifyResult res;
    char err[256] = {0};
    if (chem_cert_emit_lean(fcidump, cert_path, out_lean, mod_name, &res, err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR emit-lean: %s\n", err);
        return 1;
    }
    printf("%s\n", res.msg);
    return 0;
}

static int cmd_gpc(int argc, char **argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }
    const char *fcidump = argv[2];
    ChemMode mode = CHEM_MODE_AUTO;
    int two_s_set = 0, two_s = 0, certify_bd = 0;
    const char *witness_path = NULL;
    const char *json_path = NULL;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcmp(m, "a") == 0 || strcmp(m, "A") == 0) mode = CHEM_MODE_A_SINGLET;
            else if (strcmp(m, "b") == 0 || strcmp(m, "B") == 0) mode = CHEM_MODE_B_GENERAL;
        } else if (strcmp(argv[i], "--two-s") == 0 && i + 1 < argc) {
            two_s = atoi(argv[++i]);
            two_s_set = 1;
        } else if (strcmp(argv[i], "--witness") == 0 && i + 1 < argc) {
            witness_path = argv[++i];
        } else if (strcmp(argv[i], "--certify-bd") == 0) {
            certify_bd = 1;
        } else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown gpc option: %s\n", argv[i]);
            return 1;
        }
    }

    ChemHam ham;
    char err[256] = {0};
    if (chem_fcidump_load(fcidump, &ham, err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR loading %s: %s\n", fcidump, err);
        return 1;
    }
    ChemSolveOpts sopts = chem_solve_opts_default(&ham);
    sopts.mode = mode;
    if (two_s_set) sopts.two_s = two_s;
    sopts.tol = 1e-6;

    ChemVarResult vr;
    if (witness_path) {
        int ts = sopts.constrain_s2 ? sopts.two_s : -1;
        if (chem_var_load_witness_json(witness_path, &ham, ts, &vr, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR witness: %s\n", err);
            chem_ham_free(&ham);
            return 1;
        }
    } else {
        ChemVarOpts vopts = chem_var_opts_default();
        vopts.target_two_s = sopts.constrain_s2 ? sopts.two_s : -1;
        if (chem_var_solve(&ham, &vopts, &vr, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR variational: %s\n", err);
            chem_ham_free(&ham);
            return 1;
        }
    }

    ChemGPCResult gpc;
    if (chem_var_analyze_gpc(&ham, vr.ndet_var, vr.dets, vr.psi0, &gpc, err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR GPC analysis: %s\n", err);
        chem_var_result_free(&vr);
        chem_ham_free(&ham);
        return 1;
    }

    int has_bd_cert = 0;
    ChemObservableResult ores;
    memset(&ores, 0, sizeof(ores));
    if (certify_bd && gpc.has_bd_3e3o) {
        ChemObservable bd_obs;
        if (build_bd_3e3o_observable(&gpc, &bd_obs, err, sizeof(err)) == 0 &&
            chem_certify_observable(&ham, &bd_obs, vr.E_ub, &sopts, &ores, err, sizeof(err)) == 0) {
            has_bd_cert = 1;
        }
        chem_observable_free(&bd_obs);
    }

    printf("CHEM GPC (%s | K=%d ne=%d): S1=%.6f", fcidump, gpc.K, gpc.nelec, gpc.entanglement_entropy);
    if (gpc.has_bd_3e3o) {
        printf(" | BD36 pair_err=%.2e slack=%.6f (pinned=%d)", gpc.bd_pair_err, gpc.bd_slack, gpc.bd_pinned);
        if (has_bd_cert) printf(" | cert_BD=[%.6f, %.6f]", ores.O_lb_cert, ores.O_ub_cert);
    } else if (gpc.has_bd_2e2o) {
        printf(" | BD24 pair_err=%.2e", gpc.bd24_pair_err);
    }
    printf("\n");

    if (json_path) {
        FILE *jf = fopen(json_path, "w");
        if (jf) {
            fprintf(jf,
                    "{\n  \"K\": %d,\n  \"nelec\": %d,\n  \"E_ub\": %.17g,\n"
                    "  \"entanglement_entropy\": %.17g,\n  \"has_bd_3e3o\": %d,\n"
                    "  \"bd_pair_err\": %.17g,\n  \"bd_slack\": %.17g,\n  \"bd_pinned\": %d,\n"
                    "  \"has_bd_2e2o\": %d,\n  \"bd24_pair_err\": %.17g,\n  \"occ_sorted\": [",
                    gpc.K, gpc.nelec, vr.E_ub, gpc.entanglement_entropy, gpc.has_bd_3e3o,
                    gpc.bd_pair_err, gpc.bd_slack, gpc.bd_pinned, gpc.has_bd_2e2o, gpc.bd24_pair_err);
            for (int i = 0; i < 2 * gpc.K; i++) {
                fprintf(jf, "%s%.17g", (i > 0) ? ", " : "", gpc.occ_sorted[i]);
            }
            fprintf(jf, "]");
            if (has_bd_cert) {
                fprintf(jf,
                        ",\n  \"bd_lb_cert\": %.17g,\n  \"bd_ub_cert\": %.17g,\n  \"bd_width\": %.17g",
                        ores.O_lb_cert, ores.O_ub_cert, ores.interval_width);
            }
            fprintf(jf, "\n}\n");
            fclose(jf);
        }
    }
    chem_var_result_free(&vr);
    chem_ham_free(&ham);
    return 0;
}

/* ============================================================================
 * Subcommand: bench
 * ============================================================================ */

typedef struct {
    const char *name;
    const char *filename;
    ChemMode mode;
    int max_iters;
    double tol;
    int max_dets;
    int run_veto;
    int write_bin;
} BenchCaseSpec;

static int cmd_bench(int argc, char **argv) {
    const char *dir = "benchmarks";
    const char *out_json = "build/chem_bench.json";
    const char *cert_dir = "build/certs";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) dir = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_json = argv[++i];
        else if (strcmp(argv[i], "--cert-dir") == 0 && i + 1 < argc) cert_dir = argv[++i];
        else {
            fprintf(stderr, "Unknown bench option: %s\n", argv[i]);
            return 1;
        }
    }

    static const BenchCaseSpec CASES[] = {
        {"h2_r1.4",                 "h2_r1.4.fcidump",                 CHEM_MODE_AUTO,      500, 1e-7,  512, 0, 0},
        {"h2_r5.0",                 "h2_r5.0.fcidump",                 CHEM_MODE_AUTO,      500, 1e-7,  512, 0, 0},
        {"h3_doublet",              "h3_doublet.fcidump",              CHEM_MODE_B_GENERAL, 600, 1e-7,  512, 0, 0},
        {"h4_ry2.0",                "h4_ry2.0.fcidump",                CHEM_MODE_A_SINGLET, 600, 1e-7,  512, 1, 1},
        {"h4_ry2.0_modeB",          "h4_ry2.0.fcidump",                CHEM_MODE_B_GENERAL, 600, 1e-7,  512, 0, 0},
        {"h4_ry2.2",                "h4_ry2.2.fcidump",                CHEM_MODE_AUTO,      600, 1e-7,  512, 0, 0},
        {"h4_ry2.6",                "h4_ry2.6.fcidump",                CHEM_MODE_AUTO,      600, 1e-7,  512, 0, 0},
        {"h4_ry4.5",                "h4_ry4.5.fcidump",                CHEM_MODE_AUTO,      600, 1e-7,  512, 0, 0},
        {"h4_ry2.0_triplet",        "h4_ry2.0_triplet.fcidump",        CHEM_MODE_B_GENERAL, 600, 1e-7,  512, 0, 0},
        {"h2o_eq",                  "h2o_eq.fcidump",                  CHEM_MODE_AUTO,      600, 1e-6,  512, 0, 0},
        {"h2o_stretched",           "h2o_stretched.fcidump",           CHEM_MODE_AUTO,      600, 1e-6,  512, 0, 0},
        {"n2_eq_6e6o",              "n2_eq_6e6o.fcidump",              CHEM_MODE_AUTO,      600, 1e-6,  512, 0, 0},
        {"n2_stretched_6e6o",       "n2_stretched_6e6o.fcidump",       CHEM_MODE_AUTO,      600, 1e-6,  512, 1, 1},
        {"n2_stretched_10e8o",      "n2_stretched_10e8o.fcidump",      CHEM_MODE_AUTO,      600, 1e-6, 1024, 0, 0},
        {"tm_spin_s0_8e8o",         "tm_spin_s0_8e8o.fcidump",         CHEM_MODE_A_SINGLET, 600, 1e-6,  512, 0, 1},
        {"tm_spin_s2_8e8o",         "tm_spin_s2_8e8o.fcidump",         CHEM_MODE_B_GENERAL, 600, 1e-6,  512, 0, 0},
        {"h12_chain_12e12o",        "h12_chain_12e12o.fcidump",        CHEM_MODE_A_SINGLET, 500, 5e-6, 1024, 1, 0},
        {"h12_chain_triplet_12e12o","h12_chain_triplet_12e12o.fcidump",CHEM_MODE_B_GENERAL, 500, 5e-6, 1024, 0, 0},
        {"h16_chain_16e16o",        "h16_chain_16e16o.fcidump",        CHEM_MODE_A_SINGLET, 400, 1e-5, 1024, 1, 0},
        {"h16_chain_triplet_16e16o","h16_chain_triplet_16e16o.fcidump",CHEM_MODE_B_GENERAL, 400, 1e-5, 1024, 0, 1}
    };
    int n_cases = (int)(sizeof(CASES) / sizeof(CASES[0]));

    FILE *jf = fopen(out_json, "w");
    if (!jf) {
        fprintf(stderr, "ERROR cannot open bench output %s\n", out_json);
        return 1;
    }

    char err[256] = {0};
    fprintf(jf, "{\n  \"adjoints\": [\n");
    const int adj_dims[4][4] = {
        {4, 2, 2, CHEM_MODE_A_SINGLET},
        {8, 4, 4, CHEM_MODE_A_SINGLET},
        {16, 8, 8, CHEM_MODE_A_SINGLET},
        {8, 5, 3, CHEM_MODE_B_GENERAL}
    };
    for (int i = 0; i < 4; i++) {
        ChemAdjointReport rep;
        if (chem_test_adjoints(adj_dims[i][0], adj_dims[i][1], adj_dims[i][2],
                               adj_dims[i][1] - adj_dims[i][2], 1, (ChemMode)adj_dims[i][3],
                               UINT64_C(0xA5A5A5A5) + (uint64_t)i, &rep, err, sizeof(err)) != 0) {
            fprintf(stderr, "Adjoint benchmark failed: %s\n", err);
            fclose(jf);
            return 1;
        }
        fprintf(jf, "    {\"K\": %d, \"na\": %d, \"nb\": %d, \"mode\": \"%s\", "
                    "\"err_LQ\": %.3e, \"err_LG\": %.3e, \"err_Aeq\": %.3e, \"max_err\": %.3e}%s\n",
                adj_dims[i][0], adj_dims[i][1], adj_dims[i][2],
                (adj_dims[i][3] == CHEM_MODE_A_SINGLET) ? "A_SINGLET" : "B_GENERAL",
                rep.err_LQ, rep.err_LG, rep.err_Aeq, rep.max_err, (i + 1 < 4) ? "," : "");
    }
    fprintf(jf, "  ],\n  \"t2_adjoints\": [\n");
    const int t2a_dims[4][5] = {
        {4, 2, 2, CHEM_MODE_A_SINGLET, 4},
        {6, 3, 3, CHEM_MODE_A_SINGLET, 6},
        {4, 2, 2, CHEM_MODE_B_GENERAL, 4},
        {6, 4, 2, CHEM_MODE_B_GENERAL, 5}
    };
    for (int i = 0; i < 4; i++) {
        ChemAdjointReport rep;
        int orbs[6] = {0, 1, 2, 3, 4, 5};
        if (chem_test_adjoints_t2(t2a_dims[i][0], t2a_dims[i][1], t2a_dims[i][2],
                                  t2a_dims[i][1] - t2a_dims[i][2], 1, (ChemMode)t2a_dims[i][3],
                                  t2a_dims[i][4], orbs, UINT64_C(0xB4B4B4B4) + (uint64_t)i, &rep,
                                  err, sizeof(err)) != 0) {
            fprintf(stderr, "T2 adjoint benchmark failed: %s\n", err);
            fclose(jf);
            return 1;
        }
        fprintf(jf, "    {\"K\": %d, \"Ks\": %d, \"mode\": \"%s\", \"err_LT2\": %.3e, \"max_err\": %.3e}%s\n",
                t2a_dims[i][0], t2a_dims[i][4],
                (t2a_dims[i][3] == CHEM_MODE_A_SINGLET) ? "A_SINGLET" : "B_GENERAL",
                rep.err_LT2, rep.max_err, (i + 1 < 4) ? "," : "");
    }
    fprintf(jf, "  ],\n  \"benchmarks\": [\n");

    printf("=== CHEM Benchmark Suite (%d cases) ===\n", n_cases);
    for (int i = 0; i < n_cases; i++) {
        char fpath[1024], cpath[1024];
        snprintf(fpath, sizeof(fpath), "%s/%s", dir, CASES[i].filename);
        snprintf(cpath, sizeof(cpath), "%s/%s.cert.json", cert_dir, CASES[i].name);

        ChemHam ham;
        if (chem_fcidump_load(fpath, &ham, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR loading %s: %s\n", fpath, err);
            fclose(jf);
            return 1;
        }

        ChemSolveOpts sopts = chem_solve_opts_default(&ham);
        sopts.mode = CASES[i].mode;
        sopts.max_iters = CASES[i].max_iters;
        sopts.tol = CASES[i].tol;

        ChemSolveResult sr;
        if (chem_solve_dqg(&ham, &sopts, &sr, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR solving %s: %s\n", CASES[i].name, err);
            chem_ham_free(&ham);
            fclose(jf);
            return 1;
        }

        ChemVarOpts vopts = chem_var_opts_default();
        vopts.max_dets = CASES[i].max_dets;
        vopts.target_two_s = sopts.constrain_s2 ? sopts.two_s : -1;
        ChemVarResult vr;
        if (chem_var_solve(&ham, &vopts, &vr, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR variational %s: %s\n", CASES[i].name, err);
            chem_solve_result_free(&sr);
            chem_ham_free(&ham);
            fclose(jf);
            return 1;
        }

        if (chem_cert_write_json(cpath, &ham, fpath, &sr, &vr, &sopts, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR writing %s: %s\n", cpath, err);
            chem_solve_result_free(&sr);
            chem_var_result_free(&vr);
            chem_ham_free(&ham);
            fclose(jf);
            return 1;
        }
        if (CASES[i].write_bin) {
            char bpath[1024];
            snprintf(bpath, sizeof(bpath), "%s/%s.flowchem", cert_dir, CASES[i].name);
            if (chem_cert_write_bin(bpath, &ham, fpath, &sr, &vr, &sopts, err, sizeof(err)) != 0) {
                fprintf(stderr, "ERROR writing %s: %s\n", bpath, err);
                chem_solve_result_free(&sr);
                chem_var_result_free(&vr);
                chem_ham_free(&ham);
                fclose(jf);
                return 1;
            }
        }

        ChemSolveResult sr_veto;
        int has_veto = 0;
        double e_ai_synth = vr.E_ub - 0.15;
        if (sr.E_lb_cert - 0.05 < e_ai_synth) e_ai_synth = sr.E_lb_cert - 0.05;
        if (CASES[i].run_veto) {
            ChemSolveOpts vsopts = sopts;
            vsopts.veto_enabled = 1;
            vsopts.E_ai = e_ai_synth;
            vsopts.veto_tol = 1e-4;
            vsopts.check_cert_every = 5;
            if (chem_solve_dqg(&ham, &vsopts, &sr_veto, err, sizeof(err)) == 0) {
                has_veto = 1;
                char vcpath[1024];
                snprintf(vcpath, sizeof(vcpath), "%s/%s.veto.cert.json", cert_dir, CASES[i].name);
                chem_cert_write_json(vcpath, &ham, fpath, &sr_veto, &vr, &vsopts, err, sizeof(err));
            }
        }

        printf("  %-26s | K=%2d (%2de) | LB=%13.8f <= UB=%13.8f | gap=%7.3f mHa | it=%3d | %.2fs | RSS=%.1f MB\n",
               CASES[i].name, ham.K, ham.nelec, sr.E_lb_cert, vr.E_ub,
               (vr.E_ub - sr.E_lb_cert) * 1000.0, sr.iters, sr.elapsed_sec + vr.elapsed_sec,
               peak_rss_mb());

        write_bracket_summary_json(jf, CASES[i].name, &ham, &sr, &vr,
                                   has_veto ? &sr_veto : NULL, e_ai_synth);
        fprintf(jf, "%s\n", (i + 1 < n_cases) ? "," : "");

        if (has_veto) chem_solve_result_free(&sr_veto);
        chem_solve_result_free(&sr);
        chem_var_result_free(&vr);
        chem_ham_free(&ham);
    }

    /* v0.2.0 Gate 5a: DQG + T2 tightening showcase */
    fprintf(jf, "  ],\n  \"v02_t2_cases\": [\n");
    struct T2BenchSpec {
        const char *name;
        const char *filename;
        ChemMode mode;
        int max_iters;
        double tol;
        int max_dets;
        int t2_cap;
    };
    static const struct T2BenchSpec T2_CASES[] = {
        {"h4_ry2.2",          "h4_ry2.2.fcidump",          CHEM_MODE_A_SINGLET, 800, 1e-6, 512, 4},
        {"h2o_stretched",     "h2o_stretched.fcidump",     CHEM_MODE_A_SINGLET, 800, 1e-6, 512, 4},
        {"n2_stretched_6e6o", "n2_stretched_6e6o.fcidump", CHEM_MODE_A_SINGLET, 700, 1e-6, 512, 6},
        {"tm_spin_s0_8e8o",   "tm_spin_s0_8e8o.fcidump",   CHEM_MODE_A_SINGLET, 800, 2e-6, 512, 4}
    };
    int nt2 = (int)(sizeof(T2_CASES) / sizeof(T2_CASES[0]));
    printf("=== CHEM v0.2.0 DQG+T2 Tightening Showcase ===\n");
    for (int i = 0; i < nt2; i++) {
        char fpath[1024], cpath[1024], bpath[1024];
        snprintf(fpath, sizeof(fpath), "%s/%s", dir, T2_CASES[i].filename);
        snprintf(cpath, sizeof(cpath), "%s/%s.t2.cert.json", cert_dir, T2_CASES[i].name);
        snprintf(bpath, sizeof(bpath), "%s/%s.t2.flowchem", cert_dir, T2_CASES[i].name);

        ChemHam ham;
        if (chem_fcidump_load(fpath, &ham, err, sizeof(err)) != 0) { fclose(jf); return 1; }
        ChemVarOpts vopts = chem_var_opts_default();
        vopts.max_dets = T2_CASES[i].max_dets;
        vopts.target_two_s = 0;
        ChemVarResult vr;
        if (chem_var_solve(&ham, &vopts, &vr, err, sizeof(err)) != 0) {
            chem_ham_free(&ham); fclose(jf); return 1;
        }
        ChemSolveOpts sopts = chem_solve_opts_default(&ham);
        sopts.mode = T2_CASES[i].mode;
        sopts.max_iters = T2_CASES[i].max_iters;
        sopts.tol = T2_CASES[i].tol;
        sopts.enable_t2 = 1;
        sopts.t2_max_orbs = T2_CASES[i].t2_cap;
        select_frontier_t2_orbs(&ham, &vr, T2_CASES[i].t2_cap, &sopts);

        ChemSolveResult sr;
        if (chem_solve_dqg(&ham, &sopts, &sr, err, sizeof(err)) != 0) {
            fprintf(stderr, "ERROR DQG+T2 %s: %s\n", T2_CASES[i].name, err);
            chem_var_result_free(&vr); chem_ham_free(&ham); fclose(jf); return 1;
        }
        if (chem_cert_write_json(cpath, &ham, fpath, &sr, &vr, &sopts, err, sizeof(err)) != 0 ||
            chem_cert_write_bin(bpath, &ham, fpath, &sr, &vr, &sopts, err, sizeof(err)) != 0) {
            chem_solve_result_free(&sr); chem_var_result_free(&vr); chem_ham_free(&ham); fclose(jf); return 1;
        }
        printf("  %-26s | Ks=%d/%d | LB_T2=%13.8f <= UB=%13.8f | gap=%7.3f mHa | it=%3d | %.2fs\n",
               T2_CASES[i].name, sr.t2_norbs, ham.K, sr.E_lb_cert, vr.E_ub,
               (vr.E_ub - sr.E_lb_cert) * 1000.0, sr.iters, sr.elapsed_sec);

        fprintf(jf,
                "    {\"name\": \"%s\", \"K\": %d, \"Ks\": %d, \"E_lb_t2\": %.17g, "
                "\"E_ub\": %.17g, \"iters\": %d, \"sdp_sec\": %.6f}%s\n",
                T2_CASES[i].name, ham.K, sr.t2_norbs, sr.E_lb_cert, vr.E_ub, sr.iters,
                sr.elapsed_sec, (i + 1 < nt2) ? "," : "");
        chem_solve_result_free(&sr);
        chem_var_result_free(&vr);
        chem_ham_free(&ham);
    }

    /* v0.2.0 Gate 5b: Certified Observable & Hellmann-Feynman Force Envelope Showcase */
    fprintf(jf, "  ],\n  \"v02_observables\": [\n");
    struct ObsBenchSpec {
        const char *name;
        const char *fcidump;
        const char *spec;
        ChemMode mode;
        int two_s;
    };
    char hf_spec[1024];
    snprintf(hf_spec, sizeof(hf_spec), "hf-force:%s/h4_ry2.0_fplus.fcidump:%s/h4_ry2.0_fminus.fcidump:0.005",
             dir, dir);
    struct ObsBenchSpec OBS_CASES[4] = {
        {"h4_docc0", "h4_ry2.0.fcidump", "docc:0", CHEM_MODE_A_SINGLET, 0},
        {"h4_sdot01", "h4_ry2.0.fcidump", "sdot:0:1", CHEM_MODE_A_SINGLET, 0},
        {"h4_hf_force", "h4_ry2.0.fcidump", hf_spec, CHEM_MODE_A_SINGLET, 0},
        {"h3_occ0", "h3_doublet.fcidump", "occ:0", CHEM_MODE_B_GENERAL, 1}
    };

    printf("=== CHEM v0.2.0 Certified Observable & Force Bounds ===\n");
    for (int i = 0; i < 4; i++) {
        char fpath[1024];
        snprintf(fpath, sizeof(fpath), "%s/%s", dir, OBS_CASES[i].fcidump);
        ChemHam ham;
        if (chem_fcidump_load(fpath, &ham, err, sizeof(err)) != 0) { fclose(jf); return 1; }
        ChemObservable obs;
        if (parse_and_build_obs(ham.K, OBS_CASES[i].spec, &obs, err, sizeof(err)) != 0) {
            chem_ham_free(&ham); fclose(jf); return 1;
        }
        ChemVarOpts vopts = chem_var_opts_default();
        vopts.target_two_s = OBS_CASES[i].two_s;
        ChemVarResult vr;
        if (chem_var_solve(&ham, &vopts, &vr, err, sizeof(err)) != 0) {
            chem_observable_free(&obs); chem_ham_free(&ham); fclose(jf); return 1;
        }
        ChemSolveOpts sopts = chem_solve_opts_default(&ham);
        sopts.mode = OBS_CASES[i].mode;
        sopts.two_s = OBS_CASES[i].two_s;
        sopts.tol = 1e-6;
        double o_var = 0.0;
        ChemObservableResult ores;
        if (chem_eval_observable_witness(&ham, &obs, vr.ndet_var, vr.dets, vr.psi0, &o_var, err,
                                         sizeof(err)) != 0 ||
            chem_certify_observable(&ham, &obs, vr.E_ub, &sopts, &ores, err, sizeof(err)) != 0) {
            chem_var_result_free(&vr); chem_observable_free(&obs); chem_ham_free(&ham); fclose(jf); return 1;
        }
        printf("  %-14s | O_lb=%11.6f <= O_var=%11.6f <= O_ub=%11.6f | w=%.5f | %.2fs\n",
               OBS_CASES[i].name, ores.O_lb_cert, o_var, ores.O_ub_cert, ores.interval_width,
               ores.elapsed_sec);
        fprintf(jf,
                "    {\"name\": \"%s\", \"fcidump\": \"%s\", \"spec\": \"%s\", \"mode\": \"%s\", "
                "\"E_ub\": %.17g, \"O_lb_cert\": %.17g, \"O_var\": %.17g, \"O_ub_cert\": %.17g, "
                "\"width\": %.17g, \"elapsed_sec\": %.6f}%s\n",
                OBS_CASES[i].name, OBS_CASES[i].fcidump, OBS_CASES[i].spec,
                (OBS_CASES[i].mode == CHEM_MODE_A_SINGLET) ? "A_SINGLET" : "B_GENERAL",
                vr.E_ub, ores.O_lb_cert, o_var, ores.O_ub_cert, ores.interval_width,
                ores.elapsed_sec, (i + 1 < 4) ? "," : "");
        chem_var_result_free(&vr);
        chem_observable_free(&obs);
        chem_ham_free(&ham);
    }

    fprintf(jf, "  ],\n  \"peak_rss_mb\": %.3f\n}\n", peak_rss_mb());
    fclose(jf);
    printf("Benchmarks complete. Peak RSS = %.2f MB. Output: %s\n", peak_rss_mb(), out_json);
    return 0;
}

/* ============================================================================
 * Subcommand: atlas (Hubbard Mott Phase Diagram + N-Rep Gap + GPC + Eckart)
 * ============================================================================ */

static int cmd_atlas(int argc, char **argv) {
    const char *dir = "benchmarks";
    const char *out_json = "build/chem_atlas.json";
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) dir = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_json = argv[++i];
        else {
            fprintf(stderr, "Unknown atlas option: %s\n", argv[i]);
            return 1;
        }
    }

    FILE *jf = fopen(out_json, "w");
    if (!jf) {
        fprintf(stderr, "Cannot open %s for writing\n", out_json);
        return 1;
    }

    char err[256] = {0};
    struct HubbSpec {
        const char *name;
        double U;
        int frustrated;
    };
    static const struct HubbSpec HUBB[] = {
        {"hubbard4_u0.5",     0.5,  0},
        {"hubbard4_u2.0",     2.0,  0},
        {"hubbard4_u4.0",     4.0,  0},
        {"hubbard4_u8.0",     8.0,  0},
        {"hubbard4_u12.0",    12.0, 0},
        {"hubbard4_frust_u6", 6.0,  1}
    };
    int nhub = (int)(sizeof(HUBB) / sizeof(HUBB[0]));

    fprintf(jf, "{\n  \"hubbard_mott\": [\n");
    printf("=== CHEM v0.3.0 Hubbard Mott Phase Diagram & N-Rep Gap Atlas ===\n");
    for (int i = 0; i < nhub; i++) {
        char fpath[1024];
        snprintf(fpath, sizeof(fpath), "%s/%s.fcidump", dir, HUBB[i].name);
        ChemHam ham;
        if (chem_fcidump_load(fpath, &ham, err, sizeof(err)) != 0) {
            fprintf(stderr, "atlas load %s: %s\n", fpath, err);
            fclose(jf);
            return 1;
        }

        ChemVarOpts vopts = chem_var_opts_default();
        vopts.max_dets = 512;
        vopts.target_two_s = 0;
        ChemVarResult vr;
        if (chem_var_solve(&ham, &vopts, &vr, err, sizeof(err)) != 0) {
            chem_ham_free(&ham); fclose(jf); return 1;
        }
        double e_var_re = 0.0, var_fci = 0.0;
        if (chem_var_eval_variance(&ham, vr.ndet_var, vr.dets, vr.psi0, &e_var_re, &var_fci, err,
                                   sizeof(err)) != 0) {
            chem_var_result_free(&vr); chem_ham_free(&ham); fclose(jf); return 1;
        }
        double sigma_fci = sqrt(fmax(0.0, var_fci));

        ChemGPCResult gpc;
        if (chem_var_analyze_gpc(&ham, vr.ndet_var, vr.dets, vr.psi0, &gpc, err, sizeof(err)) != 0) {
            chem_var_result_free(&vr); chem_ham_free(&ham); fclose(jf); return 1;
        }

        ChemSolveOpts sopts = chem_solve_opts_default(&ham);
        sopts.mode = CHEM_MODE_A_SINGLET;
        sopts.max_iters = 600;
        sopts.tol = 1e-6;
        ChemSolveResult sr_dqg;
        if (chem_solve_dqg(&ham, &sopts, &sr_dqg, err, sizeof(err)) != 0) {
            chem_var_result_free(&vr); chem_ham_free(&ham); fclose(jf); return 1;
        }

        ChemSolveOpts sopts_t2 = sopts;
        sopts_t2.max_iters = 800;
        sopts_t2.enable_t2 = 1;
        sopts_t2.t2_max_orbs = ham.K;
        sopts_t2.t2_norbs = ham.K;
        for (int p = 0; p < ham.K; p++) sopts_t2.t2_orbs[p] = p;
        ChemSolveResult sr_t2;
        if (chem_solve_dqg(&ham, &sopts_t2, &sr_t2, err, sizeof(err)) != 0) {
            chem_solve_result_free(&sr_dqg); chem_var_result_free(&vr); chem_ham_free(&ham); fclose(jf); return 1;
        }

        ChemObservable obs_docc, obs_sdot;
        ChemObservableResult r_docc, r_sdot;
        double docc_var = 0.0, sdot_var = 0.0;
        ChemSolveOpts osopts = chem_solve_opts_default(&ham);
        osopts.mode = CHEM_MODE_A_SINGLET;
        osopts.max_iters = 500;
        osopts.tol = 1e-6;
        if (chem_obs_double_occ(ham.K, 0, &obs_docc, err, sizeof(err)) != 0 ||
            chem_obs_spin_corr(ham.K, 0, 1, &obs_sdot, err, sizeof(err)) != 0 ||
            chem_eval_observable_witness(&ham, &obs_docc, vr.ndet_var, vr.dets, vr.psi0, &docc_var, err, sizeof(err)) != 0 ||
            chem_eval_observable_witness(&ham, &obs_sdot, vr.ndet_var, vr.dets, vr.psi0, &sdot_var, err, sizeof(err)) != 0 ||
            chem_certify_observable(&ham, &obs_docc, vr.E_ub, &osopts, &r_docc, err, sizeof(err)) != 0 ||
            chem_certify_observable(&ham, &obs_sdot, vr.E_ub, &osopts, &r_sdot, err, sizeof(err)) != 0) {
            chem_observable_free(&obs_docc); chem_observable_free(&obs_sdot);
            chem_solve_result_free(&sr_t2); chem_solve_result_free(&sr_dqg);
            chem_var_result_free(&vr); chem_ham_free(&ham); fclose(jf); return 1;
        }
        chem_observable_free(&obs_docc);
        chem_observable_free(&obs_sdot);

        double gap_dqg = (vr.E_ci - sr_dqg.E_lb_cert) * 1000.0;
        double gap_t2 = (vr.E_ci - sr_t2.E_lb_cert) * 1000.0;
        printf("  %-18s | gap DQG=%6.3f T2=%6.3f mHa | S1=%.3f | docc0=[%.4f,%.4f] | sdot01=[%.4f,%.4f]\n",
               HUBB[i].name, gap_dqg, gap_t2, gpc.entanglement_entropy,
               r_docc.O_lb_cert, r_docc.O_ub_cert, r_sdot.O_lb_cert, r_sdot.O_ub_cert);

        fprintf(jf,
                "    {\"name\": \"%s\", \"fcidump\": \"%s.fcidump\", \"U\": %.6g, \"frustrated\": %d, "
                "\"K\": %d, \"nelec\": %d, \"E_lb_dqg\": %.17g, \"E_lb_t2\": %.17g, \"E_fci\": %.17g, "
                "\"sigma_fci\": %.17g, \"gap_dqg_mha\": %.17g, \"gap_t2_mha\": %.17g, "
                "\"entanglement_entropy\": %.17g, \"docc0_lb\": %.17g, \"docc0_var\": %.17g, "
                "\"docc0_ub\": %.17g, \"sdot01_lb\": %.17g, \"sdot01_var\": %.17g, \"sdot01_ub\": %.17g}%s\n",
                HUBB[i].name, HUBB[i].name, HUBB[i].U, HUBB[i].frustrated, ham.K, ham.nelec,
                sr_dqg.E_lb_cert, sr_t2.E_lb_cert, vr.E_ci, sigma_fci, gap_dqg, gap_t2,
                gpc.entanglement_entropy, r_docc.O_lb_cert, docc_var, r_docc.O_ub_cert,
                r_sdot.O_lb_cert, sdot_var, r_sdot.O_ub_cert, (i + 1 < nhub) ? "," : "");

        chem_solve_result_free(&sr_t2);
        chem_solve_result_free(&sr_dqg);
        chem_var_result_free(&vr);
        chem_ham_free(&ham);
    }

    fprintf(jf, "  ],\n  \"gpc_cases\": [\n");
    struct GpcSpec {
        const char *name;
        const char *fcidump;
        ChemMode mode;
        int two_s;
        int certify_bd;
    };
    static const struct GpcSpec GPCS[] = {
        {"h3_doublet", "h3_doublet.fcidump", CHEM_MODE_B_GENERAL, 1, 1},
        {"h2_r1.4",    "h2_r1.4.fcidump",    CHEM_MODE_A_SINGLET, 0, 0},
        {"h2_r5.0",    "h2_r5.0.fcidump",    CHEM_MODE_A_SINGLET, 0, 0}
    };
    for (int i = 0; i < 3; i++) {
        char fpath[1024];
        snprintf(fpath, sizeof(fpath), "%s/%s", dir, GPCS[i].fcidump);
        ChemHam ham;
        if (chem_fcidump_load(fpath, &ham, err, sizeof(err)) != 0) { fclose(jf); return 1; }
        ChemVarOpts vopts = chem_var_opts_default();
        vopts.target_two_s = GPCS[i].two_s;
        ChemVarResult vr;
        if (chem_var_solve(&ham, &vopts, &vr, err, sizeof(err)) != 0) { chem_ham_free(&ham); fclose(jf); return 1; }
        double e_v = 0.0, var_v = 0.0;
        ChemGPCResult gpc;
        if (chem_var_eval_variance(&ham, vr.ndet_var, vr.dets, vr.psi0, &e_v, &var_v, err, sizeof(err)) != 0 ||
            chem_var_analyze_gpc(&ham, vr.ndet_var, vr.dets, vr.psi0, &gpc, err, sizeof(err)) != 0) {
            chem_var_result_free(&vr); chem_ham_free(&ham); fclose(jf); return 1;
        }
        double bd_lb = 0.0, bd_var = gpc.bd_slack, bd_ub = 0.0;
        if (GPCS[i].certify_bd && gpc.has_bd_3e3o) {
            ChemObservable bd_obs;
            ChemObservableResult bd_res;
            ChemSolveOpts bsopts = chem_solve_opts_default(&ham);
            bsopts.mode = GPCS[i].mode;
            bsopts.two_s = GPCS[i].two_s;
            bsopts.max_iters = 500;
            bsopts.tol = 1e-6;
            if (build_bd_3e3o_observable(&gpc, &bd_obs, err, sizeof(err)) != 0 ||
                chem_eval_observable_witness(&ham, &bd_obs, vr.ndet_var, vr.dets, vr.psi0, &bd_var, err, sizeof(err)) != 0 ||
                chem_certify_observable(&ham, &bd_obs, vr.E_ub, &bsopts, &bd_res, err, sizeof(err)) != 0) {
                chem_observable_free(&bd_obs); chem_var_result_free(&vr); chem_ham_free(&ham); fclose(jf); return 1;
            }
            chem_observable_free(&bd_obs);
            bd_lb = bd_res.O_lb_cert;
            bd_ub = bd_res.O_ub_cert;
        }
        fprintf(jf,
                "    {\"name\": \"%s\", \"fcidump\": \"%s\", \"K\": %d, \"nelec\": %d, \"two_s\": %d, "
                "\"E_fci\": %.17g, \"sigma_fci\": %.17g, \"entanglement_entropy\": %.17g, "
                "\"has_bd_3e3o\": %d, \"bd_pair_err\": %.17g, \"bd_slack\": %.17g, \"bd_pinned\": %d, "
                "\"has_bd_2e2o\": %d, \"bd24_pair_err\": %.17g, \"bd_lb_cert\": %.17g, "
                "\"bd_var\": %.17g, \"bd_ub_cert\": %.17g}%s\n",
                GPCS[i].name, GPCS[i].fcidump, ham.K, ham.nelec, GPCS[i].two_s, vr.E_ci,
                sqrt(fmax(0.0, var_v)), gpc.entanglement_entropy, gpc.has_bd_3e3o, gpc.bd_pair_err,
                gpc.bd_slack, gpc.bd_pinned, gpc.has_bd_2e2o, gpc.bd24_pair_err, bd_lb, bd_var,
                bd_ub, (i + 1 < 3) ? "," : "");
        chem_var_result_free(&vr);
        chem_ham_free(&ham);
    }

    /* Eckart Certified Fidelity & Exact Full-Space Variance on truncated H4 (4e,4o) */
    char h4path[1024];
    snprintf(h4path, sizeof(h4path), "%s/h4_ry2.0.fcidump", dir);
    ChemHam h4ham;
    if (chem_fcidump_load(h4path, &h4ham, err, sizeof(err)) != 0) { fclose(jf); return 1; }
    ChemVarOpts vref_opts = chem_var_opts_default();
    vref_opts.max_dets = 512;
    vref_opts.target_two_s = 0;
    ChemVarOpts vtr_opts = vref_opts;
    vtr_opts.max_dets = 4;
    ChemVarResult vref, vtr;
    if (chem_var_solve(&h4ham, &vref_opts, &vref, err, sizeof(err)) != 0 ||
        chem_var_solve(&h4ham, &vtr_opts, &vtr, err, sizeof(err)) != 0) {
        chem_ham_free(&h4ham); fclose(jf); return 1;
    }
    ChemSolveOpts hsopts = chem_solve_opts_default(&h4ham);
    hsopts.mode = CHEM_MODE_A_SINGLET;
    hsopts.tol = 1e-6;
    ChemSolveResult hsr;
    if (chem_solve_dqg(&h4ham, &hsopts, &hsr, err, sizeof(err)) != 0) {
        chem_var_result_free(&vref); chem_var_result_free(&vtr); chem_ham_free(&h4ham); fclose(jf); return 1;
    }
    double e_tr = 0.0, var_tr = 0.0;
    if (chem_var_eval_variance(&h4ham, vtr.ndet_var, vtr.dets, vtr.psi0, &e_tr, &var_tr, err,
                               sizeof(err)) != 0) {
        chem_solve_result_free(&hsr); chem_var_result_free(&vref); chem_var_result_free(&vtr); chem_ham_free(&h4ham); fclose(jf); return 1;
    }
    double n2_ref = 0.0, n2_tr = 0.0, ov = 0.0;
    for (int i = 0; i < vref.ndet_var; i++) n2_ref += vref.psi0[i] * vref.psi0[i];
    for (int i = 0; i < vtr.ndet_var; i++) {
        n2_tr += vtr.psi0[i] * vtr.psi0[i];
        for (int j = 0; j < vref.ndet_var; j++) {
            if (vtr.dets[i] == vref.dets[j]) {
                ov += vtr.psi0[i] * vref.psi0[j];
                break;
            }
        }
    }
    double fid_ex = (ov * ov) / (n2_ref * n2_tr);
    double denom = vref.E_ci_excited - hsr.E_lb_cert;
    double fid_lb = (denom > 1e-12) ? (1.0 - (e_tr - hsr.E_lb_cert) / denom) : 0.0;
    double sigma_tr = sqrt(fmax(0.0, var_tr));
    double e_wein = e_tr - sigma_tr;
    fprintf(jf,
            "  ],\n  \"variance_fidelity\": {\n"
            "    \"name\": \"h4_ry2.0\", \"fcidump\": \"h4_ry2.0.fcidump\", \"ndet_tr\": %d,\n"
            "    \"E_lb_sdp\": %.17g, \"E_fci\": %.17g, \"E1_fci\": %.17g,\n"
            "    \"E_tr\": %.17g, \"sigma_tr\": %.17g, \"E_weinstein\": %.17g,\n"
            "    \"fidelity_exact\": %.17g, \"fidelity_eckart_lb\": %.17g\n  }\n}\n",
            vtr.ndet_var, hsr.E_lb_cert, vref.E_ci, vref.E_ci_excited, e_tr, sigma_tr, e_wein,
            fid_ex, fid_lb);
    fclose(jf);
    chem_solve_result_free(&hsr);
    chem_var_result_free(&vref);
    chem_var_result_free(&vtr);
    chem_ham_free(&h4ham);
    printf("Atlas complete -> %s\n", out_json);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage();
        return 0;
    }
    if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0) {
        printf("CHEM v0.3.1 (Matrix-Free DQG+T1+T2 + Auto-NO + Zero-FP-Trust Lean 4)\n");
        return 0;
    }
    if (strcmp(argv[1], "selftest") == 0) return cmd_selftest(argc, argv);
    if (strcmp(argv[1], "bracket") == 0) return cmd_bracket_or_veto(0, argc, argv);
    if (strcmp(argv[1], "veto") == 0) return cmd_bracket_or_veto(1, argc, argv);
    if (strcmp(argv[1], "observable") == 0) return cmd_observable(argc, argv);
    if (strcmp(argv[1], "cert-verify") == 0) return cmd_cert_verify(argc, argv);
    if (strcmp(argv[1], "exact-verify") == 0) return cmd_exact_verify(argc, argv);
    if (strcmp(argv[1], "emit-lean") == 0) return cmd_emit_lean(argc, argv);
    if (strcmp(argv[1], "gpc") == 0) return cmd_gpc(argc, argv);
    if (strcmp(argv[1], "atlas") == 0) return cmd_atlas(argc, argv);
    if (strcmp(argv[1], "bench") == 0) return cmd_bench(argc, argv);
    print_usage();
    return 1;
}
