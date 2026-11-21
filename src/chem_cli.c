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
            "                   [--t2] [--t2-orbs <Ks>] [--threads <n>] [--witness <file.json>]\n"
            "                   [--cert <cert.json>] [--cert-bin <cert.flowchem>] [--json <out.json>] [--verbose]\n"
            "  chem_cli veto <fcidump> --e-ai <Ha> [--veto-tol <Ha>] [--mode auto|a|b]\n"
            "                [--two-s <int>] [--no-s2] [--max-iters <int>] [--tol <double>]\n"
            "                [--max-dets <int>] [--t2] [--t2-orbs <Ks>] [--threads <n>]\n"
            "                [--witness <file.json>] [--cert <cert.json>] [--cert-bin <cert.flowchem>] [--json <out.json>] [--verbose]\n"
            "  chem_cli observable <fcidump> --type occ:<p>|docc:<p>|sdot:<p>:<q>|hf-force:<+>:<->:<dR>\n"
            "                      [--mode auto|a|b] [--two-s <int>] [--no-s2] [--max-iters <int>]\n"
            "                      [--tol <double>] [--max-dets <int>] [--t2] [--t2-orbs <Ks>]\n"
            "                      [--threads <n>] [--witness <file.json>] [--json <out.json>] [--verbose]\n"
            "  chem_cli cert-verify [<fcidump>] <cert.json|cert.flowchem>\n"
            "  chem_cli bench --dir <benchmarks_dir> --out <bench.json> --cert-dir <certs_dir>\n");
}

static void select_frontier_t2_orbs(const ChemHam *ham, const ChemVarResult *vr, int cap,
                                    ChemSolveOpts *sopts) {
    int K = ham->K;
    if (K <= cap) {
        sopts->t2_norbs = K;
        for (int i = 0; i < K; i++) sopts->t2_orbs[i] = i;
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
    for (int i = 0; i < cap; i++) sopts->t2_orbs[i] = idx[i];
    for (int i = 0; i < cap; i++) {
        for (int j = i + 1; j < cap; j++) {
            if (sopts->t2_orbs[j] < sopts->t2_orbs[i]) {
                int tmp = sopts->t2_orbs[i];
                sopts->t2_orbs[i] = sopts->t2_orbs[j];
                sopts->t2_orbs[j] = tmp;
            }
        }
    }
    sopts->t2_norbs = cap;
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

static int write_fcidump_file(const char *path, const ChemHam *ham) {
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    fprintf(fp, " &FCI NORB=%d,NELEC=%d,MS2=%d,\n", ham->K, ham->nelec, ham->ms2);
    fprintf(fp, "  ORBSYM=");
    for (int i = 0; i < ham->K; i++) fprintf(fp, "1,");
    fprintf(fp, "\n  ISYM=1,\n &END\n");
    int K = ham->K;
    for (int p = 0; p < K; p++) {
        for (int q = 0; q <= p; q++) {
            int pq = p * (p + 1) / 2 + q;
            for (int r = 0; r < K; r++) {
                for (int s = 0; s <= r; s++) {
                    int rs = r * (r + 1) / 2 + s;
                    if (rs > pq) continue;
                    double v = chem_ham_eri(ham, p, q, r, s);
                    if (fabs(v) > 1e-16) {
                        fprintf(fp, " %24.16e %3d %3d %3d %3d\n", v, p + 1, q + 1, r + 1, s + 1);
                    }
                }
            }
        }
    }
    for (int p = 0; p < K; p++) {
        for (int q = 0; q <= p; q++) {
            double v = chem_ham_h1(ham, p, q);
            if (fabs(v) > 1e-16) {
                fprintf(fp, " %24.16e %3d %3d   0   0\n", v, p + 1, q + 1);
            }
        }
    }
    fprintf(fp, " %24.16e   0   0   0   0\n", ham->E_core);
    fclose(fp);
    return 0;
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

static void write_bracket_summary_json(FILE *fp, const char *name, const ChemHam *ham,
                                       const ChemSolveResult *sr, const ChemVarResult *vr,
                                       const ChemSolveResult *sr_veto, double e_ai_synth) {
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
    printf("=== CHEM v0.2.0 Self-Test Suite ===\n");

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
    /* Check exact T2(D_FCI) PSD and trace on H4 Mode A & Mode B before freeing Da/Db */
    double me_t2_a = 0.0, te_t2_a = 0.0, me_t2_b = 0.0, te_t2_b = 0.0;
    const int orbs4[4] = {0, 1, 2, 3};
    if (chem_verify_fci_t2_psd(&h4, CHEM_MODE_A_SINGLET, 0, 4, orbs4, Da, &me_t2_a, &te_t2_a, err,
                               sizeof(err)) != 0 ||
        chem_verify_fci_t2_psd(&h4, CHEM_MODE_B_GENERAL, 0, 4, orbs4, Db, &me_t2_b, &te_t2_b, err,
                               sizeof(err)) != 0) {
        fprintf(stderr, "FAIL FCI T2 PSD: %s\n", err);
        return 1;
    }
    if (me_t2_a < -1e-13 || te_t2_a > 1e-12 || me_t2_b < -1e-13 || te_t2_b > 1e-12) {
        fprintf(stderr, "FAIL FCI T2 PSD/trace: A(eig=%.2e, tr=%.2e) B(eig=%.2e, tr=%.2e)\n",
                me_t2_a, te_t2_a, me_t2_b, te_t2_b);
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
    printf("  [PASS] H4 FCI Exact CAR & PSD (+T2): car_A=%.2e car_B=%.2e t2_eig=%.2e dE=%.2e\n",
           car_a, car_b, fmin(me_t2_a, me_t2_b), fmax(fabs(e_a - vr4.E_ci), fabs(e_b - vr4.E_ci)));

    /* 1b. T2 matrix-free adjoint checks across Mode A & Mode B */
    const int t2_adj_cases[5][5] = {
        {4, 2, 2, CHEM_MODE_A_SINGLET, 4},
        {6, 3, 3, CHEM_MODE_A_SINGLET, 6},
        {4, 2, 2, CHEM_MODE_B_GENERAL, 4},
        {5, 3, 1, CHEM_MODE_B_GENERAL, 5},
        {6, 4, 2, CHEM_MODE_B_GENERAL, 4}
    };
    double max_t2_adj = 0.0;
    for (int c = 0; c < 5; c++) {
        int norb = t2_adj_cases[c][4];
        int orbs[6] = {0, 1, 2, 3, 4, 5};
        ChemAdjointReport rep;
        int na_c = t2_adj_cases[c][1], nb_c = t2_adj_cases[c][2];
        if (chem_test_adjoints_t2(t2_adj_cases[c][0], na_c, nb_c, na_c - nb_c, 1,
                                  (ChemMode)t2_adj_cases[c][3], norb, orbs,
                                  UINT64_C(0x55551111) + (uint64_t)c, &rep, err, sizeof(err)) !=
            0) {
            fprintf(stderr, "FAIL T2 adjoint case %d: %s\n", c, err);
            return 1;
        }
        if (rep.max_err > max_t2_adj) max_t2_adj = rep.max_err;
        if (!(rep.max_err < 1e-14)) {
            fprintf(stderr, "FAIL T2 adjoint error %.3e (case %d)\n", rep.max_err, c);
            return 1;
        }
    }
    printf("  [PASS] Matrix-Free T2 Adjoints (Mode A & Mode B): max_err = %.3e (< 1e-14)\n",
           max_t2_adj);

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
    if (write_fcidump_file(tmp_fci, &h4) != 0) {
        tmp_fci = ".selftest_h4.fcidump";
        tmp_cert = ".selftest_h4.cert.json";
        tmp_bin = ".selftest_h4.flowchem";
        if (write_fcidump_file(tmp_fci, &h4) != 0) return 1;
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

    /* 5b. DQG+T2 tightening and Certified Observable Brackets (H4 Mode A & H3 Mode B) */
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
    printf("  [PASS] H4 DQG+T2 Tightening: DQG=%.8f <= DQG+T2=%.8f <= FCI=%.8f\n",
           sr4_a.E_lb_cert, sr4_t2.E_lb_cert, vr4.E_ci);

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
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--witness") == 0 && i + 1 < argc) {
            witness_path = argv[++i];
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
    sopts.num_threads = num_threads;
    sopts.verbose = verbose;

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

    if (sopts.enable_t2) {
        select_frontier_t2_orbs(&ham, &vr, t2_cap, &sopts);
    }

    ChemSolveResult sr;
    if (chem_solve_dqg(&ham, &sopts, &sr, err, sizeof(err)) != 0) {
        fprintf(stderr, "ERROR in DQG solver: %s\n", err);
        chem_var_result_free(&vr);
        chem_ham_free(&ham);
        return 1;
    }

    ChemVetoResult vres = chem_make_veto_result(&sr, vr.E_ub);
    printf("CHEM %s (%s): K=%d nelec=%d ms2=%d mode=%s%s\n",
           is_veto ? "VETO" : "BRACKET", fcidump, ham.K, ham.nelec, ham.ms2,
           (sr.mode_used == CHEM_MODE_A_SINGLET) ? "A_SINGLET" : "B_GENERAL",
           sr.t2_enabled ? "+T2" : "");
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
        if (chem_cert_write_json(cert_path, &ham, fcidump, &sr, &vr, &sopts, err, sizeof(err)) !=
            0) {
            fprintf(stderr, "ERROR writing certificate: %s\n", err);
            chem_solve_result_free(&sr);
            chem_var_result_free(&vr);
            chem_ham_free(&ham);
            return 1;
        }
        printf("  Certificate      = %s\n", cert_path);
    }
    if (cert_bin_path) {
        if (chem_cert_write_bin(cert_bin_path, &ham, fcidump, &sr, &vr, &sopts, err,
                                sizeof(err)) != 0) {
            fprintf(stderr, "ERROR writing binary certificate: %s\n", err);
            chem_solve_result_free(&sr);
            chem_var_result_free(&vr);
            chem_ham_free(&ham);
            return 1;
        }
        printf("  Binary Cert      = %s\n", cert_bin_path);
    }

    if (json_path) {
        FILE *jf = fopen(json_path, "w");
        if (jf) {
            fprintf(jf, "{\n  \"peak_rss_mb\": %.3f,\n  \"results\": [\n", peak_rss_mb());
            write_bracket_summary_json(jf, fcidump, &ham, &sr, &vr, is_veto ? &sr : NULL, e_ai);
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
    ChemMode mode = CHEM_MODE_AUTO;
    int two_s_override = -1;
    int constrain_s2 = 1;
    int max_iters = 400;
    double tol = 1e-6;
    int max_dets = 1024;
    int enable_t2 = 0;
    int t2_cap = 6;
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
    sopts.num_threads = num_threads;

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

    if (sopts.enable_t2) select_frontier_t2_orbs(&ham, &vr, t2_cap, &sopts);

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
           fcidump, ores.name, ores.O_lb_cert, ores.O_var, ores.O_ub_cert,
           ores.interval_width, ores.lam_lo, ores.lam_hi, ores.elapsed_sec);

    if (json_path) {
        FILE *jf = fopen(json_path, "w");
        if (jf) {
            fprintf(jf,
                    "{\n  \"fcidump\": \"%s\",\n  \"observable\": \"%s\",\n"
                    "  \"E_ub\": %.17g,\n  \"O_lb_cert\": %.17g,\n  \"O_var\": %.17g,\n"
                    "  \"O_ub_cert\": %.17g,\n  \"interval_width\": %.17g,\n"
                    "  \"lam_lo\": %.17g,\n  \"lam_hi\": %.17g,\n  \"elapsed_sec\": %.6f\n}\n",
                    fcidump, ores.name, ores.E_ub, ores.O_lb_cert, ores.O_var, ores.O_ub_cert,
                    ores.interval_width, ores.lam_lo, ores.lam_hi, ores.elapsed_sec);
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

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    if (strcmp(argv[1], "selftest") == 0) return cmd_selftest(argc, argv);
    if (strcmp(argv[1], "bracket") == 0) return cmd_bracket_or_veto(0, argc, argv);
    if (strcmp(argv[1], "veto") == 0) return cmd_bracket_or_veto(1, argc, argv);
    if (strcmp(argv[1], "observable") == 0) return cmd_observable(argc, argv);
    if (strcmp(argv[1], "cert-verify") == 0) return cmd_cert_verify(argc, argv);
    if (strcmp(argv[1], "bench") == 0) return cmd_bench(argc, argv);
    print_usage();
    return 1;
}
