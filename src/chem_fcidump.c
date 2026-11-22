#include "chem_fcidump.h"
#include "flow_sha256.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_err(char *err, size_t err_sz, const char *fmt, ...) {
    if (!err || err_sz == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_sz, fmt, ap);
    va_end(ap);
}

static void sha256_u32_le(flow_sha256_ctx_t *ctx, int32_t v) {
    uint32_t uv = (uint32_t)v;
    uint8_t le[4] = {
        (uint8_t)(uv & 0xFFU),
        (uint8_t)((uv >> 8U) & 0xFFU),
        (uint8_t)((uv >> 16U) & 0xFFU),
        (uint8_t)((uv >> 24U) & 0xFFU)
    };
    flow_sha256_update(ctx, le, 4U);
}

static void sha256_f64_le(flow_sha256_ctx_t *ctx, double x) {
    double norm = (x == 0.0) ? 0.0 : x;
    uint64_t u = 0;
    memcpy(&u, &norm, sizeof(u));
    uint8_t le[8];
    for (int b = 0; b < 8; b++) {
        le[b] = (uint8_t)((u >> (8 * b)) & 0xFFU);
    }
    flow_sha256_update(ctx, le, 8U);
}

void chem_sha256_hex(const ChemHam *ham, char out_hex[65]) {
    flow_sha256_ctx_t ctx;
    flow_sha256_init(&ctx);
    sha256_u32_le(&ctx, (int32_t)ham->K);
    sha256_u32_le(&ctx, (int32_t)ham->nelec);
    sha256_u32_le(&ctx, (int32_t)ham->ms2);
    sha256_f64_le(&ctx, ham->E_core);
    size_t K = (size_t)ham->K;
    size_t n1 = K * K;
    for (size_t i = 0; i < n1; i++) {
        sha256_f64_le(&ctx, ham->h[i]);
    }
    size_t n2 = n1 * n1;
    for (size_t i = 0; i < n2; i++) {
        sha256_f64_le(&ctx, ham->eri[i]);
    }
    uint8_t digest[32];
    flow_sha256_final(&ctx, digest);
    flow_sha256_digest_to_hex(digest, out_hex);
}

/* ============================================================================
 * Hamiltonian Memory Management & Finalization
 * ============================================================================ */

int chem_ham_alloc(ChemHam *ham, int K, int nelec, int ms2, char *err, size_t err_sz) {
    memset(ham, 0, sizeof(*ham));
    if (K < 1 || K > CHEM_MAX_ORBITALS) {
        set_err(err, err_sz, "invalid NORB=%d (must be in [1, %d])", K, CHEM_MAX_ORBITALS);
        return -1;
    }
    if (nelec < 0 || nelec > 2 * K) {
        set_err(err, err_sz, "invalid NELEC=%d for NORB=%d", nelec, K);
        return -1;
    }
    if (((nelec + ms2) & 1) != 0) {
        set_err(err, err_sz, "incompatible NELEC=%d and MS2=%d (parity mismatch)", nelec, ms2);
        return -1;
    }
    int na = (nelec + ms2) / 2;
    int nb = (nelec - ms2) / 2;
    if (na < 0 || nb < 0 || na > K || nb > K) {
        set_err(err, err_sz, "invalid spin occupations na=%d, nb=%d for NORB=%d", na, nb, K);
        return -1;
    }
    size_t ksz = (size_t)K;
    size_t n1 = ksz * ksz;
    size_t n2 = n1 * n1;
    ham->h = (double *)calloc(n1, sizeof(double));
    ham->eri = (double *)calloc(n2, sizeof(double));
    if (!ham->h || !ham->eri) {
        chem_ham_free(ham);
        set_err(err, err_sz, "out of memory allocating Hamiltonian for K=%d", K);
        return -1;
    }
    ham->K = K;
    ham->nelec = nelec;
    ham->ms2 = ms2;
    ham->na = na;
    ham->nb = nb;
    return 0;
}

void chem_ham_free(ChemHam *ham) {
    if (!ham) return;
    free(ham->h);
    free(ham->eri);
    memset(ham, 0, sizeof(*ham));
}

void chem_ham_finalize(ChemHam *ham) {
    int K = ham->K;
    size_t ksz = (size_t)K;
    double max_sym = 0.0;
    double max_cs = 0.0;

    /* Enforce exact h symmetry (in case caller set one triangle) & verify ERI 8-fold symmetry */
    for (int p = 0; p < K; p++) {
        for (int q = p + 1; q < K; q++) {
            double v1 = ham->h[(size_t)p * ksz + (size_t)q];
            double v2 = ham->h[(size_t)q * ksz + (size_t)p];
            double d = fabs(v1 - v2);
            if (d > max_sym) max_sym = d;
        }
    }

    for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
            double pq_pq = chem_ham_eri(ham, p, q, p, q);
            double s_pq = (pq_pq > 0.0) ? sqrt(pq_pq) : 0.0;
            for (int r = 0; r < K; r++) {
                for (int s = 0; s < K; s++) {
                    double v = chem_ham_eri(ham, p, q, r, s);
                    double perms[7] = {
                        chem_ham_eri(ham, q, p, r, s),
                        chem_ham_eri(ham, p, q, s, r),
                        chem_ham_eri(ham, q, p, s, r),
                        chem_ham_eri(ham, r, s, p, q),
                        chem_ham_eri(ham, s, r, p, q),
                        chem_ham_eri(ham, r, s, q, p),
                        chem_ham_eri(ham, s, r, q, p)
                    };
                    for (int k = 0; k < 7; k++) {
                        double d = fabs(v - perms[k]);
                        if (d > max_sym) max_sym = d;
                    }
                    double rs_rs = chem_ham_eri(ham, r, s, r, s);
                    double s_rs = (rs_rs > 0.0) ? sqrt(rs_rs) : 0.0;
                    double viol = fabs(v) - s_pq * s_rs;
                    if (viol > max_cs) max_cs = viol;
                }
            }
        }
    }

    ham->eri_sym_err = max_sym;
    ham->cs_max_viol = (max_cs > 0.0) ? max_cs : 0.0;
    chem_sha256_hex(ham, ham->sha256_hex);
}

/* ============================================================================
 * FCIDUMP Parser
 * ============================================================================ */

static void parse_header_kv(const char *hdr, const char *key, int *has_val, int *out_val) {
    size_t klen = strlen(key);
    for (const char *p = hdr; *p; p++) {
        /* Check word boundary before key */
        if (p > hdr && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) continue;
        size_t i = 0;
        while (i < klen && p[i] && toupper((unsigned char)p[i]) == key[i]) i++;
        if (i != klen) continue;
        const char *q = p + klen;
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q != '=') continue;
        q++;
        while (*q && isspace((unsigned char)*q)) q++;
        char *endp = NULL;
        long v = strtol(q, &endp, 10);
        if (endp != q) {
            *has_val = 1;
            *out_val = (int)v;
            return;
        }
    }
}

static void set_eri_8fold(ChemHam *ham, int p, int q, int r, int s, double val) {
    size_t K = (size_t)ham->K;
    int idx[8][4] = {
        {p, q, r, s}, {q, p, r, s}, {p, q, s, r}, {q, p, s, r},
        {r, s, p, q}, {s, r, p, q}, {r, s, q, p}, {s, r, q, p}
    };
    for (int k = 0; k < 8; k++) {
        size_t pos = ((((size_t)idx[k][0] * K + (size_t)idx[k][1]) * K + (size_t)idx[k][2]) * K +
                      (size_t)idx[k][3]);
        ham->eri[pos] = val;
    }
}

int chem_fcidump_load(const char *path, ChemHam *ham, char *err, size_t err_sz) {
    memset(ham, 0, sizeof(*ham));
    if (!path) {
        set_err(err, err_sz, "null FCIDUMP path");
        return -1;
    }
    FILE *fp = fopen(path, "r");
    if (!fp) {
        set_err(err, err_sz, "cannot open FCIDUMP file: %s", path);
        return -1;
    }

    char *hdr = NULL;
    size_t hdr_len = 0;
    size_t hdr_cap = 0;
    char line[4096];
    int found_end = 0;

    while (fgets(line, (int)sizeof(line), fp)) {
        size_t llen = strlen(line);
        if (hdr_len + llen + 2 > hdr_cap) {
            size_t new_cap = (hdr_cap == 0) ? 4096U : hdr_cap * 2U;
            while (new_cap < hdr_len + llen + 2) new_cap *= 2U;
            char *nb = (char *)realloc(hdr, new_cap);
            if (!nb) {
                free(hdr);
                fclose(fp);
                set_err(err, err_sz, "out of memory reading FCIDUMP header");
                return -1;
            }
            hdr = nb;
            hdr_cap = new_cap;
        }
        memcpy(hdr + hdr_len, line, llen);
        hdr_len += llen;
        hdr[hdr_len] = '\0';

        /* Check for &END or / terminating namelist header */
        for (size_t i = 0; i < llen; i++) {
            if (line[i] == '/') {
                found_end = 1;
                break;
            }
            if (line[i] == '&' && i + 3 < llen &&
                toupper((unsigned char)line[i + 1]) == 'E' &&
                toupper((unsigned char)line[i + 2]) == 'N' &&
                toupper((unsigned char)line[i + 3]) == 'D') {
                found_end = 1;
                break;
            }
        }
        if (found_end) break;
    }

    if (!found_end || !hdr) {
        free(hdr);
        fclose(fp);
        set_err(err, err_sz, "missing &END or / in FCIDUMP header: %s", path);
        return -1;
    }

    int has_norb = 0, has_nelec = 0, has_ms2 = 0;
    int K = 0, nelec = 0, ms2 = 0;
    parse_header_kv(hdr, "NORB", &has_norb, &K);
    parse_header_kv(hdr, "NELEC", &has_nelec, &nelec);
    parse_header_kv(hdr, "MS2", &has_ms2, &ms2);
    free(hdr);

    if (!has_norb || !has_nelec) {
        fclose(fp);
        set_err(err, err_sz, "FCIDUMP header missing NORB or NELEC: %s", path);
        return -1;
    }
    if (!has_ms2) ms2 = 0;

    if (chem_ham_alloc(ham, K, nelec, ms2, err, err_sz) != 0) {
        fclose(fp);
        return -1;
    }

    size_t ksz = (size_t)K;
    int line_no = 0;
    while (fgets(line, (int)sizeof(line), fp)) {
        line_no++;
        char *s = line;
        while (*s && isspace((unsigned char)*s)) s++;
        if (*s == '\0' || *s == '!' || *s == '#' || *s == '*') continue;

        /* Convert Fortran D/d exponent notation into E/e on the first token */
        char *tok = s;
        while (*tok && !isspace((unsigned char)*tok)) {
            if (*tok == 'D') *tok = 'E';
            else if (*tok == 'd') *tok = 'e';
            tok++;
        }

        char *endp = NULL;
        double val = strtod(s, &endp);
        if (endp == s) {
            chem_ham_free(ham);
            fclose(fp);
            set_err(err, err_sz, "parse error in FCIDUMP integral line %d", line_no);
            return -1;
        }
        long p = strtol(endp, &endp, 10);
        long q = strtol(endp, &endp, 10);
        long r = strtol(endp, &endp, 10);
        long s_idx = strtol(endp, &endp, 10);

        if (p == 0 && q == 0 && r == 0 && s_idx == 0) {
            ham->E_core = val;
        } else if (p > 0 && q > 0 && r == 0 && s_idx == 0) {
            if (p > K || q > K) {
                chem_ham_free(ham);
                fclose(fp);
                set_err(err, err_sz, "1e index out of bounds (%ld, %ld) > K=%d", p, q, K);
                return -1;
            }
            size_t ip = (size_t)(p - 1);
            size_t iq = (size_t)(q - 1);
            ham->h[ip * ksz + iq] = val;
            ham->h[iq * ksz + ip] = val;
        } else if (p > 0 && q == 0 && r == 0 && s_idx == 0) {
            /* Optional orbital energies in Molpro FCIDUMP; ignored for H */
            if (p > K) {
                chem_ham_free(ham);
                fclose(fp);
                set_err(err, err_sz, "orbital index out of bounds (%ld) > K=%d", p, K);
                return -1;
            }
        } else if (p > 0 && q > 0 && r > 0 && s_idx > 0) {
            if (p > K || q > K || r > K || s_idx > K) {
                chem_ham_free(ham);
                fclose(fp);
                set_err(err, err_sz, "2e index out of bounds (%ld,%ld,%ld,%ld) > K=%d",
                        p, q, r, s_idx, K);
                return -1;
            }
            set_eri_8fold(ham, (int)(p - 1), (int)(q - 1), (int)(r - 1), (int)(s_idx - 1), val);
        } else {
            chem_ham_free(ham);
            fclose(fp);
            set_err(err, err_sz, "unsupported FCIDUMP index tuple (%ld,%ld,%ld,%ld)",
                    p, q, r, s_idx);
            return -1;
        }
    }

    fclose(fp);
    chem_ham_finalize(ham);
    return 0;
}

int chem_fcidump_write(const char *path, const ChemHam *ham, char *err, size_t err_sz) {
    if (!path || !ham || !ham->h || !ham->eri) {
        set_err(err, err_sz, "invalid argument to chem_fcidump_write");
        return -1;
    }
    FILE *fp = fopen(path, "w");
    if (!fp) {
        set_err(err, err_sz, "cannot open FCIDUMP '%s' for writing", path);
        return -1;
    }
    int K = ham->K;
    fprintf(fp, " &FCI NORB=%d,NELEC=%d,MS2=%d,\n", K, ham->nelec, ham->ms2);
    fprintf(fp, "  ORBSYM=");
    for (int i = 0; i < K; i++) fprintf(fp, "1,");
    fprintf(fp, "\n  ISYM=1,\n &END\n");
    for (int p = 0; p < K; p++) {
        for (int q = 0; q <= p; q++) {
            int pq = p * (p + 1) / 2 + q;
            for (int r = 0; r < K; r++) {
                for (int s = 0; s <= r; s++) {
                    int rs = r * (r + 1) / 2 + s;
                    if (rs > pq) continue;
                    double v = chem_ham_eri(ham, p, q, r, s);
                    if (fabs(v) > 1e-15) {
                        fprintf(fp, " %25.17e %3d %3d %3d %3d\n", v, p + 1, q + 1, r + 1, s + 1);
                    }
                }
            }
        }
    }
    for (int p = 0; p < K; p++) {
        for (int q = 0; q <= p; q++) {
            double v = chem_ham_h1(ham, p, q);
            if (fabs(v) > 1e-15) {
                fprintf(fp, " %25.17e %3d %3d   0   0\n", v, p + 1, q + 1);
            }
        }
    }
    fprintf(fp, " %25.17e   0   0   0   0\n", ham->E_core);
    if (fclose(fp) != 0) {
        set_err(err, err_sz, "error flushing FCIDUMP '%s'", path);
        return -1;
    }
    return 0;
}
