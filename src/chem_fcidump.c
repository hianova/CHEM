#include "chem_fcidump.h"

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

/* ============================================================================
 * FIPS 180-4 SHA-256 (Pure C17, zero external dependencies)
 * ============================================================================ */

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t buf[64];
    size_t buflen;
} ChemSha256;

static inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32U - n));
}

static void sha256_transform(ChemSha256 *ctx, const uint8_t block[64]) {
    static const uint32_t K256[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
        0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
        0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
        0xc67178f2U
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        size_t j = (size_t)i * 4U;
        w[i] = ((uint32_t)block[j] << 24U) | ((uint32_t)block[j + 1] << 16U) |
               ((uint32_t)block[j + 2] << 8U) | ((uint32_t)block[j + 3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i - 15], 7U) ^ rotr32(w[i - 15], 18U) ^ (w[i - 15] >> 3U);
        uint32_t s1 = rotr32(w[i - 2], 17U) ^ rotr32(w[i - 2], 19U) ^ (w[i - 2] >> 10U);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6U) ^ rotr32(e, 11U) ^ rotr32(e, 25U);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = rotr32(a, 2U) ^ rotr32(a, 13U) ^ rotr32(a, 22U);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sha256_init(ChemSha256 *ctx) {
    ctx->state[0] = 0x6a09e667U;
    ctx->state[1] = 0xbb67ae85U;
    ctx->state[2] = 0x3c6ef372U;
    ctx->state[3] = 0xa54ff53aU;
    ctx->state[4] = 0x510e527fU;
    ctx->state[5] = 0x9b05688cU;
    ctx->state[6] = 0x1f83d9abU;
    ctx->state[7] = 0x5be0cd19U;
    ctx->bitlen = 0;
    ctx->buflen = 0;
}

static void sha256_update(ChemSha256 *ctx, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    ctx->bitlen += (uint64_t)len * 8ULL;
    while (len > 0) {
        size_t take = 64U - ctx->buflen;
        if (take > len) take = len;
        memcpy(ctx->buf + ctx->buflen, p, take);
        ctx->buflen += take;
        p += take;
        len -= take;
        if (ctx->buflen == 64U) {
            sha256_transform(ctx, ctx->buf);
            ctx->buflen = 0;
        }
    }
}

static void sha256_final(ChemSha256 *ctx, uint8_t out[32]) {
    size_t i = ctx->buflen;
    ctx->buf[i++] = 0x80U;
    if (i > 56U) {
        while (i < 64U) ctx->buf[i++] = 0;
        sha256_transform(ctx, ctx->buf);
        i = 0;
    }
    while (i < 56U) ctx->buf[i++] = 0;
    for (int b = 0; b < 8; b++) {
        ctx->buf[56U + (size_t)b] = (uint8_t)((ctx->bitlen >> (56 - 8 * b)) & 0xFFU);
    }
    sha256_transform(ctx, ctx->buf);
    for (int k = 0; k < 8; k++) {
        out[4 * k + 0] = (uint8_t)((ctx->state[k] >> 24U) & 0xFFU);
        out[4 * k + 1] = (uint8_t)((ctx->state[k] >> 16U) & 0xFFU);
        out[4 * k + 2] = (uint8_t)((ctx->state[k] >> 8U) & 0xFFU);
        out[4 * k + 3] = (uint8_t)(ctx->state[k] & 0xFFU);
    }
}

static void sha256_u32_le(ChemSha256 *ctx, int32_t v) {
    uint32_t uv = (uint32_t)v;
    uint8_t le[4] = {
        (uint8_t)(uv & 0xFFU),
        (uint8_t)((uv >> 8U) & 0xFFU),
        (uint8_t)((uv >> 16U) & 0xFFU),
        (uint8_t)((uv >> 24U) & 0xFFU)
    };
    sha256_update(ctx, le, 4);
}

static void sha256_f64_le(ChemSha256 *ctx, double x) {
    double norm = (x == 0.0) ? 0.0 : x;
    uint64_t u = 0;
    memcpy(&u, &norm, sizeof(u));
    uint8_t le[8];
    for (int b = 0; b < 8; b++) {
        le[b] = (uint8_t)((u >> (8 * b)) & 0xFFU);
    }
    sha256_update(ctx, le, 8);
}

void chem_sha256_hex(const ChemHam *ham, char out_hex[65]) {
    static const char HEX[] = "0123456789abcdef";
    ChemSha256 ctx;
    sha256_init(&ctx);
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
    sha256_final(&ctx, digest);
    for (int i = 0; i < 32; i++) {
        out_hex[2 * i]     = HEX[(digest[i] >> 4) & 0xF];
        out_hex[2 * i + 1] = HEX[digest[i] & 0xF];
    }
    out_hex[64] = '\0';
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
