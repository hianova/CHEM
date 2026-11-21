#ifndef CHEM_FCIDUMP_H
#define CHEM_FCIDUMP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CHEM_MAX_ORBITALS 32

typedef struct {
    int K;       /* Spatial orbitals (NORB) */
    int nelec;   /* Total active electrons (NELEC) */
    int ms2;     /* Spin projection 2*Sz = na - nb (MS2) */
    int na;      /* Alpha electrons */
    int nb;      /* Beta electrons */
    double E_core;
    double *h;   /* K * K one-electron integrals (row-major, symmetric) */
    double *eri; /* K^4 chemists' (pq|rs) two-electron integrals */
    double eri_sym_err;
    double cs_max_viol;
    char sha256_hex[65];
} ChemHam;

static inline double chem_ham_h1(const ChemHam *ham, int p, int q) {
    return ham->h[(size_t)p * (size_t)ham->K + (size_t)q];
}

static inline double chem_ham_eri(const ChemHam *ham, int p, int q, int r, int s) {
    size_t K = (size_t)ham->K;
    size_t idx = ((((size_t)p * K + (size_t)q) * K + (size_t)r) * K + (size_t)s);
    return ham->eri[idx];
}

/* Allocates h and eri zero-initialized for (K, nelec, ms2). Returns 0 on success, -1 on error. */
int chem_ham_alloc(ChemHam *ham, int K, int nelec, int ms2, char *err, size_t err_sz);

/* Frees buffers inside ham and zeroes the struct. */
void chem_ham_free(ChemHam *ham);

/* Computes canonical SHA-256 hex digest (64 chars + NUL) of (K, nelec, ms2, E_core, h, eri). */
void chem_sha256_hex(const ChemHam *ham, char out_hex[65]);

/* Recomputes eri_sym_err, cs_max_viol, and sha256_hex for an in-memory Hamiltonian. */
void chem_ham_finalize(ChemHam *ham);

/* Loads a standard Molpro/PySCF FCIDUMP file into ham. Returns 0 on success, -1 on error. */
int chem_fcidump_load(const char *path, ChemHam *ham, char *err, size_t err_sz);

/* Writes ham in canonical 8-fold symmetric Molpro/PySCF FCIDUMP format. Returns 0 on success, -1 on error. */
int chem_fcidump_write(const char *path, const ChemHam *ham, char *err, size_t err_sz);

#ifdef __cplusplus
}
#endif

#endif /* CHEM_FCIDUMP_H */
