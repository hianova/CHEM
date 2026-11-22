export PATH := $(HOME)/.elan/bin:/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/local/bin:$(PATH)

LLVM_CLANG := $(wildcard /opt/homebrew/opt/llvm/bin/clang)
CC ?= $(if $(LLVM_CLANG),$(LLVM_CLANG),clang)
PYTHON ?= python3
LEAN ?= lean

CFLAGS := -std=c17 -O3 -Wall -Wextra -Werror -Wpedantic -Wconversion -pthread -Iinclude -I../FLOW/core -I../FLOW/sdk
LDFLAGS := -framework Accelerate -lm -pthread

BUILD := build
CERT_DIR := $(BUILD)/certs
BENCH_DIR := benchmarks

SRCS := src/chem_fcidump.c src/chem_rdm2_mf.c src/chem_shci.c src/chem_cli.c
LIB_SRCS := src/chem_fcidump.c src/chem_rdm2_mf.c src/chem_shci.c
HDRS := include/chem_fcidump.h include/chem_rdm2.h include/chem_shci.h
BIN := $(BUILD)/chem_cli
LIB := $(BUILD)/libchem.dylib
LEAN_OLEAN := $(BUILD)/ChemCert.olean
BENCH_JSON := $(BUILD)/chem_bench.json
ATLAS_JSON := $(BUILD)/chem_atlas.json
REPORT := CHEM_VERIFICATION_REPORT.md

.PHONY: all lean benchmarks selftest bench atlas test-tools verify clean

all: $(BIN) $(LIB) $(LEAN_OLEAN)

$(BUILD):
	@mkdir -p $(BUILD)

$(CERT_DIR):
	@mkdir -p $(CERT_DIR)

$(BIN): $(SRCS) $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $@

$(LIB): $(LIB_SRCS) $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -dynamiclib $(LIB_SRCS) $(LDFLAGS) -o $@

lean: $(LEAN_OLEAN)

$(LEAN_OLEAN): lean/ChemCert.lean | $(BUILD)
	@test -f ../FLOW/build/lean/Flow/MolecularBundle.olean && test -f ../FLOW/build/lean/Flow/CholeskySDD.olean || $(MAKE) -C ../FLOW theory
	LEAN_PATH=../FLOW/build/lean $(LEAN) -DwarningAsError=true -o $@ $<

benchmarks: $(BENCH_DIR)/h16_chain_16e16o.fcidump $(BENCH_DIR)/h4_ry2.0_fplus.fcidump $(BENCH_DIR)/hubbard4_u12.0.fcidump

$(BENCH_DIR)/h16_chain_16e16o.fcidump $(BENCH_DIR)/h4_ry2.0_fplus.fcidump $(BENCH_DIR)/hubbard4_u12.0.fcidump: oracles/gen_benchmarks.py
	$(PYTHON) oracles/gen_benchmarks.py --out-dir $(BENCH_DIR)

selftest: $(BIN)
	./$(BIN) selftest --json $(BUILD)/selftest.json

bench: $(BIN) benchmarks | $(CERT_DIR)
	./$(BIN) bench --dir $(BENCH_DIR) --out $(BENCH_JSON) --cert-dir $(CERT_DIR)

atlas: $(BIN) benchmarks | $(BUILD)
	./$(BIN) atlas --dir $(BENCH_DIR) --out $(ATLAS_JSON)

test-tools: $(BIN) $(LIB)
	$(PYTHON) oracles/test_tools.py

verify: all selftest bench atlas
	@for c in $(CERT_DIR)/*.cert.json $(CERT_DIR)/*.flowchem; do \
		[ -e "$$c" ] || continue; \
		./$(BIN) cert-verify "$$c" || exit 1; \
		./$(BIN) exact-verify "$$c" || exit 1; \
	done
	./$(BIN) emit-lean $(BENCH_DIR)/h2_r1.4.fcidump $(CERT_DIR)/h2_r1.4.cert.json --out $(CERT_DIR)/CertH2R14.lean --name CertH2R14
	./$(BIN) emit-lean $(BENCH_DIR)/h2_r5.0.fcidump $(CERT_DIR)/h2_r5.0.cert.json --out $(CERT_DIR)/CertH2R50.lean --name CertH2R50
	./$(BIN) emit-lean $(BENCH_DIR)/h4_ry2.0.fcidump $(CERT_DIR)/h4_ry2.0.cert.json --out $(CERT_DIR)/CertH4Ry20.lean --name CertH4Ry20
	@for m in CertH2R14 CertH2R50 CertH4Ry20; do \
		LEAN_PATH=../FLOW/build/lean:$(BUILD) $(LEAN) -DwarningAsError=true -o $(BUILD)/$$m.olean $(CERT_DIR)/$$m.lean > $(CERT_DIR)/$$m.axioms.txt || exit 1; \
		cp $(BUILD)/$$m.olean $(CERT_DIR)/$$m.olean; \
		LEAN_PATH=../FLOW/build/lean:$(BUILD) leanchecker $$m || exit 1; \
		echo "LEANCHECKER VERIFIED: $$m (0 sorryAx)"; \
	done
	$(PYTHON) oracles/verify_chem_oracle.py --bench $(BENCH_JSON) --atlas $(ATLAS_JSON) --cert-dir $(CERT_DIR) --bench-dir $(BENCH_DIR) --report $(REPORT)
	$(PYTHON) oracles/test_tools.py

clean:
	rm -rf $(BUILD)
