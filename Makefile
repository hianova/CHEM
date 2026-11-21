export PATH := $(HOME)/.elan/bin:/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/local/bin:$(PATH)

LLVM_CLANG := $(wildcard /opt/homebrew/opt/llvm/bin/clang)
CC ?= $(if $(LLVM_CLANG),$(LLVM_CLANG),clang)
PYTHON ?= python3
LEAN ?= lean

CFLAGS := -std=c17 -O3 -Wall -Wextra -Werror -Wpedantic -Wconversion -pthread -Iinclude -I../FLOW/core
LDFLAGS := -framework Accelerate -lm -pthread

BUILD := build
CERT_DIR := $(BUILD)/certs
BENCH_DIR := benchmarks

SRCS := src/chem_fcidump.c src/chem_rdm2_mf.c src/chem_shci.c src/chem_cli.c
HDRS := include/chem_fcidump.h include/chem_rdm2.h include/chem_shci.h
BIN := $(BUILD)/chem_cli
LEAN_OLEAN := $(BUILD)/ChemCert.olean
BENCH_JSON := $(BUILD)/chem_bench.json
REPORT := CHEM_VERIFICATION_REPORT.md

.PHONY: all lean benchmarks selftest bench verify clean

all: $(BIN) $(LEAN_OLEAN)

$(BUILD):
	@mkdir -p $(BUILD)

$(CERT_DIR):
	@mkdir -p $(CERT_DIR)

$(BIN): $(SRCS) $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $@

lean: $(LEAN_OLEAN)

$(LEAN_OLEAN): lean/ChemCert.lean | $(BUILD)
	LEAN_PATH=../FLOW/build/lean $(LEAN) -DwarningAsError=true -o $@ $<

benchmarks: $(BENCH_DIR)/h16_chain_16e16o.fcidump $(BENCH_DIR)/h4_ry2.0_fplus.fcidump

$(BENCH_DIR)/h16_chain_16e16o.fcidump $(BENCH_DIR)/h4_ry2.0_fplus.fcidump: oracles/gen_benchmarks.py
	$(PYTHON) oracles/gen_benchmarks.py --out-dir $(BENCH_DIR)

selftest: $(BIN)
	./$(BIN) selftest --json $(BUILD)/selftest.json

bench: $(BIN) benchmarks | $(CERT_DIR)
	./$(BIN) bench --dir $(BENCH_DIR) --out $(BENCH_JSON) --cert-dir $(CERT_DIR)

verify: all selftest bench
	@for c in $(CERT_DIR)/*.cert.json $(CERT_DIR)/*.flowchem; do \
		[ -e "$$c" ] || continue; \
		./$(BIN) cert-verify "$$c" || exit 1; \
	done
	$(PYTHON) oracles/verify_chem_oracle.py --bench $(BENCH_JSON) --cert-dir $(CERT_DIR) --bench-dir $(BENCH_DIR) --report $(REPORT)

clean:
	rm -rf $(BUILD)
