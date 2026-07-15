#!/bin/bash
# Reproduce the results reported in METHODOLOGY.md.
#
# This runs ONLY experiments that exist in this repository and emits real
# outputs to results/. It does not fabricate a "matches the paper" report.
#
# Usage: bash scripts/reproduce_paper.sh

set -u

# Prefer the project virtualenv if present, else fall back to python3.
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

echo "======================================================================"
echo "Janus-1: Reproduction of METHODOLOGY.md results"
echo "Python: $($PY --version 2>&1)"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo N/A)"
echo "======================================================================"

mkdir -p results
status=0

run_step() {
    # run_step "<description>" <command...>
    local desc="$1"; shift
    echo ""
    echo "----------------------------------------------------------------------"
    echo ">>> ${desc}"
    echo "----------------------------------------------------------------------"
    if "$@"; then
        echo "[OK] ${desc}"
    else
        echo "[FAIL] ${desc}"
        status=1
    fi
}

# 1. Certify the simulator against closed-form expectations FIRST. If this
#    fails, downstream numbers are not trustworthy.
run_step "Simulator validation (closed-form checks)" \
    "$PY" experiments/validate_simulator.py

# 2. KV-cache sizing sanity (exact arithmetic).
run_step "KV-cache sizing analysis" \
    "$PY" examples/02_kv_cache_analysis.py

# 3. Prefetcher ablation (Table in METHODOLOGY.md section 3.1 + figure).
run_step "Prefetcher ablation" \
    "$PY" experiments/prefetch_ablation.py --seeds 15 --accesses 12000

# 4. Design-space sensitivity sweeps.
run_step "Prefetcher sensitivity sweeps" \
    "$PY" experiments/prefetch_sensitivity.py

# 5. Technology-selection robustness under parameter uncertainty.
run_step "Memory-technology robustness" \
    "$PY" experiments/power_robustness.py

# 6. Fast test subset (correctness + invariants). The full suite is memory-heavy
#    (see AGENTS.md); we run the fast, low-memory files here.
run_step "Fast correctness tests" \
    "$PY" -m pytest -q -o addopts="" \
        tests/test_analytical_validation.py \
        tests/test_prefetch_ablation.py \
        tests/test_memory_hierarchy.py \
        tests/test_simulator.py

echo ""
echo "======================================================================"
if [ "$status" -eq 0 ]; then
    echo "Reproduction complete. Outputs in results/:"
    ls -1 results/*.csv results/*.png 2>/dev/null | sed 's/^/  - /'
else
    echo "Reproduction finished WITH FAILURES (see [FAIL] lines above)."
fi
echo "======================================================================"
exit "$status"
