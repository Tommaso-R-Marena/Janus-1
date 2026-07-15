# AGENTS.md

## Cursor Cloud specific instructions

Janus-1 is a **pure-Python simulation/modeling library** (edge-AI processor
architecture study). There is no server, database, or web UI — the "application"
is the cycle-accurate simulator plus the analysis scripts under `examples/`.
Standard dev commands live in the `Makefile` and `README.md`; prefer those.

### Environment / running
- Dependencies are installed into a project virtualenv at `.venv` (created by the
  startup update script). There is **no `python` on PATH, only `python3`**, so
  either `source .venv/bin/activate` or call `.venv/bin/python` directly.
- Run the simulator (canonical hello-world, matches README Quick Start):
  `.venv/bin/python -m src.simulator.janus_sim` — expect `T1 Hit Rate ~100%` and
  `P50/P90/P99 = 1.0` cycles. This generates a ~67M-op trace and takes ~90s; use a
  smaller `generate_llm_trace(context_length=256, hidden_dim=1024)` for a fast run.
- Working analysis scripts: `examples/03_advanced_optimization.py`,
  `examples/04_thermal_modeling.py`.

### Known pre-existing breakage (repo code bugs, NOT environment problems)
- `janus-sim` console entry point fails — `pyproject.toml`/`setup.py` map it to a
  missing `main` in `src/simulator/janus_sim.py`. Run the module instead.
- `examples/01_basic_simulation.py` passes a `quantization=` kwarg that
  `generate_llm_trace` does not accept; `examples/02_kv_cache_analysis.py` imports
  a non-existent `calculate_kv_cache_size`.
- Several tests fail on pre-existing assertion/logic issues (e.g.
  `test_trace_generator`, `test_memory_hierarchy`, `test_statistical_analysis`).
- Do not "fix" these as part of environment work.

### Testing caveats
- **`pytest tests/` for the whole suite is memory-heavy and can be OOM-killed**
  (large traces in `test_benchmarks.py` / `test_integration.py`, ~3.5 min).
  Prefer running per-file (e.g. `pytest tests/test_models.py`) or `make test-fast`
  (`-m "not slow"`).
- Lint/format/type-check (`make lint`, `black --check`, `make type-check`) report
  pre-existing style/type issues; CI runs these with `continue-on-error`.
