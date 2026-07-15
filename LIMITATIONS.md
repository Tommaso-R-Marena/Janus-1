# Limitations, Reproducibility, and Research-Integrity Notes

This document is an honest audit of what the Janus-1 repository **does and does
not** establish. It is intended to keep the project's claims aligned with the
code that actually runs, which is a prerequisite for any publishable use of this
material. It was produced while repairing and validating the simulator.

## 1. What this artifact is

Janus-1 is a **Python simulation and analytical-modeling study** of a two-tier
(SRAM + eDRAM) memory hierarchy with a stream prefetcher for transformer
KV-cache access patterns. It is **not** an RTL design, a synthesized chip, or a
silicon measurement. All power, area, and performance figures are analytical
estimates from hand-chosen parameters unless a specific validated source is
cited (currently none are).

## 2. Substantiated by code (reproducible)

* **KV-cache sizing** (`src/models/kv_cache_sizing.py`) is exact arithmetic:
  `2 · num_layers · hidden_dim · bytes · context`. Reproduce with
  `python examples/02_kv_cache_analysis.py`.
* **Memory-technology power/area comparison** (`src/models/memory_power_model.py`)
  is an internally consistent analytical model (see §4 for the caveat on its
  constants).
* **Memory-hierarchy simulation and the prefetcher ablation** — after the fixes
  described in §3 — are reproducible via
  `python experiments/prefetch_ablation.py`, which emits
  `results/prefetch_ablation.csv` and `results/prefetch_ablation.png`.

## 3. Bugs found and fixed during this review

* **Demand-miss double counting (critical).** The original run loop re-processed
  the trace entry that caused a demand miss after its fill landed, counting it a
  second time as a T1 *hit*. A single `READ` reported `1 hit + 1 miss`; four
  identical reads reported an 80% hit rate instead of the correct 75%. This
  inflated **every** reported hit rate. Fixed by modeling an in-order core with a
  single outstanding demand miss that retires each access exactly once.
* **No baseline / control.** The simulator had no way to disable prefetching, so
  the prefetcher's contribution could never be isolated. Added
  `SimulationConfig.prefetch_mode ∈ {none, next_line, stream}`.
* **Broken entry point and examples.** The `janus-sim` console script referenced
  a non-existent `main`; `examples/01` and `examples/02` called an imagined API.
  All now run.

## 4. Claims NOT substantiated by any code (do not cite as results)

The following appear in `README.md` and other prose but are **not produced or
validated by any code in this repository**. They should be treated as untested
design targets at best, and removed or clearly labeled before publication:

* **"99.99% hit rate from the Janus-Prefetch-1 engine."** On the shipped LLM KV
  trace the prefetcher is essentially **inert**: the per-token KV stride spans
  ~128 cache lines, so the unit-stride detector never fires (`prefetch_bw = 0`).
  The high hit rate on that trace comes from the small working set fitting in the
  32 MB T1 cache (and being pre-populated by write-allocate), **not** from
  prefetching. See `examples/01_basic_simulation.py`, where demand-only and
  stream modes both reach ~100% on that trace.
* **"8.2 TOPS", "~4.05 W total system power", "~79 mm² die", "3nm GAA".** There
  is no compute-fabric model, no synthesis, and no floorplan in the repository.
  These numbers are not derived by any code.
* **"Validated INT4 quantization: Llama-2 7B at 6.04 perplexity on WikiText-103."**
  There is no quantization or perplexity-evaluation code (the referenced
  `src/benchmarks/validation.py` does not exist). This claim is unsupported.
* **Comparison table vs Google Edge TPU / NVIDIA Jetson Orin.** Not computed from
  any model here; the competitor figures are uncited literals.
* **"INT4 → 256 MB" for Llama-2 7B at 4K context.** The repository's own formula
  gives **512 MB** at INT4/4K (see `examples/02`). The 256 MB figure is an
  internal inconsistency.
* **`scripts/reproduce_paper.sh`** invokes six `experiments/*.py` scripts and a
  figure generator that **did not exist**, and pre-writes a report asserting
  "results match published paper." It cannot currently reproduce the paper.

## 5. Limitations of the simulator itself

* **Single-stream prefetcher.** The FSM tracks exactly one unit-stride stream.
  The ablation shows it scores **0%** on 8 interleaved streams and on any
  non-unit stride — a real design limitation, not just a modeling artifact.
* **Cycle model is approximate.** Bank timing uses a monotonic
  `busy_until` bookkeeping and one retirement per cycle; it is not validated
  against a reference simulator (gem5 / Ramulator) or RTL.
* **Power/area constants are illustrative.** The per-bit energies and per-MB
  leakage in `memory_power_model.py` are plausible-order but **uncited**; they
  are not calibrated against CACTI/DRAMPower or a PDK.
* **Workloads are synthetic.** Traces are generated analytically, not captured
  from a real inference engine.

## 6. What *is* defensible today

The one self-contained, reproducible, statistically supported contribution is
the **controlled prefetcher ablation** (`experiments/prefetch_ablation.py`): on
single-pass workloads with a demand-only control, it quantifies where stream
prefetching helps (unit-stride streams: 0% → ~100% at 1× bandwidth), where it is
wasteful (next-line on random: 15× bandwidth for ~2% hit rate), and where the
single-stream FSM fails (interleaved streams, non-unit strides).

## 7. Roadmap to a genuinely publishable result

1. Replace uncited power/area constants with CACTI/DRAMPower (or PDK) values and
   report validation error.
2. Validate the cycle model against an established simulator on shared traces.
3. Implement a **multi-stream** prefetcher and re-run the ablation; report the
   hardware cost honestly.
4. If TOPS/perplexity/thermal claims are to be made, add the actual compute
   model / quantization evaluation / thermal solver that produces them, with
   released scripts and data.
5. Rewrite `README.md` and `scripts/reproduce_paper.sh` so every headline number
   is emitted by a script in this repository.
