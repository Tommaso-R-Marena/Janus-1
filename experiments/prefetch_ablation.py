#!/usr/bin/env python3
"""
Prefetcher Ablation Study
=========================

A controlled, reproducible ablation of the Janus-Prefetch-1 stream prefetcher.

The central methodological point: to attribute any benefit to a prefetcher you
must compare it against a *demand-only* baseline on identical traces. This
experiment runs every workload through three prefetch policies

    * ``none``      - demand-only baseline (no prefetching)
    * ``next_line`` - always fetch the next N lines (classic next-line)
    * ``stream``    - the Janus FSM stream detector

and reports hit rate, effective read latency, and the prefetch bandwidth
overhead each policy pays. For workloads with a stochastic component we run
multiple seeds and report 95%% bootstrap confidence intervals plus a
non-parametric effect size versus the baseline.

All workloads are *single-pass* (each unique line is demanded exactly once), so
the demand-only hit rate is ~0%% by construction and any hits are attributable
to prefetching -- there is no reuse effect to confound the measurement.

Usage:
    python experiments/prefetch_ablation.py [--seeds N] [--accesses N]
                                            [--output-dir results]

Author: Janus-1 Design Team
License: MIT
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.statistical_analysis import SimulationStatistics
from src.benchmarks.trace_generator import (
    generate_mixed_trace,
    generate_random_trace,
    generate_sequential_trace,
    generate_strided_trace,
    generate_streaming_trace,
)
from src.simulator.janus_sim import JanusSim, SimulationConfig

LINE = 128  # cache line size (bytes) used throughout the study
PREFETCH_MODES = ("none", "next_line", "stream")


def _run(trace: List[Tuple[str, int]], mode: str) -> Dict[str, float]:
    """Run one trace under one prefetch mode and return summary metrics."""
    config = SimulationConfig(cache_line_size_bytes=LINE, prefetch_mode=mode)
    sim = JanusSim(config)
    sim.run(trace)
    m = sim.get_metrics()
    demand_reads = m.t1_hits + m.t1_misses
    return {
        "hit_rate": m.hit_rate,
        "mean_latency": float(np.mean(m.read_latencies)) if m.read_latencies else 0.0,
        "p99_latency": m.p99_latency,
        "demand_reads": demand_reads,
        "t1_hits": m.t1_hits,
        "prefetch_bw": m.prefetch_bandwidth,
        # Bandwidth overhead: extra T2 fetches issued per demand read.
        "bw_overhead": m.prefetch_bandwidth / demand_reads if demand_reads else 0.0,
    }


def _workloads(accesses: int) -> Dict[str, Tuple[bool, Callable[[int], list]]]:
    """Return workload builders keyed by name.

    Each value is (is_stochastic, builder(seed) -> trace).
    """
    # Sparse address span: large enough that prefetched neighbours are almost
    # never re-touched, so "random" is a clean negative control for prefetching.
    span = accesses * LINE * 512

    return {
        # Deterministic, unit-stride stream: the ideal case for prefetching.
        "sequential": (
            False,
            lambda seed: generate_sequential_trace(
                start_addr=0x100000, num_accesses=accesses, stride=LINE
            ),
        ),
        # Stride of 2 lines: the FSM detector (unit stride only) should stay
        # quiet; next-line wastes half its prefetches.
        "strided-2": (
            False,
            lambda seed: generate_strided_trace(
                start_addr=0x100000, num_accesses=accesses, stride=2 * LINE
            ),
        ),
        # Several interleaved unit-stride streams.
        "streaming-8": (
            False,
            lambda seed: generate_streaming_trace(
                start_addr=0x100000,
                num_streams=8,
                stream_length=accesses // 8,
                stride=LINE,
            ),
        ),
        # 70% sequential / 30% random (stochastic).
        "mixed-0.7": (
            True,
            lambda seed: generate_mixed_trace(
                sequential_ratio=0.7, num_accesses=accesses, seed=seed
            ),
        ),
        # No locality: prefetching cannot help and only costs bandwidth.
        "random": (
            True,
            lambda seed: generate_random_trace(
                num_accesses=accesses,
                addr_range=(0, span),
                alignment=LINE,
                seed=seed,
            ),
        ),
    }


def run_ablation(seeds: int, accesses: int) -> List[Dict]:
    """Run the full ablation grid and return one row per (workload, mode)."""
    rows: List[Dict] = []
    workloads = _workloads(accesses)

    for wl_name, (stochastic, builder) in workloads.items():
        seed_list = list(range(seeds)) if stochastic else [0]

        for mode in PREFETCH_MODES:
            hit_rates = []
            metrics_acc = {
                k: [] for k in ("mean_latency", "p99_latency", "bw_overhead")
            }
            for seed in seed_list:
                res = _run(builder(seed), mode)
                hit_rates.append(res["hit_rate"])
                for k in metrics_acc:
                    metrics_acc[k].append(res[k])

            hr = np.array(hit_rates)
            row = {
                "workload": wl_name,
                "mode": mode,
                "stochastic": stochastic,
                "n": len(seed_list),
                "hit_rate_mean": float(np.mean(hr)),
                "mean_latency": float(np.mean(metrics_acc["mean_latency"])),
                "p99_latency": float(np.mean(metrics_acc["p99_latency"])),
                "bw_overhead": float(np.mean(metrics_acc["bw_overhead"])),
                "hit_rate_ci_low": float("nan"),
                "hit_rate_ci_high": float("nan"),
            }
            if stochastic and len(hr) >= 2:
                ci = SimulationStatistics.bootstrap_ci(
                    hr, confidence_level=0.95, n_bootstrap=2000, seed=0
                )
                row["hit_rate_ci_low"] = ci.lower
                row["hit_rate_ci_high"] = ci.upper
            rows.append(row)

    return rows


def _print_table(rows: List[Dict]):
    print("\n" + "=" * 78)
    print("PREFETCHER ABLATION -- hit rate / latency / bandwidth overhead")
    print("=" * 78)
    header = (
        f"{'workload':<13}{'mode':<11}{'hit%':>8}{'95% CI':>16}"
        f"{'meanLat':>9}{'p99':>6}{'bwOvhd':>8}"
    )
    print(header)
    print("-" * 78)
    for r in rows:
        if r["n"] >= 2 and not np.isnan(r["hit_rate_ci_low"]):
            ci = f"[{r['hit_rate_ci_low']:.1f},{r['hit_rate_ci_high']:.1f}]"
        else:
            ci = "(deterministic)"
        print(
            f"{r['workload']:<13}{r['mode']:<11}"
            f"{r['hit_rate_mean']:>8.2f}{ci:>16}"
            f"{r['mean_latency']:>9.2f}{r['p99_latency']:>6.1f}"
            f"{r['bw_overhead']:>8.2f}"
        )
    print("=" * 78)


def _effect_sizes(seeds: int, accesses: int):
    """Report stream-vs-baseline effect sizes on stochastic workloads."""
    print("\nEffect size (stream vs demand-only, hit rate) on stochastic loads:")
    workloads = _workloads(accesses)
    for wl_name, (stochastic, builder) in workloads.items():
        if not stochastic:
            continue
        base = np.array([_run(builder(s), "none")["hit_rate"] for s in range(seeds)])
        strm = np.array([_run(builder(s), "stream")["hit_rate"] for s in range(seeds)])
        d = SimulationStatistics.compute_effect_size(strm, base, "cohen_d")
        test = SimulationStatistics.compare_distributions(
            strm, base, test="mann-whitney"
        )
        print(f"  {wl_name:<12} Cohen's d={d:6.2f}  {test}")


def save_csv(rows: List[Dict], path: Path):
    """Write the ablation rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved raw results to {path}")


def save_figure(rows: List[Dict], path: Path):
    """Render the hit-rate-by-workload bar chart to ``path``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping figure: matplotlib unavailable: {exc})")
        return

    workloads = sorted(
        {r["workload"] for r in rows},
        key=lambda w: [r["workload"] for r in rows].index(w),
    )
    x = np.arange(len(workloads))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, mode in enumerate(PREFETCH_MODES):
        heights, errs = [], []
        for wl in workloads:
            r = next(r for r in rows if r["workload"] == wl and r["mode"] == mode)
            heights.append(r["hit_rate_mean"])
            if not np.isnan(r["hit_rate_ci_low"]):
                errs.append(
                    (
                        r["hit_rate_mean"] - r["hit_rate_ci_low"],
                        r["hit_rate_ci_high"] - r["hit_rate_mean"],
                    )
                )
            else:
                errs.append((0.0, 0.0))
        err = np.array(errs).T
        ax.bar(x + (i - 1) * width, heights, width, label=mode, yerr=err, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=15)
    ax.set_ylabel("T1 hit rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Prefetcher ablation on single-pass workloads\n"
        "(demand-only baseline = 0% by construction)"
    )
    ax.legend(title="prefetch mode")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--accesses", type=int, default=12000)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    print(
        f"Running prefetch ablation (seeds={args.seeds}, "
        f"accesses/workload={args.accesses})..."
    )
    rows = run_ablation(args.seeds, args.accesses)
    _print_table(rows)
    _effect_sizes(args.seeds, args.accesses)
    save_csv(rows, args.output_dir / "prefetch_ablation.csv")
    save_figure(rows, args.output_dir / "prefetch_ablation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
