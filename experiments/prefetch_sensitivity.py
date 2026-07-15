#!/usr/bin/env python3
"""
Prefetcher Design-Space Sensitivity Sweeps
==========================================

Characterises how the prefetcher's effectiveness depends on its hardware
parameters, producing the design-space evidence a hardware proposal needs:

  1. Stream-table size vs. the number of concurrent streams. This quantifies
     the design law surfaced by the ablation: a multi-stream table only pays
     off once it holds at least as many entries as there are live streams.
  2. Look-ahead depth on a single stream. This shows the minimum prefetch
     distance needed to fully hide the T2 fill latency.

Outputs results/prefetch_sensitivity.csv and results/prefetch_sensitivity.png.

Usage:
    python experiments/prefetch_sensitivity.py [--output-dir results]

Author: Janus-1 Design Team
License: MIT
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.trace_generator import (
    generate_sequential_trace,
    generate_streaming_trace,
)
from src.simulator.janus_sim import JanusSim, SimulationConfig

LINE = 128


def _hit_rate(trace, **cfg) -> float:
    sim = JanusSim(SimulationConfig(cache_line_size_bytes=LINE, **cfg))
    sim.run(trace)
    return sim.get_metrics().hit_rate


def sweep_table_size(concurrencies=(4, 8, 12), table_sizes=range(1, 17)) -> List[Dict]:
    """Sweep multi-stream table size against several stream counts."""
    rows = []
    for c in concurrencies:
        trace = generate_streaming_trace(num_streams=c, stream_length=200, stride=LINE)
        for size in table_sizes:
            hr = _hit_rate(trace, prefetch_mode="multi_stream", prefetch_streams=size)
            rows.append(
                {
                    "sweep": "table_size",
                    "concurrency": c,
                    "table_size": size,
                    "hit_rate": hr,
                }
            )
    return rows


def sweep_look_ahead(depths=(1, 2, 3, 4, 6, 8, 12, 16, 32)) -> List[Dict]:
    """Sweep look-ahead depth on a single cold unit-stride stream."""
    trace = generate_sequential_trace(num_accesses=3000, stride=LINE)
    rows = []
    for depth in depths:
        hr = _hit_rate(
            trace,
            prefetch_mode="stream",
            prefetch_look_ahead=depth,
            prefetch_issue_width=4,
        )
        rows.append({"sweep": "look_ahead", "look_ahead": depth, "hit_rate": hr})
    return rows


def save_csv(rows: List[Dict], path: Path):
    """Write sweep rows to CSV (union of all keys as header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved sweep data to {path}")


def save_figure(table_rows, la_rows, path: Path):
    """Render the two sensitivity panels."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping figure: matplotlib unavailable: {exc})")
        return

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))

    concurrencies = sorted({r["concurrency"] for r in table_rows})
    for c in concurrencies:
        xs = [r["table_size"] for r in table_rows if r["concurrency"] == c]
        ys = [r["hit_rate"] for r in table_rows if r["concurrency"] == c]
        axa.plot(xs, ys, marker="o", label=f"{c} concurrent streams")
        axa.axvline(c, color="grey", ls=":", lw=0.7)
    axa.set_xlabel("multi-stream table size (entries)")
    axa.set_ylabel("T1 hit rate (%)")
    axa.set_title("Table size must reach the concurrent-stream count")
    axa.legend()
    axa.grid(alpha=0.3)

    xs = [r["look_ahead"] for r in la_rows]
    ys = [r["hit_rate"] for r in la_rows]
    axb.plot(xs, ys, marker="s", color="tab:red")
    axb.set_xlabel("look-ahead depth (lines)")
    axb.set_ylabel("T1 hit rate (%)")
    axb.set_title("Look-ahead needed to hide T2 latency (single stream)")
    axb.set_xscale("log", base=2)
    axb.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    print("Sweeping multi-stream table size vs concurrency...")
    table_rows = sweep_table_size()
    print("Sweeping look-ahead depth...")
    la_rows = sweep_look_ahead()

    print("\nTable-size sweep (hit rate %):")
    print(
        f"  {'table':>6}"
        + "".join(
            f"{f'c={c}':>10}" for c in sorted({r["concurrency"] for r in table_rows})
        )
    )
    for size in sorted({r["table_size"] for r in table_rows}):
        cells = []
        for c in sorted({r["concurrency"] for r in table_rows}):
            hr = next(
                r["hit_rate"]
                for r in table_rows
                if r["table_size"] == size and r["concurrency"] == c
            )
            cells.append(f"{hr:>10.1f}")
        print(f"  {size:>6}" + "".join(cells))

    print("\nLook-ahead sweep (single stream):")
    for r in la_rows:
        print(f"  depth={r['look_ahead']:>3}  hit_rate={r['hit_rate']:.2f}%")

    save_csv(table_rows + la_rows, args.output_dir / "prefetch_sensitivity.csv")
    save_figure(table_rows, la_rows, args.output_dir / "prefetch_sensitivity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
