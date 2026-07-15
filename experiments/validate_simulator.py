#!/usr/bin/env python3
"""
Simulator Validation Report
===========================

Prints a PASS/FAIL table comparing the simulator against closed-form analytical
expectations for the cycle model. This is the human-readable companion to
``tests/test_analytical_validation.py`` and is invoked by the reproduction
script so every run re-certifies the simulator before results are trusted.

Usage:
    python experiments/validate_simulator.py

Author: Janus-1 Design Team
License: MIT
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.janus_sim import JanusSim, SimulationConfig

LINE = 128


def _sim(**kw):
    return JanusSim(SimulationConfig(cache_line_size_bytes=LINE, **kw))


def _checks():
    """Yield (name, measured, expected, ok) tuples."""
    # 1) Cold demand miss latency == t1 + t2.
    s = _sim(prefetch_mode="none", t1_latency_cycles=1, t2_latency_cycles=3)
    s.run([("READ", 0x40000)])
    lat = s.get_metrics().read_latencies[0]
    yield ("cold miss latency (t1+t2)", lat, 4, lat == 4)

    # 2) Warm hit latency == t1.
    s = _sim(prefetch_mode="none")
    s.run([("READ", 0x40000)] * 4)
    hits = s.get_metrics().read_latencies[1:]
    yield ("warm hit latency (t1)", set(hits), {1}, set(hits) == {1})

    # 3) Cold sequential, demand-only => 0% hit rate.
    s = _sim(prefetch_mode="none")
    s.run([("READ", 0x100000 + i * LINE) for i in range(500)])
    hr = s.get_metrics().hit_rate
    yield ("cold sequential hit rate", f"{hr:.1f}%", "0.0%", hr == 0.0)

    # 4) LRU thrash (working set = capacity+1) => 100% miss.
    cap = (1 * 1024 * 1024) // LINE
    ws = cap + 1
    s = _sim(prefetch_mode="none", t1_sram_size_mb=1)
    s.run([("READ", (i % ws) * LINE) for i in range(3 * ws)])
    hr = s.get_metrics().hit_rate
    yield ("LRU thrash hit rate", f"{hr:.1f}%", "0.0%", hr == 0.0)

    # 5) Ideal stream, stream prefetch => near-perfect steady state.
    s = _sim(prefetch_mode="stream")
    s.run([("READ", 0x100000 + i * LINE) for i in range(2000)])
    hr = s.get_metrics().hit_rate
    yield ("ideal stream hit rate (>99%)", f"{hr:.2f}%", ">99%", hr > 99.0)

    # 6) multi_stream recovers 8 interleaved streams (single FSM => 0%).
    base = [0x100000 + k * 0x100000 for k in range(8)]
    trace = []
    for step in range(250):
        for k in range(8):
            trace.append(("READ", base[k] + step * LINE))
    s_single = _sim(prefetch_mode="stream")
    s_single.run(trace)
    s_multi = _sim(prefetch_mode="multi_stream", prefetch_streams=8)
    s_multi.run(trace)
    hr_s = s_single.get_metrics().hit_rate
    hr_m = s_multi.get_metrics().hit_rate
    ok = hr_s < 5.0 and hr_m > 95.0
    yield (
        "multi_stream recovers 8 streams",
        f"single={hr_s:.1f}% multi={hr_m:.1f}%",
        "single<5% multi>95%",
        ok,
    )


def main() -> int:
    print("=" * 74)
    print("SIMULATOR VALIDATION vs CLOSED-FORM EXPECTATIONS")
    print("=" * 74)
    print(f"{'check':<34}{'measured':>22}{'expected':>12}  ok")
    print("-" * 74)
    all_ok = True
    for name, measured, expected, ok in _checks():
        all_ok &= ok
        print(
            f"{name:<34}{str(measured):>22}{str(expected):>12}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    print("=" * 74)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
