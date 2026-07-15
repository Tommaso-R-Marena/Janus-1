#!/usr/bin/env python3
"""
Analytical Validation of the Cycle Model
========================================

Cross-checks the simulator against closed-form expectations. These are the
correctness guarantees the earlier version silently violated (it reported
phantom hits and impossible latencies), so they are asserted here to prevent
regressions and to give the timing model an auditable specification.

Author: Janus-1 Design Team
License: MIT
"""

import numpy as np
import pytest

from src.simulator.janus_sim import JanusSim, SimulationConfig

LINE = 128


def _sim(**kw):
    return JanusSim(SimulationConfig(cache_line_size_bytes=LINE, **kw))


def test_cold_demand_miss_latency_equals_t1_plus_t2():
    """A single cold demand miss must cost exactly t1_latency + t2_latency."""
    for t1, t2 in [(1, 3), (1, 5), (2, 7)]:
        sim = _sim(prefetch_mode="none", t1_latency_cycles=t1, t2_latency_cycles=t2)
        sim.run([("READ", 0x40000)])
        m = sim.get_metrics()
        assert m.t1_hits == 0 and m.t1_misses == 1
        assert m.read_latencies == [t1 + t2]


def test_warm_hit_latency_equals_t1():
    """Every read of an already-resident line costs exactly t1_latency."""
    sim = _sim(prefetch_mode="none", t1_latency_cycles=1)
    # Warm the line, then re-read it many times.
    trace = [("READ", 0x40000)] * 10
    sim.run(trace)
    m = sim.get_metrics()
    assert m.t1_misses == 1 and m.t1_hits == 9
    # latencies: first is the miss (t1+t2=4), rest are hits (1 cycle).
    assert m.read_latencies[0] == 4
    assert all(lat == 1 for lat in m.read_latencies[1:])


def test_cold_sequential_demand_only_is_all_misses():
    """N distinct cold lines with no prefetch => N misses, 0 hits, mean=t1+t2."""
    n = 500
    sim = _sim(prefetch_mode="none")
    sim.run([("READ", 0x100000 + i * LINE) for i in range(n)])
    m = sim.get_metrics()
    assert m.t1_hits == 0
    assert m.t1_misses == n
    assert m.hit_rate == 0.0
    assert float(np.mean(m.read_latencies)) == pytest.approx(4.0)


def test_lru_thrashing_working_set_just_over_capacity():
    """A cyclic working set one line larger than the cache => 100% misses."""
    # 1 MB cache with 128 B lines = 8192 lines.
    cap_lines = (1 * 1024 * 1024) // LINE
    ws = cap_lines + 1  # one past capacity forces eviction of the next-needed line
    sim = _sim(prefetch_mode="none", t1_sram_size_mb=1)
    trace = [("READ", (i % ws) * LINE) for i in range(3 * ws)]
    sim.run(trace)
    m = sim.get_metrics()
    # After the first sweep every re-reference misses (classic LRU thrash).
    assert m.t1_hits == 0
    assert m.t1_misses == 3 * ws


def test_conservation_reads_accounted_exactly_once():
    """hits + misses must equal the number of READ operations in the trace."""
    trace = [("READ", i * LINE) for i in range(200)]
    trace += [("WRITE", 999 * LINE)]  # writes are not counted as reads
    trace += [("READ", i * LINE) for i in range(200)]  # a second (warm) sweep
    n_reads = sum(1 for op, _ in trace if op == "READ")
    sim = _sim(prefetch_mode="stream")
    sim.run(trace)
    m = sim.get_metrics()
    assert m.t1_hits + m.t1_misses == n_reads
    assert len(m.read_latencies) == n_reads


def test_stream_detector_confirms_after_two_consecutive_lines():
    """The FSM must not prefetch on the first access, only after 2 in a row."""
    # Two consecutive lines then a big gap: exactly one confirmed step.
    sim = _sim(prefetch_mode="stream")
    sim.run([("READ", 0x200000), ("READ", 0x200000 + LINE)])
    # Prefetches were issued only after the second (confirming) access.
    assert sim.get_metrics().prefetch_bandwidth > 0

    sim2 = _sim(prefetch_mode="stream")
    sim2.run([("READ", 0x200000), ("READ", 0x900000)])  # non-consecutive
    assert sim2.get_metrics().prefetch_bandwidth == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
