#!/usr/bin/env python3
"""
Prefetcher Ablation Invariants
==============================

Locks in the corrected simulator semantics and the qualitative findings of the
prefetch ablation study (see experiments/prefetch_ablation.py).

Author: Janus-1 Design Team
License: MIT
"""

import pytest

from src.simulator.janus_sim import JanusSim, SimulationConfig
from src.benchmarks.trace_generator import (
    generate_sequential_trace,
    generate_strided_trace,
    generate_streaming_trace,
    generate_random_trace,
)

LINE = 128


def _hit_rate(trace, mode):
    sim = JanusSim(SimulationConfig(cache_line_size_bytes=LINE, prefetch_mode=mode))
    sim.run(trace)
    return sim.get_metrics().hit_rate


def test_invalid_prefetch_mode_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(prefetch_mode="magic")


def test_demand_only_single_pass_has_no_hits():
    """A cold single-pass stream must be 0% hits without prefetching."""
    trace = generate_sequential_trace(num_accesses=2000, stride=LINE)
    assert _hit_rate(trace, "none") == 0.0


def test_prefetch_converts_sequential_misses_to_hits():
    """Both prefetchers should turn a unit-stride stream into ~all hits."""
    trace = generate_sequential_trace(num_accesses=2000, stride=LINE)
    assert _hit_rate(trace, "next_line") > 99.0
    assert _hit_rate(trace, "stream") > 99.0


def test_stream_fsm_ignores_non_unit_stride():
    """The unit-stride FSM must not fire on a 2-line stride (no false hits)."""
    trace = generate_strided_trace(num_accesses=2000, stride=2 * LINE)
    assert _hit_rate(trace, "stream") == 0.0
    # next-line still covers it (at a bandwidth cost measured in the study).
    assert _hit_rate(trace, "next_line") > 99.0


def test_single_stream_fsm_fails_on_interleaved_streams():
    """Documented limitation: the single-stream FSM cannot track 8 streams."""
    trace = generate_streaming_trace(num_streams=8, stream_length=250, stride=LINE)
    assert _hit_rate(trace, "stream") < 5.0
    assert _hit_rate(trace, "next_line") > 95.0


def test_prefetch_does_not_help_sparse_random():
    """On sparse random access the stream prefetcher matches the baseline."""
    trace = generate_random_trace(
        num_accesses=2000,
        addr_range=(0, 2000 * LINE * 512),
        alignment=LINE,
        seed=0,
    )
    assert _hit_rate(trace, "stream") == pytest.approx(_hit_rate(trace, "none"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
