#!/usr/bin/env python3
"""
Basic Memory Hierarchy Simulation
==================================

Runs a cycle-approximate simulation of the Janus-1 two-tier memory hierarchy
on a synthetic LLM KV-cache trace, and contrasts a demand-only baseline with
the stream prefetcher so the prefetcher's actual contribution is visible.

Author: Janus-1 Research Team
License: MIT
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.janus_sim import JanusSim, SimulationConfig
from src.benchmarks.trace_generator import generate_llm_trace


def main():
    """Run a basic memory hierarchy simulation with a prefetch ablation."""

    print("=" * 60)
    print("Janus-1 Basic Memory Hierarchy Simulation")
    print("=" * 60)
    print()

    context_length = 256
    hidden_dim = 1024
    num_layers = 32

    print("Configuration:")
    print(f"  Context Length: {context_length} tokens")
    print(f"  Hidden Dimension: {hidden_dim}")
    print(f"  Layers: {num_layers}")
    print()

    print("Generating LLM inference trace (INT4 KV cache)...")
    trace = generate_llm_trace(
        context_length=context_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        quantization="int4",
    )
    print(f"Generated {len(trace):,} memory accesses")
    print()

    for mode in ("none", "stream"):
        sim = JanusSim(SimulationConfig(prefetch_mode=mode))
        sim.run(trace)
        m = sim.get_metrics()
        print(
            f"prefetch_mode={mode:<6}  "
            f"hit_rate={m.hit_rate:6.2f}%  "
            f"P99_latency={m.p99_latency:4.1f} cyc  "
            f"prefetch_bw={m.prefetch_bandwidth:,}"
        )

    print()
    print("Note: on this KV-cache pattern the per-token stride spans many cache")
    print("lines, so the stream prefetcher rarely triggers. See")
    print("experiments/prefetch_ablation.py for workloads where it helps.")


if __name__ == "__main__":
    main()
