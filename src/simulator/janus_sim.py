"""Janus-Sim: Cycle-Accurate Memory Hierarchy Simulator

This module implements the core simulation engine for the Janus-1 memory hierarchy.
It models the two-tier SRAM+eDRAM system with the Janus-Prefetch-1 engine.

Author: The Janus-1 Design Team
License: MIT
"""

import collections
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration parameters for Janus-Sim."""

    t1_sram_size_mb: int = 32
    t1_sram_banks: int = 4
    t2_edram_banks: int = 14
    cache_line_size_bytes: int = 128
    t1_latency_cycles: int = 1
    t2_latency_cycles: int = 3
    bank_conflict_penalty_cycles: int = 5
    prefetch_issue_width: int = 4
    prefetch_look_ahead: int = 16
    # Prefetch policy: "none" (demand-only baseline), "next_line" (always
    # prefetch the next lines), or "stream" (FSM stream detector). Exposing
    # this as a first-class knob is what makes controlled ablations possible.
    prefetch_mode: str = "stream"

    def __post_init__(self):
        valid_modes = {"none", "next_line", "stream"}
        if self.prefetch_mode not in valid_modes:
            raise ValueError(
                f"prefetch_mode must be one of {sorted(valid_modes)}, "
                f"got {self.prefetch_mode!r}"
            )


@dataclass
class SimulationMetrics:
    """Performance metrics from simulation run."""

    t1_hits: int
    t1_misses: int
    total_cycles: int
    read_latencies: List[int]
    prefetch_bandwidth: int
    compute_bandwidth: int

    @property
    def hit_rate(self) -> float:
        """Calculate T1 cache hit rate."""
        total = self.t1_hits + self.t1_misses
        return (self.t1_hits / total * 100) if total > 0 else 0.0

    @property
    def p50_latency(self) -> float:
        """Calculate P50 (median) latency."""
        return np.percentile(self.read_latencies, 50) if self.read_latencies else 0.0

    @property
    def p90_latency(self) -> float:
        """Calculate P90 latency."""
        return np.percentile(self.read_latencies, 90) if self.read_latencies else 0.0

    @property
    def p99_latency(self) -> float:
        """Calculate P99 latency."""
        return np.percentile(self.read_latencies, 99) if self.read_latencies else 0.0


class JanusSim:
    """Cycle-accurate simulator for Janus-1 memory hierarchy.

    This simulator models:
    - 32 MB SRAM Tier-1 active cache (4 banks)
    - 224 MB eDRAM Tier-2 main store (14 banks)
    - Janus-Prefetch-1 FSM-based stream prefetcher
    - Bank conflicts and queuing delays

    Example:
        >>> sim = JanusSim()
        >>> trace = [("READ", 0x1000), ("READ", 0x1080), ...]
        >>> sim.run(trace)
        >>> sim.report()
        T1 Hit Rate: 99.99% (65520 hits / 65536 reads)
        Latencies (cycles): P50=1.0, P90=1.0, P99=1.0
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize simulator with configuration.

        Args:
            config: Simulation configuration. Uses defaults if None.
        """
        self.config = config or SimulationConfig()
        self._init_memory_hierarchy()
        self._init_prefetcher()
        self._init_metrics()

    def _init_memory_hierarchy(self):
        """Initialize memory hierarchy state."""
        # T1 SRAM cache (LRU replacement)
        self.t1_sram_size_bytes = self.config.t1_sram_size_mb * 1024 * 1024
        self.t1_max_lines = self.t1_sram_size_bytes // self.config.cache_line_size_bytes
        self.t1_cache = collections.OrderedDict()

        # Bank busy tracking
        self.t1_bank_busy_until = [0] * self.config.t1_sram_banks
        self.t2_bank_busy_until = [0] * self.config.t2_edram_banks

        # Event queue for T2 responses
        self.pending_events = []
        self.pending_cpu_read = None
        self.pending_cpu_read_start_cycle = None

        # Prefetch tracking
        self.inflight_prefetches = set()

    def _init_prefetcher(self):
        """Initialize prefetcher FSM state."""
        self.prefetch_stream_addr = -1
        self.prefetch_stream_detected = False

    def _init_metrics(self):
        """Initialize performance metrics."""
        self.cycle = 0
        self.t1_hits = 0
        self.t1_misses = 0
        self.read_latencies = []
        self.prefetch_bw_count = 0
        self.compute_bw_count = 0

    def get_t1_bank(self, addr: int) -> int:
        """Calculate T1 SRAM bank ID for address."""
        line_num = addr // self.config.cache_line_size_bytes
        return line_num % self.config.t1_sram_banks

    def get_t2_bank(self, addr: int) -> int:
        """Calculate T2 eDRAM bank ID for address."""
        line_num = addr // self.config.cache_line_size_bytes
        return line_num % self.config.t2_edram_banks

    def run(self, trace: List[Tuple[str, int]]):
        """Run simulation on memory access trace.

        Models an in-order core with a single outstanding demand miss: a demand
        read that misses in T1 stalls the core until the line is filled from T2,
        while the prefetch engine issues requests opportunistically. Each trace
        entry is accounted for exactly once (a demand miss is counted as a miss
        when issued and is *not* re-counted as a hit when the fill lands).

        Args:
            trace: List of (operation, address) tuples.
                  Operations: "READ" or "WRITE"
        """
        trace_iterator = iter(trace)
        current_trace_entry = next(trace_iterator, None)

        while (
            current_trace_entry is not None
            or self.pending_cpu_read is not None
            or self.pending_events
        ):
            self._process_pending_events()

            # Complete an outstanding demand miss once its line has arrived,
            # then advance past the entry that caused the miss.
            if self.pending_cpu_read is not None:
                if self._complete_pending_read():
                    current_trace_entry = next(trace_iterator, None)

            # Only issue a new demand access when the core is not stalled on a
            # miss. A miss sets ``pending_cpu_read`` and holds the entry until
            # the fill completes above (no re-processing => no double counting).
            if self.pending_cpu_read is None and current_trace_entry is not None:
                if self._handle_entry(current_trace_entry):
                    current_trace_entry = next(trace_iterator, None)

            self._maybe_issue_prefetches()

            self.cycle += 1

    def _process_pending_events(self):
        """Process T2 responses arriving this cycle (fills into T1)."""
        arrived = [ev for ev in self.pending_events if self.cycle >= ev[1]]
        if not arrived:
            return

        self.pending_events = [ev for ev in self.pending_events if self.cycle < ev[1]]

        for addr, _arrival_time, _is_prefetch in arrived:
            self.inflight_prefetches.discard(addr)
            if addr in self.t1_cache:
                continue
            if len(self.t1_cache) >= self.t1_max_lines:
                self.t1_cache.popitem(last=False)
            self.t1_cache[addr] = True

    def _complete_pending_read(self) -> bool:
        """Complete the outstanding demand read if its line is now in T1.

        Returns:
            True if the read completed this cycle, False otherwise.
        """
        addr = self.pending_cpu_read
        if addr not in self.t1_cache:
            return False

        bank_id = self.get_t1_bank(addr)
        service_time = (
            max(self.cycle, self.t1_bank_busy_until[bank_id])
            + self.config.t1_latency_cycles
        )
        self.read_latencies.append(service_time - self.pending_cpu_read_start_cycle)
        self.t1_bank_busy_until[bank_id] = service_time
        self.t1_cache.move_to_end(addr)
        self.pending_cpu_read = None
        self.pending_cpu_read_start_cycle = None
        return True

    def _handle_entry(self, entry: Tuple[str, int]) -> bool:
        """Handle a single trace entry.

        Returns:
            True if the trace should advance to the next entry (hit or write),
            False if the core is now stalled on a demand miss.
        """
        op, addr = entry

        if op == "READ":
            self.compute_bw_count += 1
            self._update_prefetch_state(addr)

            if addr in self.t1_cache:
                self.t1_hits += 1
                bank_id = self.get_t1_bank(addr)
                service_time = (
                    max(self.cycle, self.t1_bank_busy_until[bank_id])
                    + self.config.t1_latency_cycles
                )
                self.read_latencies.append(service_time - self.cycle)
                self.t1_bank_busy_until[bank_id] = service_time
                self.t1_cache.move_to_end(addr)
                return True

            # T1 miss - stall the core and fetch from T2.
            self.t1_misses += 1
            self.pending_cpu_read = addr
            self.pending_cpu_read_start_cycle = self.cycle
            self.issue_to_t2(addr, is_prefetch=False)
            return False

        if op == "WRITE":
            if addr not in self.t1_cache:
                if len(self.t1_cache) >= self.t1_max_lines:
                    self.t1_cache.popitem(last=False)
                self.t1_cache[addr] = True
            return True

        raise ValueError(f"Unknown operation: {op}")

    def _update_prefetch_state(self, addr: int):
        """Update the stream detector on a demand read address."""
        prev = self.prefetch_stream_addr
        self.prefetch_stream_detected = (
            prev >= 0 and prev + self.config.cache_line_size_bytes == addr
        )
        self.prefetch_stream_addr = addr

    def _maybe_issue_prefetches(self):
        """Issue prefetches for the active policy, if any."""
        mode = self.config.prefetch_mode
        if mode == "none" or self.prefetch_stream_addr < 0:
            return
        # "next_line" prefetches unconditionally; "stream" waits until the FSM
        # has observed a run of consecutive cache lines.
        if mode == "next_line" or self.prefetch_stream_detected:
            self._issue_prefetches()

    def _issue_prefetches(self):
        """Issue up to ``prefetch_issue_width`` lines ahead of the stream."""
        issued = 0
        for i in range(1, self.config.prefetch_look_ahead + 1):
            if issued >= self.config.prefetch_issue_width:
                break

            pf_addr = self.prefetch_stream_addr + i * self.config.cache_line_size_bytes

            if pf_addr not in self.t1_cache and pf_addr not in self.inflight_prefetches:
                self.issue_to_t2(pf_addr, is_prefetch=True)
                self.inflight_prefetches.add(pf_addr)
                issued += 1

    def issue_to_t2(self, addr: int, is_prefetch: bool):
        """Issue request to T2 eDRAM.

        Args:
            addr: Address to fetch
            is_prefetch: True if this is a prefetch, False if demand
        """
        if is_prefetch:
            self.prefetch_bw_count += 1
        else:
            self.compute_bw_count += 1

        bank_id = self.get_t2_bank(addr)

        # Calculate arrival time accounting for bank conflicts
        base_arrival = (
            max(self.cycle, self.t2_bank_busy_until[bank_id])
            + self.config.t2_latency_cycles
        )

        # Add conflict penalty if bank is busy
        if base_arrival > self.cycle + self.config.t2_latency_cycles:
            base_arrival += self.config.bank_conflict_penalty_cycles

        self.t2_bank_busy_until[bank_id] = base_arrival
        self.pending_events.append((addr, base_arrival, is_prefetch))

    def get_metrics(self) -> SimulationMetrics:
        """Return simulation metrics."""
        return SimulationMetrics(
            t1_hits=self.t1_hits,
            t1_misses=self.t1_misses,
            total_cycles=self.cycle,
            read_latencies=self.read_latencies,
            prefetch_bandwidth=self.prefetch_bw_count,
            compute_bandwidth=self.compute_bw_count,
        )

    def report(self):
        """Print formatted simulation results."""
        metrics = self.get_metrics()
        total_reads = metrics.t1_hits + metrics.t1_misses

        print(f"\n{'='*60}")
        print("Janus-1 Memory Hierarchy Simulation Results")
        print(f"{'='*60}")
        print(f"\nCache Performance:")
        print(
            f"  T1 Hit Rate: {metrics.hit_rate:.2f}% "
            f"({metrics.t1_hits} hits / {total_reads} reads)"
        )
        print(f"\nLatency Distribution (cycles):")
        print(f"  P50: {metrics.p50_latency:.1f}")
        print(f"  P90: {metrics.p90_latency:.1f}")
        print(f"  P99: {metrics.p99_latency:.1f}")
        print(f"\nBandwidth Utilization:")
        print(f"  Compute BW: {metrics.compute_bandwidth} accesses")
        print(f"  Prefetch BW: {metrics.prefetch_bandwidth} accesses")
        print(f"  Total Cycles: {metrics.total_cycles}")
        print(f"\n{'='*60}\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Console entry point: run a demo simulation and print a report.

    Usage:
        janus-sim [context_length] [hidden_dim] [prefetch_mode]
    """
    import argparse

    from src.benchmarks.trace_generator import generate_llm_trace

    parser = argparse.ArgumentParser(
        description="Run a Janus-1 memory-hierarchy simulation."
    )
    parser.add_argument("context_length", nargs="?", type=int, default=256)
    parser.add_argument("hidden_dim", nargs="?", type=int, default=1024)
    parser.add_argument(
        "prefetch_mode",
        nargs="?",
        choices=["none", "next_line", "stream"],
        default="stream",
    )
    args = parser.parse_args(argv)

    print("Generating LLM inference trace...")
    trace = generate_llm_trace(
        context_length=args.context_length, hidden_dim=args.hidden_dim
    )

    print(
        f"Running simulation on {len(trace):,} memory operations "
        f"(prefetch_mode={args.prefetch_mode})..."
    )
    sim = JanusSim(SimulationConfig(prefetch_mode=args.prefetch_mode))
    sim.run(trace)
    sim.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
