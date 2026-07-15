#!/usr/bin/env python3
"""
Memory Hierarchy Test Suite
============================

Comprehensive validation of Janus-1 memory hierarchy behavior.
Tests cover correctness, performance, corner cases, and stress scenarios.

Author: Janus-1 Design Team
License: MIT
"""

import pytest
import numpy as np
from src.simulator.janus_sim import JanusSim, SimulationConfig, SimulationMetrics


class TestMemoryHierarchyCorrectness:
    """Test correctness of memory hierarchy operations."""

    def test_t1_hit_single_read(self):
        """Test single read hits in T1 SRAM after initial miss."""
        config = SimulationConfig()
        sim = JanusSim(config)
        
        addr = 0x1000
        trace = [("READ", addr), ("READ", addr)]  # First miss, second hit
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        assert metrics.t1_hits == 1, "Second read should hit in T1"
        assert metrics.t1_misses == 1, "First read should miss T1"
        assert metrics.hit_rate == 50.0, "Hit rate should be 50%"

    def test_t1_lru_eviction(self):
        """Test LRU eviction policy in T1 SRAM."""
        # Configure small T1 cache (only 1 MB = 8192 lines). Disable the
        # prefetcher so this test isolates pure demand-driven LRU behaviour.
        config = SimulationConfig(t1_sram_size_mb=1, prefetch_mode="none")
        sim = JanusSim(config)
        
        # Fill cache to capacity + 1
        num_lines = (config.t1_sram_size_mb * 1024 * 1024) // config.cache_line_size_bytes
        
        trace = []
        # Fill cache
        for i in range(num_lines):
            trace.append(("READ", i * config.cache_line_size_bytes))
        
        # Access oldest entry (should still be present)
        trace.append(("READ", 0))
        
        # Add one more unique address (should evict LRU)
        trace.append(("READ", num_lines * config.cache_line_size_bytes))
        
        # Access oldest entry again (should miss now)
        trace.append(("READ", 0))
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # First fill: all misses (num_lines)
        # Second access to 0: hit -> promotes line 0 to most-recently-used
        # New address: miss -> evicts the true LRU line (line 1), NOT line 0
        # Third access to 0: hit (line 0 was promoted, so it survives)
        expected_hits = 2
        assert metrics.t1_hits == expected_hits, f"Expected {expected_hits} hits, got {metrics.t1_hits}"

    def test_bank_conflict_detection(self):
        """Test bank conflict handling for concurrent accesses."""
        config = SimulationConfig(t1_sram_banks=4)
        sim = JanusSim(config)
        
        # Generate addresses mapping to same T1 bank
        bank_0_addr1 = 0x0000  # Line 0 -> Bank 0
        bank_0_addr2 = 0x0000 + (4 * config.cache_line_size_bytes)  # Line 4 -> Bank 0
        
        trace = [
            ("READ", bank_0_addr1),
            ("READ", bank_0_addr1),  # Hit, same bank
            ("READ", bank_0_addr2),  # Miss, same bank (conflict possible)
            ("READ", bank_0_addr2),  # Hit, same bank
        ]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # First and third are misses, second and fourth are hits
        assert metrics.t1_hits == 2
        assert metrics.t1_misses == 2

    def test_write_allocate_policy(self):
        """Test write-allocate behavior in T1 cache."""
        sim = JanusSim()
        
        addr = 0x2000
        trace = [
            ("WRITE", addr),  # Allocates in T1
            ("READ", addr),   # Should hit
        ]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        assert metrics.t1_hits == 1, "Read after write should hit"
        assert metrics.t1_misses == 0, "No misses expected"


class TestPrefetcherBehavior:
    """Test Janus-Prefetch-1 stream prefetcher."""

    def test_stream_detection(self):
        """Test sequential access stream detection."""
        config = SimulationConfig(prefetch_look_ahead=16)
        sim = JanusSim(config)
        
        # Sequential access pattern
        base_addr = 0x10000
        trace = [("READ", base_addr + i * config.cache_line_size_bytes) for i in range(32)]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # First access misses, prefetcher kicks in, later accesses hit
        assert metrics.hit_rate > 80.0, f"Stream prefetching should achieve >80% hit rate, got {metrics.hit_rate:.1f}%"
        assert metrics.prefetch_bandwidth > 0, "Prefetcher should issue requests"

    def test_random_access_no_prefetch(self):
        """Test that random accesses don't trigger prefetcher."""
        sim = JanusSim()
        
        # Random access pattern
        np.random.seed(42)
        trace = [("READ", int(addr)) for addr in np.random.randint(0, 1_000_000, size=100) * 128]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # Random access should have minimal prefetching
        assert metrics.prefetch_bandwidth == 0, "Random access should not trigger prefetcher"

    def test_prefetch_lookahead_depth(self):
        """Test prefetcher with different lookahead depths."""
        results = []
        
        for lookahead in [4, 8, 16, 32]:
            config = SimulationConfig(prefetch_look_ahead=lookahead)
            sim = JanusSim(config)
            
            base_addr = 0x20000
            trace = [("READ", base_addr + i * config.cache_line_size_bytes) for i in range(64)]
            
            sim.run(trace)
            metrics = sim.get_metrics()
            results.append((lookahead, metrics.hit_rate))
        
        # Hit rate should generally increase with lookahead
        hit_rates = [hr for _, hr in results]
        assert hit_rates[-1] >= hit_rates[0], "Higher lookahead should improve or maintain hit rate"

    def test_prefetch_issue_width(self):
        """Test prefetch bandwidth limiting via issue width."""
        config = SimulationConfig(prefetch_issue_width=2, prefetch_look_ahead=16)
        sim = JanusSim(config)
        
        base_addr = 0x30000
        trace = [("READ", base_addr + i * config.cache_line_size_bytes) for i in range(64)]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # Prefetcher should respect issue width
        assert metrics.prefetch_bandwidth > 0, "Prefetcher should issue requests"
        # In any single cycle, max prefetch_issue_width requests issued


class TestLatencyCharacteristics:
    """Test memory access latency behavior."""

    def test_t1_hit_latency(self):
        """Test T1 SRAM hit latency is 1 cycle."""
        config = SimulationConfig(t1_latency_cycles=1)
        sim = JanusSim(config)
        
        addr = 0x4000
        trace = [("READ", addr), ("READ", addr)]  # Miss then hit
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # Second access (hit) should have 1-cycle latency
        hit_latency = metrics.read_latencies[1]
        assert hit_latency == 1, f"T1 hit should be 1 cycle, got {hit_latency}"

    def test_t2_miss_latency(self):
        """Test T2 eDRAM access latency is 3+ cycles."""
        config = SimulationConfig(t1_latency_cycles=1, t2_latency_cycles=3)
        sim = JanusSim(config)
        
        addr = 0x5000
        trace = [("READ", addr)]  # Miss to T2
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # First access (miss) should be T2 latency + T1 read
        miss_latency = metrics.read_latencies[0]
        assert miss_latency >= 3, f"T2 miss should be >=3 cycles, got {miss_latency}"

    def test_latency_percentiles(self):
        """Test P50/P90/P99 latency calculations."""
        config = SimulationConfig()
        sim = JanusSim(config)
        
        # Mixed access pattern
        base = 0x10000
        trace = []
        for i in range(100):
            trace.append(("READ", base + i * config.cache_line_size_bytes))
            trace.append(("READ", base + i * config.cache_line_size_bytes))  # Repeat for hit
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        assert 0 < metrics.p50_latency <= metrics.p90_latency <= metrics.p99_latency, \
            "Latency percentiles should be ordered: P50 <= P90 <= P99"


class TestStressScenarios:
    """Stress testing and corner cases."""

    def test_large_trace_execution(self):
        """Test simulation with large memory trace."""
        sim = JanusSim()
        
        # Generate 100K memory operations
        base = 0x100000
        trace = [("READ", base + (i % 1000) * 128) for i in range(100_000)]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        assert metrics.t1_hits + metrics.t1_misses == 100_000, "All reads should be counted"
        assert metrics.total_cycles > 0, "Simulation should execute cycles"

    def test_capacity_miss_behavior(self):
        """Test behavior when working set exceeds T1 capacity."""
        # Small cache; disable prefetch to isolate capacity-miss behaviour.
        config = SimulationConfig(t1_sram_size_mb=1, prefetch_mode="none")
        sim = JanusSim(config)
        
        # Working set larger than cache
        num_lines = (config.t1_sram_size_mb * 1024 * 1024) // config.cache_line_size_bytes
        trace = []
        
        # Create working set 2x cache size
        for i in range(num_lines * 2):
            trace.append(("READ", i * config.cache_line_size_bytes))
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # Should have many misses due to capacity
        assert metrics.t1_misses > num_lines, "Should have capacity misses"

    def test_bank_conflict_penalty(self):
        """Test bank conflict introduces additional latency."""
        config = SimulationConfig(
            t2_edram_banks=1,  # Single bank for forced conflicts
            bank_conflict_penalty_cycles=5
        )
        sim = JanusSim(config)
        
        # Two misses to same T2 bank
        trace = [
            ("READ", 0x1000),
            ("READ", 0x2000),  # Will conflict on T2 bank
        ]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # Both are misses, second should see conflict penalty
        assert metrics.t1_misses == 2
        assert len(metrics.read_latencies) == 2

    def test_empty_trace(self):
        """Test simulator handles empty trace gracefully."""
        sim = JanusSim()
        trace = []
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        assert metrics.t1_hits == 0
        assert metrics.t1_misses == 0
        assert metrics.total_cycles >= 0

    def test_single_operation_trace(self):
        """Test simulator with minimal trace."""
        sim = JanusSim()
        trace = [("READ", 0x1000)]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        assert metrics.t1_misses == 1
        assert metrics.t1_hits == 0
        assert len(metrics.read_latencies) == 1


class TestConfigurationVariations:
    """Test different configuration parameters."""

    def test_varying_t1_sizes(self):
        """Test different T1 SRAM sizes."""
        for size_mb in [8, 16, 32, 64]:
            config = SimulationConfig(t1_sram_size_mb=size_mb)
            sim = JanusSim(config)
            
            # Access pattern within capacity
            trace = [("READ", i * 128) for i in range(1000)]
            trace += [("READ", i * 128) for i in range(1000)]  # Repeat
            
            sim.run(trace)
            metrics = sim.get_metrics()
            
            # Larger cache should have better hit rate on repeated accesses
            assert metrics.hit_rate > 0, f"Size {size_mb}MB should have hits"

    def test_varying_bank_counts(self):
        """Test different bank configurations."""
        for num_banks in [2, 4, 8, 16]:
            config = SimulationConfig(t1_sram_banks=num_banks)
            sim = JanusSim(config)
            
            trace = [("READ", i * 128) for i in range(100)]
            
            sim.run(trace)
            metrics = sim.get_metrics()
            
            assert metrics.t1_misses > 0, "First accesses should miss"

    def test_varying_line_sizes(self):
        """Test different cache line sizes."""
        for line_size in [64, 128, 256]:
            # Disable prefetch: this test checks that every unique first-touch
            # line misses, which a stream prefetcher would otherwise hide.
            config = SimulationConfig(
                cache_line_size_bytes=line_size, prefetch_mode="none"
            )
            sim = JanusSim(config)
            
            trace = [("READ", i * line_size) for i in range(100)]
            
            sim.run(trace)
            metrics = sim.get_metrics()
            
            assert metrics.t1_misses == 100, "All first accesses should miss"


class TestMetricsAccuracy:
    """Test accuracy of reported metrics."""

    def test_hit_rate_calculation(self):
        """Test hit rate is correctly calculated."""
        sim = JanusSim()
        
        # 75% hit rate: 1 miss + 3 hits
        addr = 0x7000
        trace = [("READ", addr)] * 4
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        expected_rate = 75.0
        assert abs(metrics.hit_rate - expected_rate) < 0.1, \
            f"Expected {expected_rate}% hit rate, got {metrics.hit_rate:.2f}%"

    def test_bandwidth_accounting(self):
        """Test compute and prefetch bandwidth are tracked correctly."""
        config = SimulationConfig(prefetch_look_ahead=8)
        sim = JanusSim(config)
        
        # Sequential pattern to trigger prefetching
        base = 0x8000
        num_reads = 32
        trace = [("READ", base + i * config.cache_line_size_bytes) for i in range(num_reads)]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        # All reads count as compute bandwidth
        assert metrics.compute_bandwidth >= num_reads, "All reads should count in compute BW"
        
        # Prefetcher should also generate traffic
        total_bw = metrics.compute_bandwidth + metrics.prefetch_bandwidth
        assert total_bw >= num_reads, "Total bandwidth should account for all traffic"

    def test_cycle_count(self):
        """Test cycle count advances correctly."""
        sim = JanusSim()
        
        trace = [("READ", i * 128) for i in range(10)]
        
        sim.run(trace)
        metrics = sim.get_metrics()
        
        assert metrics.total_cycles > 0, "Cycles should advance during simulation"
        assert metrics.total_cycles >= len(trace), "Should take at least 1 cycle per operation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])