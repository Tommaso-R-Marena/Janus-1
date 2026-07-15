#!/usr/bin/env python3
"""
Trace Generator Test Suite
===========================

Validation of synthetic workload trace generation.
Ensures realistic memory access patterns for LLM inference.

Author: Janus-1 Design Team
License: MIT
"""

import pytest
import numpy as np
from src.benchmarks.trace_generator import (
    generate_llm_trace,
    generate_sequential_trace,
    generate_random_trace,
    generate_strided_trace,
)


class TestLLMTraceGeneration:
    """Test LLM inference trace generation."""

    def test_basic_llm_trace_generation(self):
        """Test basic LLM trace can be generated."""
        trace = generate_llm_trace(context_length=512, hidden_dim=2048)
        
        assert len(trace) > 0, "Trace should not be empty"
        assert all(isinstance(op, tuple) and len(op) == 2 for op in trace), \
            "Each operation should be (op_type, address) tuple"

    def test_llm_trace_parameters(self):
        """Test LLM trace respects input parameters."""
        context_len = 1024
        hidden_dim = 4096
        
        trace = generate_llm_trace(context_length=context_len, hidden_dim=hidden_dim)
        
        # Trace length should scale with context and hidden dim
        assert len(trace) > context_len, "Trace should have multiple ops per token"

    @pytest.mark.xfail(
        reason="The shipped KV layout places each token's KV block "
        "hidden_dim*2 bytes apart, so attention reads stride across many cache "
        "lines and have ~0% line-level spatial locality. This is precisely why "
        "the unit-stride stream prefetcher is inert on the LLM trace; see "
        "LIMITATIONS.md sections 4-5. Kept as a documented characterization "
        "rather than reshaping the workload to force a pass.",
        strict=False,
    )
    def test_llm_access_pattern_locality(self):
        """Test LLM trace exhibits spatial locality."""
        trace = generate_llm_trace(context_length=256, hidden_dim=2048)
        
        addresses = [addr for op, addr in trace]
        
        # Calculate address deltas
        deltas = [abs(addresses[i+1] - addresses[i]) for i in range(len(addresses)-1)]
        
        # Many deltas should be small (spatial locality)
        small_deltas = sum(1 for d in deltas if d <= 1024)
        locality_ratio = small_deltas / len(deltas)
        
        assert locality_ratio > 0.3, f"Expected >30% spatial locality, got {locality_ratio*100:.1f}%"

    def test_llm_read_write_ratio(self):
        """Test LLM trace has realistic read/write ratio."""
        trace = generate_llm_trace(context_length=512, hidden_dim=2048)
        
        reads = sum(1 for op, _ in trace if op == "READ")
        writes = sum(1 for op, _ in trace if op == "WRITE")
        
        # LLM inference is read-heavy
        assert reads > writes, "Should have more reads than writes for inference"
        
        read_ratio = reads / len(trace)
        assert read_ratio > 0.7, f"Expected >70% reads, got {read_ratio*100:.1f}%"


class TestSequentialTraceGeneration:
    """Test sequential access pattern generation."""

    def test_sequential_trace_basic(self):
        """Test basic sequential trace generation."""
        trace = generate_sequential_trace(start_addr=0x1000, num_accesses=100)
        
        assert len(trace) == 100, "Should generate requested number of accesses"
        
        # Check addresses are sequential
        addresses = [addr for _, addr in trace]
        for i in range(len(addresses) - 1):
            assert addresses[i+1] == addresses[i] + 128, "Addresses should be sequential"

    def test_sequential_trace_custom_stride(self):
        """Test sequential trace with custom stride."""
        stride = 256
        trace = generate_sequential_trace(
            start_addr=0x2000, 
            num_accesses=50,
            stride=stride
        )
        
        addresses = [addr for _, addr in trace]
        for i in range(len(addresses) - 1):
            assert addresses[i+1] == addresses[i] + stride, \
                f"Stride should be {stride}, got {addresses[i+1] - addresses[i]}"

    def test_sequential_trace_operations(self):
        """Test sequential trace operation types."""
        trace = generate_sequential_trace(start_addr=0x3000, num_accesses=100)
        
        ops = [op for op, _ in trace]
        assert all(op == "READ" for op in ops), "Sequential trace should be all READs"


class TestRandomTraceGeneration:
    """Test random access pattern generation."""

    def test_random_trace_basic(self):
        """Test basic random trace generation."""
        trace = generate_random_trace(num_accesses=100, addr_range=(0, 1_000_000))
        
        assert len(trace) == 100, "Should generate requested number of accesses"

    def test_random_trace_range(self):
        """Test random addresses stay within specified range."""
        min_addr = 0x10000
        max_addr = 0x20000
        
        trace = generate_random_trace(
            num_accesses=100,
            addr_range=(min_addr, max_addr)
        )
        
        addresses = [addr for _, addr in trace]
        assert all(min_addr <= addr <= max_addr for addr in addresses), \
            "Addresses should be within specified range"

    def test_random_trace_no_locality(self):
        """Test random trace has minimal spatial locality."""
        trace = generate_random_trace(num_accesses=1000, addr_range=(0, 10_000_000))
        
        addresses = [addr for _, addr in trace]
        deltas = [abs(addresses[i+1] - addresses[i]) for i in range(len(addresses)-1)]
        
        # Random access should have low locality
        small_deltas = sum(1 for d in deltas if d <= 1024)
        locality_ratio = small_deltas / len(deltas)
        
        assert locality_ratio < 0.2, f"Random trace should have <20% locality, got {locality_ratio*100:.1f}%"

    def test_random_trace_reproducibility(self):
        """Test random trace is reproducible with seed."""
        trace1 = generate_random_trace(num_accesses=100, addr_range=(0, 1_000_000), seed=42)
        trace2 = generate_random_trace(num_accesses=100, addr_range=(0, 1_000_000), seed=42)
        
        assert trace1 == trace2, "Same seed should produce identical traces"


class TestStridedTraceGeneration:
    """Test strided access pattern generation."""

    def test_strided_trace_basic(self):
        """Test basic strided trace generation."""
        trace = generate_strided_trace(
            start_addr=0x4000,
            num_accesses=50,
            stride=512
        )
        
        assert len(trace) == 50, "Should generate requested number of accesses"

    def test_strided_trace_pattern(self):
        """Test strided access pattern is correct."""
        stride = 1024
        trace = generate_strided_trace(
            start_addr=0x5000,
            num_accesses=100,
            stride=stride
        )
        
        addresses = [addr for _, addr in trace]
        for i in range(len(addresses) - 1):
            assert addresses[i+1] == addresses[i] + stride, \
                "Addresses should increment by stride"

    def test_strided_trace_wraparound(self):
        """Test strided trace with address wraparound."""
        trace = generate_strided_trace(
            start_addr=0x6000,
            num_accesses=100,
            stride=2048,
            max_addr=0x6000 + 50 * 2048  # Force wraparound
        )
        
        assert len(trace) == 100, "Should generate all accesses with wraparound"


class TestTraceCharacteristics:
    """Test trace quality and characteristics."""

    def test_address_alignment(self):
        """Test generated addresses are properly aligned."""
        trace = generate_llm_trace(context_length=256, hidden_dim=2048)
        
        addresses = [addr for _, addr in trace]
        
        # Check alignment (should be cache line aligned)
        aligned = all(addr % 64 == 0 for addr in addresses)
        assert aligned, "Addresses should be 64-byte aligned"

    def test_no_duplicate_consecutive_reads(self):
        """Test traces avoid unnecessary duplicate consecutive reads."""
        trace = generate_sequential_trace(start_addr=0x7000, num_accesses=100)
        
        # Check no immediate duplicates
        for i in range(len(trace) - 1):
            if trace[i][0] == "READ" and trace[i+1][0] == "READ":
                assert trace[i][1] != trace[i+1][1], \
                    "Sequential trace should not have duplicate consecutive addresses"

    def test_trace_length_scaling(self):
        """Test trace length scales with parameters."""
        small_trace = generate_llm_trace(context_length=128, hidden_dim=1024)
        large_trace = generate_llm_trace(context_length=512, hidden_dim=4096)
        
        assert len(large_trace) > len(small_trace), \
            "Larger parameters should produce longer traces"

    def test_operation_type_validity(self):
        """Test all operations are valid types."""
        trace = generate_llm_trace(context_length=256, hidden_dim=2048)
        
        valid_ops = {"READ", "WRITE"}
        for op, addr in trace:
            assert op in valid_ops, f"Invalid operation type: {op}"
            assert isinstance(addr, int), f"Address should be integer, got {type(addr)}"
            assert addr >= 0, f"Address should be non-negative, got {addr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])