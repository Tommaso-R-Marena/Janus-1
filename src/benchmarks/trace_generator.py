#!/usr/bin/env python3
"""
Memory Trace Generator
======================

Generates realistic memory access traces for Janus-1 simulation.
Supports various access patterns including LLM inference workloads.

Author: Janus-1 Design Team
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional

_QUANTIZATION_BYTES = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}


def generate_llm_trace(
    context_length: int = 2048,
    hidden_dim: int = 4096,
    num_layers: int = 32,
    batch_size: int = 1,
    quantization: str = "fp16",
) -> List[Tuple[str, int]]:
    """
    Generate memory trace for LLM inference (autoregressive generation).

    Models KV-cache access pattern during token generation:
    - Sequential reads of KV cache for attention
    - Write of new KV pair
    - Sequential reads of embeddings/weights

    Args:
        context_length: Number of tokens in context
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        batch_size: Batch size
        quantization: KV element precision ('fp32', 'fp16', 'int8', 'int4');
            controls the per-token KV footprint and therefore the access stride.

    Returns:
        List of (operation, address) tuples
    """
    if quantization not in _QUANTIZATION_BYTES:
        raise ValueError(
            f"quantization must be one of {sorted(_QUANTIZATION_BYTES)}, "
            f"got {quantization!r}"
        )

    trace = []

    # Base address for KV cache
    kv_cache_base = 0x100000

    # Size of one KV pair (key + value): 2 tensors (K, V) * bytes-per-element.
    bytes_per_kv = int(hidden_dim * 2 * _QUANTIZATION_BYTES[quantization])

    # Cache line size
    cache_line_size = 128

    # Simulate autoregressive generation
    for token_idx in range(context_length):
        for layer in range(num_layers):
            layer_offset = layer * context_length * bytes_per_kv

            # Read existing KV cache for attention (sequential)
            for past_token in range(token_idx):
                kv_addr = kv_cache_base + layer_offset + (past_token * bytes_per_kv)
                # Align to cache line
                kv_addr = (kv_addr // cache_line_size) * cache_line_size
                trace.append(("READ", kv_addr))

            # Write new KV pair
            new_kv_addr = kv_cache_base + layer_offset + (token_idx * bytes_per_kv)
            new_kv_addr = (new_kv_addr // cache_line_size) * cache_line_size
            trace.append(("WRITE", new_kv_addr))

            # Read query projection (simulated as sequential access)
            if token_idx % 4 == 0:  # Every 4th token
                query_base = 0x500000
                query_addr = query_base + (layer * hidden_dim * 2)
                query_addr = (query_addr // cache_line_size) * cache_line_size
                trace.append(("READ", query_addr))

    return trace


def generate_sequential_trace(
    start_addr: int = 0x10000,
    num_accesses: int = 1000,
    stride: int = 128,
    operation: str = "READ",
) -> List[Tuple[str, int]]:
    """
    Generate sequential memory access trace.

    Args:
        start_addr: Starting address
        num_accesses: Number of sequential accesses
        stride: Stride between accesses (bytes)
        operation: "READ" or "WRITE"

    Returns:
        List of (operation, address) tuples
    """
    trace = []
    addr = start_addr

    for _ in range(num_accesses):
        trace.append((operation, addr))
        addr += stride

    return trace


def generate_random_trace(
    num_accesses: int = 1000,
    addr_range: Tuple[int, int] = (0, 10_000_000),
    alignment: int = 128,
    operation: str = "READ",
    seed: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Generate random memory access trace.

    Args:
        num_accesses: Number of random accesses
        addr_range: (min_addr, max_addr) tuple
        alignment: Address alignment (bytes)
        operation: "READ" or "WRITE"
        seed: Random seed for reproducibility

    Returns:
        List of (operation, address) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    trace = []
    min_addr, max_addr = addr_range

    # Generate random addresses
    num_aligned_blocks = (max_addr - min_addr) // alignment
    random_blocks = np.random.randint(0, num_aligned_blocks, size=num_accesses)

    for block in random_blocks:
        addr = min_addr + block * alignment
        trace.append((operation, addr))

    return trace


def generate_strided_trace(
    start_addr: int = 0x10000,
    num_accesses: int = 1000,
    stride: int = 256,
    max_addr: Optional[int] = None,
    operation: str = "READ",
) -> List[Tuple[str, int]]:
    """
    Generate strided memory access trace with wraparound.

    Args:
        start_addr: Starting address
        num_accesses: Number of accesses
        stride: Stride between accesses (bytes)
        max_addr: Maximum address (wraps to start_addr)
        operation: "READ" or "WRITE"

    Returns:
        List of (operation, address) tuples
    """
    trace = []
    addr = start_addr

    for _ in range(num_accesses):
        trace.append((operation, addr))
        addr += stride

        if max_addr is not None and addr >= max_addr:
            addr = start_addr

    return trace


def generate_streaming_trace(
    start_addr: int = 0x10000,
    num_streams: int = 4,
    stream_length: int = 256,
    stride: int = 128,
) -> List[Tuple[str, int]]:
    """
    Generate multiple interleaved streaming access patterns.

    Useful for testing prefetcher with multiple concurrent streams.

    Args:
        start_addr: Base starting address
        num_streams: Number of concurrent streams
        stream_length: Length of each stream
        stride: Stride within each stream

    Returns:
        List of (operation, address) tuples
    """
    trace = []

    # Initialize stream positions
    stream_positions = [
        start_addr + (i * stream_length * stride * 10) for i in range(num_streams)
    ]

    # Interleave accesses from different streams
    for _ in range(stream_length):
        for stream_idx in range(num_streams):
            trace.append(("READ", stream_positions[stream_idx]))
            stream_positions[stream_idx] += stride

    return trace


def generate_mixed_trace(
    sequential_ratio: float = 0.7,
    num_accesses: int = 1000,
    base_addr: int = 0x10000,
    seed: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Generate mixed sequential and random access trace.

    Args:
        sequential_ratio: Fraction of accesses that are sequential (0.0-1.0)
        num_accesses: Total number of accesses
        base_addr: Base address
        seed: Random seed

    Returns:
        List of (operation, address) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    trace = []
    num_sequential = int(num_accesses * sequential_ratio)
    num_random = num_accesses - num_sequential

    # Generate sequential portion
    seq_trace = generate_sequential_trace(
        start_addr=base_addr, num_accesses=num_sequential
    )

    # Generate random portion
    rand_trace = generate_random_trace(
        num_accesses=num_random,
        addr_range=(base_addr, base_addr + 10_000_000),
        seed=seed,
    )

    # Interleave
    seq_idx = 0
    rand_idx = 0

    for _ in range(num_accesses):
        if seq_idx < num_sequential and (
            rand_idx >= num_random or np.random.rand() < sequential_ratio
        ):
            trace.append(seq_trace[seq_idx])
            seq_idx += 1
        else:
            trace.append(rand_trace[rand_idx])
            rand_idx += 1

    return trace


def analyze_trace(trace: List[Tuple[str, int]]) -> dict:
    """
    Analyze memory trace characteristics.

    Args:
        trace: Memory trace to analyze

    Returns:
        Dictionary with trace statistics
    """
    addresses = [addr for _, addr in trace]
    operations = [op for op, _ in trace]

    # Calculate deltas (spatial locality)
    deltas = [abs(addresses[i + 1] - addresses[i]) for i in range(len(addresses) - 1)]

    # Operation counts
    num_reads = sum(1 for op in operations if op == "READ")
    num_writes = sum(1 for op in operations if op == "WRITE")

    # Locality analysis
    small_deltas = sum(1 for d in deltas if d <= 1024)
    locality_ratio = small_deltas / len(deltas) if deltas else 0.0

    return {
        "num_operations": len(trace),
        "num_reads": num_reads,
        "num_writes": num_writes,
        "read_ratio": num_reads / len(trace) if trace else 0.0,
        "unique_addresses": len(set(addresses)),
        "min_address": min(addresses) if addresses else 0,
        "max_address": max(addresses) if addresses else 0,
        "address_range": max(addresses) - min(addresses) if addresses else 0,
        "avg_delta": np.mean(deltas) if deltas else 0.0,
        "median_delta": np.median(deltas) if deltas else 0.0,
        "spatial_locality": locality_ratio,
    }


if __name__ == "__main__":
    # Example usage
    print("Generating LLM inference trace...")
    llm_trace = generate_llm_trace(context_length=512, hidden_dim=2048)
    print(f"Generated {len(llm_trace)} operations")

    stats = analyze_trace(llm_trace)
    print("\nTrace Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nGenerating sequential trace...")
    seq_trace = generate_sequential_trace(num_accesses=100)
    print(f"Generated {len(seq_trace)} operations")

    print("\nGenerating random trace...")
    rand_trace = generate_random_trace(num_accesses=100, seed=42)
    print(f"Generated {len(rand_trace)} operations")
