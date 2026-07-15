#!/usr/bin/env python3
"""
KV-Cache Memory Requirement Analysis
====================================

Calculate KV-cache memory requirements for different model sizes,
context lengths, and quantization levels.

This helps determine the optimal memory hierarchy configuration
for your target workload.

Author: Janus-1 Research Team
License: MIT
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.kv_cache_sizing import calculate_kv_cache_size


def analyze_model_sizes():
    """Analyze KV-cache requirements across model sizes."""
    
    print("=" * 70)
    print("KV-Cache Memory Requirement Analysis")
    print("=" * 70)
    print()
    
    # Model configurations
    models = {
        'Llama-2 7B': {'layers': 32, 'hidden_dim': 4096, 'heads': 32},
        'Llama-2 13B': {'layers': 40, 'hidden_dim': 5120, 'heads': 40},
        'Llama-2 70B': {'layers': 80, 'hidden_dim': 8192, 'heads': 64},
    }
    
    context_length = 4096  # Standard context window
    
    print(f"Analysis for {context_length}-token context:\n")
    
    # Quantization levels
    quant_bits = [16, 8, 4]
    
    for model_name, config in models.items():
        print(f"\n{model_name}:")
        print("-" * 70)
        
        for bits in quant_bits:
            size_mb = calculate_kv_cache_size(
                num_layers=config['layers'],
                hidden_dim=config['hidden_dim'],
                context_length=context_length,
                bits_per_param=bits
            )
            
            quant_name = f"FP{bits}" if bits == 16 else f"INT{bits}"
            print(f"  {quant_name:6s}: {size_mb:8.1f} MB")
            
            # Janus-1 feasibility check
            janus_capacity = 256  # MB (32 SRAM + 224 eDRAM)
            if size_mb <= janus_capacity:
                print(f"           ✅ Fits in Janus-1 ({size_mb/janus_capacity*100:.1f}% utilization)")
            else:
                print(f"           ❌ Exceeds Janus-1 capacity ({size_mb/janus_capacity:.1f}x over)")
    
    print("\n" + "=" * 70)


def analyze_context_scaling():
    """Analyze how KV-cache scales with context length."""
    
    print("\nContext Length Scaling Analysis (Llama-2 7B):")
    print("=" * 70)
    
    context_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    
    # Llama-2 7B config
    config = {'layers': 32, 'hidden_dim': 4096}
    
    print(f"\n{'Context':>10s} {'FP16':>12s} {'INT8':>12s} {'INT4':>12s} {'Janus-1 Fit':>15s}")
    print("-" * 70)
    
    for ctx_len in context_lengths:
        sizes = {}
        for bits in [16, 8, 4]:
            sizes[bits] = calculate_kv_cache_size(
                num_layers=config['layers'],
                hidden_dim=config['hidden_dim'],
                context_length=ctx_len,
                bits_per_param=bits
            )
        
        fit_symbol = "✅" if sizes[4] <= 256 else "❌"
        
        print(f"{ctx_len:10d} {sizes[16]:10.1f} MB {sizes[8]:10.1f} MB "
              f"{sizes[4]:10.1f} MB {fit_symbol:>12s}")
    
    print("\n" + "=" * 70)


def plot_memory_tradeoffs():
    """Generate visualization of memory vs. quantization tradeoffs."""
    
    print("\nGenerating visualization...")
    
    context_lengths = np.array([512, 1024, 2048, 4096, 8192])
    config = {'layers': 32, 'hidden_dim': 4096}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for bits, color, label in [(16, 'red', 'FP16'), (8, 'blue', 'INT8'), (4, 'green', 'INT4')]:
        sizes = [calculate_kv_cache_size(
            num_layers=config['layers'],
            hidden_dim=config['hidden_dim'],
            context_length=ctx,
            bits_per_param=bits
        ) for ctx in context_lengths]
        
        ax.plot(context_lengths, sizes, marker='o', color=color, 
                label=label, linewidth=2, markersize=8)
    
    # Janus-1 capacity line
    ax.axhline(y=256, color='black', linestyle='--', linewidth=2, 
               label='Janus-1 Capacity (256 MB)')
    
    ax.set_xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('KV-Cache Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('KV-Cache Memory Requirements: Llama-2 7B', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('results/kv_cache_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to results/kv_cache_analysis.png")
    
    return fig


def main():
    """Run complete KV-cache analysis."""
    
    analyze_model_sizes()
    analyze_context_scaling()
    plot_memory_tradeoffs()
    
    print("\n✅ Analysis complete!")
    print("\nKey Takeaway:")
    print("  INT4 quantization cuts the KV-cache footprint 4x vs FP16, but")
    print("  Llama-2 7B at 4K context is still ~512 MB in INT4 -- it exceeds a")
    print("  256 MB on-chip budget. Shorter contexts (<=2K) do fit in INT4.")
    print()


if __name__ == "__main__":
    main()
