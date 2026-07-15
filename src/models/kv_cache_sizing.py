"""KV-Cache Size Calculation

Calculates theoretical Key-Value cache requirements for Transformer models.

Author: The Janus-1 Design Team
"""

from typing import Dict
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """LLM model configuration parameters."""

    num_layers: int = 32
    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    context_length: int = 4096


class KVCacheSizer:
    """Calculate KV-cache memory requirements.

    The KV cache stores Keys and Values for all tokens in the context
    window across all Transformer layers during autoregressive inference.

    For each token, each layer stores:
    - Key vector: [hidden_dim] elements
    - Value vector: [hidden_dim] elements

    Example:
        >>> sizer = KVCacheSizer()
        >>> result = sizer.calculate(precision='INT8')
        >>> print(f"Cache size: {result['size_mb']:.2f} MB")
        Cache size: 1024.00 MB
    """

    PRECISION_BYTES = {"FP32": 4, "FP16": 2, "INT8": 1, "INT4": 0.5}

    def __init__(self, config: ModelConfig = None):
        """Initialize with model configuration.

        Args:
            config: Model parameters. Uses Llama-2 7B defaults if None.
        """
        self.config = config or ModelConfig()

    def calculate(self, precision: str = "INT8") -> Dict:
        """Calculate KV-cache size for given precision.

        Args:
            precision: Data precision ('FP32', 'FP16', 'INT8', 'INT4')

        Returns:
            Dictionary with cache size in bytes and MB
        """
        if precision not in self.PRECISION_BYTES:
            raise ValueError(f"Unknown precision: {precision}")

        bytes_per_element = self.PRECISION_BYTES[precision]

        # Size per token = layers * hidden_dim * 2 (K and V) * bytes
        bytes_per_token = (
            self.config.num_layers * self.config.hidden_dim * 2 * bytes_per_element
        )

        # Total cache = size per token * context length
        total_bytes = bytes_per_token * self.config.context_length
        total_mb = total_bytes / (1024 * 1024)
        total_gb = total_mb / 1024

        return {
            "precision": precision,
            "bytes_per_element": bytes_per_element,
            "bytes_per_token": bytes_per_token,
            "total_bytes": total_bytes,
            "size_mb": total_mb,
            "size_gb": total_gb,
            "context_length": self.config.context_length,
            "num_layers": self.config.num_layers,
            "hidden_dim": self.config.hidden_dim,
        }

    def calculate_all_precisions(self) -> Dict[str, Dict]:
        """Calculate cache sizes for all supported precisions.

        Returns:
            Dictionary mapping precision to cache size info
        """
        return {
            precision: self.calculate(precision)
            for precision in self.PRECISION_BYTES.keys()
        }

    def print_report(self):
        """Print formatted report of cache requirements."""
        print(f"\n{'='*70}")
        print("KV-Cache Size Analysis")
        print(f"{'='*70}")
        print(f"\nModel Configuration:")
        print(f"  Layers: {self.config.num_layers}")
        print(f"  Hidden Dim: {self.config.hidden_dim}")
        print(f"  Context Length: {self.config.context_length} tokens")
        print(f"\nMemory Requirements by Precision:")
        print(
            f"\n{'Precision':<10} {'Bytes/Token':<15} {'Total MB':<15} {'Total GB':<10}"
        )
        print(f"{'-'*60}")

        for precision, info in self.calculate_all_precisions().items():
            print(
                f"{precision:<10} {info['bytes_per_token']:<15.1f} "
                f"{info['size_mb']:<15.2f} {info['size_gb']:<10.3f}"
            )

        print(f"\n{'='*70}\n")


_BITS_TO_PRECISION = {32: "FP32", 16: "FP16", 8: "INT8", 4: "INT4"}


def calculate_kv_cache_size(
    num_layers: int,
    hidden_dim: int,
    context_length: int,
    bits_per_param: int = 16,
) -> float:
    """Convenience wrapper returning KV-cache size in MB.

    Args:
        num_layers: Number of transformer layers.
        hidden_dim: Model hidden dimension.
        context_length: Context window length in tokens.
        bits_per_param: Precision in bits (32, 16, 8 or 4).

    Returns:
        KV-cache size in megabytes.
    """
    if bits_per_param not in _BITS_TO_PRECISION:
        raise ValueError(
            f"bits_per_param must be one of {sorted(_BITS_TO_PRECISION)}, "
            f"got {bits_per_param}"
        )
    config = ModelConfig(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        context_length=context_length,
    )
    result = KVCacheSizer(config).calculate(_BITS_TO_PRECISION[bits_per_param])
    return float(result["size_mb"])


if __name__ == "__main__":
    # Example: Llama-2 7B cache analysis
    sizer = KVCacheSizer()
    sizer.print_report()

    # Show impact of quantization
    int8_result = sizer.calculate("INT8")
    int4_result = sizer.calculate("INT4")

    print(f"\nQuantization Impact:")
    print(
        f"  INT8 -> INT4 reduction: {int8_result['size_mb'] / int4_result['size_mb']:.1f}x"
    )
    print(f"  Memory saved: {int8_result['size_mb'] - int4_result['size_mb']:.1f} MB")
