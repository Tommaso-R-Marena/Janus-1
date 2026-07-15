#!/usr/bin/env python3
"""
Robustness of the Memory-Technology Selection Under Parameter Uncertainty
=========================================================================

The per-bit energies and per-MB leakage in ``MemoryPowerModel`` are
order-of-magnitude estimates, not calibrated (CACTI/PDK) values -- see
LIMITATIONS.md. Absolute power numbers therefore should not be trusted. What we
*can* test honestly is whether the qualitative design decision ("use eDRAM, not
HD-SRAM, for the 224 MB leakage-dominated Tier-2 cache") survives wide
uncertainty in those constants.

This script perturbs every technology constant by a large relative factor
(default +/-50%, log-uniform) and Monte-Carlo-propagates it to the total-power
comparison. It reports the probability that eDRAM remains lower-power than
HD-SRAM, and the distribution of the power ratio. The conclusion is robust not
because the constants are precise, but because SRAM's per-MB leakage is ~16x
eDRAM's, a gap far larger than the uncertainty.

Usage:
    python experiments/power_robustness.py [--samples N] [--rel-uncertainty F]

Author: Janus-1 Design Team
License: MIT
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.memory_power_model import MemoryPowerModel

CACHE_MB = 224.0
BANDWIDTH_GB_S = 20.0


def _total_power(
    size_mb,
    static_mw_per_mb,
    read_pj,
    write_pj,
    bandwidth_gb_s=BANDWIDTH_GB_S,
    read_ratio=0.9,
    write_ratio=0.1,
):
    """Total power (W) from perturbed constants; mirrors MemoryPowerModel."""
    bw_bits = bandwidth_gb_s * 8 * 1024**3
    dyn = (bw_bits * read_ratio * read_pj + bw_bits * write_ratio * write_pj) * 1e-12
    static = size_mb * static_mw_per_mb / 1000.0
    return dyn + static


def _log_uniform(rng, nominal, rel, n):
    """Sample n values log-uniformly within nominal * [1-rel, 1+rel]."""
    lo, hi = np.log(nominal * (1 - rel)), np.log(nominal * (1 + rel))
    return np.exp(rng.uniform(lo, hi, n))


def run(samples: int, rel: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    techs = MemoryPowerModel.TECHNOLOGIES
    sram, edram = techs["HD_SRAM"], techs["eDRAM"]

    sram_p = _total_power(
        CACHE_MB,
        _log_uniform(rng, sram.static_power_per_mb_mw, rel, samples),
        _log_uniform(rng, sram.read_energy_pj, rel, samples),
        _log_uniform(rng, sram.write_energy_pj, rel, samples),
    )
    edram_p = _total_power(
        CACHE_MB,
        _log_uniform(rng, edram.static_power_per_mb_mw, rel, samples),
        _log_uniform(rng, edram.read_energy_pj, rel, samples),
        _log_uniform(rng, edram.write_energy_pj, rel, samples),
    )
    ratio = sram_p / edram_p
    win_prob = float(np.mean(edram_p < sram_p))
    return sram_p, edram_p, ratio, win_prob


def save_figure(ratio, path: Path):
    """Histogram of the SRAM/eDRAM total-power ratio."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping figure: matplotlib unavailable: {exc})")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratio, bins=60, color="tab:purple", alpha=0.85)
    ax.axvline(1.0, color="black", ls="--", label="parity (ratio = 1)")
    ax.axvline(
        float(np.median(ratio)),
        color="tab:orange",
        label=f"median = {np.median(ratio):.1f}x",
    )
    ax.set_xlabel("HD-SRAM / eDRAM total power ratio (224 MB)")
    ax.set_ylabel("Monte-Carlo samples")
    ax.set_title("Technology selection is robust to +/-50% constant uncertainty")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--rel-uncertainty", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    sram_p, edram_p, ratio, win_prob = run(args.samples, args.rel_uncertainty)

    print("=" * 70)
    print("MEMORY-TECHNOLOGY SELECTION ROBUSTNESS (224 MB Tier-2)")
    print("=" * 70)
    print(f"Monte-Carlo samples : {args.samples:,}")
    print(f"Per-constant uncertainty : +/-{args.rel_uncertainty:.0%} (log-uniform)")
    print()
    print(
        f"HD-SRAM total power : {sram_p.mean():6.2f} W "
        f"[{np.percentile(sram_p, 2.5):.2f}, {np.percentile(sram_p, 97.5):.2f}]"
    )
    print(
        f"eDRAM   total power : {edram_p.mean():6.2f} W "
        f"[{np.percentile(edram_p, 2.5):.2f}, {np.percentile(edram_p, 97.5):.2f}]"
    )
    print(
        f"Power ratio (SRAM/eDRAM): median {np.median(ratio):.1f}x "
        f"[{np.percentile(ratio, 2.5):.1f}, {np.percentile(ratio, 97.5):.1f}]"
    )
    print()
    print(f"P(eDRAM lower power than HD-SRAM) = {win_prob:.4f}")
    print()
    print("Interpretation: the ABSOLUTE watts are not calibrated and must not be")
    print("cited, but the ORDERING (eDRAM << HD-SRAM for a leakage-dominated")
    print("224 MB cache) is robust because SRAM per-MB leakage is ~16x eDRAM's,")
    print("far exceeding the assumed uncertainty.")
    print("=" * 70)

    save_figure(ratio, args.output_dir / "power_robustness.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
