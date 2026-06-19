#!/usr/bin/env python3
"""Compare v1 baseline vs v2 improved pipeline across all methods."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.compression.fft_compression import compress_fft, decompress_fft
from src.compression.wavelet_compression import compress_wavelet, decompress_wavelet
from src.compression.quantization import compress_quantization, decompress_quantization
from src.data_loader.loader import load_signal
from src.generate_synthetic import create_synthetic_dataset
from src.metrics.compression_ratio import compression_ratio
from src.metrics.mse import mse
from src.metrics.snr import snr_db
from src.preprocessing.filtering import butterworth_lowpass
from src.preprocessing.noise import apply_all_noise
from src.v2.compression.adaptive_fft import compress_adaptive_fft, decompress_adaptive_fft
from src.v2.compression.soft_wavelet import compress_soft_wavelet, decompress_soft_wavelet
from src.v2.compression.mulaw_quantization import compress_mulaw, decompress_mulaw
from src.v2.compression.hybrid import compress_hybrid, decompress_hybrid
from src.v2.denoising.pipeline import denoise_multistage


def benchmark(name, original, compress_fn, decompress_fn):
    """Run a single compression benchmark."""
    compressed = compress_fn(original)
    reconstructed = decompress_fn(compressed)
    return {
        "method": name,
        "mse": round(mse(original, reconstructed), 6),
        "snr_db": round(snr_db(original, reconstructed), 2),
        "compression_ratio": round(compression_ratio(original, compressed), 2),
    }


def main():
    data_path = "data/raw/synthetic_rocket.csv"
    if not Path(data_path).exists():
        create_synthetic_dataset(data_path)

    time_arr, signal, metadata = load_signal(data_path)
    fs = metadata["sampling_frequency"]

    noisy, _ = apply_all_noise(signal, seed=42)

    # v1 denoising
    v1_filtered = butterworth_lowpass(noisy, fs, cutoff_frequency=15.0)
    # v2 denoising
    v2_filtered, _ = denoise_multistage(noisy, fs)

    results = {"denoising": {}, "compression_v1": [], "compression_v2": []}

    # Denoising comparison (vs clean original)
    results["denoising"] = {
        "v1_butterworth": {
            "snr_db": round(snr_db(signal, v1_filtered), 2),
            "mse": round(mse(signal, v1_filtered), 6),
        },
        "v2_multistage": {
            "snr_db": round(snr_db(signal, v2_filtered), 2),
            "mse": round(mse(signal, v2_filtered), 6),
        },
    }

    working_v1 = v1_filtered
    working_v2 = v2_filtered

    # v1 compression benchmarks
    results["compression_v1"] = [
        benchmark("fft_10pct", working_v1,
                  lambda s: compress_fft(s, 0.1), decompress_fft),
        benchmark("wavelet_db4_10pct", working_v1,
                  lambda s: compress_wavelet(s, "db4", 0.1), decompress_wavelet),
        benchmark("quantization_8bit", working_v1,
                  lambda s: compress_quantization(s, 8), decompress_quantization),
    ]

    # v2 compression benchmarks
    results["compression_v2"] = [
        benchmark("adaptive_fft_90pct_energy", working_v2,
                  lambda s: compress_adaptive_fft(s, 0.90), decompress_adaptive_fft),
        benchmark("soft_wavelet_10pct", working_v2,
                  lambda s: compress_soft_wavelet(s, "db4", 0.1), decompress_soft_wavelet),
        benchmark("mulaw_8bit", working_v2,
                  lambda s: compress_mulaw(s, 8), decompress_mulaw),
        benchmark("hybrid_85pct", working_v2,
                  lambda s: compress_hybrid(s, 0.85), decompress_hybrid),
    ]

    # Print summary table
    print("\n=== Denoising Comparison (vs clean signal) ===")
    print(f"{'Method':<20} {'SNR (dB)':>10} {'MSE':>12}")
    for name, metrics in results["denoising"].items():
        print(f"{name:<20} {metrics['snr_db']:>10.1f} {metrics['mse']:>12.6f}")

    print("\n=== v1 Compression (on v1-filtered signal) ===")
    print(f"{'Method':<30} {'Ratio':>8} {'SNR (dB)':>10} {'MSE':>12}")
    for r in results["compression_v1"]:
        print(f"{r['method']:<30} {r['compression_ratio']:>7.1f}x {r['snr_db']:>10.1f} {r['mse']:>12.6f}")

    print("\n=== v2 Compression (on v2-denoised signal) ===")
    print(f"{'Method':<30} {'Ratio':>8} {'SNR (dB)':>10} {'MSE':>12}")
    for r in results["compression_v2"]:
        print(f"{r['method']:<30} {r['compression_ratio']:>7.1f}x {r['snr_db']:>10.1f} {r['mse']:>12.6f}")

    output_path = Path("results/metrics/v1_v2_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
