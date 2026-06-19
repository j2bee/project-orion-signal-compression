#!/usr/bin/env python3
"""Compare v1 baseline vs v2 refined pipeline with ratio-matched and segmented metrics."""

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
from src.metrics.segmented import compute_segment_metrics, default_rocket_segments
from src.metrics.snr import snr_db
from src.preprocessing.filtering import butterworth_lowpass
from src.preprocessing.noise import apply_all_noise
from src.v2.compression.adaptive_fft import compress_adaptive_fft, decompress_adaptive_fft
from src.v2.compression.soft_wavelet import compress_soft_wavelet, decompress_soft_wavelet
from src.v2.compression.mulaw_quantization import compress_mulaw, decompress_mulaw
from src.v2.compression.hybrid import compress_hybrid, decompress_hybrid
from src.v2.denoising.pipeline import denoise_multistage


def benchmark(name, clean, working, compress_fn, decompress_fn):
    """Run compression benchmark with dual reference metrics."""
    compressed = compress_fn(working)
    reconstructed = decompress_fn(compressed)
    return {
        "method": name,
        "mse_vs_working": round(mse(working, reconstructed), 8),
        "snr_vs_working_db": round(snr_db(working, reconstructed), 2),
        "mse_vs_clean": round(mse(clean, reconstructed), 8),
        "snr_vs_clean_db": round(snr_db(clean, reconstructed), 2),
        "compression_ratio": round(compression_ratio(working, compressed), 2),
    }


def main():
    data_path = "data/raw/synthetic_rocket.csv"
    if not Path(data_path).exists():
        create_synthetic_dataset(data_path)

    time_arr, clean, metadata = load_signal(data_path)
    fs = metadata["sampling_frequency"]
    segments = default_rocket_segments(metadata["duration"])

    noisy, _ = apply_all_noise(clean, seed=42)
    v1_filtered = butterworth_lowpass(noisy, fs, cutoff_frequency=15.0)
    v2_filtered, _ = denoise_multistage(noisy, fs)

    results = {
        "denoising": {},
        "compression_v1": [],
        "compression_v2": [],
        "ratio_matched_v2": [],
        "segment_hybrid": {},
    }

    results["denoising"] = {
        "v1_butterworth": {
            "snr_vs_clean_db": round(snr_db(clean, v1_filtered), 2),
            "mse_vs_clean": round(mse(clean, v1_filtered), 6),
        },
        "v2_multistage": {
            "snr_vs_clean_db": round(snr_db(clean, v2_filtered), 2),
            "mse_vs_clean": round(mse(clean, v2_filtered), 6),
        },
    }

    results["compression_v1"] = [
        benchmark("fft_10pct", clean, v1_filtered,
                  lambda s: compress_fft(s, 0.1), decompress_fft),
        benchmark("wavelet_db4_10pct", clean, v1_filtered,
                  lambda s: compress_wavelet(s, "db4", 0.1), decompress_wavelet),
        benchmark("quantization_8bit", clean, v1_filtered,
                  lambda s: compress_quantization(s, 8), decompress_quantization),
    ]

    results["compression_v2"] = [
        benchmark("adaptive_fft_85pct", clean, v2_filtered,
                  lambda s: compress_adaptive_fft(s, 0.85), decompress_adaptive_fft),
        benchmark("soft_wavelet_10pct", clean, v2_filtered,
                  lambda s: compress_soft_wavelet(s, "db4", 0.1), decompress_soft_wavelet),
        benchmark("mulaw_8bit", clean, v2_filtered,
                  lambda s: compress_mulaw(s, 8), decompress_mulaw),
        benchmark("hybrid_85pct", clean, v2_filtered,
                  lambda s: compress_hybrid(s, 0.85), decompress_hybrid),
    ]

    # Ratio-matched comparison at ~6.7x (v1 FFT baseline ratio)
    target = 6.7
    results["ratio_matched_v2"] = [
        benchmark("hybrid_target_6.7x", clean, v2_filtered,
                  lambda s: compress_hybrid(s, 0.85, target_ratio=target),
                  decompress_hybrid),
        benchmark("adaptive_fft_capped", clean, v2_filtered,
                  lambda s: compress_adaptive_fft(s, 0.90, max_keep_fraction=0.15),
                  decompress_adaptive_fft),
    ]

    # Segmented metrics for best v2 method (hybrid)
    hybrid_comp = compress_hybrid(v2_filtered, 0.85)
    hybrid_recon = decompress_hybrid(hybrid_comp)
    results["segment_hybrid"] = compute_segment_metrics(
        time_arr, clean, hybrid_recon, segments
    )

    # Print summary
    print("\n=== Denoising (vs clean signal) ===")
    print(f"{'Method':<20} {'SNR (dB)':>10} {'MSE':>12}")
    for name, m in results["denoising"].items():
        print(f"{name:<20} {m['snr_vs_clean_db']:>10.1f} {m['mse_vs_clean']:>12.6f}")

    print("\n=== v2 Compression (vs working + vs clean) ===")
    print(f"{'Method':<25} {'Ratio':>7} {'SNR work':>9} {'SNR clean':>10} {'MSE clean':>12}")
    for r in results["compression_v2"]:
        print(f"{r['method']:<25} {r['compression_ratio']:>6.1f}x "
              f"{r['snr_vs_working_db']:>9.1f} {r['snr_vs_clean_db']:>10.1f} "
              f"{r['mse_vs_clean']:>12.8f}")

    print("\n=== Ratio-Matched v2 (~6.7x) ===")
    for r in results["ratio_matched_v2"]:
        print(f"{r['method']:<25} {r['compression_ratio']:>6.1f}x "
              f"SNR clean={r['snr_vs_clean_db']:.1f} dB  MSE clean={r['mse_vs_clean']:.8f}")

    print("\n=== Hybrid Segment Metrics (vs clean) ===")
    for phase, sm in results["segment_hybrid"].items():
        print(f"  {phase:12s}  SNR={sm['snr_db']:6.1f} dB  MSE={sm['mse']:.8f}")

    output_path = Path("results/metrics/v1_v2_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
