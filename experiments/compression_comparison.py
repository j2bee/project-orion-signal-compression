#!/usr/bin/env python3
"""
Compression benchmark — compare all methods with ratio-matched metrics.

Outputs results/compression_comparison.csv with Method, Compression Ratio,
MSE, RMSE, SNR, Runtime, File Size.
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.compression.fft_compression import compress_fft, decompress_fft
from src.compression.wavelet_compression import compress_wavelet, decompress_wavelet
from src.compression.quantization import compress_quantization, decompress_quantization
from src.v2.compression.adaptive_fft import compress_adaptive_fft, decompress_adaptive_fft
from src.v2.compression.soft_wavelet import compress_soft_wavelet, decompress_soft_wavelet
from src.v2.compression.hybrid import compress_hybrid, decompress_hybrid
from src.v2.compression.event_aware import compress_event_aware, decompress_event_aware
from src.compression.windowed import compress_windowed_fft, decompress_windowed_fft
from src.data_loader.loader import load_signal
from src.generate_synthetic import create_synthetic_dataset
from src.metrics.compression_ratio import compression_ratio, estimate_compressed_size
from src.metrics.mse import mse, rmse
from src.metrics.snr import snr_db
from src.preprocessing.filtering import butterworth_lowpass, moving_average, savgol_filter
from src.preprocessing.noise import apply_all_noise
from src.v2.denoising.pipeline import denoise_multistage


def benchmark_row(name, signal, clean, compress_fn, decompress_fn):
    """Run one compression benchmark and return CSV row dict."""
    t0 = time.time()
    compressed = compress_fn(signal)
    reconstructed = decompress_fn(compressed)
    runtime = time.time() - t0
    return {
        "Method": name,
        "Compression_Ratio": round(compression_ratio(signal, compressed), 2),
        "MSE_vs_working": round(mse(signal, reconstructed), 8),
        "RMSE_vs_working": round(rmse(signal, reconstructed), 8),
        "SNR_vs_working_dB": round(snr_db(signal, reconstructed), 2),
        "MSE_vs_clean": round(mse(clean, reconstructed), 8),
        "SNR_vs_clean_dB": round(snr_db(clean, reconstructed), 2),
        "Runtime_s": round(runtime, 4),
        "File_Size_bytes": estimate_compressed_size(compressed),
    }


def main():
    data_path = "data/raw/synthetic_rocket.csv"
    if not Path(data_path).exists():
        create_synthetic_dataset(data_path)

    time_arr, clean, meta = load_signal(data_path)
    fs = meta["sampling_frequency"]
    noisy, _ = apply_all_noise(clean, seed=42)
    working = denoise_multistage(noisy, fs)[0]

    rows = []

    # FFT threshold sweep
    for pct in [0.01, 0.05, 0.10, 0.20, 0.50]:
        rows.append(benchmark_row(
            f"FFT_{int(pct*100)}pct", working, clean,
            lambda s, p=pct: compress_fft(s, p), decompress_fft,
        ))

    # Adaptive FFT energy sweep
    for energy in [0.90, 0.95, 0.99]:
        rows.append(benchmark_row(
            f"AdaptiveFFT_{int(energy*100)}pct_energy", working, clean,
            lambda s, e=energy: compress_adaptive_fft(s, e, max_keep_fraction=0.15),
            decompress_adaptive_fft,
        ))

    # Wavelet sweep
    for wavelet in ["haar", "db4", "db8"]:
        rows.append(benchmark_row(
            f"Wavelet_{wavelet}_10pct", working, clean,
            lambda s, w=wavelet: compress_wavelet(s, w, 0.10), decompress_wavelet,
        ))
        rows.append(benchmark_row(
            f"SoftWavelet_{wavelet}_10pct", working, clean,
            lambda s, w=wavelet: compress_soft_wavelet(s, w, 0.10), decompress_soft_wavelet,
        ))

    # Quantization (8 and 16 bit supported)
    for bits in [8, 16]:
        rows.append(benchmark_row(
            f"Quantization_{bits}bit", working, clean,
            lambda s, b=bits: compress_quantization(s, b),
            decompress_quantization,
        ))

    # PCA baseline
    try:
        from src.research.classical_extensions import compress_pca, decompress_pca
        rows.append(benchmark_row(
            "PCA_32components", working, clean,
            lambda s: compress_pca(s, n_components=32), decompress_pca,
        ))
    except ImportError:
        pass

    # Hybrid and event-aware
    rows.append(benchmark_row(
        "Hybrid_85pct", working, clean,
        lambda s: compress_hybrid(s, 0.85), decompress_hybrid,
    ))
    rows.append(benchmark_row(
        "EventAware_hybrid", working, clean,
        lambda s: compress_event_aware(s, fs), decompress_event_aware,
    ))

    # Windowed compression
    for ws in [512, 1024, 2048]:
        rows.append(benchmark_row(
            f"WindowedFFT_{ws}", working, clean,
            lambda s, w=ws: compress_windowed_fft(s, window_size=w, keep_percentage=0.10),
            decompress_windowed_fft,
        ))

    # Pre-filtering comparison (Improvement 1)
    for filt_name, filt_fn in [
        ("Butterworth", lambda s: butterworth_lowpass(s, fs, 15.0)),
        ("MovingAvg", lambda s: moving_average(s, 11)),
        ("SavitzkyGolay", lambda s: savgol_filter(s, 11, 3)),
    ]:
        filtered = filt_fn(working)
        rows.append(benchmark_row(
            f"Prefilter_{filt_name}_FFT10pct", filtered, clean,
            lambda s: compress_fft(s, 0.10), decompress_fft,
        ))

    # Write CSV
    out_path = Path("results/compression_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print("\nTop 5 by SNR vs working:")
    sorted_rows = sorted(rows, key=lambda r: r["SNR_vs_working_dB"], reverse=True)
    for r in sorted_rows[:5]:
        print(f"  {r['Method']:35s}  ratio={r['Compression_Ratio']:5.1f}x  "
              f"SNR={r['SNR_vs_working_dB']:6.1f} dB  MSE={r['MSE_vs_working']:.2e}")


if __name__ == "__main__":
    main()
