#!/usr/bin/env python3
"""
Baseline run — execute current pipeline without modifications.

Loads example signal, runs compression + reconstruction, saves artifacts
to results/baseline/ for research comparison.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.compression.fft_compression import compress_fft, decompress_fft
from src.data_loader.loader import load_signal, save_signal
from src.generate_synthetic import create_synthetic_dataset
from src.metrics.compression_ratio import compression_ratio, estimate_compressed_size
from src.metrics.mse import mse, rmse
from src.metrics.snr import snr_db
from src.preprocessing.filtering import butterworth_lowpass
from src.preprocessing.noise import apply_all_noise
from src.analysis.report import generate_analysis_report
from src.visualization.error_plots import plot_error_analysis
from src.visualization.plots import plot_comparison, plot_signal_overview


def main():
    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = "data/raw/synthetic_rocket.csv"
    if not Path(data_path).exists():
        create_synthetic_dataset(data_path)

    print("[1/6] Loading signal...")
    time_arr, clean, metadata = load_signal(data_path)
    fs = metadata["sampling_frequency"]

    print("[2/6] Applying noise and filtering...")
    noisy, _ = apply_all_noise(clean, seed=42)
    filtered = butterworth_lowpass(noisy, fs, cutoff_frequency=15.0)

    print("[3/6] Running baseline FFT compression (10% keep)...")
    t0 = time.time()
    compressed = compress_fft(filtered, keep_percentage=0.10)
    reconstructed = decompress_fft(compressed)
    runtime = time.time() - t0

    print("[4/6] Computing metrics...")
    metrics = {
        "method": "fft_10pct_baseline",
        "compression_ratio": round(compression_ratio(filtered, compressed), 2),
        "compressed_bytes": estimate_compressed_size(compressed),
        "mse_vs_filtered": round(mse(filtered, reconstructed), 8),
        "rmse_vs_filtered": round(rmse(filtered, reconstructed), 8),
        "snr_vs_filtered_db": round(snr_db(filtered, reconstructed), 2),
        "mse_vs_clean": round(mse(clean, reconstructed), 8),
        "snr_vs_clean_db": round(snr_db(clean, reconstructed), 2),
        "runtime_seconds": round(runtime, 4),
        "metadata": metadata,
    }

    print("[5/6] Saving artifacts...")
    save_signal(output_dir / "original_clean.csv", time_arr, clean)
    save_signal(output_dir / "working_filtered.csv", time_arr, filtered)
    save_signal(output_dir / "reconstructed.csv", time_arr[: len(reconstructed)], reconstructed)

    with open(output_dir / "compressed_meta.json", "w") as f:
        json.dump({
            "method": compressed["method"],
            "n_coeffs_kept": int(compressed["n_coeffs_kept"]),
            "n_coeffs_total": int(compressed["n_coeffs_total"]),
            "keep_percentage": compressed["keep_percentage"],
        }, f, indent=2)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    generate_analysis_report(clean, fs, time_arr, output_dir / "signal_analysis_report.json")

    plot_signal_overview(
        time_arr, clean, fs, "Clean Original",
        save_path=str(output_dir / "plot_original.png"),
    )
    plot_comparison(
        time_arr, filtered, reconstructed,
        "Filtered (pre-compression)", "Reconstructed",
        "Baseline FFT 10% Reconstruction",
        save_path=str(output_dir / "plot_comparison.png"),
    )
    plot_error_analysis(
        time_arr, filtered, reconstructed,
        save_path=str(output_dir / "plot_error.png"),
    )

    print("[6/6] Done.")
    print(json.dumps(metrics, indent=2))
    print(f"\nArtifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()
