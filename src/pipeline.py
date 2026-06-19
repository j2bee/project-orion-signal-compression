#!/usr/bin/env python3
"""
Project Orion — Automated Signal Compression Pipeline.

End-to-end experiment runner:
    Load → Noise → Filter → Compress → Reconstruct → Metrics → Save

Usage:
    python src/pipeline.py --input data/raw/synthetic_rocket.csv \\
        --compression fft --compression_rate 0.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.compression.fft_compression import compress_fft
from src.compression.wavelet_compression import compress_wavelet
from src.compression.quantization import compress_quantization
from src.data_loader.loader import load_signal, save_signal
from src.data_loader.validation import validate_signal
from src.metrics.compression_ratio import compression_ratio, format_metrics_report
from src.metrics.mse import mse, rmse
from src.metrics.snr import snr_db
from src.preprocessing.filtering import butterworth_lowpass
from src.preprocessing.noise import apply_all_noise
from src.reconstruction.reconstruction import reconstruct
from src.visualization.plots import plot_comparison, plot_pipeline_stages


def compress_signal(signal: np.ndarray, method: str, rate: float, wavelet: str = "db4") -> dict:
    """
    Dispatch compression to the selected algorithm.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    method : str
        'fft', 'wavelet', or 'quantization'.
    rate : float
        Compression parameter (keep_percentage for fft/wavelet, bits for quantization).
    wavelet : str
        Wavelet type for wavelet compression.

    Returns
    -------
    dict
        Compressed representation.
    """
    if method == "fft":
        return compress_fft(signal, keep_percentage=rate)
    elif method == "wavelet":
        return compress_wavelet(signal, wavelet=wavelet, keep_percentage=rate)
    elif method == "quantization":
        bits = int(rate) if rate >= 1 else 8
        return compress_quantization(signal, bits=bits)
    else:
        raise ValueError(f"Unknown compression method: {method}")


def run_pipeline(
    input_path: str,
    compression_method: str = "fft",
    compression_rate: float = 0.1,
    wavelet: str = "db4",
    add_noise: bool = True,
    filter_signal: bool = True,
    output_dir: str = "results",
) -> dict:
    """
    Execute the full signal compression pipeline.

    Parameters
    ----------
    input_path : str
        Path to input signal file.
    compression_method : str
        Compression algorithm: fft, wavelet, or quantization.
    compression_rate : float
        Keep percentage (fft/wavelet) or bit depth (quantization).
    wavelet : str
        Wavelet type for wavelet compression.
    add_noise : bool
        Whether to add realistic noise.
    filter_signal : bool
        Whether to apply Butterworth low-pass filter.
    output_dir : str
        Directory for plots and metrics.

    Returns
    -------
    dict
        Full experiment results including metrics and plot paths.
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    metrics_dir = output_dir / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Load ---
    print(f"[1/7] Loading signal from {input_path}")
    time_arr, signal, metadata = load_signal(input_path)
    is_valid, errors = validate_signal(time_arr, signal)
    if not is_valid:
        raise ValueError(f"Signal validation failed: {errors}")
    fs = metadata["sampling_frequency"]
    print(f"      {metadata['n_samples']} samples, {metadata['duration']:.1f}s, fs={fs:.1f} Hz")

    stages = {"original": signal.copy()}

    # --- Stage 2: Add noise ---
    if add_noise:
        print("[2/7] Adding realistic noise (Gaussian + impulse + drift)")
        noisy, noise_info = apply_all_noise(signal, seed=42)
        stages["noisy"] = noisy
        working_signal = noisy
        save_signal("data/noisy/noisy_signal.csv", time_arr, noisy)
    else:
        print("[2/7] Skipping noise")
        working_signal = signal.copy()
        stages["noisy"] = signal.copy()

    # --- Stage 3: Filter ---
    if filter_signal:
        print("[3/7] Applying Butterworth low-pass filter")
        filtered = butterworth_lowpass(working_signal, fs, cutoff_frequency=15.0)
        stages["filtered"] = filtered
        working_signal = filtered
    else:
        print("[3/7] Skipping filter")
        stages["filtered"] = working_signal.copy()

    # --- Stage 4: Compress ---
    print(f"[4/7] Compressing with {compression_method} (rate={compression_rate})")
    compressed = compress_signal(working_signal, compression_method, compression_rate, wavelet)

    # --- Stage 5: Reconstruct ---
    print("[5/7] Reconstructing signal")
    reconstructed = reconstruct(compressed, method=compression_method)
    stages["reconstructed"] = reconstructed

    # --- Stage 6: Metrics ---
    print("[6/7] Calculating metrics")
    mse_val = mse(working_signal, reconstructed)
    rmse_val = rmse(working_signal, reconstructed)
    snr_val = snr_db(working_signal, reconstructed)
    ratio_val = compression_ratio(working_signal, compressed)

    report = format_metrics_report(
        compression_method, compressed, working_signal, reconstructed,
        mse_val, rmse_val, snr_val,
    )
    print(f"\n{report}\n")

    # --- Stage 7: Save plots and results ---
    print("[7/7] Saving plots and results")
    plot_paths = plot_pipeline_stages(
        time_arr, stages, fs, save_dir=str(plots_dir), prefix="pipeline"
    )

    comparison_path = plots_dir / "pipeline_comparison.png"
    plot_comparison(
        time_arr,
        working_signal,
        reconstructed,
        original_label="Pre-compression",
        processed_label="Reconstructed",
        title=f"{compression_method.upper()} Compression (rate={compression_rate})",
        save_path=str(comparison_path),
    )
    plot_paths["comparison"] = str(comparison_path)

    elapsed = time.time() - start_time
    results = {
        "input": str(input_path),
        "method": compression_method,
        "compression_rate": compression_rate,
        "wavelet": wavelet if compression_method == "wavelet" else None,
        "metrics": {
            "compression_ratio": round(ratio_val, 2),
            "mse": round(mse_val, 6),
            "rmse": round(rmse_val, 6),
            "snr_db": round(snr_val, 2),
        },
        "metadata": metadata,
        "noise_applied": add_noise,
        "filter_applied": filter_signal,
        "execution_time_seconds": round(elapsed, 3),
        "plots": plot_paths,
    }

    report_path = output_dir / "final_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {report_path}")

    # Save reconstructed signal
    save_signal(
        "data/reconstructed/reconstructed_signal.csv",
        time_arr[: len(reconstructed)],
        reconstructed,
    )

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Project Orion Signal Compression Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/synthetic_rocket.csv",
        help="Input signal file path",
    )
    parser.add_argument(
        "--compression", "-c",
        choices=["fft", "wavelet", "quantization"],
        default="fft",
        help="Compression method",
    )
    parser.add_argument(
        "--compression_rate", "-r",
        type=float,
        default=0.1,
        help="Keep percentage (0.01-1.0) for fft/wavelet, or bit depth (8/16) for quantization",
    )
    parser.add_argument(
        "--wavelet", "-w",
        default="db4",
        help="Wavelet type for wavelet compression",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Skip noise addition",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip filtering",
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic rocket data before running",
    )

    args = parser.parse_args()

    if args.generate_data or not Path(args.input).exists():
        from src.generate_synthetic import create_synthetic_dataset
        print("Generating synthetic rocket telemetry...")
        create_synthetic_dataset(args.input)

    results = run_pipeline(
        input_path=args.input,
        compression_method=args.compression,
        compression_rate=args.compression_rate,
        wavelet=args.wavelet,
        add_noise=not args.no_noise,
        filter_signal=not args.no_filter,
        output_dir=args.output,
    )

    print("\n=== Pipeline Complete ===")
    print(f"Compression ratio: {results['metrics']['compression_ratio']}x")
    print(f"MSE:  {results['metrics']['mse']}")
    print(f"RMSE: {results['metrics']['rmse']}")
    print(f"SNR:  {results['metrics']['snr_db']} dB")
    print(f"Time: {results['execution_time_seconds']}s")


if __name__ == "__main__":
    main()
