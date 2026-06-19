#!/usr/bin/env python3
"""
Project Orion — Automated Signal Compression Pipeline.

End-to-end experiment runner supporting v1 (baseline) and v2 (improved):
    Load → Noise → Filter/Denoise → Compress → Reconstruct → Metrics → Save

Usage:
    # v2 improved pipeline (default)
    python src/pipeline.py --compression adaptive_fft --compression_rate 0.90

    # v1 baseline for comparison
    python src/pipeline.py --version v1 --compression fft --compression_rate 0.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

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


V1_METHODS = ["fft", "wavelet", "quantization"]
V2_METHODS = ["adaptive_fft", "soft_wavelet", "mulaw", "hybrid", "ml"]


def compress_signal_v1(signal: np.ndarray, method: str, rate: float, wavelet: str = "db4") -> dict:
    """v1 compression dispatch (preserved baseline)."""
    if method == "fft":
        return compress_fft(signal, keep_percentage=rate)
    elif method == "wavelet":
        return compress_wavelet(signal, wavelet=wavelet, keep_percentage=rate)
    elif method == "quantization":
        bits = int(rate) if rate >= 1 else 8
        return compress_quantization(signal, bits=bits)
    raise ValueError(f"Unknown v1 method: {method}")


def compress_signal_v2(
    signal: np.ndarray,
    method: str,
    rate: float,
    wavelet: str = "db4",
    ml_epochs: int = 50,
) -> dict:
    """v2 improved compression dispatch."""
    if method == "adaptive_fft":
        from src.v2.compression.adaptive_fft import compress_adaptive_fft
        return compress_adaptive_fft(signal, energy_keep_fraction=rate)
    elif method == "soft_wavelet":
        from src.v2.compression.soft_wavelet import compress_soft_wavelet
        return compress_soft_wavelet(signal, wavelet=wavelet, keep_percentage=rate)
    elif method == "mulaw":
        from src.v2.compression.mulaw_quantization import compress_mulaw
        bits = int(rate) if rate >= 1 else 8
        return compress_mulaw(signal, bits=bits)
    elif method == "hybrid":
        from src.v2.compression.hybrid import compress_hybrid
        return compress_hybrid(signal, energy_keep_fraction=rate, wavelet=wavelet)
    elif method == "ml":
        from src.ml.autoencoder import compress_ml
        latent_dim = int(rate) if rate >= 1 else 32
        return compress_ml(signal, latent_dim=latent_dim, epochs=ml_epochs)
    raise ValueError(f"Unknown v2 method: {method}")


def compress_signal(
    signal: np.ndarray,
    method: str,
    rate: float,
    wavelet: str = "db4",
    version: str = "v2",
    ml_epochs: int = 50,
) -> dict:
    """Dispatch compression to v1 or v2 implementation."""
    if version == "v1":
        return compress_signal_v1(signal, method, rate, wavelet)
    return compress_signal_v2(signal, method, rate, wavelet, ml_epochs)


def run_pipeline(
    input_path: str,
    compression_method: str = "adaptive_fft",
    compression_rate: float = 0.90,
    wavelet: str = "db4",
    add_noise: bool = True,
    filter_signal: bool = True,
    output_dir: str = "results",
    version: str = "v2",
    ml_epochs: int = 50,
) -> dict:
    """
    Execute the full signal compression pipeline.

    Parameters
    ----------
    input_path : str
        Path to input signal file.
    compression_method : str
        Compression algorithm (v1 or v2 methods).
    compression_rate : float
        Method-specific rate parameter.
    wavelet : str
        Wavelet type for wavelet-based methods.
    add_noise : bool
        Whether to add realistic noise.
    filter_signal : bool
        Whether to apply denoising/filtering.
    output_dir : str
        Directory for plots and metrics.
    version : str
        Pipeline version: 'v1' (baseline) or 'v2' (improved, default).
    ml_epochs : int
        Training epochs for ML compression.

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

    print(f"=== Project Orion Pipeline ({version.upper()}) ===")

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

    # --- Stage 3: Filter / Denoise ---
    if filter_signal:
        if version == "v2":
            print("[3/7] Applying v2 multi-stage denoising")
            from src.v2.denoising.pipeline import denoise_multistage
            filtered, denoise_info = denoise_multistage(working_signal, fs)
            stages["filtered"] = filtered
            working_signal = filtered
        else:
            print("[3/7] Applying v1 Butterworth low-pass filter")
            filtered = butterworth_lowpass(working_signal, fs, cutoff_frequency=15.0)
            stages["filtered"] = filtered
            working_signal = filtered
    else:
        print("[3/7] Skipping filter/denoise")
        stages["filtered"] = working_signal.copy()

    # --- Stage 4: Compress ---
    print(f"[4/7] Compressing with {compression_method} ({version}, rate={compression_rate})")
    compressed = compress_signal(
        working_signal, compression_method, compression_rate, wavelet, version, ml_epochs
    )

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
        f"{compression_method} ({version})",
        compressed, working_signal, reconstructed,
        mse_val, rmse_val, snr_val,
    )
    print(f"\n{report}\n")

    # --- Stage 7: Save plots and results ---
    print("[7/7] Saving plots and results")
    plot_paths = plot_pipeline_stages(
        time_arr, stages, fs, save_dir=str(plots_dir), prefix=f"pipeline_{version}"
    )

    comparison_path = plots_dir / f"pipeline_{version}_comparison.png"
    plot_comparison(
        time_arr,
        working_signal,
        reconstructed,
        original_label="Pre-compression",
        processed_label="Reconstructed",
        title=f"{compression_method.upper()} ({version}) rate={compression_rate}",
        save_path=str(comparison_path),
    )
    plot_paths["comparison"] = str(comparison_path)

    elapsed = time.time() - start_time
    results = {
        "version": version,
        "input": str(input_path),
        "method": compression_method,
        "compression_rate": compression_rate,
        "wavelet": wavelet if "wavelet" in compression_method else None,
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
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {report_path}")

    save_signal(
        "data/reconstructed/reconstructed_signal.csv",
        time_arr[: len(reconstructed)],
        reconstructed,
    )

    return results


def main():
    """CLI entry point."""
    all_methods = V1_METHODS + V2_METHODS
    parser = argparse.ArgumentParser(
        description="Project Orion Signal Compression Pipeline (v1 + v2)"
    )
    parser.add_argument("--input", "-i", default="data/raw/synthetic_rocket.csv")
    parser.add_argument(
        "--version", "-V", choices=["v1", "v2"], default="v2",
        help="Pipeline version: v1 (baseline) or v2 (improved, default)",
    )
    parser.add_argument(
        "--compression", "-c", choices=all_methods, default=None,
        help="Compression method (default: fft for v1, adaptive_fft for v2)",
    )
    parser.add_argument(
        "--compression_rate", "-r", type=float, default=None,
        help="Rate parameter (default: 0.1 for v1, 0.90 for v2 adaptive_fft)",
    )
    parser.add_argument("--wavelet", "-w", default="db4")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--output", "-o", default="results")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument(
        "--ml-epochs", type=int, default=50,
        help="Training epochs for ML compression",
    )

    args = parser.parse_args()

    # Set version-appropriate defaults
    if args.compression is None:
        args.compression = "fft" if args.version == "v1" else "adaptive_fft"
    if args.compression_rate is None:
        args.compression_rate = 0.1 if args.version == "v1" else 0.90

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
        version=args.version,
        ml_epochs=args.ml_epochs,
    )

    print(f"\n=== Pipeline Complete ({results['version'].upper()}) ===")
    print(f"Compression ratio: {results['metrics']['compression_ratio']}x")
    print(f"MSE:  {results['metrics']['mse']}")
    print(f"RMSE: {results['metrics']['rmse']}")
    print(f"SNR:  {results['metrics']['snr_db']} dB")
    print(f"Time: {results['execution_time_seconds']}s")


if __name__ == "__main__":
    main()
