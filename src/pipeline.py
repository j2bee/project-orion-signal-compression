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
from src.metrics.segmented import compute_segment_metrics, default_rocket_segments
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
    ml_epochs: int = 80,
    residual_rate: float = 0.15,
    target_ratio: float = None,
    ml_checkpoint: str = None,
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
        return compress_hybrid(
            signal,
            energy_keep_fraction=rate,
            residual_keep_percentage=residual_rate,
            wavelet=wavelet,
            target_ratio=target_ratio,
        )
    elif method == "ml":
        from src.ml.autoencoder import compress_ml, DEFAULT_CHECKPOINT
        latent_dim = int(rate) if rate >= 1 else 32
        ckpt = ml_checkpoint or DEFAULT_CHECKPOINT
        return compress_ml(signal, latent_dim=latent_dim, epochs=ml_epochs, checkpoint_path=ckpt)
    raise ValueError(f"Unknown v2 method: {method}")


def compress_signal(
    signal: np.ndarray,
    method: str,
    rate: float,
    wavelet: str = "db4",
    version: str = "v2",
    ml_epochs: int = 80,
    residual_rate: float = 0.15,
    target_ratio: float = None,
    ml_checkpoint: str = None,
) -> dict:
    """Dispatch compression to v1 or v2 implementation."""
    if version == "v1":
        return compress_signal_v1(signal, method, rate, wavelet)
    return compress_signal_v2(
        signal, method, rate, wavelet, ml_epochs, residual_rate, target_ratio, ml_checkpoint
    )


def run_pipeline(
    input_path: str,
    compression_method: str = "hybrid",
    compression_rate: float = 0.85,
    wavelet: str = "db4",
    add_noise: bool = True,
    filter_signal: bool = True,
    output_dir: str = "results",
    version: str = "v2",
    ml_epochs: int = 80,
    residual_rate: float = 0.15,
    target_ratio: float = None,
    ml_checkpoint: str = None,
    skip_plots: bool = False,
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
        working_signal, compression_method, compression_rate, wavelet, version,
        ml_epochs, residual_rate, target_ratio, ml_checkpoint,
    )

    # --- Stage 5: Reconstruct ---
    print("[5/7] Reconstructing signal")
    reconstructed = reconstruct(compressed, method=compression_method)
    stages["reconstructed"] = reconstructed

    # --- Stage 6: Metrics ---
    print("[6/7] Calculating metrics")
    # Metrics vs filtered (pre-compression) input
    mse_val = mse(working_signal, reconstructed)
    rmse_val = rmse(working_signal, reconstructed)
    snr_val = snr_db(working_signal, reconstructed)
    ratio_val = compression_ratio(working_signal, compressed)

    # Metrics vs clean original (end-to-end fidelity)
    mse_clean = mse(signal, reconstructed)
    snr_clean = snr_db(signal, reconstructed)

    # Per-phase segmented metrics vs clean original
    segments = default_rocket_segments(metadata["duration"])
    segment_metrics = compute_segment_metrics(
        time_arr, signal, reconstructed, segments
    )

    report = format_metrics_report(
        f"{compression_method} ({version})",
        compressed, working_signal, reconstructed,
        mse_val, rmse_val, snr_val,
    )
    print(f"\n{report}")
    print(f"  vs clean original: MSE={mse_clean:.6f}, SNR={snr_clean:.1f} dB\n")
    if segment_metrics:
        print("  Segment SNR (vs clean):")
        for phase, sm in segment_metrics.items():
            print(f"    {phase:12s}  SNR={sm['snr_db']:6.1f} dB  MSE={sm['mse']:.6f}")
        print()

    # --- Stage 7: Save plots and results ---
    plot_paths = {}
    if not skip_plots:
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
    else:
        print("[7/7] Skipping plots (--skip-plots)")

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
            "vs_clean": {
                "mse": round(mse_clean, 6),
                "snr_db": round(snr_clean, 2),
            },
            "segments": segment_metrics,
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
        help="Compression method (default: fft for v1, hybrid for v2)",
    )
    parser.add_argument(
        "--compression_rate", "-r", type=float, default=None,
        help="Rate parameter (default: 0.1 for v1, 0.85 for v2 hybrid energy fraction)",
    )
    parser.add_argument(
        "--residual-rate", type=float, default=0.15,
        help="Wavelet residual keep fraction for hybrid compression",
    )
    parser.add_argument(
        "--target-ratio", type=float, default=None,
        help="Target compression ratio for hybrid (auto-tunes internal params)",
    )
    parser.add_argument("--wavelet", "-w", default="db4")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--output", "-o", default="results")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--ml-epochs", type=int, default=80,
        help="Training epochs for ML compression",
    )
    parser.add_argument(
        "--ml-checkpoint", type=str, default=None,
        help="Path for ML model checkpoint weights",
    )

    args = parser.parse_args()

    # Set version-appropriate defaults
    if args.compression is None:
        args.compression = "fft" if args.version == "v1" else "hybrid"
    if args.compression_rate is None:
        args.compression_rate = 0.1 if args.version == "v1" else 0.85

    if args.generate_data or not Path(args.input).exists():
        from src.generate_synthetic import create_synthetic_dataset
        print("Generating clean synthetic rocket telemetry...")
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
        residual_rate=args.residual_rate,
        target_ratio=args.target_ratio,
        ml_checkpoint=args.ml_checkpoint,
        skip_plots=args.skip_plots,
    )

    print(f"\n=== Pipeline Complete ({results['version'].upper()}) ===")
    print(f"Compression ratio: {results['metrics']['compression_ratio']}x")
    print(f"MSE:  {results['metrics']['mse']}")
    print(f"RMSE: {results['metrics']['rmse']}")
    print(f"SNR:  {results['metrics']['snr_db']} dB")
    print(f"Time: {results['execution_time_seconds']}s")


if __name__ == "__main__":
    main()
