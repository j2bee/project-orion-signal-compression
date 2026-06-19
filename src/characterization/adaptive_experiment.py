"""Importance-weighted vs uniform compression experiment."""

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.compression.fft_compression import compress_fft, decompress_fft
from src.importance import compute_importance_mask, importance_to_keep_percentage
from src.metrics.compression_ratio import compression_ratio
from src.metrics.mse import mse
from src.metrics.snr import snr_db


def _compress_windowed_variable(
    signal: np.ndarray,
    keep_per_window: np.ndarray,
    window_size: int,
    step: int,
) -> tuple:
    """Compress signal with per-window keep percentages, overlap-add reconstruct."""
    n = len(signal)
    output = np.zeros(n)
    weight = np.zeros(n)
    total_kept = 0
    total_coeffs = 0

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        chunk = signal[start:end]
        center = start + window_size // 2
        keep_pct = float(keep_per_window[min(center, n - 1)])
        keep_pct = np.clip(keep_pct, 0.01, 1.0)

        compressed = compress_fft(chunk, keep_percentage=keep_pct)
        recon = decompress_fft(compressed)
        total_kept += compressed["n_coeffs_kept"]
        total_coeffs += compressed["n_coeffs_total"]

        hann = np.hanning(window_size)
        output[start:end] += recon * hann
        weight[start:end] += hann

    weight = np.maximum(weight, 1e-8)
    reconstructed = output / weight

    pseudo_compressed = {
        "method": "windowed_variable_fft",
        "n_coeffs_kept": total_kept,
        "n_coeffs_total": total_coeffs,
    }
    return reconstructed, pseudo_compressed


def run_adaptive_compression_experiment(
    signal: np.ndarray,
    window_size: int = 512,
    uniform_keep: float = 0.10,
    high_keep: float = 0.50,
    low_keep: float = 0.05,
    output_path: str = "results/characterization/adaptive_experiment.json",
) -> Dict:
    """
    Compare uniform compression vs importance-weighted compression.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_size : int
        Window size for local compression.
    uniform_keep : float
        Keep percentage for uniform baseline.
    high_keep : float
        Keep percentage for high-importance windows.
    low_keep : float
        Keep percentage for low-importance windows.
    output_path : str
        JSON results path.

    Returns
    -------
    dict
        Comparison metrics for both methods.
    """
    step = window_size // 2
    n = len(signal)

    # Uniform baseline
    uniform_keep_arr = np.full(n, uniform_keep)
    recon_uniform, comp_uniform = _compress_windowed_variable(
        signal, uniform_keep_arr, window_size, step
    )

    # Importance-weighted
    importance, _ = compute_importance_mask(signal, window_size=window_size)
    weighted_keep = importance_to_keep_percentage(importance, high_keep, low_keep)
    recon_weighted, comp_weighted = _compress_windowed_variable(
        signal, weighted_keep, window_size, step
    )

    results = {
        "uniform": {
            "keep_percentage": uniform_keep,
            "mse": round(mse(signal, recon_uniform), 8),
            "snr_db": round(snr_db(signal, recon_uniform), 2),
            "compression_ratio": round(compression_ratio(signal, comp_uniform), 2),
        },
        "importance_weighted": {
            "high_keep": high_keep,
            "low_keep": low_keep,
            "mse": round(mse(signal, recon_weighted), 8),
            "snr_db": round(snr_db(signal, recon_weighted), 2),
            "compression_ratio": round(compression_ratio(signal, comp_weighted), 2),
        },
        "improvement": {
            "snr_gain_db": round(
                snr_db(signal, recon_weighted) - snr_db(signal, recon_uniform), 2
            ),
            "mse_reduction_factor": round(
                mse(signal, recon_uniform) / max(mse(signal, recon_weighted), 1e-12), 2
            ),
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
