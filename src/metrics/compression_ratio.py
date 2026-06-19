"""Compression ratio estimation."""

from typing import Dict

import numpy as np


def estimate_compressed_size(compressed: Dict) -> int:
    """
    Estimate the storage size of a compressed representation in bytes.

    Parameters
    ----------
    compressed : dict
        Compressed signal data structure.

    Returns
    -------
    int
        Estimated byte count.
    """
    method = compressed.get("method", "")

    if method == "fft":
        # Complex128 values + int64 indices
        n_kept = compressed["n_coeffs_kept"]
        return n_kept * (16 + 8)

    elif method == "wavelet":
        n_kept = compressed["n_coeffs_kept"]
        return n_kept * 8  # float64 per kept coefficient

    elif method == "quantization":
        bits = compressed["bits"]
        n = compressed["n_samples"]
        return n * bits // 8 + 16  # +16 for min/max metadata

    return 0


def compression_ratio(original: np.ndarray, compressed: Dict) -> float:
    """
    Compute compression ratio as original_size / compressed_size.

    A ratio of 10 means the compressed form is 10× smaller.

    Parameters
    ----------
    original : np.ndarray
        Original signal array.
    compressed : dict
        Compressed representation.

    Returns
    -------
    float
        Compression ratio (≥ 1.0 means compression occurred).
    """
    original_bytes = original.nbytes
    compressed_bytes = estimate_compressed_size(compressed)

    if compressed_bytes == 0:
        return 1.0
    return float(original_bytes / compressed_bytes)


def format_metrics_report(
    method: str,
    compressed: Dict,
    original: np.ndarray,
    reconstructed: np.ndarray,
    mse_val: float,
    rmse_val: float,
    snr_val: float,
) -> str:
    """
    Format a human-readable metrics summary.

    Parameters
    ----------
    method : str
        Compression method name.
    compressed : dict
        Compressed data.
    original, reconstructed : np.ndarray
        Signal arrays.
    mse_val, rmse_val, snr_val : float
        Computed metrics.

    Returns
    -------
    str
        Formatted report string.
    """
    ratio = compression_ratio(original, compressed)
    lines = [
        f"Method: {method}",
        f"Compression: {ratio:.1f}x",
        f"MSE: {mse_val:.6f}",
        f"RMSE: {rmse_val:.6f}",
        f"SNR: {snr_val:.1f} dB",
    ]
    return "\n".join(lines)
