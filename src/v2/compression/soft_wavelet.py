"""Soft-threshold wavelet compression (v2, refined).

Uses Donoho-Johnstone universal threshold with noise estimated from the
finest detail subband (standard practice). Threshold is tuned to hit a
target sparsity level for predictable compression ratios.
"""

from typing import Dict, Optional

import numpy as np
import pywt


def _estimate_sigma_from_detail(coeffs_list: list) -> float:
    """
    Estimate noise std from finest detail subband via MAD.

    Standard wavelet denoising practice: detail coefficients at the
    highest scale are mostly noise-dominated.
    """
    detail = coeffs_list[-1]
    sigma = float(np.median(np.abs(detail)) / 0.6745)
    if sigma == 0:
        sigma = float(np.std(detail))
    return sigma


def _universal_threshold(n: int, sigma: float) -> float:
    """Donoho-Johnstone universal threshold: sigma * sqrt(2 * log(n))."""
    return sigma * np.sqrt(2.0 * np.log(max(n, 2)))


def _soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft thresholding: sign(x) * max(|x| - T, 0)."""
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)


def _threshold_for_sparsity(magnitudes: np.ndarray, keep_percentage: float) -> float:
    """Find threshold that leaves approximately keep_percentage non-zero."""
    n_keep = max(1, int(len(magnitudes) * keep_percentage))
    sorted_mag = np.sort(magnitudes)
    return float(sorted_mag[-n_keep])


def compress_soft_wavelet(
    signal: np.ndarray,
    wavelet: str = "db4",
    keep_percentage: float = 0.1,
    level: int = None,
) -> Dict:
    """
    Compress using wavelet decomposition with soft thresholding.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    wavelet : str
        Wavelet family ('haar', 'db4', 'db8', etc.).
    keep_percentage : float
        Target fraction of non-zero coefficients after thresholding.
    level : int, optional
        Decomposition level.

    Returns
    -------
    dict
        Compressed wavelet representation.
    """
    keep_percentage = np.clip(keep_percentage, 0.001, 1.0)
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    if level is None:
        level = min(max_level, 6)
    level = min(level, max_level)

    coeffs_list = pywt.wavedec(signal, wavelet, level=level)
    coeffs, coeff_slices = pywt.coeffs_to_array(coeffs_list)

    sigma = _estimate_sigma_from_detail(coeffs_list)
    universal = _universal_threshold(len(coeffs), sigma)
    sparsity_threshold = _threshold_for_sparsity(np.abs(coeffs), keep_percentage)
    threshold = min(universal, sparsity_threshold)

    thresholded = _soft_threshold(coeffs, threshold)
    kept_mask = np.abs(thresholded) > 0

    return {
        "method": "soft_wavelet",
        "version": 2,
        "wavelet": wavelet,
        "n_samples": len(signal),
        "keep_percentage": keep_percentage,
        "threshold": float(threshold),
        "coeffs": thresholded,
        "coeff_slices": coeff_slices,
        "kept_mask": kept_mask,
        "n_coeffs_total": len(coeffs),
        "n_coeffs_kept": int(np.sum(kept_mask)),
    }


def decompress_soft_wavelet(compressed: Dict) -> np.ndarray:
    """
    Reconstruct signal from soft-thresholded wavelet coefficients.

    Parameters
    ----------
    compressed : dict
        Output of compress_soft_wavelet().

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    coeffs_list = pywt.array_to_coeffs(
        compressed["coeffs"], compressed["coeff_slices"], output_format="wavedec"
    )
    reconstructed = pywt.waverec(coeffs_list, compressed["wavelet"])
    n = compressed["n_samples"]
    return reconstructed[:n]
