"""Soft-threshold wavelet compression (v2).

Uses Donoho-Johnstone universal threshold with soft shrinkage instead of
v1's hard keep-percentage cutoff. Soft thresholding reduces artifacts
and improves SNR for the same number of non-zero coefficients.
"""

from typing import Dict, Optional

import numpy as np
import pywt


def _universal_threshold(coeffs: np.ndarray, sigma: Optional[float] = None) -> float:
    """
    Compute Donoho-Johnstone universal threshold: sigma * sqrt(2 * log(n)).

    Parameters
    ----------
    coeffs : np.ndarray
        Wavelet coefficients.
    sigma : float, optional
        Noise std estimate. Uses MAD of finest detail if None.

    Returns
    -------
    float
        Threshold value.
    """
    n = len(coeffs)
    if sigma is None:
        # Median absolute deviation estimate (robust to outliers)
        sigma = float(np.median(np.abs(coeffs)) / 0.6745)
        if sigma == 0:
            sigma = float(np.std(coeffs))
    return sigma * np.sqrt(2.0 * np.log(max(n, 2)))


def _soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft thresholding: sign(x) * max(|x| - T, 0)."""
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)


def compress_soft_wavelet(
    signal: np.ndarray,
    wavelet: str = "db4",
    keep_percentage: float = 0.1,
    level: int = None,
) -> Dict:
    """
    Compress using wavelet decomposition with soft thresholding.

    Threshold is scaled so approximately `keep_percentage` of coefficients
    remain non-zero after soft shrinkage.

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

    # Scale universal threshold to hit target sparsity
    base_threshold = _universal_threshold(coeffs)
    magnitudes = np.abs(coeffs)
    sorted_mag = np.sort(magnitudes)
    n_keep = max(1, int(len(coeffs) * keep_percentage))
    target_mag = sorted_mag[-n_keep]
    threshold = min(base_threshold, target_mag)

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
