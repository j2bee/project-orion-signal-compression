"""Wavelet-based compression using PyWavelets.

Wavelets capture both time and frequency localization, making them effective
for non-stationary rocket telemetry with transient events (launch spike).
"""

from typing import Dict, Tuple

import numpy as np
import pywt


def wavelet_transform(
    signal: np.ndarray, wavelet: str = "db4", level: int = None
) -> Tuple[np.ndarray, list]:
    """
    Perform discrete wavelet decomposition.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    wavelet : str
        Wavelet family name (e.g. 'haar', 'db4', 'db8').
    level : int, optional
        Decomposition level. Auto-selected if None.

    Returns
    -------
    coeffs : np.ndarray
        Flattened concatenation of all coefficient arrays.
    coeff_slices : list
        Slice objects to reconstruct the coefficient list structure.
    """
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    if level is None:
        level = min(max_level, 6)
    level = min(level, max_level)

    coeffs_list = pywt.wavedec(signal, wavelet, level=level)
    coeffs, coeff_slices = pywt.coeffs_to_array(coeffs_list)
    return coeffs, coeff_slices


def threshold_coefficients(
    coeffs: np.ndarray, keep_percentage: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero out small-magnitude wavelet coefficients (hard thresholding).

    Parameters
    ----------
    coeffs : np.ndarray
        Flattened wavelet coefficients.
    keep_percentage : float
        Fraction of coefficients to retain.

    Returns
    -------
    thresholded : np.ndarray
        Coefficients with small values zeroed.
    kept_mask : np.ndarray
        Boolean mask of retained coefficients.
    """
    keep_percentage = np.clip(keep_percentage, 0.001, 1.0)
    magnitudes = np.abs(coeffs)
    n_keep = max(1, int(len(coeffs) * keep_percentage))
    threshold = np.sort(magnitudes)[-n_keep]

    kept_mask = magnitudes >= threshold
    thresholded = coeffs * kept_mask
    return thresholded, kept_mask


def inverse_wavelet(
    thresholded_coeffs: np.ndarray,
    coeff_slices: list,
    wavelet: str = "db4",
    n_samples: int = None,
) -> np.ndarray:
    """
    Reconstruct signal from thresholded wavelet coefficients.

    Parameters
    ----------
    thresholded_coeffs : np.ndarray
        Thresholded flattened coefficients.
    coeff_slices : list
        Slice structure from wavelet_transform().
    wavelet : str
        Wavelet name used in decomposition.
    n_samples : int, optional
        Original signal length for trimming.

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    coeffs_list = pywt.array_to_coeffs(thresholded_coeffs, coeff_slices, output_format="wavedec")
    reconstructed = pywt.waverec(coeffs_list, wavelet)
    if n_samples is not None:
        reconstructed = reconstructed[:n_samples]
    return reconstructed


def compress_wavelet(
    signal: np.ndarray,
    wavelet: str = "db4",
    keep_percentage: float = 0.1,
    level: int = None,
) -> Dict:
    """
    Compress signal using wavelet decomposition and coefficient thresholding.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    wavelet : str
        Wavelet type ('haar', 'db4', 'db8', etc.).
    keep_percentage : float
        Fraction of coefficients to retain.
    level : int, optional
        Decomposition level.

    Returns
    -------
    dict
        Compressed wavelet representation.
    """
    coeffs, coeff_slices = wavelet_transform(signal, wavelet, level)
    thresholded, kept_mask = threshold_coefficients(coeffs, keep_percentage)

    return {
        "method": "wavelet",
        "wavelet": wavelet,
        "n_samples": len(signal),
        "keep_percentage": keep_percentage,
        "coeffs": thresholded,
        "coeff_slices": coeff_slices,
        "kept_mask": kept_mask,
        "n_coeffs_total": len(coeffs),
        "n_coeffs_kept": int(np.sum(kept_mask)),
    }


def decompress_wavelet(compressed: Dict) -> np.ndarray:
    """
    Reconstruct signal from wavelet compressed representation.

    Parameters
    ----------
    compressed : dict
        Output of compress_wavelet().

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    return inverse_wavelet(
        compressed["coeffs"],
        compressed["coeff_slices"],
        wavelet=compressed["wavelet"],
        n_samples=compressed["n_samples"],
    )
