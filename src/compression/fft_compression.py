"""FFT-based frequency-domain compression.

Removes low-magnitude frequency coefficients to achieve bandwidth reduction.
The inverse FFT reconstructs an approximate time-domain signal.
"""

from typing import Dict, Tuple

import numpy as np


def compress_fft(signal: np.ndarray, keep_percentage: float = 0.1) -> Dict:
    """
    Compress a signal by retaining only the largest FFT magnitude coefficients.

    Pipeline: signal → FFT → threshold small coefficients → store sparse spectrum.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal.
    keep_percentage : float
        Fraction of coefficients to retain (0.01 = 1%, 0.5 = 50%).

    Returns
    -------
    dict
        Compressed representation with indices, values, and metadata.
    """
    keep_percentage = np.clip(keep_percentage, 0.001, 1.0)
    n = len(signal)

    # Compute full complex spectrum
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum)

    # Determine how many coefficients to keep
    n_coeffs = len(spectrum)
    n_keep = max(1, int(n_coeffs * keep_percentage))

    # Select indices of largest magnitude coefficients
    top_indices = np.argsort(magnitudes)[-n_keep:]
    top_indices = np.sort(top_indices)

    compressed = {
        "method": "fft",
        "n_samples": n,
        "keep_percentage": keep_percentage,
        "indices": top_indices,
        "values": spectrum[top_indices],
        "n_coeffs_total": n_coeffs,
        "n_coeffs_kept": n_keep,
    }
    return compressed


def decompress_fft(compressed: Dict) -> np.ndarray:
    """
    Reconstruct signal from sparse FFT representation via inverse FFT.

    Parameters
    ----------
    compressed : dict
        Output of compress_fft().

    Returns
    -------
    np.ndarray
        Reconstructed time-domain signal.
    """
    n = compressed["n_samples"]
    n_coeffs = compressed["n_coeffs_total"]

    # Rebuild full spectrum (zeros for discarded coefficients)
    spectrum = np.zeros(n_coeffs, dtype=np.complex128)
    spectrum[compressed["indices"]] = compressed["values"]

    reconstructed = np.fft.irfft(spectrum, n=n)
    return reconstructed


def compress_fft_sweep(
    signal: np.ndarray,
    keep_percentages: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50),
) -> Dict[float, Dict]:
    """
    Compress at multiple keep-percentage levels for comparison.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    keep_percentages : tuple of float
        Fractions of coefficients to retain.

    Returns
    -------
    dict
        Mapping from keep_percentage to compressed representation.
    """
    return {pct: compress_fft(signal, pct) for pct in keep_percentages}
