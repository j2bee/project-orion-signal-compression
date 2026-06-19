"""Hybrid FFT + residual wavelet compression (v2).

First pass: energy-adaptive FFT captures main spectral content.
Second pass: soft wavelet on the reconstruction residual captures
time-localized errors (e.g. launch spike edges) missed by FFT alone.
"""

from typing import Dict

import numpy as np

from .adaptive_fft import compress_adaptive_fft, decompress_adaptive_fft
from .soft_wavelet import compress_soft_wavelet, decompress_soft_wavelet


def compress_hybrid(
    signal: np.ndarray,
    energy_keep_fraction: float = 0.85,
    residual_keep_percentage: float = 0.15,
    wavelet: str = "db4",
) -> Dict:
    """
    Two-stage hybrid compression: adaptive FFT + residual wavelet.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    energy_keep_fraction : float
        Energy fraction for FFT stage.
    residual_keep_percentage : float
        Coefficient keep fraction for wavelet on residual.
    wavelet : str
        Wavelet for residual stage.

    Returns
    -------
    dict
        Combined compressed representation.
    """
    fft_compressed = compress_adaptive_fft(signal, energy_keep_fraction)
    fft_recon = decompress_adaptive_fft(fft_compressed)
    residual = signal - fft_recon

    wavelet_compressed = compress_soft_wavelet(
        residual, wavelet=wavelet, keep_percentage=residual_keep_percentage
    )

    return {
        "method": "hybrid",
        "version": 2,
        "n_samples": len(signal),
        "fft": fft_compressed,
        "wavelet_residual": wavelet_compressed,
        "n_coeffs_kept": (
            fft_compressed["n_coeffs_kept"] + wavelet_compressed["n_coeffs_kept"]
        ),
        "n_coeffs_total": fft_compressed["n_coeffs_total"] + wavelet_compressed["n_coeffs_total"],
    }


def decompress_hybrid(compressed: Dict) -> np.ndarray:
    """
    Reconstruct hybrid compressed signal.

    Parameters
    ----------
    compressed : dict
        Output of compress_hybrid().

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    fft_recon = decompress_adaptive_fft(compressed["fft"])
    residual_recon = decompress_soft_wavelet(compressed["wavelet_residual"])
    n = compressed["n_samples"]
    return fft_recon[:n] + residual_recon[:n]
