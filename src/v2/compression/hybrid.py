"""Hybrid FFT + residual wavelet compression (v2, refined).

Two-stage compression with optional target-ratio budget splitting between
the FFT energy stage and wavelet residual stage.
"""

from typing import Dict, Optional

import numpy as np

from .adaptive_fft import compress_adaptive_fft, decompress_adaptive_fft
from .soft_wavelet import compress_soft_wavelet, decompress_soft_wavelet


def compress_hybrid(
    signal: np.ndarray,
    energy_keep_fraction: float = 0.85,
    residual_keep_percentage: float = 0.15,
    wavelet: str = "db4",
    target_ratio: Optional[float] = None,
) -> Dict:
    """
    Two-stage hybrid compression: adaptive FFT + residual wavelet.

    When target_ratio is set, adjusts max_keep_fraction and residual
    sparsity to approximate the desired overall compression ratio.

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
    target_ratio : float, optional
        Desired overall compression ratio (auto-tunes internal params).

    Returns
    -------
    dict
        Combined compressed representation.
    """
    max_keep = 0.15
    residual_rate = residual_keep_percentage

    if target_ratio is not None and target_ratio > 1:
        # Split budget: ~70% FFT, ~30% wavelet residual
        total_coeffs = len(np.fft.rfft(signal))
        target_kept = max(2, int(total_coeffs * 2 / target_ratio))
        fft_budget = int(target_kept * 0.7)
        max_keep = np.clip(fft_budget / total_coeffs, 0.02, 0.30)
        wavelet_budget = max(1, target_kept - fft_budget)
        residual_rate = np.clip(wavelet_budget / total_coeffs, 0.005, 0.25)

    fft_compressed = compress_adaptive_fft(
        signal, energy_keep_fraction, max_keep_fraction=max_keep
    )
    fft_recon = decompress_adaptive_fft(fft_compressed)
    residual = signal - fft_recon

    wavelet_compressed = compress_soft_wavelet(
        residual, wavelet=wavelet, keep_percentage=residual_rate
    )

    return {
        "method": "hybrid",
        "version": 2,
        "n_samples": len(signal),
        "target_ratio": target_ratio,
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
