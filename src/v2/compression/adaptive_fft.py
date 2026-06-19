"""Energy-adaptive FFT compression (v2).

Instead of keeping a fixed fraction of coefficients by count, retains
coefficients until a target fraction of total spectral energy is captured.
This preserves perceptually important low-frequency rocket dynamics while
achieving similar compression ratios with better reconstruction quality.
"""

from typing import Dict

import numpy as np


def compress_adaptive_fft(
    signal: np.ndarray, energy_keep_fraction: float = 0.90
) -> Dict:
    """
    Compress by retaining FFT coefficients until target energy is captured.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal.
    energy_keep_fraction : float
        Fraction of total spectral energy to preserve (0.90 = 90% energy).
        Lower values → higher compression, more loss.

    Returns
    -------
    dict
        Compressed sparse spectrum with energy metadata.
    """
    energy_keep_fraction = np.clip(energy_keep_fraction, 0.5, 0.999)
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum)
    energy = magnitudes ** 2
    total_energy = np.sum(energy)

    if total_energy == 0:
        return {
            "method": "adaptive_fft",
            "version": 2,
            "n_samples": n,
            "energy_keep_fraction": energy_keep_fraction,
            "indices": np.array([0]),
            "values": spectrum[:1],
            "n_coeffs_total": len(spectrum),
            "n_coeffs_kept": 1,
            "actual_energy_fraction": 1.0,
        }

    # Sort by descending energy and accumulate until target reached
    sorted_idx = np.argsort(energy)[::-1]
    cumulative = np.cumsum(energy[sorted_idx])
    target_energy = energy_keep_fraction * total_energy
    n_keep = int(np.searchsorted(cumulative, target_energy)) + 1
    n_keep = min(n_keep, len(spectrum))

    kept_indices = np.sort(sorted_idx[:n_keep])
    actual_energy = float(np.sum(energy[kept_indices]) / total_energy)

    return {
        "method": "adaptive_fft",
        "version": 2,
        "n_samples": n,
        "energy_keep_fraction": energy_keep_fraction,
        "indices": kept_indices,
        "values": spectrum[kept_indices],
        "n_coeffs_total": len(spectrum),
        "n_coeffs_kept": n_keep,
        "actual_energy_fraction": actual_energy,
    }


def decompress_adaptive_fft(compressed: Dict) -> np.ndarray:
    """
    Reconstruct signal from energy-adaptive FFT representation.

    Parameters
    ----------
    compressed : dict
        Output of compress_adaptive_fft().

    Returns
    -------
    np.ndarray
        Reconstructed time-domain signal.
    """
    n = compressed["n_samples"]
    n_coeffs = compressed["n_coeffs_total"]
    spectrum = np.zeros(n_coeffs, dtype=np.complex128)
    spectrum[compressed["indices"]] = compressed["values"]
    return np.fft.irfft(spectrum, n=n)
