"""Energy-adaptive FFT compression (v2, refined).

Retains coefficients until a target spectral energy fraction is captured,
with safeguards: DC/low-frequency preservation, band weighting for rocket
dynamics, and a maximum coefficient cap for predictable compression ratios.
"""

from typing import Dict, Optional

import numpy as np


def compress_adaptive_fft(
    signal: np.ndarray,
    energy_keep_fraction: float = 0.90,
    max_keep_fraction: float = 0.15,
    low_freq_bins: int = 5,
    band_weight: float = 2.0,
) -> Dict:
    """
    Compress by retaining weighted FFT coefficients until target energy is captured.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal.
    energy_keep_fraction : float
        Fraction of total weighted spectral energy to preserve.
    max_keep_fraction : float
        Maximum fraction of coefficients to retain (caps compression ratio).
    low_freq_bins : int
        Number of lowest frequency bins always retained (DC + baseline).
    band_weight : float
        Weight multiplier for low-frequency bins (rocket dynamics band).

    Returns
    -------
    dict
        Compressed sparse spectrum with energy metadata.
    """
    energy_keep_fraction = np.clip(energy_keep_fraction, 0.5, 0.999)
    max_keep_fraction = np.clip(max_keep_fraction, 0.01, 1.0)
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    n_coeffs = len(spectrum)

    magnitudes = np.abs(spectrum)
    energy = magnitudes ** 2
    total_energy = np.sum(energy)

    if total_energy == 0:
        return _empty_result(n, n_coeffs, energy_keep_fraction)

    # Weight low-frequency bins more heavily (rocket dynamics live below ~15 Hz
    # at typical 200 Hz sampling — roughly first n/27 bins)
    weights = np.ones(n_coeffs)
    lf = min(low_freq_bins, n_coeffs)
    weights[:lf] = band_weight
    weighted_energy = energy * weights

    # Always retain DC and lowest bins
    mandatory = set(range(lf))

    # Greedy selection by weighted energy until target reached
    sorted_idx = np.argsort(weighted_energy)[::-1]
    cumulative = np.cumsum(weighted_energy[sorted_idx])
    target_energy = energy_keep_fraction * np.sum(weighted_energy)
    n_energy = int(np.searchsorted(cumulative, target_energy)) + 1

    # Cap total kept coefficients
    max_keep = max(lf, int(n_coeffs * max_keep_fraction))
    n_keep = min(n_energy, max_keep, n_coeffs)

    kept_set = set(sorted_idx[:n_keep]) | mandatory
    kept_indices = np.sort(list(kept_set))
    actual_energy = float(np.sum(energy[kept_indices]) / total_energy)

    return {
        "method": "adaptive_fft",
        "version": 2,
        "n_samples": n,
        "energy_keep_fraction": energy_keep_fraction,
        "max_keep_fraction": max_keep_fraction,
        "indices": kept_indices,
        "values": spectrum[kept_indices],
        "n_coeffs_total": n_coeffs,
        "n_coeffs_kept": len(kept_indices),
        "actual_energy_fraction": actual_energy,
    }


def _empty_result(n: int, n_coeffs: int, energy_keep_fraction: float) -> Dict:
    """Return minimal compressed representation for zero signal."""
    return {
        "method": "adaptive_fft",
        "version": 2,
        "n_samples": n,
        "energy_keep_fraction": energy_keep_fraction,
        "max_keep_fraction": 0.15,
        "indices": np.array([0]),
        "values": np.zeros(1, dtype=np.complex128),
        "n_coeffs_total": n_coeffs,
        "n_coeffs_kept": 1,
        "actual_energy_fraction": 1.0,
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
