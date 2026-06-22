"""Frequency-domain signal analysis via FFT."""

from typing import Dict, List

import numpy as np


def analyze_frequency_domain(
    signal: np.ndarray,
    sampling_frequency: float,
    n_dominant: int = 10,
) -> Dict:
    """
    Compute spectral characteristics using FFT.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal.
    sampling_frequency : float
        Sampling rate in Hz.
    n_dominant : int
        Number of dominant frequencies to report.

    Returns
    -------
    dict
        frequencies, magnitudes, dominant_frequencies, spectral_energy,
        noise_band_energy (high-frequency tail estimate).
    """
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum) / n
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_frequency)
    power = magnitudes ** 2
    total_energy = float(np.sum(power))

    top_idx = np.argsort(magnitudes)[-n_dominant:][::-1]
    dominant = [
        {"frequency_hz": float(frequencies[i]), "magnitude": float(magnitudes[i])}
        for i in top_idx
    ]

    # Estimate noise band as energy above 80% of Nyquist
    nyquist = sampling_frequency / 2.0
    noise_mask = frequencies > 0.8 * nyquist
    noise_energy = float(np.sum(power[noise_mask])) if np.any(noise_mask) else 0.0
    signal_band_mask = frequencies <= 0.8 * nyquist
    signal_energy = float(np.sum(power[signal_band_mask]))

    return {
        "dominant_frequencies": dominant,
        "total_spectral_energy": total_energy,
        "signal_band_energy": signal_energy,
        "noise_band_energy": noise_energy,
        "noise_fraction": noise_energy / total_energy if total_energy > 0 else 0.0,
        "frequencies": frequencies,
        "magnitudes": magnitudes,
    }
