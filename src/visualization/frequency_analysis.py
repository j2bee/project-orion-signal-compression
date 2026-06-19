"""Frequency-domain analysis utilities."""

from typing import Dict, Tuple

import numpy as np


def compute_spectrum(
    signal: np.ndarray, sampling_frequency: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute single-sided FFT magnitude and phase spectra.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal.
    sampling_frequency : float
        Sampling rate in Hz.

    Returns
    -------
    frequencies : np.ndarray
        Frequency axis (Hz).
    magnitudes : np.ndarray
        Normalized magnitude spectrum.
    phases : np.ndarray
        Phase spectrum (radians).
    """
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum) / n
    phases = np.angle(spectrum)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_frequency)
    return frequencies, magnitudes, phases


def find_dominant_frequencies(
    signal: np.ndarray,
    sampling_frequency: float,
    n_peaks: int = 5,
) -> Dict:
    """
    Identify the top-N dominant frequency components.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sampling_frequency : float
        Sampling rate in Hz.
    n_peaks : int
        Number of peaks to return.

    Returns
    -------
    dict
        frequencies, magnitudes, and indices of dominant components.
    """
    frequencies, magnitudes, _ = compute_spectrum(signal, sampling_frequency)
    top_indices = np.argsort(magnitudes)[-n_peaks:][::-1]
    return {
        "frequencies": frequencies[top_indices],
        "magnitudes": magnitudes[top_indices],
        "indices": top_indices,
    }


def band_energy(
    signal: np.ndarray, sampling_frequency: float, low_hz: float, high_hz: float
) -> float:
    """
    Compute signal energy in a frequency band.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sampling_frequency : float
        Sampling rate in Hz.
    low_hz, high_hz : float
        Band edges in Hz.

    Returns
    -------
    float
        Energy (sum of squared magnitudes) in the band.
    """
    frequencies, magnitudes, _ = compute_spectrum(signal, sampling_frequency)
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    return float(np.sum(magnitudes[mask] ** 2))
