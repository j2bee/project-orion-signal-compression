"""Signal-to-Noise Ratio metric."""

import numpy as np


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio in decibels.

    SNR = 10 * log10(signal_power / noise_power)

    where noise = original - reconstructed.

    Higher is better (more signal energy relative to reconstruction error).

    Parameters
    ----------
    original : np.ndarray
        Reference signal.
    reconstructed : np.ndarray
        Reconstructed signal.

    Returns
    -------
    float
        SNR in dB. Returns inf if noise power is zero.
    """
    n = min(len(original), len(reconstructed))
    signal = original[:n]
    noise = signal - reconstructed[:n]

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")

    return float(10.0 * np.log10(signal_power / noise_power))
