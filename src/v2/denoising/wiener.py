"""Wiener filter for optimal MMSE denoising (v2).

The Wiener filter minimizes mean-squared error in the frequency domain
by applying gain proportional to signal-to-noise ratio at each frequency.
"""

import numpy as np


def wiener_denoise(
    signal: np.ndarray,
    sampling_frequency: float,
    noise_fraction: float = 0.05,
) -> np.ndarray:
    """
    Apply frequency-domain Wiener filtering.

    Parameters
    ----------
    signal : np.ndarray
        Noisy input signal.
    sampling_frequency : float
        Sampling rate in Hz.
    noise_fraction : float
        Fraction of signal used to estimate noise PSD.

    Returns
    -------
    np.ndarray
        Wiener-filtered signal.
    """
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2

    # Estimate noise from signal edges
    segment_len = max(32, int(n * noise_fraction))
    noise_seg = np.concatenate([signal[:segment_len], signal[-segment_len:]])
    noise_spectrum = np.fft.rfft(noise_seg)
    noise_power = np.abs(noise_spectrum) ** 2
    full_len = len(spectrum)
    if len(noise_power) != full_len:
        x_old = np.linspace(0, 1, len(noise_power))
        x_new = np.linspace(0, 1, full_len)
        noise_power = np.interp(x_new, x_old, noise_power)

    # Wiener gain: |X|^2 / (|X|^2 + |N|^2)
    gain = power / (power + noise_power + 1e-12)
    cleaned = spectrum * gain
    return np.fft.irfft(cleaned, n=n)
