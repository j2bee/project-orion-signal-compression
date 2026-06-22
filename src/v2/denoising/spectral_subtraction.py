"""Spectral subtraction for Gaussian noise reduction (v2).

Estimates noise power spectrum from signal edges (assumed quiet regions)
and subtracts it from the full spectrum, reducing broadband noise while
preserving tonal rocket vibration components.
"""

import numpy as np


def _estimate_noise_spectrum(
    signal: np.ndarray, sampling_frequency: float, noise_fraction: float = 0.05
) -> np.ndarray:
    """Estimate noise PSD from first/last segments of the signal."""
    n = len(signal)
    segment_len = max(32, int(n * noise_fraction))
    noise_segments = np.concatenate([signal[:segment_len], signal[-segment_len:]])
    noise_spectrum = np.abs(np.fft.rfft(noise_segments)) ** 2
    # Interpolate to full signal length spectrum bins
    full_len = len(np.fft.rfft(signal))
    if len(noise_spectrum) != full_len:
        x_old = np.linspace(0, 1, len(noise_spectrum))
        x_new = np.linspace(0, 1, full_len)
        noise_spectrum = np.interp(x_new, x_old, noise_spectrum)
    return noise_spectrum


def spectral_subtract(
    signal: np.ndarray,
    sampling_frequency: float,
    alpha: float = 2.0,
    floor: float = 0.01,
) -> np.ndarray:
    """
    Reduce Gaussian noise via spectral subtraction.

    Parameters
    ----------
    signal : np.ndarray
        Noisy input signal.
    sampling_frequency : float
        Sampling rate in Hz.
    alpha : float
        Over-subtraction factor (higher = more aggressive).
    floor : float
        Minimum gain to avoid musical noise artifacts.

    Returns
    -------
    np.ndarray
        Denoised signal.
    """
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2
    noise_power = _estimate_noise_spectrum(signal, sampling_frequency)

    # Compute gain: G = max(1 - alpha * noise/power, floor)
    gain = 1.0 - alpha * noise_power / (power + 1e-12)
    gain = np.maximum(gain, floor)

    cleaned_spectrum = spectrum * gain
    return np.fft.irfft(cleaned_spectrum, n=n)
