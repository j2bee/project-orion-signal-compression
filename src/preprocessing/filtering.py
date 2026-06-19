"""Signal filtering methods for noise reduction."""

import numpy as np
from scipy import signal as sp_signal


def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a simple moving average filter (low-pass smoothing).

    Uses convolution with a uniform kernel — effective for high-frequency
    noise but blurs sharp transients like launch acceleration spikes.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_size : int
        Number of samples in the averaging window (must be odd for symmetry).

    Returns
    -------
    np.ndarray
        Smoothed signal, same length as input.
    """
    window_size = max(1, int(window_size))
    if window_size == 1:
        return signal.copy()
    kernel = np.ones(window_size) / window_size
    # 'same' mode preserves signal length
    return np.convolve(signal, kernel, mode="same")


def butterworth_lowpass(
    signal: np.ndarray,
    sampling_frequency: float,
    cutoff_frequency: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass IIR filter.

    Butterworth filters have maximally flat passband response, making them
    suitable for preserving low-frequency rocket dynamics while removing
    high-frequency sensor noise.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sampling_frequency : float
        Sampling rate in Hz.
    cutoff_frequency : float
        Cutoff frequency in Hz.
    order : int
        Filter order (higher = sharper rolloff).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyquist = sampling_frequency / 2.0
    if cutoff_frequency >= nyquist:
        cutoff_frequency = nyquist * 0.99
    normalized_cutoff = cutoff_frequency / nyquist
    b, a = sp_signal.butter(order, normalized_cutoff, btype="low")
    # filtfilt applies zero-phase filtering (forward + backward pass)
    return sp_signal.filtfilt(b, a, signal)


def savgol_filter(
    signal: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter for smooth derivative-preserving smoothing.

    Fits local polynomials — better at preserving peak shapes (e.g. launch
    acceleration spike) compared to moving average.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_length : int
        Length of the filter window (must be odd and > polyorder).
    polyorder : int
        Order of the fitted polynomial.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    window_length = max(polyorder + 2, window_length)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(signal):
        window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    return sp_signal.savgol_filter(signal, window_length, polyorder)
