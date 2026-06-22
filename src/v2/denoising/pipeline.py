"""Multi-stage denoising pipeline (v2).

Applies specialized filters in sequence, each targeting a different
noise type found in rocket telemetry:

    Impulse spikes  → Median filter
    Gaussian noise  → Spectral subtraction
    Residual noise  → Wiener filter
    Baseline drift  → Polynomial detrending
    High-frequency  → Adaptive Butterworth low-pass
"""

from typing import Dict, Tuple

import numpy as np
from scipy import signal as sp_signal

from .drift_correction import correct_sensor_drift
from .median_filter import remove_impulse_median
from .spectral_subtraction import spectral_subtract
from .wiener import wiener_denoise


def _adaptive_cutoff(signal: np.ndarray, sampling_frequency: float) -> float:
    """
    Estimate optimal low-pass cutoff from spectral rolloff.

    Finds the frequency below which 95% of signal energy is contained.
    """
    spectrum = np.abs(np.fft.rfft(signal))
    energy = spectrum ** 2
    cumulative = np.cumsum(energy) / (np.sum(energy) + 1e-12)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_frequency)
    idx = int(np.searchsorted(cumulative, 0.95))
    idx = min(idx, len(freqs) - 1)
    cutoff = float(freqs[idx])
    # Clamp to reasonable range for rocket telemetry
    return np.clip(cutoff, 5.0, sampling_frequency * 0.4)


def denoise_multistage(
    signal: np.ndarray,
    sampling_frequency: float,
    median_kernel: int = 5,
    spectral_alpha: float = 2.0,
    apply_drift_correction: bool = True,
    apply_adaptive_lowpass: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Run the full v2 multi-stage denoising pipeline.

    Parameters
    ----------
    signal : np.ndarray
        Noisy input signal.
    sampling_frequency : float
        Sampling rate in Hz.
    median_kernel : int
        Median filter window for impulse removal.
    spectral_alpha : float
        Spectral subtraction over-subtraction factor.
    apply_drift_correction : bool
        Whether to apply polynomial detrending.
    apply_adaptive_lowpass : bool
        Whether to apply adaptive Butterworth at end.

    Returns
    -------
    denoised : np.ndarray
        Multi-stage denoised signal.
    stage_info : dict
        Metadata about each stage applied.
    """
    stages_applied = []

    # Stage 1: Remove impulse spikes
    current = remove_impulse_median(signal, kernel_size=median_kernel)
    stages_applied.append("median_impulse")

    # Stage 2: Spectral subtraction for Gaussian noise
    current = spectral_subtract(current, sampling_frequency, alpha=spectral_alpha)
    stages_applied.append("spectral_subtraction")

    # Stage 3: Wiener filter for residual noise
    current = wiener_denoise(current, sampling_frequency)
    stages_applied.append("wiener")

    # Stage 4: Drift correction
    if apply_drift_correction:
        current = correct_sensor_drift(current, order=2)
        stages_applied.append("drift_correction")

    # Stage 5: Adaptive low-pass
    cutoff = None
    if apply_adaptive_lowpass:
        cutoff = _adaptive_cutoff(current, sampling_frequency)
        nyquist = sampling_frequency / 2.0
        normalized = min(cutoff, nyquist * 0.99) / nyquist
        b, a = sp_signal.butter(4, normalized, btype="low")
        current = sp_signal.filtfilt(b, a, current)
        stages_applied.append("adaptive_butterworth")

    return current, {
        "stages": stages_applied,
        "adaptive_cutoff_hz": cutoff,
        "version": 2,
    }
