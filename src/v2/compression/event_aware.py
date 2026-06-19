"""Event-aware compression — allocate more coefficients to high-activity regions.

Rocket telemetry contains transient events (ignition, acceleration spikes)
that must be preserved even at high compression. This module splits the
signal into activity regions and applies tighter compression to quiet segments
while retaining more detail during events.

Research principle: preserve transients over maximizing compression ratio.
"""

from typing import Dict, Tuple

import numpy as np

from .adaptive_fft import compress_adaptive_fft, decompress_adaptive_fft
from .soft_wavelet import compress_soft_wavelet, decompress_soft_wavelet


def _activity_envelope(signal: np.ndarray, window: int = 51) -> np.ndarray:
    """
    Compute local activity envelope via sliding RMS.

    High values indicate regions with transients or vibration.
    """
    window = max(3, window | 1)
    squared = signal ** 2
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(squared, kernel, mode="same"))


def _segment_by_activity(
    signal: np.ndarray,
    envelope: np.ndarray,
    threshold_percentile: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split signal indices into high-activity and low-activity regions.

    Returns
    -------
    high_mask, low_mask : boolean arrays
    """
    threshold = np.percentile(envelope, threshold_percentile)
    high_mask = envelope >= threshold
    low_mask = ~high_mask
    return high_mask, low_mask


def compress_event_aware(
    signal: np.ndarray,
    sampling_frequency: float,
    high_activity_energy: float = 0.95,
    low_activity_energy: float = 0.70,
    high_activity_max_keep: float = 0.25,
    low_activity_max_keep: float = 0.05,
    wavelet: str = "db4",
) -> Dict:
    """
    Compress with region-dependent budgets: more detail during events.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sampling_frequency : float
        Sampling rate (Hz) — used for metadata.
    high_activity_energy : float
        Energy fraction for FFT in high-activity regions.
    low_activity_energy : float
        Energy fraction for FFT in quiet regions.
    high_activity_max_keep : float
        Max coefficient fraction in high-activity regions.
    low_activity_max_keep : float
        Max coefficient fraction in quiet regions.
    wavelet : str
        Wavelet for residual encoding on high-activity segments.

    Returns
    -------
    dict
        Event-aware compressed representation.
    """
    n = len(signal)
    envelope = _activity_envelope(signal)
    high_mask, low_mask = _segment_by_activity(signal, envelope)

    # Build region-specific FFT compressions on masked signals
    high_signal = signal * high_mask.astype(float)
    low_signal = signal * low_mask.astype(float)

    fft_high = compress_adaptive_fft(
        high_signal, high_activity_energy, max_keep_fraction=high_activity_max_keep
    )
    fft_low = compress_adaptive_fft(
        low_signal, low_activity_energy, max_keep_fraction=low_activity_max_keep
    )

    # Residual wavelet on full signal for event edges
    fft_combined_recon = decompress_adaptive_fft(fft_high) + decompress_adaptive_fft(fft_low)
    residual = signal - fft_combined_recon
    wavelet_res = compress_soft_wavelet(residual, wavelet=wavelet, keep_percentage=0.20)

    return {
        "method": "event_aware",
        "version": 2,
        "n_samples": n,
        "sampling_frequency": sampling_frequency,
        "activity_threshold_percentile": 60.0,
        "high_mask_fraction": float(np.mean(high_mask)),
        "fft_high": fft_high,
        "fft_low": fft_low,
        "wavelet_residual": wavelet_res,
        "n_coeffs_kept": (
            fft_high["n_coeffs_kept"] + fft_low["n_coeffs_kept"] + wavelet_res["n_coeffs_kept"]
        ),
        "n_coeffs_total": fft_high["n_coeffs_total"] * 2 + wavelet_res["n_coeffs_total"],
    }


def decompress_event_aware(compressed: Dict) -> np.ndarray:
    """
    Reconstruct event-aware compressed signal.

    Parameters
    ----------
    compressed : dict
        Output of compress_event_aware().

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    fft_high_recon = decompress_adaptive_fft(compressed["fft_high"])
    fft_low_recon = decompress_adaptive_fft(compressed["fft_low"])
    residual_recon = decompress_soft_wavelet(compressed["wavelet_residual"])
    n = compressed["n_samples"]
    return (fft_high_recon + fft_low_recon + residual_recon)[:n]
