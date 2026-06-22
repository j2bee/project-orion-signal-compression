"""Time-domain signal analysis."""

from typing import Dict

import numpy as np


def analyze_time_domain(signal: np.ndarray, time: np.ndarray = None) -> Dict:
    """
    Compute time-domain statistics for a signal.

    Parameters
    ----------
    signal : np.ndarray
        Signal amplitude values.
    time : np.ndarray, optional
        Time axis (for duration metadata).

    Returns
    -------
    dict
        mean, variance, std, rms, peak, peak_to_peak, min, max, n_samples.
    """
    return {
        "mean": float(np.mean(signal)),
        "variance": float(np.var(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "peak": float(np.max(np.abs(signal))),
        "peak_to_peak": float(np.max(signal) - np.min(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "n_samples": len(signal),
        "duration_s": float(time[-1] - time[0]) if time is not None and len(time) > 1 else None,
    }
