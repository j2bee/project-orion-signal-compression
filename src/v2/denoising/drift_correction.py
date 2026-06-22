"""Sensor drift correction via polynomial detrending (v2).

Removes slow baseline drift by fitting and subtracting a low-order
polynomial, preserving high-frequency rocket dynamics.
"""

import numpy as np


def correct_sensor_drift(signal: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Remove sensor drift by subtracting a polynomial trend.

    Parameters
    ----------
    signal : np.ndarray
        Signal with superimposed drift.
    order : int
        Polynomial order for detrending (2 = quadratic).

    Returns
    -------
    np.ndarray
        Drift-corrected signal.
    """
    n = len(signal)
    t = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(t, signal, order)
    trend = np.polyval(coeffs, t)
    return signal - trend
