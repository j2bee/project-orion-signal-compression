"""Importance scoring for adaptive compression allocation.

Computes a per-sample importance mask (0–1) from local signal characteristics.
High-importance regions should retain more coefficients during compression.
"""

from typing import Dict, Tuple

import numpy as np


def compute_importance_mask(
    signal: np.ndarray,
    window_size: int = 101,
    weights: Dict[str, float] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute importance score for each sample as weighted combination of:
    - local variance
    - local energy (RMS²)
    - peak density (local max prominence)
    - derivative magnitude

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_size : int
        Sliding window for local statistics (odd integer).
    weights : dict, optional
        Weights for variance, energy, peaks, derivative. Default equal mix.

    Returns
    -------
    importance : np.ndarray
        Importance mask in [0, 1], same length as signal.
    metadata : dict
        Component maps and weight used.
    """
    if weights is None:
        weights = {"variance": 0.25, "energy": 0.25, "peaks": 0.25, "derivative": 0.25}

    window_size = max(5, window_size | 1)
    half = window_size // 2
    n = len(signal)
    kernel = np.ones(window_size) / window_size

    # Local variance
    local_mean = np.convolve(signal, kernel, mode="same")
    local_var = np.convolve(signal ** 2, kernel, mode="same") - local_mean ** 2
    local_var = np.maximum(local_var, 0)

    # Local energy
    local_energy = np.convolve(signal ** 2, kernel, mode="same")

    # Derivative magnitude (smoothed)
    derivative = np.abs(np.gradient(signal))
    local_deriv = np.convolve(derivative, kernel, mode="same")

    # Peak density: count local maxima in window via rolling comparison
    peak_density = np.zeros(n)
    for i in range(half, n - half):
        seg = signal[i - half : i + half + 1]
        center = seg[half]
        if center >= np.max(seg) * 0.99:
            peak_density[i] = 1.0
    local_peaks = np.convolve(peak_density, kernel, mode="same")

    def _normalize(x: np.ndarray) -> np.ndarray:
        xmin, xmax = np.min(x), np.max(x)
        if xmax - xmin < 1e-12:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    comp_var = _normalize(local_var)
    comp_energy = _normalize(local_energy)
    comp_peaks = _normalize(local_peaks)
    comp_deriv = _normalize(local_deriv)

    importance = (
        weights["variance"] * comp_var
        + weights["energy"] * comp_energy
        + weights["peaks"] * comp_peaks
        + weights["derivative"] * comp_deriv
    )
    importance = np.clip(importance, 0.0, 1.0)

    return importance, {
        "weights": weights,
        "window_size": window_size,
        "components": {
            "variance": comp_var,
            "energy": comp_energy,
            "peaks": comp_peaks,
            "derivative": comp_deriv,
        },
    }


def importance_to_keep_percentage(
    importance: np.ndarray,
    high_keep: float = 0.50,
    low_keep: float = 0.05,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Map importance mask to per-sample keep-percentage for windowed compression.

    Parameters
    ----------
    importance : np.ndarray
        Importance mask [0, 1].
    high_keep : float
        Keep fraction for high-importance regions.
    low_keep : float
        Keep fraction for low-importance regions.
    threshold : float
        Importance above this uses high_keep.

    Returns
    -------
    np.ndarray
        Per-sample keep percentage (interpolated between low and high).
    """
    return np.where(
        importance >= threshold,
        high_keep,
        low_keep + (high_keep - low_keep) * (importance / threshold),
    )
