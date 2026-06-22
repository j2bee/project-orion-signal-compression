"""Signal normalization methods."""

from typing import Dict, Tuple

import numpy as np


def min_max_scale(signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Scale signal to [0, 1] using min-max normalization.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    scaled : np.ndarray
        Normalized signal in [0, 1].
    params : dict
        Min and max values for inverse transform.
    """
    sig_min = float(np.min(signal))
    sig_max = float(np.max(signal))
    span = sig_max - sig_min
    if span == 0:
        scaled = np.zeros_like(signal)
    else:
        scaled = (signal - sig_min) / span
    return scaled, {"method": "min_max", "min": sig_min, "max": sig_max}


def inverse_min_max_scale(scaled: np.ndarray, params: Dict) -> np.ndarray:
    """Reverse min-max scaling using stored parameters."""
    span = params["max"] - params["min"]
    return scaled * span + params["min"]


def standardize(signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Standardize signal to zero mean and unit variance (z-score).

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    standardized : np.ndarray
        Zero-mean, unit-variance signal.
    params : dict
        Mean and std for inverse transform.
    """
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    if std == 0:
        standardized = np.zeros_like(signal)
    else:
        standardized = (signal - mean) / std
    return standardized, {"method": "standardize", "mean": mean, "std": std}


def inverse_standardize(standardized: np.ndarray, params: Dict) -> np.ndarray:
    """Reverse z-score standardization using stored parameters."""
    return standardized * params["std"] + params["mean"]
