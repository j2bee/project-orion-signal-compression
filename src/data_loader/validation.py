"""Signal validation utilities."""

from typing import Dict, List, Tuple

import numpy as np


def validate_signal(
    time: np.ndarray, signal: np.ndarray
) -> Tuple[bool, List[str]]:
    """
    Validate that time and signal arrays are well-formed.

    Parameters
    ----------
    time : np.ndarray
        Time axis.
    signal : np.ndarray
        Signal values.

    Returns
    -------
    is_valid : bool
        True if all checks pass.
    errors : list of str
        Descriptions of validation failures.
    """
    errors: List[str] = []

    if len(time) != len(signal):
        errors.append(
            f"Length mismatch: time has {len(time)} samples, "
            f"signal has {len(signal)} samples."
        )

    if len(time) == 0:
        errors.append("Signal is empty (zero samples).")

    if np.any(~np.isfinite(time)):
        errors.append("Time array contains NaN or Inf values.")

    if np.any(~np.isfinite(signal)):
        errors.append("Signal array contains NaN or Inf values.")

    if len(time) > 1 and not np.all(np.diff(time) >= 0):
        errors.append("Time array is not monotonically increasing.")

    return len(errors) == 0, errors


def check_sampling_uniformity(time: np.ndarray, tolerance: float = 0.01) -> Dict:
    """
    Check whether the time axis is uniformly sampled.

    Parameters
    ----------
    time : np.ndarray
        Time axis in seconds.
    tolerance : float
        Relative tolerance for dt variation (fraction of mean dt).

    Returns
    -------
    dict
        is_uniform flag, mean dt, and coefficient of variation.
    """
    if len(time) < 2:
        return {"is_uniform": True, "mean_dt": 0.0, "cv": 0.0}

    dt = np.diff(time)
    mean_dt = float(np.mean(dt))
    cv = float(np.std(dt) / mean_dt) if mean_dt > 0 else 0.0

    return {
        "is_uniform": cv <= tolerance,
        "mean_dt": mean_dt,
        "cv": cv,
    }
