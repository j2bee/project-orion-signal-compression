"""Segmented evaluation metrics for rocket flight phases."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .mse import mse, rmse
from .snr import snr_db


def compute_segment_metrics(
    time: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
    segments: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict]:
    """
    Compute MSE, RMSE, and SNR for each time segment.

    Parameters
    ----------
    time : np.ndarray
        Time axis in seconds.
    original : np.ndarray
        Reference signal.
    reconstructed : np.ndarray
        Reconstructed signal.
    segments : dict
        Phase name → (start_s, end_s).

    Returns
    -------
    dict
        Phase name → {mse, rmse, snr_db, n_samples}.
    """
    results = {}
    n = min(len(original), len(reconstructed), len(time))

    for name, (start, end) in segments.items():
        mask = (time[:n] >= start) & (time[:n] < end)
        if not np.any(mask):
            continue
        orig_seg = original[:n][mask]
        recon_seg = reconstructed[:n][mask]
        results[name] = {
            "mse": round(mse(orig_seg, recon_seg), 8),
            "rmse": round(rmse(orig_seg, recon_seg), 8),
            "snr_db": round(snr_db(orig_seg, recon_seg), 2),
            "n_samples": int(np.sum(mask)),
        }
    return results


def default_rocket_segments(duration: float = 100.0) -> Dict[str, Tuple[float, float]]:
    """Standard rocket phase boundaries for segmented metrics."""
    return {
        "pre_launch": (0.0, 10.0),
        "launch": (10.0, 20.0),
        "flight": (20.0, 90.0),
        "descent": (90.0, duration),
    }
