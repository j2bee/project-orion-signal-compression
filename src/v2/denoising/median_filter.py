"""Median filter for impulse noise removal (v2).

Impulse spikes are isolated outliers — median filtering replaces each
sample with the median of its neighborhood, effectively removing spikes
while preserving edges better than mean filtering.
"""

import numpy as np
from scipy import ndimage


def remove_impulse_median(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Remove impulse noise using a median filter.

    Parameters
    ----------
    signal : np.ndarray
        Noisy signal with potential impulse spikes.
    kernel_size : int
        Median filter window size (odd integer).

    Returns
    -------
    np.ndarray
        Signal with impulse spikes suppressed.
    """
    kernel_size = max(3, kernel_size | 1)  # ensure odd, >= 3
    return ndimage.median_filter(signal, size=kernel_size)
