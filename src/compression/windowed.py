"""Sliding-window compression for streaming telemetry research.

Compares whole-signal vs window-based FFT compression to study
local vs global tradeoffs for rocket event preservation.
"""

from typing import Dict, List

import numpy as np

from .fft_compression import compress_fft, decompress_fft


def compress_windowed_fft(
    signal: np.ndarray,
    window_size: int = 1024,
    keep_percentage: float = 0.10,
    overlap: float = 0.5,
) -> Dict:
    """
    Compress signal in overlapping windows via FFT thresholding.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_size : int
        Samples per window.
    keep_percentage : float
        Fraction of FFT coefficients kept per window.
    overlap : float
        Fraction of window overlap (0.5 = 50%).

    Returns
    -------
    dict
        List of per-window compressed representations.
    """
    n = len(signal)
    step = max(1, int(window_size * (1.0 - overlap)))
    windows: List[Dict] = []
    starts: List[int] = []

    for start in range(0, n, step):
        end = min(start + window_size, n)
        chunk = signal[start:end]
        if len(chunk) < 4:
            break
        # Pad short final window
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        comp = compress_fft(chunk, keep_percentage=keep_percentage)
        windows.append(comp)
        starts.append(start)

    return {
        "method": "windowed_fft",
        "n_samples": n,
        "window_size": window_size,
        "keep_percentage": keep_percentage,
        "overlap": overlap,
        "step": step,
        "windows": windows,
        "starts": starts,
        "n_windows": len(windows),
        "n_coeffs_kept": sum(w["n_coeffs_kept"] for w in windows),
        "n_coeffs_total": sum(w["n_coeffs_total"] for w in windows),
    }


def decompress_windowed_fft(compressed: Dict) -> np.ndarray:
    """
    Reconstruct signal from windowed FFT compression via overlap-add.

    Parameters
    ----------
    compressed : dict
        Output of compress_windowed_fft().

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    n = compressed["n_samples"]
    window_size = compressed["window_size"]
    step = compressed["step"]
    output = np.zeros(n)
    weight = np.zeros(n)
    hann = np.hanning(window_size)

    for comp, start in zip(compressed["windows"], compressed["starts"]):
        chunk_recon = decompress_fft(comp)
        end = min(start + window_size, n)
        plen = end - start
        w = hann[:plen]
        output[start:end] += chunk_recon[:plen] * w
        weight[start:end] += w

    weight = np.maximum(weight, 1e-8)
    return output / weight
