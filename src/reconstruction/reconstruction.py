"""Signal reconstruction from compressed representations."""

from typing import Dict

import numpy as np

from ..compression.fft_compression import decompress_fft
from ..compression.wavelet_compression import decompress_wavelet
from ..compression.quantization import decompress_quantization


def reconstruct(compressed: Dict, method: str = None) -> np.ndarray:
    """
    Unified reconstruction interface for all compression methods.

    Parameters
    ----------
    compressed : dict
        Compressed signal representation (must contain 'method' key).
    method : str, optional
        Override method: 'fft', 'wavelet', or 'quantization'.
        If None, inferred from compressed['method'].

    Returns
    -------
    np.ndarray
        Reconstructed time-domain signal.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    method = method or compressed.get("method")
    if method is None:
        raise ValueError("Cannot determine compression method from compressed data.")

    dispatch = {
        "fft": decompress_fft,
        "wavelet": decompress_wavelet,
        "quantization": decompress_quantization,
    }

    if method not in dispatch:
        raise ValueError(
            f"Unknown method '{method}'. Supported: {list(dispatch.keys())}"
        )

    return dispatch[method](compressed)


def reconstruct_and_compare(
    original: np.ndarray, compressed: Dict, method: str = None
) -> Dict:
    """
    Reconstruct signal and return side-by-side comparison arrays.

    Parameters
    ----------
    original : np.ndarray
        Original signal before compression.
    compressed : dict
        Compressed representation.
    method : str, optional
        Compression method override.

    Returns
    -------
    dict
        original, reconstructed, and error arrays.
    """
    reconstructed = reconstruct(compressed, method=method)
    n = min(len(original), len(reconstructed))
    error = original[:n] - reconstructed[:n]
    return {
        "original": original[:n],
        "reconstructed": reconstructed[:n],
        "error": error,
    }
