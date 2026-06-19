"""Signal reconstruction from compressed representations."""

from typing import Dict

import numpy as np

from ..compression.fft_compression import decompress_fft
from ..compression.wavelet_compression import decompress_wavelet
from ..compression.quantization import decompress_quantization


def reconstruct(compressed: Dict, method: str = None) -> np.ndarray:
    """
    Unified reconstruction interface for all compression methods (v1 and v2).

    Parameters
    ----------
    compressed : dict
        Compressed signal representation (must contain 'method' key).
    method : str, optional
        Override method. If None, inferred from compressed['method'].

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

    # v1 methods
    dispatch = {
        "fft": decompress_fft,
        "wavelet": decompress_wavelet,
        "quantization": decompress_quantization,
    }

    # v2 methods (lazy import to keep v1 dependency-free)
    v2_methods = {
        "adaptive_fft", "soft_wavelet", "mulaw", "hybrid", "ml",
    }
    if method in v2_methods:
        return _reconstruct_v2(compressed, method)

    if method not in dispatch:
        raise ValueError(
            f"Unknown method '{method}'. Supported: {list(dispatch.keys()) + list(v2_methods)}"
        )

    return dispatch[method](compressed)


def _reconstruct_v2(compressed: Dict, method: str) -> np.ndarray:
    """Dispatch v2 reconstruction methods."""
    if method == "adaptive_fft":
        from ..v2.compression.adaptive_fft import decompress_adaptive_fft
        return decompress_adaptive_fft(compressed)
    elif method == "soft_wavelet":
        from ..v2.compression.soft_wavelet import decompress_soft_wavelet
        return decompress_soft_wavelet(compressed)
    elif method == "mulaw":
        from ..v2.compression.mulaw_quantization import decompress_mulaw
        return decompress_mulaw(compressed)
    elif method == "hybrid":
        from ..v2.compression.hybrid import decompress_hybrid
        return decompress_hybrid(compressed)
    elif method == "ml":
        from ..ml.autoencoder import decompress_ml
        return decompress_ml(compressed)
    raise ValueError(f"Unknown v2 method: {method}")


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
