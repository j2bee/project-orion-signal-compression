"""Bit-depth quantization compression.

Reduces storage by mapping float64 samples to lower bit-width integers.
Useful as a baseline for measuring pure precision loss without transform coding.
"""

from typing import Dict

import numpy as np


def quantize_signal(signal: np.ndarray, bits: int = 8) -> Dict:
    """
    Quantize a float signal to a specified bit depth.

    Maps signal values linearly to [0, 2^bits - 1] integer range.

    Parameters
    ----------
    signal : np.ndarray
        Input float signal.
    bits : int
        Target bit depth (8 or 16).

    Returns
    -------
    dict
        Quantized integers and scaling parameters for dequantization.
    """
    bits = int(bits)
    if bits not in (8, 16):
        raise ValueError("Supported bit depths: 8, 16")

    sig_min = float(np.min(signal))
    sig_max = float(np.max(signal))
    span = sig_max - sig_min
    max_int = 2**bits - 1

    if span == 0:
        quantized = np.zeros(len(signal), dtype=np.int32)
    else:
        normalized = (signal - sig_min) / span
        quantized = np.round(normalized * max_int).astype(np.int32)

    return {
        "method": "quantization",
        "bits": bits,
        "quantized": quantized,
        "sig_min": sig_min,
        "sig_max": sig_max,
        "n_samples": len(signal),
    }


def dequantize_signal(compressed: Dict) -> np.ndarray:
    """
    Reconstruct float signal from quantized integers.

    Parameters
    ----------
    compressed : dict
        Output of quantize_signal().

    Returns
    -------
    np.ndarray
        Dequantized float signal.
    """
    bits = compressed["bits"]
    max_int = 2**bits - 1
    sig_min = compressed["sig_min"]
    sig_max = compressed["sig_max"]
    span = sig_max - sig_min

    if span == 0:
        return np.full(compressed["n_samples"], sig_min, dtype=np.float64)

    normalized = compressed["quantized"].astype(np.float64) / max_int
    return normalized * span + sig_min


def compress_quantization(signal: np.ndarray, bits: int = 8) -> Dict:
    """Alias for quantize_signal — unified compression interface."""
    return quantize_signal(signal, bits)


def decompress_quantization(compressed: Dict) -> np.ndarray:
    """Alias for dequantize_signal — unified decompression interface."""
    return dequantize_signal(compressed)


def estimate_storage_bytes(n_samples: int, bits: int = 64) -> int:
    """
    Estimate storage requirement in bytes for a signal.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    bits : int
        Bits per sample.

    Returns
    -------
    int
        Total bytes.
    """
    return n_samples * bits // 8
