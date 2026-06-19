"""μ-law companding quantization (v2).

Applies μ-law compression before bit reduction to preserve dynamic range
of telemetry spikes (launch acceleration) better than v1 linear quantization.
"""

from typing import Dict

import numpy as np


def _mulaw_compress(signal: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """
    μ-law companding: maps signal to [-1, 1] with logarithmic compression.

    Large amplitudes are compressed less aggressively relative to small ones,
    preserving spike detail in quantization.
    """
    # Normalize to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak == 0:
        return np.zeros_like(signal)
    x = signal / peak
    sign = np.sign(x)
    compressed = sign * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return compressed, peak


def _mulaw_expand(compressed: np.ndarray, peak: float, mu: float = 255.0) -> np.ndarray:
    """Inverse μ-law expansion."""
    sign = np.sign(compressed)
    x = sign * (np.expm1(np.abs(compressed) * np.log1p(mu))) / mu
    return x * peak


def compress_mulaw(signal: np.ndarray, bits: int = 8, mu: float = 255.0) -> Dict:
    """
    Quantize signal using μ-law companding before bit reduction.

    Parameters
    ----------
    signal : np.ndarray
        Input float signal.
    bits : int
        Target bit depth (8 or 16).
    mu : float
        μ-law parameter (255 is standard telephony value).

    Returns
    -------
    dict
        Quantized representation with companding metadata.
    """
    bits = int(bits)
    if bits not in (8, 16):
        raise ValueError("Supported bit depths: 8, 16")

    companded, peak = _mulaw_compress(signal, mu)
    max_int = 2**bits - 1
    # Map [-1, 1] to [0, max_int]
    normalized = (companded + 1.0) / 2.0
    quantized = np.round(normalized * max_int).astype(np.int32)

    return {
        "method": "mulaw",
        "version": 2,
        "bits": bits,
        "mu": mu,
        "peak": float(peak),
        "quantized": quantized,
        "n_samples": len(signal),
    }


def decompress_mulaw(compressed: Dict) -> np.ndarray:
    """
    Reconstruct float signal from μ-law quantized representation.

    Parameters
    ----------
    compressed : dict
        Output of compress_mulaw().

    Returns
    -------
    np.ndarray
        Dequantized signal.
    """
    bits = compressed["bits"]
    mu = compressed["mu"]
    peak = compressed["peak"]
    max_int = 2**bits - 1

    normalized = compressed["quantized"].astype(np.float64) / max_int
    companded = normalized * 2.0 - 1.0
    return _mulaw_expand(companded, peak, mu)
