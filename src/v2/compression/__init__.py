"""v2 compression algorithms with improved accuracy."""

from .adaptive_fft import compress_adaptive_fft, decompress_adaptive_fft
from .soft_wavelet import compress_soft_wavelet, decompress_soft_wavelet
from .mulaw_quantization import compress_mulaw, decompress_mulaw
from .hybrid import compress_hybrid, decompress_hybrid

__all__ = [
    "compress_adaptive_fft",
    "decompress_adaptive_fft",
    "compress_soft_wavelet",
    "decompress_soft_wavelet",
    "compress_mulaw",
    "decompress_mulaw",
    "compress_hybrid",
    "decompress_hybrid",
]
