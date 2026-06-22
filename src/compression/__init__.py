"""Signal compression algorithms."""

from .fft_compression import compress_fft, decompress_fft
from .wavelet_compression import wavelet_transform, threshold_coefficients, inverse_wavelet, compress_wavelet, decompress_wavelet
from .quantization import quantize_signal, dequantize_signal, compress_quantization, decompress_quantization

__all__ = [
    "compress_fft",
    "decompress_fft",
    "wavelet_transform",
    "threshold_coefficients",
    "inverse_wavelet",
    "compress_wavelet",
    "decompress_wavelet",
    "quantize_signal",
    "dequantize_signal",
    "compress_quantization",
    "decompress_quantization",
]
