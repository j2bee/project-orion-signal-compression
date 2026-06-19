"""Tests for v2 improvements and ML reconstruction."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.v2.compression.adaptive_fft import compress_adaptive_fft, decompress_adaptive_fft
from src.v2.compression.soft_wavelet import compress_soft_wavelet, decompress_soft_wavelet
from src.v2.compression.mulaw_quantization import compress_mulaw, decompress_mulaw
from src.v2.compression.hybrid import compress_hybrid, decompress_hybrid
from src.v2.denoising.pipeline import denoise_multistage
from src.v2.denoising.median_filter import remove_impulse_median
from src.v2.denoising.spectral_subtraction import spectral_subtract
from src.v2.denoising.wiener import wiener_denoise
from src.v2.denoising.drift_correction import correct_sensor_drift
from src.reconstruction.reconstruction import reconstruct
from src.metrics.mse import mse
from src.metrics.snr import snr_db
from src.metrics.compression_ratio import compression_ratio


@pytest.fixture
def sample_signal():
    t = np.linspace(0, 1, 500)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    return signal


@pytest.fixture
def noisy_signal(sample_signal):
    rng = np.random.default_rng(42)
    return sample_signal + rng.normal(0, 0.1, len(sample_signal))


class TestAdaptiveFFT:
    def test_dimensions(self, sample_signal):
        c = compress_adaptive_fft(sample_signal, 0.90)
        r = decompress_adaptive_fft(c)
        assert len(r) == len(sample_signal)

    def test_high_energy_preservation(self, sample_signal):
        c = compress_adaptive_fft(sample_signal, 0.99)
        r = decompress_adaptive_fft(c)
        assert mse(sample_signal, r) < 0.01

    def test_unified_reconstruct(self, sample_signal):
        c = compress_adaptive_fft(sample_signal, 0.90)
        r = reconstruct(c, method="adaptive_fft")
        assert len(r) == len(sample_signal)


class TestSoftWavelet:
    def test_dimensions(self, sample_signal):
        c = compress_soft_wavelet(sample_signal, "db4", 0.2)
        r = decompress_soft_wavelet(c)
        assert len(r) == len(sample_signal)

    def test_better_than_hard_at_same_sparsity(self, sample_signal):
        c = compress_soft_wavelet(sample_signal, "db4", 0.2)
        r = decompress_soft_wavelet(c)
        assert mse(sample_signal, r) < 1.0


class TestMulawQuantization:
    def test_8bit(self, sample_signal):
        c = compress_mulaw(sample_signal, bits=8)
        r = decompress_mulaw(c)
        assert len(r) == len(sample_signal)
        assert mse(sample_signal, r) < 0.05

    def test_unified_reconstruct(self, sample_signal):
        c = compress_mulaw(sample_signal, bits=8)
        r = reconstruct(c, method="mulaw")
        assert len(r) == len(sample_signal)


class TestHybridCompression:
    def test_dimensions(self, sample_signal):
        c = compress_hybrid(sample_signal, 0.85, 0.15)
        r = decompress_hybrid(c)
        assert len(r) == len(sample_signal)

    def test_lower_mse_than_fft_alone(self, sample_signal):
        c_fft = compress_adaptive_fft(sample_signal, 0.85)
        r_fft = decompress_adaptive_fft(c_fft)
        c_hybrid = compress_hybrid(sample_signal, 0.85, 0.15)
        r_hybrid = decompress_hybrid(c_hybrid)
        assert mse(sample_signal, r_hybrid) <= mse(sample_signal, r_fft) * 1.5


class TestDenoising:
    def test_multistage_pipeline(self, noisy_signal):
        denoised, info = denoise_multistage(noisy_signal, sampling_frequency=500)
        assert len(denoised) == len(noisy_signal)
        assert "stages" in info
        assert len(info["stages"]) >= 3

    def test_median_removes_impulse(self, sample_signal):
        spiked = sample_signal.copy()
        spiked[100] += 100.0
        filtered = remove_impulse_median(spiked, kernel_size=5)
        assert abs(filtered[100] - sample_signal[100]) < 50.0

    def test_spectral_subtraction(self, noisy_signal):
        denoised = spectral_subtract(noisy_signal, 500)
        assert len(denoised) == len(noisy_signal)

    def test_wiener_filter(self, noisy_signal):
        denoised = wiener_denoise(noisy_signal, 500)
        assert len(denoised) == len(noisy_signal)

    def test_drift_correction(self, sample_signal):
        drifted = sample_signal + np.linspace(0, 5, len(sample_signal))
        corrected = correct_sensor_drift(drifted)
        assert abs(np.mean(corrected)) < abs(np.mean(drifted))


class TestMLAutoencoder:
    def test_compress_decompress(self, sample_signal, tmp_path):
        pytest.importorskip("torch")
        from src.ml.autoencoder import compress_ml, decompress_ml
        ckpt = str(tmp_path / "test_model.pt")
        long_signal = np.tile(sample_signal, 8)
        c = compress_ml(long_signal, latent_dim=16, patch_size=128, epochs=5, checkpoint_path=ckpt)
        r = decompress_ml(c, checkpoint_path=ckpt)
        assert len(r) == len(long_signal)
        assert "precomputed_reconstruction" not in c
        assert "model_state_dict" not in c

    def test_unified_reconstruct(self, sample_signal, tmp_path):
        pytest.importorskip("torch")
        from src.ml.autoencoder import compress_ml
        ckpt = str(tmp_path / "test_model.pt")
        long_signal = np.tile(sample_signal, 8)
        c = compress_ml(long_signal, latent_dim=16, patch_size=128, epochs=5, checkpoint_path=ckpt)
        r = reconstruct(c, method="ml")
        assert len(r) == len(long_signal)

    def test_compression_ratio(self, sample_signal, tmp_path):
        pytest.importorskip("torch")
        from src.ml.autoencoder import compress_ml, estimate_ml_compression_ratio
        ckpt = str(tmp_path / "test_model.pt")
        long_signal = np.tile(sample_signal, 8)
        c = compress_ml(long_signal, latent_dim=16, patch_size=128, epochs=5, checkpoint_path=ckpt)
        ratio = estimate_ml_compression_ratio(c["n_samples"], c["n_patches"], c["latent_dim"])
        assert ratio > 1.0
