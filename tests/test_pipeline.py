"""Tests for Project Orion Signal Compression pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compression.fft_compression import compress_fft, decompress_fft
from src.compression.wavelet_compression import compress_wavelet, decompress_wavelet
from src.compression.quantization import compress_quantization, decompress_quantization
from src.data_loader.loader import load_signal, save_signal
from src.data_loader.validation import validate_signal
from src.metrics.compression_ratio import compression_ratio
from src.metrics.mse import mse, rmse
from src.metrics.snr import snr_db
from src.preprocessing.noise import add_gaussian_noise, add_impulse_noise, add_sensor_drift
from src.preprocessing.normalization import min_max_scale, standardize
from src.preprocessing.filtering import moving_average, butterworth_lowpass
from src.reconstruction.reconstruction import reconstruct


@pytest.fixture
def sample_signal():
    """Generate a simple test signal."""
    t = np.linspace(0, 1, 500)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    return t, signal


@pytest.fixture
def temp_csv(tmp_path, sample_signal):
    """Create a temporary CSV signal file."""
    t, signal = sample_signal
    filepath = tmp_path / "test_signal.csv"
    save_signal(filepath, t, signal)
    return filepath


# --- Loader tests ---

class TestLoader:
    def test_load_csv(self, temp_csv):
        time, signal, metadata = load_signal(temp_csv)
        assert len(time) == len(signal) == 500
        assert metadata["n_samples"] == 500
        assert metadata["sampling_frequency"] > 0

    def test_load_npy(self, tmp_path, sample_signal):
        t, signal = sample_signal
        filepath = tmp_path / "test.npy"
        save_signal(filepath, t, signal, format="npy")
        time, signal_out, metadata = load_signal(filepath)
        assert len(signal_out) == 500

    def test_metadata_statistics(self, temp_csv):
        _, signal, metadata = load_signal(temp_csv)
        assert "mean" in metadata
        assert "std" in metadata
        assert metadata["min"] <= metadata["mean"] <= metadata["max"]


class TestValidation:
    def test_valid_signal(self, sample_signal):
        t, signal = sample_signal
        is_valid, errors = validate_signal(t, signal)
        assert is_valid
        assert len(errors) == 0

    def test_length_mismatch(self, sample_signal):
        t, signal = sample_signal
        is_valid, errors = validate_signal(t[:100], signal)
        assert not is_valid

    def test_nan_detection(self, sample_signal):
        t, signal = sample_signal
        signal_with_nan = signal.copy()
        signal_with_nan[10] = np.nan
        is_valid, errors = validate_signal(t, signal_with_nan)
        assert not is_valid


# --- Compression tests ---

class TestFFTCompression:
    def test_compress_decompress_dimensions(self, sample_signal):
        _, signal = sample_signal
        compressed = compress_fft(signal, keep_percentage=0.1)
        reconstructed = decompress_fft(compressed)
        assert len(reconstructed) == len(signal)

    def test_perfect_reconstruction_at_100_percent(self, sample_signal):
        _, signal = sample_signal
        compressed = compress_fft(signal, keep_percentage=1.0)
        reconstructed = decompress_fft(compressed)
        assert mse(signal, reconstructed) < 1e-10

    @pytest.mark.parametrize("keep_pct", [0.01, 0.05, 0.10, 0.25, 0.50])
    def test_various_keep_percentages(self, sample_signal, keep_pct):
        _, signal = sample_signal
        compressed = compress_fft(signal, keep_percentage=keep_pct)
        reconstructed = decompress_fft(compressed)
        assert len(reconstructed) == len(signal)


class TestWaveletCompression:
    def test_haar_wavelet(self, sample_signal):
        _, signal = sample_signal
        compressed = compress_wavelet(signal, wavelet="haar", keep_percentage=0.2)
        reconstructed = decompress_wavelet(compressed)
        assert len(reconstructed) == len(signal)

    def test_daubechies_wavelet(self, sample_signal):
        _, signal = sample_signal
        compressed = compress_wavelet(signal, wavelet="db4", keep_percentage=0.2)
        reconstructed = decompress_wavelet(compressed)
        assert len(reconstructed) == len(signal)


class TestQuantization:
    def test_8bit_quantization(self, sample_signal):
        _, signal = sample_signal
        compressed = compress_quantization(signal, bits=8)
        reconstructed = decompress_quantization(compressed)
        assert len(reconstructed) == len(signal)
        assert mse(signal, reconstructed) < 0.01

    def test_16bit_quantization(self, sample_signal):
        _, signal = sample_signal
        compressed = compress_quantization(signal, bits=16)
        reconstructed = decompress_quantization(compressed)
        assert mse(signal, reconstructed) < 1e-6


class TestReconstruction:
    def test_unified_reconstruct_fft(self, sample_signal):
        _, signal = sample_signal
        compressed = compress_fft(signal, 0.2)
        reconstructed = reconstruct(compressed, method="fft")
        assert len(reconstructed) == len(signal)

    def test_unified_reconstruct_wavelet(self, sample_signal):
        _, signal = sample_signal
        compressed = compress_wavelet(signal, keep_percentage=0.2)
        reconstructed = reconstruct(compressed, method="wavelet")
        assert len(reconstructed) == len(signal)


# --- Metrics tests ---

class TestMetrics:
    def test_mse_identical_signals(self, sample_signal):
        _, signal = sample_signal
        assert mse(signal, signal) == 0.0

    def test_rmse_identical_signals(self, sample_signal):
        _, signal = sample_signal
        assert rmse(signal, signal) == 0.0

    def test_snr_identical_signals(self, sample_signal):
        _, signal = sample_signal
        assert snr_db(signal, signal) == float("inf")

    def test_snr_degraded_signal(self, sample_signal):
        _, signal = sample_signal
        noisy = add_gaussian_noise(signal, noise_level=0.5, seed=0)
        snr = snr_db(signal, noisy)
        assert snr < 10  # Should be low SNR with high noise

    def test_compression_ratio(self, sample_signal):
        _, signal = sample_signal
        compressed = compress_fft(signal, keep_percentage=0.05)
        ratio = compression_ratio(signal, compressed)
        assert ratio > 1.0


# --- Preprocessing tests ---

class TestPreprocessing:
    def test_gaussian_noise(self, sample_signal):
        _, signal = sample_signal
        noisy = add_gaussian_noise(signal, noise_level=0.1, seed=42)
        assert len(noisy) == len(signal)
        assert not np.allclose(noisy, signal)

    def test_impulse_noise(self, sample_signal):
        _, signal = sample_signal
        noisy = add_impulse_noise(signal, probability=0.05, seed=42)
        assert len(noisy) == len(signal)

    def test_sensor_drift(self, sample_signal):
        _, signal = sample_signal
        drifted = add_sensor_drift(signal, seed=42)
        assert len(drifted) == len(signal)

    def test_min_max_scale(self, sample_signal):
        _, signal = sample_signal
        scaled, params = min_max_scale(signal)
        assert np.min(scaled) >= 0.0
        assert np.max(scaled) <= 1.0

    def test_standardize(self, sample_signal):
        _, signal = sample_signal
        std_signal, params = standardize(signal)
        assert abs(np.mean(std_signal)) < 1e-10
        assert abs(np.std(std_signal) - 1.0) < 1e-10

    def test_moving_average(self, sample_signal):
        _, signal = sample_signal
        filtered = moving_average(signal, window_size=5)
        assert len(filtered) == len(signal)

    def test_butterworth_filter(self, sample_signal):
        _, signal = sample_signal
        filtered = butterworth_lowpass(signal, sampling_frequency=500, cutoff_frequency=10)
        assert len(filtered) == len(signal)
