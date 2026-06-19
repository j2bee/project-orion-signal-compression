"""Tests for signal characterization study."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.importance import compute_importance_mask, importance_to_keep_percentage
from src.characterization.events import detect_events
from src.characterization.frequency_study import analyze_frequency_content
from src.characterization.sensitivity import local_compression_sensitivity
from src.characterization.noise_characterization import characterize_noise
from src.characterization.adaptive_experiment import run_adaptive_compression_experiment
from src.characterization.dataset import characterize_signal


@pytest.fixture
def rocket_like_signal():
    t = np.linspace(0, 100, 2000)
    signal = 9.81 * np.ones(200)  # pre-launch
    signal = np.concatenate([
        signal,
        9.81 + 30 * np.exp(-0.3 * np.linspace(0, 10, 200)) + np.random.randn(200) * 0.1,
        np.sin(2 * np.pi * 3.5 * np.linspace(0, 70, 1400)) * 0.5 + 1.0,
        np.zeros(200),
    ])
    return t[: len(signal)], signal


class TestImportance:
    def test_mask_range(self, rocket_like_signal):
        _, signal = rocket_like_signal
        mask, _ = compute_importance_mask(signal)
        assert len(mask) == len(signal)
        assert np.min(mask) >= 0.0
        assert np.max(mask) <= 1.0

    def test_keep_percentage_mapping(self, rocket_like_signal):
        _, signal = rocket_like_signal
        mask, _ = compute_importance_mask(signal)
        keep = importance_to_keep_percentage(mask)
        assert np.min(keep) >= 0.05
        assert np.max(keep) <= 0.50


class TestEvents:
    def test_detects_events(self, rocket_like_signal):
        t, signal = rocket_like_signal
        result = detect_events(t, signal, 20.0)
        assert result["n_events"] > 0


class TestFrequency:
    def test_frequency_analysis(self, rocket_like_signal):
        _, signal = rocket_like_signal
        result = analyze_frequency_content(signal, 20.0, "test", "results/frequency_analysis")
        assert "dominant_frequencies" in result
        assert "fft_compression_appropriate" in result


class TestSensitivity:
    def test_sensitivity_study(self, rocket_like_signal):
        _, signal = rocket_like_signal
        result = local_compression_sensitivity(signal[:2000], window_size=256, output_dir="results/characterization")
        assert result["n_windows"] > 0
        assert result["max_error"] >= result["mean_error"]


class TestNoiseChar:
    def test_noise_types(self, rocket_like_signal):
        _, signal = rocket_like_signal
        noisy = signal + np.random.randn(len(signal)) * 0.5
        result = characterize_noise(signal, noisy, 20.0)
        assert "snr_db" in result
        assert len(result["detected_noise_types"]) > 0


class TestAdaptiveExperiment:
    def test_uniform_vs_weighted(self, rocket_like_signal):
        _, signal = rocket_like_signal
        result = run_adaptive_compression_experiment(signal[:2000], window_size=256)
        assert "uniform" in result
        assert "importance_weighted" in result


class TestDataset:
    def test_characterize_synthetic(self):
        from src.generate_synthetic import create_synthetic_dataset
        path = create_synthetic_dataset("data/raw/test_char.csv")
        result = characterize_signal(Path(path))
        assert result["n_samples"] > 0
        assert "skewness" in result
