"""Tests for segmented metrics and refined synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate_synthetic import generate_synthetic_rocket, get_phase_segments
from src.metrics.segmented import compute_segment_metrics, default_rocket_segments
from src.v2.compression.adaptive_fft import compress_adaptive_fft, decompress_adaptive_fft
from src.v2.compression.hybrid import compress_hybrid, decompress_hybrid


class TestSyntheticData:
    def test_clean_signal_no_nan(self):
        time, signal = generate_synthetic_rocket(duration=100, sampling_frequency=200)
        assert len(time) == 20000
        assert np.all(np.isfinite(signal))

    def test_pre_launch_near_gravity(self):
        time, signal = generate_synthetic_rocket()
        pre_launch = signal[time < 9.0]
        assert np.mean(pre_launch) > 8.0

    def test_smooth_transitions(self):
        time, signal = generate_synthetic_rocket()
        # No single-sample jumps larger than physically plausible
        max_jump = np.max(np.abs(np.diff(signal)))
        assert max_jump < 15.0

    def test_phase_segments(self):
        segs = get_phase_segments()
        assert "launch" in segs
        assert segs["launch"] == (10.0, 20.0)


class TestSegmentedMetrics:
    def test_segment_metrics(self):
        time = np.linspace(0, 100, 2000)
        original = np.sin(2 * np.pi * 0.5 * time)
        reconstructed = original + 0.01 * np.random.randn(len(time))
        segs = default_rocket_segments(100.0)
        results = compute_segment_metrics(time, original, reconstructed, segs)
        assert "launch" in results
        assert results["launch"]["snr_db"] > 20


class TestAdaptiveFFTCap:
    def test_max_keep_fraction_caps_ratio(self):
        signal = np.random.randn(2000)
        c_loose = compress_adaptive_fft(signal, 0.99, max_keep_fraction=0.50)
        c_tight = compress_adaptive_fft(signal, 0.99, max_keep_fraction=0.05)
        assert c_tight["n_coeffs_kept"] < c_loose["n_coeffs_kept"]

    def test_hybrid_target_ratio(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 1, 2000)
        c = compress_hybrid(signal, 0.85, target_ratio=5.0)
        r = decompress_hybrid(c)
        assert len(r) == len(signal)
