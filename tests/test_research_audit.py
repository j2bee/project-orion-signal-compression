"""Tests for research audit modules."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.time_domain import analyze_time_domain
from src.analysis.frequency_domain import analyze_frequency_domain
from src.compression.windowed import compress_windowed_fft, decompress_windowed_fft
from src.noise_analysis import add_gaussian_noise_at_snr, add_frequency_interference
from src.reconstruction.reconstruction import reconstruct
from src.streaming import packetize_signal, simulate_stream
from src.v2.compression.event_aware import compress_event_aware, decompress_event_aware


@pytest.fixture
def signal():
    t = np.linspace(0, 1, 1000)
    return t, np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)


class TestAnalysis:
    def test_time_domain(self, signal):
        t, s = signal
        stats = analyze_time_domain(s, t)
        assert stats["rms"] > 0
        assert stats["peak_to_peak"] > 0

    def test_frequency_domain(self, signal):
        _, s = signal
        stats = analyze_frequency_domain(s, 1000.0)
        assert len(stats["dominant_frequencies"]) > 0


class TestNoiseAnalysis:
    def test_gaussian_at_snr(self, signal):
        _, s = signal
        noisy, info = add_gaussian_noise_at_snr(s, 20.0, seed=0)
        assert abs(info["actual_snr_db"] - 20.0) < 2.0

    def test_frequency_interference(self, signal):
        _, s = signal
        noisy, info = add_frequency_interference(s, 1000.0, 50.0)
        assert len(noisy) == len(s)


class TestEventAware:
    def test_roundtrip(self, signal):
        _, s = signal
        c = compress_event_aware(s, 1000.0)
        r = decompress_event_aware(c)
        assert len(r) == len(s)

    def test_unified_reconstruct(self, signal):
        _, s = signal
        c = compress_event_aware(s, 1000.0)
        r = reconstruct(c)
        assert len(r) == len(s)


class TestWindowed:
    def test_roundtrip(self, signal):
        _, s = signal
        c = compress_windowed_fft(s, window_size=256, keep_percentage=0.2)
        r = decompress_windowed_fft(c)
        assert len(r) == len(s)


class TestStreaming:
    def test_packetize(self, signal):
        _, s = signal
        packets = packetize_signal(s, 1000.0, packet_size=128)
        assert len(packets) > 0

    def test_simulate_stream(self, signal):
        _, s = signal
        recon, stats = simulate_stream(s, 1000.0, packet_size=128)
        assert len(recon) == len(s)
        assert stats.n_packets > 0


class TestPCA:
    def test_pca_roundtrip(self, signal):
        pytest.importorskip("sklearn")
        from src.research.classical_extensions import compress_pca, decompress_pca
        _, s = signal
        c = compress_pca(s, n_components=8, window_size=128)
        r = decompress_pca(c)
        assert len(r) == len(s)
