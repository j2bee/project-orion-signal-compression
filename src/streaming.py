"""Streaming simulation for rocket telemetry packet compression.

Simulates: Sensor → Packet → Compress → Transmit → Decompress → Reconstruct
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.compression.fft_compression import compress_fft, decompress_fft
from src.metrics.compression_ratio import estimate_compressed_size


@dataclass
class Packet:
    """Single telemetry packet."""
    packet_id: int
    timestamp: float
    samples: np.ndarray
    compressed: Optional[Dict] = None
    compressed_bytes: int = 0


@dataclass
class StreamStats:
    """Aggregated streaming performance statistics."""
    n_packets: int = 0
    total_samples: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    total_compress_time_s: float = 0.0
    total_decompress_time_s: float = 0.0
    max_latency_ms: float = 0.0
    peak_memory_samples: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.total_compressed_bytes == 0:
            return 1.0
        return self.total_original_bytes / self.total_compressed_bytes

    @property
    def avg_latency_ms(self) -> float:
        if self.n_packets == 0:
            return 0.0
        return (self.total_compress_time_s + self.total_decompress_time_s) / self.n_packets * 1000

    def to_dict(self) -> Dict:
        return {
            "n_packets": self.n_packets,
            "total_samples": self.total_samples,
            "compression_ratio": round(self.compression_ratio, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "max_latency_ms": round(self.max_latency_ms, 3),
            "total_compress_time_s": round(self.total_compress_time_s, 4),
            "total_decompress_time_s": round(self.total_decompress_time_s, 4),
            "peak_memory_samples": self.peak_memory_samples,
        }


def packetize_signal(
    signal: np.ndarray,
    sampling_frequency: float,
    packet_size: int = 512,
) -> List[Packet]:
    """
    Split a continuous signal into fixed-size packets.

    Parameters
    ----------
    signal : np.ndarray
        Full signal.
    sampling_frequency : float
        Sampling rate in Hz.
    packet_size : int
        Samples per packet.

    Returns
    -------
    list of Packet
    """
    packets = []
    for i, start in enumerate(range(0, len(signal), packet_size)):
        chunk = signal[start : start + packet_size]
        if len(chunk) == 0:
            break
        packets.append(Packet(
            packet_id=i,
            timestamp=start / sampling_frequency,
            samples=chunk,
        ))
    return packets


def simulate_stream(
    signal: np.ndarray,
    sampling_frequency: float,
    packet_size: int = 512,
    keep_percentage: float = 0.10,
) -> tuple:
    """
    Run full streaming simulation with per-packet FFT compression.

    Parameters
    ----------
    signal : np.ndarray
        Input telemetry signal.
    sampling_frequency : float
        Sampling rate in Hz.
    packet_size : int
        Samples per packet.
    keep_percentage : float
        FFT keep fraction per packet.

    Returns
    -------
    reconstructed : np.ndarray
        Full reconstructed signal.
    stats : StreamStats
        Performance statistics.
    """
    packets = packetize_signal(signal, sampling_frequency, packet_size)
    stats = StreamStats()
    reconstructed_chunks = []

    for pkt in packets:
        stats.peak_memory_samples = max(stats.peak_memory_samples, len(pkt.samples))

        t0 = time.time()
        pkt.compressed = compress_fft(pkt.samples, keep_percentage=keep_percentage)
        compress_time = time.time() - t0

        t1 = time.time()
        recon = decompress_fft(pkt.compressed)
        decompress_time = time.time() - t1

        pkt.compressed_bytes = estimate_compressed_size(pkt.compressed)
        latency_ms = (compress_time + decompress_time) * 1000

        stats.n_packets += 1
        stats.total_samples += len(pkt.samples)
        stats.total_original_bytes += pkt.samples.nbytes
        stats.total_compressed_bytes += pkt.compressed_bytes
        stats.total_compress_time_s += compress_time
        stats.total_decompress_time_s += decompress_time
        stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)

        reconstructed_chunks.append(recon[: len(pkt.samples)])

    reconstructed = np.concatenate(reconstructed_chunks)
    return reconstructed, stats


def run_streaming_benchmark(
    signal: np.ndarray,
    sampling_frequency: float,
    output_path: str = "results/streaming_benchmark.json",
) -> Dict:
    """
    Run streaming benchmark and save results.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sampling_frequency : float
        Sampling rate.
    output_path : str
        JSON output path.

    Returns
    -------
    dict
        Benchmark results.
    """
    recon, stats = simulate_stream(signal, sampling_frequency)
    from src.metrics.mse import mse
    from src.metrics.snr import snr_db

    results = stats.to_dict()
    results["mse"] = round(float(mse(signal[: len(recon)], recon)), 8)
    results["snr_db"] = round(float(snr_db(signal[: len(recon)], recon)), 2)
    results["packet_size"] = 512
    results["keep_percentage"] = 0.10

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
