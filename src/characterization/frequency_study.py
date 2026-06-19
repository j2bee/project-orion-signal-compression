"""Frequency content investigation — FFT sparsity and band energy analysis."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze_frequency_content(
    signal: np.ndarray,
    sampling_frequency: float,
    name: str = "signal",
    output_dir: str = "results/frequency_analysis",
) -> Dict:
    """
    Full FFT analysis: dominant frequencies, harmonics, noise bands, sparsity.

    Answers:
    1. Is most energy in few frequencies?
    2. Is the signal sparse in frequency space?
    3. Is FFT compression theoretically appropriate?

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal.
    sampling_frequency : float
        Sampling rate in Hz.
    name : str
        Signal name for output files.
    output_dir : str
        Directory for plots and JSON.

    Returns
    -------
    dict
        Frequency analysis results and FFT appropriateness assessment.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(signal)
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum) / n
    power = magnitudes ** 2
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_frequency)
    total_power = float(np.sum(power))

    # Dominant frequencies (top 15)
    top_n = min(15, len(magnitudes))
    top_idx = np.argsort(magnitudes)[-top_n:][::-1]
    dominant = [
        {"hz": round(float(frequencies[i]), 3), "magnitude": round(float(magnitudes[i]), 6),
         "power_fraction": round(float(power[i] / total_power), 4) if total_power > 0 else 0}
        for i in top_idx
    ]

    # Energy concentration: fraction of power in top 1%, 5%, 10% of bins
    sorted_power = np.sort(power)[::-1]
    n_bins = len(sorted_power)
    concentration = {}
    for pct_label, frac in [("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10)]:
        n_keep = max(1, int(n_bins * frac))
        concentration[pct_label] = round(float(np.sum(sorted_power[:n_keep]) / total_power), 4)

    # Band energy (rocket-relevant bands)
    bands = {
        "dc_sub_hz_0_5": (0.0, 0.5),
        "launch_dynamics_0_5_15": (0.5, 15.0),
        "vibration_15_50": (15.0, 50.0),
        "high_freq_noise_50_plus": (50.0, sampling_frequency / 2),
    }
    band_energy = {}
    for band_name, (lo, hi) in bands.items():
        mask = (frequencies >= lo) & (frequencies < hi)
        band_energy[band_name] = round(float(np.sum(power[mask]) / total_power), 4) if total_power > 0 else 0

    # Sparsity: Gini coefficient of magnitude distribution
    sorted_mag = np.sort(magnitudes)
    n_m = len(sorted_mag)
    gini = (2 * np.sum((np.arange(1, n_m + 1)) * sorted_mag) / (n_m * np.sum(sorted_mag)) - (n_m + 1) / n_m) if np.sum(sorted_mag) > 0 else 0

    # FFT appropriateness heuristic
    energy_concentrated = concentration["top_5pct"] > 0.80
    fft_appropriate = energy_concentrated and gini > 0.5

    results = {
        "name": name,
        "n_samples": n,
        "sampling_frequency_hz": sampling_frequency,
        "dominant_frequencies": dominant,
        "energy_concentration": concentration,
        "band_energy_fraction": band_energy,
        "gini_coefficient": round(float(gini), 4),
        "is_frequency_sparse": bool(gini > 0.5),
        "fft_compression_appropriate": bool(fft_appropriate),
        "assessment": (
            "YES — energy concentrated in few bins; FFT thresholding should work well."
            if fft_appropriate else
            "PARTIAL — energy spread across bands; consider wavelet or event-aware methods for transients."
        ),
    }

    # Plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(frequencies, magnitudes, linewidth=0.6, color="darkorange")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"{name} — FFT Magnitude Spectrum")
    axes[0].set_xlim(0, min(100, sampling_frequency / 2))
    axes[0].grid(True, alpha=0.3)

    # Mark dominant peaks
    for d in dominant[:5]:
        axes[0].axvline(d["hz"], color="red", alpha=0.3, linestyle="--")

    axes[1].semilogy(frequencies, power + 1e-20, linewidth=0.6, color="purple")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (log)")
    axes[1].set_title("Power Spectrum (log scale) — broadband noise visible at high frequencies")
    axes[1].set_xlim(0, sampling_frequency / 2)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_fft_analysis.png", dpi=150)
    plt.close(fig)

    # Spectrogram
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.specgram(signal, NFFT=256, Fs=sampling_frequency, noverlap=128, cmap="viridis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{name} — Spectrogram")
    ax.set_ylim(0, min(80, sampling_frequency / 2))
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_spectrogram.png", dpi=150)
    plt.close(fig)

    with open(output_dir / f"{name}_frequency_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
