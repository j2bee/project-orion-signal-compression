"""Noise characterization and controlled noise injection for research experiments."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.metrics.snr import snr_db


def add_gaussian_noise_at_snr(
    signal: np.ndarray, target_snr_db: float, seed: int = None
) -> Tuple[np.ndarray, Dict]:
    """
    Add Gaussian noise to achieve a target SNR in dB.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal.
    target_snr_db : float
        Desired SNR (e.g. 5, 10, 20, 40).
    seed : int, optional
        Random seed.

    Returns
    -------
    noisy : np.ndarray
        Noisy signal.
    info : dict
        Actual SNR and noise power used.
    """
    rng = np.random.default_rng(seed)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (target_snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    noisy = signal + noise
    actual_snr = snr_db(signal, noisy)
    return noisy, {"target_snr_db": target_snr_db, "actual_snr_db": actual_snr, "type": "gaussian"}


def add_impulse_noise(
    signal: np.ndarray, probability: float = 0.005, amplitude: float = None, seed: int = None
) -> Tuple[np.ndarray, Dict]:
    """Add random impulse spikes."""
    rng = np.random.default_rng(seed)
    noisy = signal.copy()
    if amplitude is None:
        amplitude = 3.0 * (np.max(signal) - np.min(signal))
    mask = rng.random(len(signal)) < probability
    signs = rng.choice([-1.0, 1.0], size=int(mask.sum()))
    noisy[mask] += signs * amplitude
    return noisy, {"type": "impulse", "probability": probability, "n_spikes": int(mask.sum())}


def add_frequency_interference(
    signal: np.ndarray,
    sampling_frequency: float,
    interference_hz: float = 50.0,
    amplitude: float = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Add sinusoidal frequency interference (e.g. power-line or EMI).

    Parameters
    ----------
    signal : np.ndarray
        Clean signal.
    sampling_frequency : float
        Sampling rate in Hz.
    interference_hz : float
        Interference frequency.
    amplitude : float, optional
        Interference amplitude. Defaults to 10% of signal std.

    Returns
    -------
    noisy : np.ndarray
        Signal with interference.
    info : dict
        Interference parameters.
    """
    n = len(signal)
    t = np.arange(n) / sampling_frequency
    if amplitude is None:
        amplitude = 0.1 * np.std(signal)
    interference = amplitude * np.sin(2 * np.pi * interference_hz * t)
    noisy = signal + interference
    return noisy, {
        "type": "frequency_interference",
        "frequency_hz": interference_hz,
        "amplitude": amplitude,
        "actual_snr_db": snr_db(signal, noisy),
    }


def run_noise_sweep(
    signal: np.ndarray,
    sampling_frequency: float,
    snr_levels: List[float] = None,
    output_dir: str = "results/noise_analysis",
) -> Dict:
    """
    Generate noise variants and comparison plots.

    Parameters
    ----------
    signal : np.ndarray
        Clean reference signal.
    sampling_frequency : float
        Sampling rate in Hz.
    snr_levels : list of float
        Target SNR levels for Gaussian noise.
    output_dir : str
        Directory for plots and JSON.

    Returns
    -------
    dict
        Results for each noise type.
    """
    if snr_levels is None:
        snr_levels = [5, 10, 20, 40]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(5000, len(signal))  # plot subset for speed
    t = np.arange(n) / sampling_frequency
    clean = signal[:n]

    results = {"gaussian_sweep": [], "impulse": {}, "frequency_interference": {}}

    # Gaussian SNR sweep
    fig, axes = plt.subplots(len(snr_levels), 1, figsize=(12, 3 * len(snr_levels)), sharex=True)
    if len(snr_levels) == 1:
        axes = [axes]

    for i, snr_target in enumerate(snr_levels):
        noisy, info = add_gaussian_noise_at_snr(signal, snr_target, seed=42 + i)
        results["gaussian_sweep"].append(info)
        axes[i].plot(t, clean, alpha=0.5, label="Clean", linewidth=0.8)
        axes[i].plot(t, noisy[:n], alpha=0.7, label=f"Noisy SNR={info['actual_snr_db']:.1f} dB", linewidth=0.8)
        axes[i].set_ylabel("Amplitude")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Gaussian Noise at Target SNR Levels")
    fig.tight_layout()
    fig.savefig(output_dir / "gaussian_noise_sweep.png", dpi=150)
    plt.close(fig)

    # Impulse noise
    impulse_noisy, impulse_info = add_impulse_noise(signal, probability=0.005, seed=99)
    results["impulse"] = impulse_info
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, clean, alpha=0.5, label="Clean")
    ax.plot(t, impulse_noisy[:n], alpha=0.7, label="Impulse noise")
    ax.legend()
    ax.set_title("Impulse Noise")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "impulse_noise.png", dpi=150)
    plt.close(fig)

    # Frequency interference
    freq_noisy, freq_info = add_frequency_interference(signal, sampling_frequency, 50.0)
    results["frequency_interference"] = freq_info
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, clean, alpha=0.5, label="Clean")
    ax.plot(t, freq_noisy[:n], alpha=0.7, label=f"Interference @ {freq_info['frequency_hz']} Hz")
    ax.legend()
    ax.set_title("Frequency Interference")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "frequency_interference.png", dpi=150)
    plt.close(fig)

    import json
    with open(output_dir / "noise_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
