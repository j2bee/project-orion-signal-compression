"""Signal visualization tools for time and frequency domain analysis."""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_time_domain(
    time: np.ndarray,
    signal: np.ndarray,
    title: str = "Time Domain Signal",
    ylabel: str = "Amplitude",
    ax: Optional[plt.Axes] = None,
    color: str = "steelblue",
) -> plt.Axes:
    """
    Plot signal amplitude versus time.

    Parameters
    ----------
    time : np.ndarray
        Time axis in seconds.
    signal : np.ndarray
        Signal values.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    ax : matplotlib Axes, optional
        Axes to plot on. Created if None.
    color : str
        Line color.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, signal, color=color, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_frequency_domain(
    signal: np.ndarray,
    sampling_frequency: float,
    title: str = "Frequency Spectrum",
    ax: Optional[plt.Axes] = None,
    color: str = "darkorange",
) -> Tuple[np.ndarray, np.ndarray, plt.Axes]:
    """
    Plot single-sided magnitude spectrum via FFT.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal.
    sampling_frequency : float
        Sampling rate in Hz.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on.
    color : str
        Line color.

    Returns
    -------
    frequencies : np.ndarray
        Frequency axis in Hz.
    magnitudes : np.ndarray
        Magnitude spectrum.
    ax : matplotlib Axes
    """
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum) / n
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_frequency)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frequencies, magnitudes, color=color, linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return frequencies, magnitudes, ax


def plot_signal_overview(
    time: np.ndarray,
    signal: np.ndarray,
    sampling_frequency: float,
    title_prefix: str = "Signal",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate time-domain and frequency-domain plots side by side.

    Parameters
    ----------
    time : np.ndarray
        Time axis.
    signal : np.ndarray
        Signal values.
    sampling_frequency : float
        Sampling rate in Hz.
    title_prefix : str
        Prefix for subplot titles.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure
    """
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 8))
    plot_time_domain(time, signal, title=f"{title_prefix} — Time Domain", ax=ax_time)
    plot_frequency_domain(
        signal, sampling_frequency, title=f"{title_prefix} — Frequency Spectrum", ax=ax_freq
    )
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_comparison(
    time: np.ndarray,
    original: np.ndarray,
    processed: np.ndarray,
    original_label: str = "Original",
    processed_label: str = "Processed",
    title: str = "Signal Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay original and processed signals for visual comparison.

    Parameters
    ----------
    time : np.ndarray
        Time axis.
    original : np.ndarray
        Reference signal.
    processed : np.ndarray
        Modified signal (filtered, reconstructed, etc.).
    original_label, processed_label : str
        Legend labels.
    title : str
        Plot title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(time, original, color="steelblue", linewidth=0.8, label=original_label)
    ax1.plot(time[: len(processed)], processed, color="tomato", linewidth=0.8, alpha=0.7, label=processed_label)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    error = original[: len(processed)] - processed
    ax2.plot(time[: len(error)], error, color="purple", linewidth=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error")
    ax2.set_title("Reconstruction Error")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pipeline_stages(
    time: np.ndarray,
    stages: dict,
    sampling_frequency: float,
    save_dir: str = "results/plots",
    prefix: str = "pipeline",
) -> dict:
    """
    Generate and save plots for each pipeline stage.

    Parameters
    ----------
    time : np.ndarray
        Time axis.
    stages : dict
        Mapping of stage name → signal array.
    sampling_frequency : float
        Sampling rate in Hz.
    save_dir : str
        Directory to save plots.
    prefix : str
        Filename prefix.

    Returns
    -------
    dict
        Mapping of stage name → saved filepath.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    for stage_name, signal in stages.items():
        filepath = save_dir / f"{prefix}_{stage_name}.png"
        plot_signal_overview(
            time[: len(signal)],
            signal,
            sampling_frequency,
            title_prefix=stage_name.replace("_", " ").title(),
            save_path=str(filepath),
        )
        saved[stage_name] = str(filepath)

    return saved
