"""Reconstruction error visualization for research analysis."""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_error_analysis(
    time: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "Reconstruction Error Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot difference signal and error magnitude over time.

    Helps identify whether reconstruction failures occur during
    spikes, events, or quiet regions.

    Parameters
    ----------
    time : np.ndarray
        Time axis.
    original : np.ndarray
        Reference signal.
    reconstructed : np.ndarray
        Reconstructed signal.
    title : str
        Plot title.
    save_path : str, optional
        Save path for figure.

    Returns
    -------
    matplotlib Figure
    """
    n = min(len(time), len(original), len(reconstructed))
    t = time[:n]
    orig = original[:n]
    recon = reconstructed[:n]
    error = orig - recon
    abs_error = np.abs(error)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, orig, color="steelblue", linewidth=0.8, label="Original")
    axes[0].plot(t, recon, color="tomato", linewidth=0.8, alpha=0.7, label="Reconstructed")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"{title} — Overlay")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, error, color="purple", linewidth=0.8)
    axes[1].set_ylabel("Error")
    axes[1].set_title("Difference Signal (original − reconstructed)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, abs_error, color="darkred", linewidth=0.8)
    axes[2].fill_between(t, 0, abs_error, alpha=0.3, color="darkred")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("|Error|")
    axes[2].set_title("Absolute Error Over Time")
    axes[2].grid(True, alpha=0.3)

    # Mark top-5 error peaks
    peak_idx = np.argsort(abs_error)[-5:]
    for idx in peak_idx:
        axes[2].axvline(t[idx], color="orange", alpha=0.4, linestyle="--")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_error_by_segment(
    time: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
    segments: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of mean absolute error per flight phase segment.

    Parameters
    ----------
    time, original, reconstructed : np.ndarray
        Signal arrays.
    segments : dict
        Phase name → (start_s, end_s).
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    n = min(len(time), len(original), len(reconstructed))
    error = np.abs(original[:n] - reconstructed[:n])
    t = time[:n]

    names, mae_values = [], []
    for name, (start, end) in segments.items():
        mask = (t >= start) & (t < end)
        if np.any(mask):
            names.append(name)
            mae_values.append(float(np.mean(error[mask])))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, mae_values, color="steelblue", edgecolor="navy")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Reconstruction Error by Flight Phase")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
