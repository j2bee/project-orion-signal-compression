"""Local compression sensitivity — which signal regions are fragile?"""

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.compression.fft_compression import compress_fft, decompress_fft
from src.metrics.mse import mse
from src.metrics.snr import snr_db


def local_compression_sensitivity(
    signal: np.ndarray,
    window_size: int = 512,
    step: int = None,
    aggressive_keep: float = 0.05,
    output_dir: str = "results/characterization",
    name: str = "signal",
) -> Dict:
    """
    Compress each window aggressively and measure reconstruction error.

    Produces heatmap: Window Index vs Reconstruction Error.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_size : int
        Analysis window length in samples.
    step : int, optional
        Window step (default window_size // 2).
    aggressive_keep : float
        FFT keep percentage for sensitivity test (low = aggressive).
    output_dir : str
        Plot output directory.
    name : str
        Signal name for filenames.

    Returns
    -------
    dict
        Per-window errors, fragile region indices, heatmap path.
    """
    if step is None:
        step = window_size // 2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(signal)
    window_errors = []
    window_centers = []
    window_indices = []

    for i, start in enumerate(range(0, n - window_size + 1, step)):
        chunk = signal[start : start + window_size]
        compressed = compress_fft(chunk, keep_percentage=aggressive_keep)
        reconstructed = decompress_fft(compressed)
        err = mse(chunk, reconstructed)
        snr = snr_db(chunk, reconstructed)
        center = start + window_size // 2
        window_errors.append(err)
        window_centers.append(center)
        window_indices.append(i)

    errors = np.array(window_errors)
    centers = np.array(window_centers)

    # Identify fragile windows (top 10% error)
    fragile_threshold = np.percentile(errors, 90)
    fragile_mask = errors >= fragile_threshold
    fragile_centers = centers[fragile_mask].tolist()

    results = {
        "window_size": window_size,
        "step": step,
        "aggressive_keep_pct": aggressive_keep,
        "n_windows": len(window_errors),
        "mean_error": round(float(np.mean(errors)), 8),
        "max_error": round(float(np.max(errors)), 8),
        "fragile_threshold": round(float(fragile_threshold), 8),
        "fragile_window_centers_samples": fragile_centers,
        "window_errors": [round(float(e), 8) for e in window_errors],
        "window_center_times_s": None,  # filled by caller if time known
    }

    # Heatmap: window index vs error
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].bar(window_indices, window_errors, color="steelblue", width=1.0, alpha=0.8)
    axes[0].axhline(fragile_threshold, color="red", linestyle="--", label=f"Fragile threshold (90th pct)")
    axes[0].set_xlabel("Window Index")
    axes[0].set_ylabel("MSE")
    axes[0].set_title(f"{name} — Local Compression Sensitivity (FFT keep={aggressive_keep*100:.0f}%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2D heatmap along signal timeline
    error_timeline = np.zeros(n)
    weight = np.zeros(n)
    for center, err in zip(centers, errors):
        lo = max(0, center - window_size // 2)
        hi = min(n, center + window_size // 2)
        error_timeline[lo:hi] += err
        weight[lo:hi] += 1
    weight = np.maximum(weight, 1)
    error_timeline /= weight

    im = axes[1].imshow(
        error_timeline.reshape(1, -1),
        aspect="auto",
        cmap="hot",
        extent=[0, n, 0, 1],
    )
    axes[1].set_xlabel("Sample Index")
    axes[1].set_title("Reconstruction Error Heatmap Along Signal")
    plt.colorbar(im, ax=axes[1], label="MSE")
    fig.tight_layout()
    heatmap_path = output_dir / f"{name}_sensitivity_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    results["heatmap_path"] = str(heatmap_path)

    with open(output_dir / f"{name}_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
