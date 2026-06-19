"""Dataset investigation — statistical characterization of all available signals."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from src.data_loader.loader import load_signal

# Known dataset units (m/s² for synthetic rocket accelerometer)
DATASET_UNITS = {
    "synthetic_rocket": "m/s² (acceleration)",
    "noisy_signal": "m/s² (acceleration)",
    "reconstructed_signal": "m/s² (acceleration)",
}


def _discover_datasets(data_dir: str = "data") -> List[Path]:
    """Find all loadable signal files under data/."""
    root = Path(data_dir)
    patterns = ["**/*.csv", "**/*.txt", "**/*.npy", "**/*.npz"]
    files = []
    for pat in patterns:
        files.extend(root.glob(pat))
    return sorted(set(files))


def characterize_signal(
    filepath: Path,
    units: str = "unknown",
) -> Dict:
    """
    Compute full statistical characterization for one signal file.

    Parameters
    ----------
    filepath : Path
        Path to signal file.
    units : str
        Physical units of the signal.

    Returns
    -------
    dict
        Sample count, duration, fs, min/max, mean, variance, RMS, skewness, kurtosis.
    """
    time, signal, metadata = load_signal(filepath)
    n = len(signal)
    duration = float(time[-1] - time[0]) if n > 1 else 0.0
    fs = metadata.get("sampling_frequency", (n - 1) / duration if duration > 0 else 1.0)

    return {
        "filepath": str(filepath),
        "name": filepath.stem,
        "units": units,
        "n_samples": n,
        "duration_s": round(duration, 4),
        "sampling_frequency_hz": round(float(fs), 2),
        "min": round(float(np.min(signal)), 6),
        "max": round(float(np.max(signal)), 6),
        "mean": round(float(np.mean(signal)), 6),
        "variance": round(float(np.var(signal)), 6),
        "std": round(float(np.std(signal)), 6),
        "rms": round(float(np.sqrt(np.mean(signal ** 2))), 6),
        "skewness": round(float(stats.skew(signal)), 4),
        "kurtosis": round(float(stats.kurtosis(signal)), 4),
        "peak_to_peak": round(float(np.max(signal) - np.min(signal)), 6),
    }


def characterize_all_datasets(data_dir: str = "data") -> List[Dict]:
    """
    Characterize every signal dataset found under data/.

    Parameters
    ----------
    data_dir : str
        Root data directory.

    Returns
    -------
    list of dict
        One characterization dict per file.
    """
    results = []
    for filepath in _discover_datasets(data_dir):
        units = DATASET_UNITS.get(filepath.stem, "unknown (assumed m/s² for rocket telemetry)")
        try:
            results.append(characterize_signal(filepath, units))
        except Exception as e:
            results.append({"filepath": str(filepath), "error": str(e)})
    return results


def write_dataset_report(
    characterizations: List[Dict],
    output_path: str = "reports/dataset_characterization.md",
    plot_dir: str = "results/characterization",
) -> str:
    """
    Write markdown dataset characterization report with summary table.

    Parameters
    ----------
    characterizations : list
        Output of characterize_all_datasets().
    output_path : str
        Markdown report path.
    plot_dir : str
        Directory for per-dataset plots.

    Returns
    -------
    str
        Report text.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data_loader.loader import load_signal

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Dataset Characterization",
        "",
        "Statistical investigation of every available signal dataset.",
        "",
        "## Summary Table",
        "",
        "| Dataset | Samples | Duration (s) | fs (Hz) | Units | Mean | Std | RMS | Skew | Kurt | Min | Max |",
        "|---------|---------|--------------|---------|-------|------|-----|-----|------|------|-----|-----|",
    ]

    for c in characterizations:
        if "error" in c:
            lines.append(f"| {c['filepath']} | ERROR | — | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {c['name']} | {c['n_samples']} | {c['duration_s']} | {c['sampling_frequency_hz']} "
            f"| {c['units']} | {c['mean']} | {c['std']} | {c['rms']} | {c['skewness']} "
            f"| {c['kurtosis']} | {c['min']} | {c['max']} |"
        )

        # Generate time-domain plot per dataset
        try:
            time, signal, _ = load_signal(c["filepath"])
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(time, signal, linewidth=0.6, color="steelblue")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(c["units"])
            ax.set_title(f"{c['name']} — {c['n_samples']} samples, fs={c['sampling_frequency_hz']} Hz")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{c['name']}_overview.png", dpi=120)
            plt.close(fig)
        except Exception:
            pass

    lines += [
        "",
        "## Interpretation",
        "",
        "- **High kurtosis** (>3) indicates heavy tails or impulsive content (launch spikes, impulse noise).",
        "- **Skewness** away from zero suggests asymmetric dynamics (e.g. thrust bias during launch).",
        "- **RMS vs std** divergence indicates DC offset (gravity baseline ~9.81 m/s² in pre-launch).",
        "",
        f"Plots saved to `{plot_dir}/`.",
        "",
    ]

    report = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report
