"""Generate combined signal analysis reports."""

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np

from .frequency_domain import analyze_frequency_domain
from .time_domain import analyze_time_domain


def generate_analysis_report(
    signal: np.ndarray,
    sampling_frequency: float,
    time: np.ndarray = None,
    output_path: Union[str, Path] = "results/signal_analysis_report.json",
) -> Dict:
    """
    Generate and save a combined time + frequency analysis report.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sampling_frequency : float
        Sampling rate in Hz.
    time : np.ndarray, optional
        Time axis.
    output_path : str or Path
        JSON output path.

    Returns
    -------
    dict
        Full analysis report.
    """
    time_stats = analyze_time_domain(signal, time)
    freq_stats = analyze_frequency_domain(signal, sampling_frequency)

    # Exclude large arrays from JSON
    report = {
        "time_domain": time_stats,
        "frequency_domain": {
            "dominant_frequencies": freq_stats["dominant_frequencies"],
            "total_spectral_energy": freq_stats["total_spectral_energy"],
            "signal_band_energy": freq_stats["signal_band_energy"],
            "noise_band_energy": freq_stats["noise_band_energy"],
            "noise_fraction": freq_stats["noise_fraction"],
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
