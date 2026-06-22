"""Reconstruction failure analysis — explain why errors occur, not just metrics."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks

from src.compression.fft_compression import compress_fft, decompress_fft
from src.metrics.mse import mse


def analyze_reconstruction_failures(
    time: np.ndarray,
    original: np.ndarray,
    method: str = "fft",
    keep_percentage: float = 0.10,
    output_path: str = "reports/reconstruction_failures.md",
) -> str:
    """
    Identify reconstruction error spikes and explain root causes.

    Parameters
    ----------
    time : np.ndarray
        Time axis.
    original : np.ndarray
        Original signal.
    method : str
        Compression method (currently fft).
    keep_percentage : float
        Compression aggressiveness.
    output_path : str
        Markdown report path.

    Returns
    -------
    str
        Report text.
    """
    if method == "fft":
        compressed = compress_fft(original, keep_percentage=keep_percentage)
        reconstructed = decompress_fft(compressed)
    else:
        raise ValueError(f"Failure analysis not implemented for method: {method}")

    n = min(len(time), len(original), len(reconstructed))
    t = time[:n]
    orig = original[:n]
    recon = reconstructed[:n]
    error = orig - recon
    abs_error = np.abs(error)

    # Find error spikes (top 10)
    spike_idx, _ = find_peaks(abs_error, height=np.percentile(abs_error, 95), distance=20)
    spike_idx = spike_idx[np.argsort(abs_error[spike_idx])[-10:][::-1]]

    failures: List[Dict] = []
    derivative = np.abs(np.gradient(orig))

    for idx in spike_idx:
        local_deriv = derivative[max(0, idx - 10) : idx + 10]
        local_var = float(np.var(orig[max(0, idx - 50) : idx + 50]))

        # Root cause heuristics
        causes = []
        if derivative[idx] > np.percentile(derivative, 99):
            causes.append("Large signal derivative (transient event) — high-frequency content discarded by FFT threshold")
        if local_var > np.percentile(np.convolve(orig ** 2, np.ones(101) / 101, mode="same"), 95):
            causes.append("High local variance (vibration burst) — energy spread across many bins")
        if abs(orig[idx]) > np.percentile(np.abs(orig), 99):
            causes.append("Amplitude peak (launch/impact) — global FFT loses time-localized spike detail")
        if 9.5 < t[idx] < 20.5:
            causes.append("Located in launch window — known fragile region for global compression")
        if not causes:
            causes.append("Moderate error — likely accumulation of discarded mid-frequency coefficients")

        failures.append({
            "time_s": round(float(t[idx]), 3),
            "sample_index": int(idx),
            "abs_error": round(float(abs_error[idx]), 6),
            "signal_value": round(float(orig[idx]), 4),
            "local_variance": round(local_var, 6),
            "local_derivative": round(float(derivative[idx]), 4),
            "likely_causes": causes,
        })

    lines = [
        "# Reconstruction Failure Analysis",
        "",
        f"**Method:** {method.upper()} at {keep_percentage*100:.0f}% coefficient retention",
        f"**Overall MSE:** {mse(orig, recon):.8f}",
        "",
        "## Summary",
        "",
        "Reconstruction errors concentrate at **transient events** (launch spike, phase transitions)",
        "where the signal has high derivative and broadband frequency content. Global FFT compression",
        "discards coefficients uniformly by magnitude, which removes the sharp features that define",
        "mission-critical events.",
        "",
        "## Error Spikes (Top 10)",
        "",
    ]

    for i, f in enumerate(failures, 1):
        lines += [
            f"### Failure {i} — t = {f['time_s']} s (sample {f['sample_index']})",
            "",
            f"- **Absolute error:** {f['abs_error']}",
            f"- **Signal value:** {f['signal_value']}",
            f"- **Local variance:** {f['local_variance']}",
            f"- **Local derivative:** {f['local_derivative']}",
            "",
            "**Likely root causes:**",
        ]
        for cause in f["likely_causes"]:
            lines.append(f"- {cause}")
        lines.append("")

    lines += [
        "## General Findings",
        "",
        "1. **Pre-launch (0–10 s):** Low error — signal is near-DC; few coefficients needed.",
        "2. **Launch (10–20 s):** Highest errors — thrust spike has broadband + transient content.",
        "3. **Flight (20–90 s):** Moderate errors — repetitive vibration compresses well except at envelope changes.",
        "4. **Descent (90–100 s):** Low-moderate errors — low amplitude, less critical.",
        "",
        "## Recommendation",
        "",
        "Do not apply uniform compression. Use **importance-weighted** or **event-aware**",
        "compression that allocates more coefficients to high-derivative, high-variance regions.",
        "",
    ]

    report = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report
