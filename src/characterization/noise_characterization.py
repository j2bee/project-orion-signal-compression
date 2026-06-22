"""Noise characterization — identify noise type and estimate SNR."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import stats


def characterize_noise(
    clean: np.ndarray,
    noisy: np.ndarray,
    sampling_frequency: float,
    output_path: str = "results/characterization/noise_characterization.json",
) -> Dict:
    """
    Characterize noise between clean and noisy signals.

    Tests for:
    - Gaussian (normality of residual)
    - Impulse (kurtosis of residual, spike count)
    - Sensor drift (linear trend in residual)
    - Periodic interference (peaks in residual spectrum)
    - Quantization (staircase in residual histogram)

    Parameters
    ----------
    clean : np.ndarray
        Reference clean signal.
    noisy : np.ndarray
        Observed noisy signal.
    sampling_frequency : float
        Sampling rate in Hz.
    output_path : str
        JSON output path.

    Returns
    -------
    dict
        Noise type assessment, power estimates, SNR.
    """
    n = min(len(clean), len(noisy))
    clean = clean[:n]
    noisy = noisy[:n]
    residual = noisy - clean

    signal_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean(residual ** 2))
    snr_linear = signal_power / noise_power if noise_power > 0 else float("inf")
    snr_db = 10.0 * np.log10(snr_linear) if noise_power > 0 else float("inf")

    # Gaussian test: Shapiro-Wilk on subsample
    subsample = residual[::max(1, len(residual) // 5000)][:5000]
    if len(subsample) >= 8:
        _, gaussian_p = stats.shapiro(subsample)
        is_gaussian = gaussian_p > 0.05
    else:
        gaussian_p = None
        is_gaussian = False

    # Impulse: count outliers beyond 4 sigma
    sigma = np.std(residual)
    impulse_count = int(np.sum(np.abs(residual) > 4 * sigma))
    impulse_fraction = impulse_count / n
    is_impulse = impulse_fraction > 0.001

    # Drift: linear trend in residual
    t = np.arange(n)
    slope, intercept = np.polyfit(t, residual, 1)
    drift_r2 = float(np.corrcoef(t, residual)[0, 1] ** 2)
    is_drift = drift_r2 > 0.3 and abs(slope) > 1e-6

    # Periodic interference: dominant frequency in residual
    resid_spectrum = np.abs(np.fft.rfft(residual))
    resid_freqs = np.fft.rfftfreq(n, d=1.0 / sampling_frequency)
    if len(resid_spectrum) > 1:
        peak_idx = np.argmax(resid_spectrum[1:]) + 1
        interference_hz = float(resid_freqs[peak_idx])
        interference_strength = float(resid_spectrum[peak_idx] / np.mean(resid_spectrum))
        is_periodic = interference_strength > 5.0
    else:
        interference_hz = 0.0
        interference_strength = 0.0
        is_periodic = False

    # Quantization: detect discrete levels in residual histogram
    hist, bin_edges = np.histogram(residual, bins=50)
    peak_ratio = np.max(hist) / (np.mean(hist) + 1)
    is_quantization = peak_ratio > 3.0 and np.std(residual) < 0.1 * np.std(clean)

    noise_types = []
    if is_gaussian:
        noise_types.append("gaussian")
    if is_impulse:
        noise_types.append("impulse")
    if is_drift:
        noise_types.append("sensor_drift")
    if is_periodic:
        noise_types.append("periodic_interference")
    if is_quantization:
        noise_types.append("quantization")
    if not noise_types:
        noise_types.append("mixed_or_unknown")

    results = {
        "signal_power": round(signal_power, 6),
        "noise_power": round(noise_power, 6),
        "snr_db": round(snr_db, 2),
        "residual_std": round(float(np.std(residual)), 6),
        "detected_noise_types": noise_types,
        "tests": {
            "gaussian": {"is_gaussian": bool(is_gaussian), "shapiro_p": round(float(gaussian_p), 4) if gaussian_p else None},
            "impulse": {"is_impulse": bool(is_impulse), "spike_fraction": round(impulse_fraction, 6), "spike_count": impulse_count},
            "drift": {"is_drift": bool(is_drift), "slope": round(float(slope), 8), "r_squared": round(drift_r2, 4)},
            "periodic": {"is_periodic": bool(is_periodic), "dominant_hz": round(interference_hz, 2), "strength_ratio": round(interference_strength, 2)},
            "quantization": {"is_quantization": bool(is_quantization), "hist_peak_ratio": round(float(peak_ratio), 2)},
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def write_noise_report(
    results: Dict,
    output_path: str = "reports/noise_characterization.md",
    plot_dir: str = "results/characterization",
) -> str:
    """
    Write markdown noise characterization report with residual diagnostic plot.

    Parameters
    ----------
    results : dict
        Output of characterize_noise().
    output_path : str
        Markdown report path.
    plot_dir : str
        Directory for diagnostic plots.

    Returns
    -------
    str
        Report text.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tests = results.get("tests", {})
    types = results.get("detected_noise_types", [])

    lines = [
        "# Noise Characterization Report",
        "",
        "Assessment of noise between clean reference and noisy observation.",
        "",
        "## Power and SNR",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Signal power | {results.get('signal_power', 'N/A')} |",
        f"| Noise power | {results.get('noise_power', 'N/A')} |",
        f"| SNR | **{results.get('snr_db', 'N/A')} dB** |",
        f"| Residual std | {results.get('residual_std', 'N/A')} |",
        "",
        "## Detected Noise Types",
        "",
    ]
    for t in types:
        lines.append(f"- **{t}**")
    lines += [
        "",
        "## Test Results",
        "",
        "### Gaussian",
        f"- Is Gaussian: {tests.get('gaussian', {}).get('is_gaussian', 'N/A')}",
        f"- Shapiro-Wilk p-value: {tests.get('gaussian', {}).get('shapiro_p', 'N/A')}",
        "",
        "### Impulse",
        f"- Is impulse: {tests.get('impulse', {}).get('is_impulse', 'N/A')}",
        f"- Spike count: {tests.get('impulse', {}).get('spike_count', 'N/A')}",
        f"- Spike fraction: {tests.get('impulse', {}).get('spike_fraction', 'N/A')}",
        "",
        "### Sensor Drift",
        f"- Is drift: {tests.get('drift', {}).get('is_drift', 'N/A')}",
        f"- Slope: {tests.get('drift', {}).get('slope', 'N/A')}",
        f"- R²: {tests.get('drift', {}).get('r_squared', 'N/A')}",
        "",
        "### Periodic Interference",
        f"- Is periodic: {tests.get('periodic', {}).get('is_periodic', 'N/A')}",
        f"- Dominant frequency: {tests.get('periodic', {}).get('dominant_hz', 'N/A')} Hz",
        f"- Strength ratio: {tests.get('periodic', {}).get('strength_ratio', 'N/A')}",
        "",
        "### Quantization",
        f"- Is quantization: {tests.get('quantization', {}).get('is_quantization', 'N/A')}",
        f"- Histogram peak ratio: {tests.get('quantization', {}).get('hist_peak_ratio', 'N/A')}",
        "",
        "## Interpretation",
        "",
        "- **Gaussian** residuals suggest additive white noise — Wiener/spectral subtraction effective.",
        "- **Impulse** spikes require median filtering before spectral methods.",
        "- **Sensor drift** needs detrending or high-pass filtering.",
        "- **Periodic interference** may need notch filtering at the dominant frequency.",
        "- **Quantization** noise limits effective bit depth — consider μ-law companding.",
        "",
        f"Diagnostic plot: `{plot_dir}/noise_residual_diagnostics.png`",
        "",
    ]

    report = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report


def plot_noise_diagnostics(
    clean: np.ndarray,
    noisy: np.ndarray,
    results: Dict,
    output_path: str = "results/characterization/noise_residual_diagnostics.png",
) -> str:
    """Plot residual histogram and spectrum for noise report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(clean), len(noisy))
    residual = noisy[:n] - clean[:n]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(residual[: min(2000, n)], linewidth=0.5, color="steelblue")
    axes[0, 0].set_title("Residual (first 2000 samples)")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(residual, bins=60, color="darkorange", edgecolor="white", alpha=0.8)
    axes[0, 1].set_title("Residual Histogram")
    axes[0, 1].set_xlabel("Amplitude")

    resid_spectrum = np.abs(np.fft.rfft(residual))
    axes[1, 0].plot(resid_spectrum[: len(resid_spectrum) // 4], linewidth=0.6, color="green")
    axes[1, 0].set_title("Residual Spectrum (low quarter)")
    axes[1, 0].set_xlabel("Bin")

    types = ", ".join(results.get("detected_noise_types", []))
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05, 0.85,
        f"SNR: {results.get('snr_db', 'N/A')} dB\n"
        f"Signal power: {results.get('signal_power', 'N/A')}\n"
        f"Noise power: {results.get('noise_power', 'N/A')}\n"
        f"Types: {types}",
        fontsize=12, verticalalignment="top", family="monospace",
    )

    fig.suptitle("Noise Characterization Diagnostics", fontsize=13)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)
