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
