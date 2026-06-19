"""Automatic detection of mission-critical physical events in rocket telemetry."""

from typing import Dict, List

import numpy as np
from scipy.signal import find_peaks


def detect_events(
    time: np.ndarray,
    signal: np.ndarray,
    sampling_frequency: float,
) -> Dict:
    """
    Detect rocket mission events using derivative, peak, energy, and variance methods.

    Event types:
    - launch / ignition (large positive derivative + peak)
    - vibration onset (variance step increase)
    - energy burst (local RMS spike)
    - impact / separation (sharp peak in |derivative|)

    Parameters
    ----------
    time : np.ndarray
        Time axis in seconds.
    signal : np.ndarray
        Signal values.
    sampling_frequency : float
        Sampling rate in Hz.

    Returns
    -------
    dict
        events list with type, time_s, index, confidence, method.
    """
    n = len(signal)
    dt = 1.0 / sampling_frequency
    events: List[Dict] = []

    # Derivative-based detection
    derivative = np.gradient(signal, dt)
    deriv_threshold = np.percentile(np.abs(derivative), 99.0)
    deriv_peaks, _ = find_peaks(np.abs(derivative), height=deriv_threshold, distance=int(fs * 0.5) if (fs := sampling_frequency) else 50)

    for idx in deriv_peaks:
        events.append({
            "type": "rapid_change",
            "time_s": round(float(time[idx]), 3),
            "index": int(idx),
            "confidence": round(float(np.abs(derivative[idx]) / deriv_threshold), 2),
            "method": "derivative_threshold",
        })

    # Amplitude peak detection (launch spike, impacts)
    peak_threshold = np.percentile(signal, 99.5)
    peaks, props = find_peaks(signal, height=peak_threshold, distance=int(sampling_frequency * 0.3))
    for idx in peaks:
        events.append({
            "type": "amplitude_peak",
            "time_s": round(float(time[idx]), 3),
            "index": int(idx),
            "amplitude": round(float(signal[idx]), 4),
            "confidence": round(float(signal[idx] / peak_threshold), 2),
            "method": "peak_detection",
        })

    # Moving variance bursts (vibration onset)
    window = max(11, int(sampling_frequency * 0.1) | 1)
    kernel = np.ones(window) / window
    local_var = np.convolve(signal ** 2, kernel, mode="same") - np.convolve(signal, kernel, mode="same") ** 2
    var_diff = np.diff(local_var, prepend=local_var[0])
    var_threshold = np.percentile(var_diff, 98.0)
    var_peaks, _ = find_peaks(var_diff, height=var_threshold, distance=int(sampling_frequency * 2))

    for idx in var_peaks:
        events.append({
            "type": "vibration_onset",
            "time_s": round(float(time[idx]), 3),
            "index": int(idx),
            "confidence": round(float(var_diff[idx] / var_threshold), 2),
            "method": "moving_variance",
        })

    # Energy burst (local RMS)
    rms_window = max(21, int(sampling_frequency * 0.2) | 1)
    local_rms = np.sqrt(np.convolve(signal ** 2, np.ones(rms_window) / rms_window, mode="same"))
    rms_threshold = np.percentile(local_rms, 95.0)
    rms_peaks, _ = find_peaks(local_rms, height=rms_threshold, distance=int(sampling_frequency * 1.0))

    for idx in rms_peaks:
        events.append({
            "type": "energy_burst",
            "time_s": round(float(time[idx]), 3),
            "index": int(idx),
            "confidence": round(float(local_rms[idx] / rms_threshold), 2),
            "method": "local_rms",
        })

    # Label known phases from synthetic rocket timeline (reference for clean signals)
    phase_labels = _label_known_phases(time)

    # Deduplicate events within 0.5s
    events.sort(key=lambda e: e["time_s"])
    deduped = []
    last_t = -999
    for e in events:
        if e["time_s"] - last_t > 0.5:
            deduped.append(e)
            last_t = e["time_s"]

    return {
        "n_events": len(deduped),
        "events": deduped,
        "phase_reference": phase_labels,
        "mission_critical_regions": [
            {"phase": "launch", "start_s": 10.0, "end_s": 20.0,
             "description": "Ignition and thrust spike — highest compression sensitivity"},
            {"phase": "flight", "start_s": 20.0, "end_s": 90.0,
             "description": "Vibration — moderate importance, repetitive structure"},
            {"phase": "descent", "start_s": 90.0, "end_s": 100.0,
             "description": "Engine cutoff / descent — lower amplitude"},
        ],
    }


def _label_known_phases(time: np.ndarray) -> List[Dict]:
    """Reference phase boundaries for synthetic rocket (100s mission)."""
    duration = time[-1] if len(time) > 0 else 100.0
    return [
        {"phase": "pre_launch", "start_s": 0.0, "end_s": 10.0},
        {"phase": "launch", "start_s": 10.0, "end_s": 20.0},
        {"phase": "flight", "start_s": 20.0, "end_s": min(90.0, duration)},
        {"phase": "descent", "start_s": 90.0, "end_s": duration},
    ]
