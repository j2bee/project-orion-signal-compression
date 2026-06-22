"""Synthetic rocket telemetry signal generator (refined).

Generates clean physics-based acceleration profiles without embedded sensor
noise — noise is applied separately by the pipeline for realistic control.
Phases use smooth sigmoid transitions to avoid artificial high-frequency artifacts.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data_loader.loader import save_signal

G = 9.81  # m/s² gravitational baseline


def _smoothstep(t: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Smooth 0→1 transition between edge0 and edge1 (C1 continuous)."""
    x = np.clip((t - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def generate_synthetic_rocket(
    duration: float = 100.0,
    sampling_frequency: float = 200.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a clean synthetic rocket acceleration telemetry signal.

    Phases (smooth transitions at boundaries):
    - 0–10 s:   Pre-launch stationary (~1g baseline)
    - 10–20 s:  Launch thrust spike with damped structural oscillation
    - 20–90 s:  Flight vibration (multi-frequency, decaying)
    - 90–100 s: Descent / engine cutoff

    No sensor noise is added here — use preprocessing/noise.py in the pipeline.

    Parameters
    ----------
    duration : float
        Total signal duration in seconds.
    sampling_frequency : float
        Sampling rate in Hz.
    seed : int
        Random seed (reserved for future stochastic physics extensions).

    Returns
    -------
    time : np.ndarray
        Time axis in seconds.
    signal : np.ndarray
        Clean acceleration in m/s².
    """
    del seed  # reserved; signal is deterministic physics model
    n_samples = int(duration * sampling_frequency)
    time = np.linspace(0, duration, n_samples, endpoint=False)
    t = time

    # --- Phase envelopes (smooth transitions) ---
    pre_launch = 1.0 - _smoothstep(t, 9.5, 10.5)
    launch = _smoothstep(t, 9.5, 10.5) * (1.0 - _smoothstep(t, 19.5, 20.5))
    flight = _smoothstep(t, 19.5, 20.5) * (1.0 - _smoothstep(t, 89.5, 90.5))
    descent = _smoothstep(t, 89.5, 90.5)

    # --- Phase components ---
    # Pre-launch: quiet 1g baseline (ideal for noise PSD estimation)
    pre_launch_sig = np.full(n_samples, G) * pre_launch

    # Launch: thrust curve + damped oscillation (structural mode ~8 Hz)
    t_launch = np.maximum(t - 10.0, 0.0)
    thrust = 35.0 * (1.0 - np.exp(-1.5 * t_launch)) * np.exp(-0.25 * t_launch)
    oscillation = 2.5 * np.sin(2 * np.pi * 8.0 * t_launch) * np.exp(-0.6 * t_launch)
    launch_sig = (G + thrust + oscillation) * launch

    # Flight: multi-frequency vibration with exponential decay
    t_flight = np.maximum(t - 20.0, 0.0)
    vibration = (
        1.8 * np.sin(2 * np.pi * 3.5 * t_flight)
        + 1.0 * np.sin(2 * np.pi * 12.0 * t_flight)
        + 0.4 * np.sin(2 * np.pi * 25.0 * t_flight)
    )
    flight_decay = np.exp(-0.008 * t_flight)
    flight_sig = (G * 0.15 + vibration * flight_decay) * flight

    # Descent: engine cutoff, decay toward low acceleration
    t_descent = np.maximum(t - 90.0, 0.0)
    descent_sig = (G * 0.08 * np.exp(-0.15 * t_descent)) * descent

    signal = pre_launch_sig + launch_sig + flight_sig + descent_sig
    return time, signal


def get_phase_segments(
    duration: float = 100.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Return time boundaries for rocket flight phases (for segmented metrics).

    Returns
    -------
    dict
        Phase name → (start_s, end_s).
    """
    return {
        "pre_launch": (0.0, 10.0),
        "launch": (10.0, 20.0),
        "flight": (20.0, 90.0),
        "descent": (90.0, min(duration, 100.0)),
    }


def create_synthetic_dataset(
    output_path: str = "data/raw/synthetic_rocket.csv",
    duration: float = 100.0,
    sampling_frequency: float = 200.0,
    seed: int = 42,
) -> str:
    """
    Generate and save clean synthetic rocket telemetry to CSV.

    Parameters
    ----------
    output_path : str
        Output file path.
    duration : float
        Signal duration in seconds.
    sampling_frequency : float
        Sampling rate in Hz.
    seed : int
        Random seed (passed through for API compatibility).

    Returns
    -------
    str
        Path to saved file.
    """
    time, signal = generate_synthetic_rocket(duration, sampling_frequency, seed)
    save_signal(output_path, time, signal, format="csv")
    return output_path


def create_synthetic_corpus(
    output_dir: str = "data/raw",
    n_signals: int = 5,
    duration: float = 100.0,
    sampling_frequency: float = 200.0,
    seeds: List[int] = None,
) -> List[str]:
    """
    Generate multiple synthetic signals for ML training and validation splits.

    Parameters
    ----------
    output_dir : str
        Directory for output CSV files.
    n_signals : int
        Number of signals to generate.
    duration : float
        Duration per signal in seconds.
    sampling_frequency : float
        Sampling rate in Hz.
    seeds : list of int, optional
        Seeds for each signal. Auto-generated if None.

    Returns
    -------
    list of str
        Paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = list(range(42, 42 + n_signals))

    paths = []
    for i, seed in enumerate(seeds[:n_signals]):
        path = output_dir / f"synthetic_rocket_{i:02d}.csv"
        create_synthetic_dataset(str(path), duration, sampling_frequency, seed)
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    path = create_synthetic_dataset()
    print(f"Clean synthetic rocket telemetry saved to {path}")
