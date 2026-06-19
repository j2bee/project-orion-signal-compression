"""Synthetic rocket telemetry signal generator."""

from pathlib import Path

import numpy as np

from src.data_loader.loader import save_signal


def generate_synthetic_rocket(
    duration: float = 100.0,
    sampling_frequency: float = 200.0,
    seed: int = 42,
) -> tuple:
    """
    Generate a synthetic rocket acceleration telemetry signal.

    Phases:
    - 0–10 s:   Pre-launch stationary (gravity + minor sensor noise)
    - 10–20 s:  Launch acceleration spike with damped oscillation
    - 20–100 s: Flight vibration (multi-frequency oscillation + decay)
    - ~100 s:   Descent phase (decreasing acceleration)

    Parameters
    ----------
    duration : float
        Total signal duration in seconds.
    sampling_frequency : float
        Sampling rate in Hz.
    seed : int
        Random seed.

    Returns
    -------
    time : np.ndarray
        Time axis.
    signal : np.ndarray
        Acceleration in m/s² (approximate).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sampling_frequency)
    time = np.linspace(0, duration, n_samples, endpoint=False)
    signal = np.zeros(n_samples)

    g = 9.81  # gravitational baseline

    for i, t in enumerate(time):
        if t < 10.0:
            # Pre-launch: stationary at ~1g with minor noise
            signal[i] = g + rng.normal(0, 0.05)

        elif t < 20.0:
            # Launch: rapid acceleration spike with damped oscillation
            t_launch = t - 10.0
            thrust = 30.0 * np.exp(-0.3 * t_launch)  # decaying thrust
            oscillation = 2.0 * np.sin(2 * np.pi * 8 * t_launch) * np.exp(-0.5 * t_launch)
            signal[i] = g + thrust + oscillation + rng.normal(0, 0.5)

        elif t < 90.0:
            # Flight: vibration from aerodynamic forces and engine
            t_flight = t - 20.0
            vibration = (
                1.5 * np.sin(2 * np.pi * 3.5 * t_flight)
                + 0.8 * np.sin(2 * np.pi * 12 * t_flight)
                + 0.3 * np.sin(2 * np.pi * 25 * t_flight)
            )
            decay = np.exp(-0.01 * t_flight)
            signal[i] = g * 0.1 + vibration * decay + rng.normal(0, 0.2)

        else:
            # Descent: decreasing acceleration toward free-fall
            t_descent = t - 90.0
            signal[i] = g * 0.05 * np.exp(-0.1 * t_descent) + rng.normal(0, 0.1)

    return time, signal


def create_synthetic_dataset(
    output_path: str = "data/raw/synthetic_rocket.csv",
    duration: float = 100.0,
    sampling_frequency: float = 200.0,
    seed: int = 42,
) -> str:
    """
    Generate and save synthetic rocket telemetry to CSV.

    Parameters
    ----------
    output_path : str
        Output file path.
    duration : float
        Signal duration in seconds.
    sampling_frequency : float
        Sampling rate in Hz.
    seed : int
        Random seed.

    Returns
    -------
    str
        Path to saved file.
    """
    time, signal = generate_synthetic_rocket(duration, sampling_frequency, seed)
    save_signal(output_path, time, signal, format="csv")
    return output_path


if __name__ == "__main__":
    path = create_synthetic_dataset()
    print(f"Synthetic rocket telemetry saved to {path}")
