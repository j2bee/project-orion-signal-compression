"""Realistic noise models for rocket telemetry simulation."""

from typing import Tuple

import numpy as np


def add_gaussian_noise(
    signal: np.ndarray, noise_level: float = 0.05, seed: int = None
) -> np.ndarray:
    """
    Add zero-mean Gaussian (white) noise to the signal.

    Models thermal noise in analog sensor front-ends. noise_level is
    specified as a fraction of the signal's standard deviation.

    Parameters
    ----------
    signal : np.ndarray
        Clean input signal.
    noise_level : float
        Noise amplitude as fraction of signal std (e.g. 0.05 = 5%).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy signal.
    """
    rng = np.random.default_rng(seed)
    sigma = noise_level * np.std(signal)
    noise = rng.normal(0.0, sigma, size=signal.shape)
    return signal + noise


def add_impulse_noise(
    signal: np.ndarray, probability: float = 0.01, amplitude: float = None, seed: int = None
) -> np.ndarray:
    """
    Add random impulse (spike) noise.

    Simulates electromagnetic interference or ADC glitches — sudden large
    deviations at random sample locations.

    Parameters
    ----------
    signal : np.ndarray
        Clean input signal.
    probability : float
        Fraction of samples affected (0 to 1).
    amplitude : float, optional
        Spike magnitude. Defaults to 3× signal peak-to-peak.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Signal with impulse spikes.
    """
    rng = np.random.default_rng(seed)
    noisy = signal.copy()
    if amplitude is None:
        amplitude = 3.0 * (np.max(signal) - np.min(signal))

    mask = rng.random(len(signal)) < probability
    spike_signs = rng.choice([-1.0, 1.0], size=int(mask.sum()))
    noisy[mask] += spike_signs * amplitude
    return noisy


def add_sensor_drift(
    signal: np.ndarray,
    drift_rate: float = None,
    seed: int = None,
) -> np.ndarray:
    """
    Add slow sensor bias drift over time.

    Models temperature-dependent baseline shift in accelerometers during
    long flights. Drift is a low-frequency linear or sinusoidal trend.

    Parameters
    ----------
    signal : np.ndarray
        Clean input signal.
    drift_rate : float, optional
        Maximum drift amplitude. Defaults to 10% of signal peak-to-peak.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Signal with superimposed drift.
    """
    rng = np.random.default_rng(seed)
    n = len(signal)
    if drift_rate is None:
        drift_rate = 0.1 * (np.max(signal) - np.min(signal))

    # Combine linear trend with slow sinusoidal component
    t_norm = np.linspace(0, 1, n)
    linear = drift_rate * t_norm
    sinusoidal = (drift_rate * 0.3) * np.sin(2 * np.pi * 0.5 * t_norm + rng.uniform(0, 2 * np.pi))
    drift = linear + sinusoidal
    return signal + drift


def apply_all_noise(
    signal: np.ndarray,
    gaussian_level: float = 0.05,
    impulse_prob: float = 0.005,
    drift: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Apply a composite realistic noise model.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal.
    gaussian_level : float
        Gaussian noise level (fraction of std).
    impulse_prob : float
        Impulse noise probability.
    drift : bool
        Whether to add sensor drift.
    seed : int
        Random seed.

    Returns
    -------
    noisy_signal : np.ndarray
        Composite noisy signal.
    noise_info : dict
        Parameters used for each noise component.
    """
    rng_seed = seed
    noisy = add_gaussian_noise(signal, gaussian_level, seed=rng_seed)
    noisy = add_impulse_noise(noisy, impulse_prob, seed=rng_seed + 1)
    if drift:
        noisy = add_sensor_drift(noisy, seed=rng_seed + 2)
    return noisy, {
        "gaussian_level": gaussian_level,
        "impulse_probability": impulse_prob,
        "sensor_drift": drift,
        "seed": seed,
    }
