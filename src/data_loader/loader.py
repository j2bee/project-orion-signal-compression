"""Universal signal loader supporting CSV, TXT, and NumPy array formats."""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd


def load_signal(
    filepath: Union[str, Path],
    time_column: str = "time",
    signal_column: str = "signal",
    sampling_frequency: float = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a time-series signal from disk.

    Supports CSV, TXT (whitespace-delimited), and NumPy (.npy / .npz) files.

    Parameters
    ----------
    filepath : str or Path
        Path to the signal file.
    time_column : str
        Column name for time values in CSV files.
    signal_column : str
        Column name for signal amplitude in CSV files.
    sampling_frequency : float, optional
        Override sampling rate in Hz. If None, inferred from time array.

    Returns
    -------
    time : np.ndarray
        Time axis in seconds.
    signal : np.ndarray
        Signal amplitude values.
    metadata : dict
        Sampling frequency, sample count, duration, and signal statistics.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Signal file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        time, signal = _load_csv(filepath, time_column, signal_column)
    elif suffix in (".txt", ".dat"):
        time, signal = _load_txt(filepath)
    elif suffix == ".npy":
        time, signal = _load_npy(filepath)
    elif suffix == ".npz":
        time, signal = _load_npz(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    metadata = _build_metadata(time, signal, sampling_frequency)
    return time, signal, metadata


def _load_csv(
    filepath: Path, time_column: str, signal_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load time and signal columns from a CSV file."""
    df = pd.read_csv(filepath)
    if time_column not in df.columns or signal_column not in df.columns:
        # Fall back to first two numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError(
                f"CSV must contain '{time_column}' and '{signal_column}' columns "
                f"or at least two numeric columns."
            )
        time = df[numeric_cols[0]].values.astype(np.float64)
        signal = df[numeric_cols[1]].values.astype(np.float64)
    else:
        time = df[time_column].values.astype(np.float64)
        signal = df[signal_column].values.astype(np.float64)
    return time, signal


def _load_txt(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load whitespace-delimited two-column TXT file (time, signal)."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        # Single column — synthesize uniform time axis
        signal = data.astype(np.float64)
        time = np.arange(len(signal), dtype=np.float64)
    elif data.shape[1] >= 2:
        time = data[:, 0].astype(np.float64)
        signal = data[:, 1].astype(np.float64)
    else:
        raise ValueError("TXT file must have at least one column of data.")
    return time, signal


def _load_npy(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NumPy array; synthesize time if only signal is stored."""
    arr = np.load(filepath)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        time = arr[:, 0].astype(np.float64)
        signal = arr[:, 1].astype(np.float64)
    else:
        signal = arr.flatten().astype(np.float64)
        time = np.arange(len(signal), dtype=np.float64)
    return time, signal


def _load_npz(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load time and signal arrays from an NPZ archive."""
    data = np.load(filepath)
    if "time" in data and "signal" in data:
        return data["time"].astype(np.float64), data["signal"].astype(np.float64)
    keys = list(data.keys())
    if len(keys) >= 2:
        return data[keys[0]].astype(np.float64), data[keys[1]].astype(np.float64)
    signal = data[keys[0]].flatten().astype(np.float64)
    time = np.arange(len(signal), dtype=np.float64)
    return time, signal


def _build_metadata(
    time: np.ndarray, signal: np.ndarray, sampling_frequency: float = None
) -> Dict[str, Any]:
    """Compute signal metadata including sampling rate and statistics."""
    n_samples = len(signal)
    duration = float(time[-1] - time[0]) if n_samples > 1 else 0.0

    if sampling_frequency is not None:
        fs = float(sampling_frequency)
    elif n_samples > 1 and duration > 0:
        fs = (n_samples - 1) / duration
    else:
        fs = 1.0

    return {
        "sampling_frequency": fs,
        "n_samples": n_samples,
        "duration": duration,
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "peak_to_peak": float(np.max(signal) - np.min(signal)),
    }


def save_signal(
    filepath: Union[str, Path],
    time: np.ndarray,
    signal: np.ndarray,
    format: str = "csv",
) -> None:
    """
    Save a time-series signal to disk.

    Parameters
    ----------
    filepath : str or Path
        Output path.
    time : np.ndarray
        Time axis.
    signal : np.ndarray
        Signal values.
    format : str
        Output format: 'csv', 'txt', or 'npy'.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df = pd.DataFrame({"time": time, "signal": signal})
        df.to_csv(filepath, index=False)
    elif format == "txt":
        np.savetxt(filepath, np.column_stack([time, signal]))
    elif format == "npy":
        np.save(filepath, np.column_stack([time, signal]))
    else:
        raise ValueError(f"Unsupported save format: {format}")
