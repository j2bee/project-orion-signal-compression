"""Mean Squared Error and Root Mean Squared Error metrics."""

import numpy as np


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Mean Squared Error between original and reconstructed signals.

    MSE = mean((original - reconstructed)^2)

    Lower is better. Sensitive to large errors (quadratic penalty).

    Parameters
    ----------
    original : np.ndarray
        Reference signal.
    reconstructed : np.ndarray
        Reconstructed signal.

    Returns
    -------
    float
        MSE value.
    """
    n = min(len(original), len(reconstructed))
    diff = original[:n] - reconstructed[:n]
    return float(np.mean(diff ** 2))


def rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    RMSE = sqrt(MSE) — same units as the signal amplitude.

    Parameters
    ----------
    original : np.ndarray
        Reference signal.
    reconstructed : np.ndarray
        Reconstructed signal.

    Returns
    -------
    float
        RMSE value.
    """
    return float(np.sqrt(mse(original, reconstructed)))
