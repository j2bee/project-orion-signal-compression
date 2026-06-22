"""Classical research extensions: PCA and sparse-style compression baselines."""

from typing import Dict, Tuple

import numpy as np


def compress_pca(
    signal: np.ndarray,
    n_components: int = 32,
    window_size: int = 256,
) -> Dict:
    """
    PCA compression on overlapping signal windows.

    Treats each window as a vector; retains top principal components.
    Classical baseline for comparison with FFT and wavelet methods.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    n_components : int
        Number of PCA components to retain.
    window_size : int
        Window length for patch extraction.

    Returns
    -------
    dict
        PCA compressed representation.
    """
    from sklearn.decomposition import PCA

    n = len(signal)
    stride = window_size // 2
    patches = []
    for start in range(0, n - window_size + 1, stride):
        patches.append(signal[start : start + window_size])
    if not patches:
        padded = np.pad(signal, (0, window_size - n))
        patches = [padded]

    X = np.array(patches)
    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(X)

    return {
        "method": "pca",
        "n_samples": n,
        "window_size": window_size,
        "n_components": n_comp,
        "n_patches": len(patches),
        "stride": stride,
        "components": pca.components_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
        "transformed": transformed.astype(np.float32),
    }


def decompress_pca(compressed: Dict) -> np.ndarray:
    """
    Reconstruct signal from PCA representation via overlap-add.

    Parameters
    ----------
    compressed : dict
        Output of compress_pca().

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    n = compressed["n_samples"]
    window_size = compressed["window_size"]
    stride = compressed["stride"]
    components = compressed["components"]
    mean = compressed["mean"]
    transformed = compressed["transformed"]

    output = np.zeros(n)
    weight = np.zeros(n)
    hann = np.hanning(window_size)

    for i, coeffs in enumerate(transformed):
        start = i * stride
        patch = coeffs @ components + mean
        end = min(start + window_size, n)
        plen = end - start
        w = hann[:plen]
        output[start:end] += patch[:plen] * w
        weight[start:end] += w

    weight = np.maximum(weight, 1e-8)
    return output / weight
