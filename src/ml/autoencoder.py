"""Machine learning compression via 1D convolutional autoencoder (refined).

Self-supervised patch-based autoencoder with:
- Temporal conv encoder (no global pooling — preserves transients)
- Hann-window overlap-add reconstruction
- Combined time + spectral training loss
- Honest latent-only compressed payload (no precomputed reconstruction)
- Optional checkpoint path for model weights (separate from payload)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEFAULT_CHECKPOINT = "data/processed/ml_autoencoder.pt"


class SignalAutoencoder(nn.Module):
    """
    1D convolutional autoencoder preserving temporal structure.

    Uses strided convolutions without global pooling so launch spikes
    and oscillations retain spatial information in the latent code.
    """

    def __init__(self, patch_size: int = 512, latent_dim: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self._enc_out_len = patch_size // 8

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self._enc_out_len, latent_dim),
        )

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
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out[..., : self.patch_size]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def _hann_window(size: int) -> np.ndarray:
    """Hann window for smooth overlap-add."""
    return np.hanning(size)


def _extract_patches(signal: np.ndarray, patch_size: int, stride: int = None) -> np.ndarray:
    """Extract overlapping patches from a 1D signal."""
    if stride is None:
        stride = patch_size // 4
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
    """Hann-window overlap-add reconstruction from decoded patches."""
    if stride is None:
        stride = patch_size // 4
    window = _hann_window(patch_size)
    output = np.zeros(n_samples)
    weight = np.zeros(n_samples)

    for i, patch in enumerate(patches):
        start = i * stride
        end = min(start + patch_size, n_samples)
        plen = end - start
        w = window[:plen]
        output[start:end] += patch[:plen] * w
        weight[start:end] += w

    weight = np.maximum(weight, 1e-8)
    return output / weight


def _spectral_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """FFT magnitude MSE — penalizes smeared frequency content."""
    r_fft = torch.fft.rfft(reconstructed.squeeze(1), dim=-1)
    t_fft = torch.fft.rfft(target.squeeze(1), dim=-1)
    return torch.mean((torch.abs(r_fft) - torch.abs(t_fft)) ** 2)


def _train_model(
    model: "SignalAutoencoder",
    tensor_patches: torch.Tensor,
    epochs: int,
    learning_rate: float,
    batch_size: int = 32,
) -> float:
    """Train autoencoder with time + spectral loss using mini-batches."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    time_criterion = nn.MSELoss()
    n = tensor_patches.shape[0]
    final_loss = 0.0

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = tensor_patches[idx]
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = time_criterion(recon, batch) + 0.3 * _spectral_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        final_loss = epoch_loss / max(n_batches, 1)
    return final_loss


def compress_ml(
    signal: np.ndarray,
    latent_dim: int = 32,
    patch_size: int = 512,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    seed: int = 42,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
) -> Dict:
    """
    Compress signal using a trained convolutional autoencoder.

    Stores only latent vectors in the compressed payload. Model weights
    are saved separately to checkpoint_path for reconstruction.

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
    checkpoint_path : str
        Path to save/load model weights (not counted in payload size).

    Returns
    -------
    dict
        Compressed latent vectors and normalization metadata.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for ML compression. Install: pip install torch")

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(signal)
    sig_mean = float(np.mean(signal))
    sig_std = float(np.std(signal)) or 1.0
    normalized = (signal - sig_mean) / sig_std

    stride = patch_size // 4
    patches = _extract_patches(normalized, patch_size, stride)
    n_patches = patches.shape[0]

    device = torch.device("cpu")
    model = SignalAutoencoder(patch_size=patch_size, latent_dim=latent_dim).to(device)
    tensor_patches = torch.tensor(patches, dtype=torch.float32).to(device)

    final_loss = _train_model(model, tensor_patches, epochs, learning_rate)

    # Save model weights separately from compressed payload
    ckpt = Path(checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)

    model.eval()
    with torch.no_grad():
        _, latents = model(tensor_patches)
        latents_np = latents.cpu().numpy().astype(np.float32)

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
        "stride": stride,
        "training_epochs": epochs,
        "final_loss": float(final_loss),
        "checkpoint_path": str(ckpt),
    }


def decompress_ml(compressed: Dict, checkpoint_path: str = None) -> np.ndarray:
    """
    Reconstruct signal from latent vectors via saved model checkpoint.

    Parameters
    ----------
    compressed : dict
        Output of compress_ml() (latents only).
    checkpoint_path : str, optional
        Override path to model weights.

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for ML decompression.")

    patch_size = compressed["patch_size"]
    latent_dim = compressed["latent_dim"]
    n = compressed["n_samples"]
    sig_mean = compressed["sig_mean"]
    sig_std = compressed["sig_std"]
    latents = compressed["latents"]
    stride = compressed.get("stride", patch_size // 4)
    ckpt = checkpoint_path or compressed.get("checkpoint_path", DEFAULT_CHECKPOINT)

    device = torch.device("cpu")
    model = SignalAutoencoder(patch_size=patch_size, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        z = torch.tensor(latents, dtype=torch.float32).to(device)
        recon_patches = model.decode(z).cpu().numpy().squeeze(1)

    full_recon = _reconstruct_from_patches(recon_patches, n, patch_size, stride)
    return full_recon * sig_std + sig_mean


def estimate_ml_compression_ratio(
    n_samples: int, n_patches: int, latent_dim: int
) -> float:
    """Estimate compression ratio from latent payload only (excludes model weights)."""
    original_bytes = n_samples * 8
    compressed_bytes = n_patches * latent_dim * 4
    return original_bytes / max(compressed_bytes, 1)
