# Project Orion — Development Progress

This document tracks the evolution of the signal compression pipeline. **v1 code is preserved unchanged** in `src/` as a baseline reference; v2 improvements live in `src/v2/` and `src/ml/`.

## Version History

| Version | Status | Description |
|---------|--------|-------------|
| **v1** | Complete (baseline) | End-to-end pipeline with FFT, wavelet, quantization; basic Butterworth filtering |
| **v2** | Complete | Improved compression accuracy, multi-stage denoising, ML autoencoder |
| **v3** | Planned | Real-time streaming, adaptive bitrate, on-board deployment |

---

## v1 Baseline (Preserved)

Location: `src/compression/`, `src/preprocessing/`, `src/reconstruction/`

| Component | Approach | Limitation |
|-----------|----------|------------|
| FFT compression | Top-N magnitude coefficient selection | Ignores energy distribution; fixed count |
| Wavelet compression | Hard threshold by keep-percentage | Discontinuities at threshold boundary |
| Quantization | Linear 8/16-bit mapping | Poor dynamic range for telemetry spikes |
| Denoising | Single Butterworth low-pass (15 Hz) | Cannot remove impulse noise or drift |
| Reconstruction | Inverse transform only | No learned priors for missing data |

See [v1_architecture.md](v1_architecture.md) for full v1 API reference.

---

## v2 Improvements

Location: `src/v2/`, `src/ml/`

### Compression Accuracy

| Method | v1 | v2 Improvement | Expected Gain |
|--------|----|----------------|---------------|
| FFT | Fixed keep-% | **Energy-adaptive selection** — retain coefficients until target energy captured | +3–8 dB SNR at same ratio |
| Wavelet | Hard threshold | **Soft thresholding** (Donoho-Johnstone universal threshold) | Smoother reconstruction |
| Quantization | Linear | **μ-law companding** before quantization | Better spike preservation |
| Hybrid | — | **FFT + residual wavelet** on reconstruction error | Lower MSE at moderate ratios |

### Noise Reduction

| Stage | Technique | Target |
|-------|-----------|--------|
| 1 | Median filter | Impulse/spike removal |
| 2 | Spectral subtraction | Gaussian noise reduction |
| 3 | Wiener filter | Optimal MMSE frequency-domain denoising |
| 4 | Adaptive Butterworth | Auto cutoff from spectral rolloff |
| 5 | Drift correction | Polynomial detrending |

Pipeline: `src/v2/denoising/pipeline.py` → `denoise_multistage()`

### Machine Learning Reconstruction

| Component | Description |
|-----------|-------------|
| Architecture | 1D convolutional autoencoder with latent bottleneck |
| Training | Self-supervised on signal patches (no external dataset) |
| Compression | Latent vector dimension controls ratio |
| Integration | `reconstruct(compressed, method="ml")` |

Location: `src/ml/autoencoder.py`

---

## Metrics Comparison (Synthetic Rocket, benchmarked)

Run with `python3 experiments/compare_v1_v2.py`.

| Metric | v1 FFT 10% | v2 Hybrid | v2 Soft Wavelet |
|--------|------------|-----------|-----------------|
| SNR (dB) | 26.9 | **51.0** | **51.0** |
| MSE | 1.030 | **0.000026** | **0.000027** |
| Compression | 6.7× | 2.7× | 2.7× |

| Denoising | v1 Butterworth | v2 Multi-stage |
|-----------|----------------|----------------|
| SNR vs clean (dB) | -8.3 | **0.4** |
| MSE vs clean | 366.1 | **49.7** |

*v1 code preserved in `src/compression/`, `src/preprocessing/` — unchanged.*

---

## How to Run Each Version

```bash
# v1 baseline (original behavior)
python3 src/pipeline.py --version v1 --compression fft --compression_rate 0.1

# v2 improved pipeline (default)
python3 src/pipeline.py --version v2 --compression adaptive_fft --compression_rate 0.1

# v2 with ML reconstruction
python3 src/pipeline.py --version v2 --compression ml --compression_rate 32

# Compare v1 vs v2
python3 experiments/compare_v1_v2.py
```

---

## File Map

```
src/
├── compression/          # v1 — DO NOT MODIFY (baseline reference)
├── preprocessing/        # v1 — DO NOT MODIFY
├── reconstruction/       # v1 — DO NOT MODIFY
├── v2/
│   ├── compression/      # adaptive_fft, soft_wavelet, mulaw, hybrid
│   └── denoising/        # multistage pipeline, wiener, spectral, median
├── ml/
│   └── autoencoder.py    # PyTorch 1D conv autoencoder
└── pipeline.py           # Supports --version v1|v2
```
