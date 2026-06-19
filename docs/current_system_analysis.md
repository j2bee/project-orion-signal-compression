# Current System Analysis — Project Orion Signal Compression

Research audit of the existing implementation. This document describes what is built, how data flows, and known limitations.

## Current Pipeline

```
Input signal (CSV / TXT / NPY)
    ↓
load_signal() — src/data_loader/loader.py
    ↓
validate_signal() — src/data_loader/validation.py
    ↓
[Optional] apply_all_noise() — src/preprocessing/noise.py
    ↓
[Optional] Denoise / Filter
    │  v1: butterworth_lowpass() — src/preprocessing/filtering.py
    │  v2: denoise_multistage() — src/v2/denoising/pipeline.py
    ↓
Compress (v1 or v2 method)
    │  v1: fft, wavelet, quantization — src/compression/
    │  v2: adaptive_fft, soft_wavelet, mulaw, hybrid, ml — src/v2/, src/ml/
    ↓
Store compressed dict (indices/values, coeffs, latents, etc.)
    ↓
reconstruct() — src/reconstruction/reconstruction.py
    ↓
Metrics: mse, rmse, snr_db, compression_ratio — src/metrics/
    ↓
Visualization — src/visualization/plots.py
    ↓
JSON report — results/final_report.json
```

Orchestrated by `src/pipeline.py` (CLI: `--version v1|v2`, `--compression METHOD`).

---

## Module Reference

### `src/data_loader/loader.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `load_signal()` | Universal loader | filepath, column names | time, signal, metadata | No streaming/chunked load |
| `save_signal()` | Save CSV/TXT/NPY | time, signal, format | file on disk | — |

### `src/data_loader/validation.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `validate_signal()` | Check array integrity | time, signal | is_valid, errors | No quality scoring |
| `check_sampling_uniformity()` | Detect irregular dt | time | dict | — |

### `src/preprocessing/noise.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `add_gaussian_noise()` | White noise | signal, noise_level | noisy signal | Level is fraction of std |
| `add_impulse_noise()` | Random spikes | signal, probability | noisy signal | Fixed amplitude default |
| `add_sensor_drift()` | Slow baseline shift | signal | drifted signal | Linear + sinusoidal only |
| `apply_all_noise()` | Composite model | signal | noisy, info dict | Applied once in pipeline |

### `src/preprocessing/filtering.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `moving_average()` | Smoothing | signal, window | filtered | Blurs transients |
| `butterworth_lowpass()` | IIR low-pass | signal, fs, cutoff | filtered | Zero-phase (non-causal) |
| `savgol_filter()` | Polynomial smooth | signal, window | filtered | — |

### `src/compression/fft_compression.py` (v1)

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `compress_fft()` | Top-N magnitude coeffs | signal, keep_percentage | dict | Fixed count, ignores energy |
| `decompress_fft()` | Inverse FFT | compressed dict | reconstructed | — |

### `src/compression/wavelet_compression.py` (v1)

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `compress_wavelet()` | Hard threshold | signal, wavelet, keep% | dict | Discontinuities at threshold |
| `decompress_wavelet()` | Inverse wavelet | compressed | reconstructed | — |

### `src/compression/quantization.py` (v1)

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `compress_quantization()` | Linear bit reduction | signal, bits | dict | Poor dynamic range for spikes |

### `src/v2/compression/adaptive_fft.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `compress_adaptive_fft()` | Energy-adaptive selection | signal, energy_fraction, max_keep | dict | Global spectrum, not event-local |
| `decompress_adaptive_fft()` | Inverse FFT | compressed | reconstructed | — |

### `src/v2/compression/soft_wavelet.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `compress_soft_wavelet()` | Soft DJ threshold | signal, wavelet, keep% | dict | — |
| `decompress_soft_wavelet()` | Inverse wavelet | compressed | reconstructed | — |

### `src/v2/compression/hybrid.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `compress_hybrid()` | FFT + residual wavelet | signal, energy%, residual% | dict | Best accuracy, moderate ratio |
| `decompress_hybrid()` | Sum of stages | compressed | reconstructed | — |

### `src/v2/denoising/pipeline.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `denoise_multistage()` | 5-stage denoising | signal, fs | denoised, info | Non-causal; ~0.4 dB vs clean |

### `src/ml/autoencoder.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `compress_ml()` | Train + encode patches | signal, latent_dim | latents dict | Requires separate checkpoint |
| `decompress_ml()` | Decode from latents | compressed, checkpoint | reconstructed | Needs training time |

### `src/reconstruction/reconstruction.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `reconstruct()` | Unified dispatch | compressed, method | signal | — |

### `src/metrics/`

| Module | Functions | Purpose |
|--------|-----------|---------|
| `mse.py` | mse, rmse | Reconstruction error |
| `snr.py` | snr_db | Signal-to-noise ratio |
| `compression_ratio.py` | compression_ratio, estimate_compressed_size | Storage reduction |
| `segmented.py` | compute_segment_metrics | Per-phase metrics |

### `src/generate_synthetic.py`

| Function | Purpose | Inputs | Outputs | Limitations |
|----------|---------|--------|---------|-------------|
| `generate_synthetic_rocket()` | Clean physics model | duration, fs | time, signal | Single-axis accelerometer only |
| `create_synthetic_dataset()` | Save CSV | path | filepath | — |

### `src/pipeline.py`

| Function | Purpose | Defaults |
|----------|---------|----------|
| `run_pipeline()` | End-to-end experiment | v2, hybrid, rate=0.85 |
| `main()` | CLI entry | — |

---

## Research Question

> How much information can we remove from rocket telemetry while still recovering important physical events (ignition, vibration spikes, acceleration changes)?

**Design principle:** Preserve transients over maximizing compression ratio.

---

## Known Bottlenecks (Pre-Research)

1. **Denoising vs clean fidelity** — v2 multistage denoising limits end-to-end SNR vs clean (~0.3 dB).
2. **Global compression** — FFT/wavelet treat entire signal uniformly; launch spike same budget as quiet pre-launch.
3. **Non-streaming** — Full signal loaded and processed; no packet-level simulation until `src/streaming.py`.
4. **ML baseline** — Autoencoder requires per-signal training; not yet compared fairly to classical methods in one CSV.

---

## Existing Experiments

| Script | Purpose |
|--------|---------|
| `experiments/compare_v1_v2.py` | Side-by-side v1/v2 benchmarks |
| `experiments/baseline_run.py` | Baseline run with saved artifacts |
| `experiments/compression_comparison.py` | Full method sweep → CSV |

---

## File Map

```
src/
├── data_loader/       # Load & validate
├── preprocessing/     # v1 noise & filters
├── compression/       # v1 FFT, wavelet, quant (preserved)
├── v2/compression/    # Improved algorithms
├── v2/denoising/      # Multi-stage denoising
├── ml/                # Autoencoder
├── analysis/          # Time/frequency signal analysis (NEW)
├── research/          # PCA, sparse extensions (NEW)
├── compression/       # windowed compression (NEW)
├── streaming.py       # Packet simulation (NEW)
├── noise_analysis.py  # Noise characterization (NEW)
└── pipeline.py        # CLI runner
```
