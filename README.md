# Project Orion: Signal Compression & Reconstruction

**Reducing telemetry bandwidth while maintaining reconstruction accuracy.**

Project Orion is a modular signal processing pipeline for rocket telemetry compression. It loads sensor data, simulates realistic noise, filters and preprocesses signals, applies compression algorithms, reconstructs the original waveform, and quantifies degradation with standard metrics.

> **Version note:** v1 baseline code is preserved unchanged in `src/compression/`, `src/preprocessing/`, and `src/reconstruction/` as a progress reference. v2 improvements live in `src/v2/` and `src/ml/`. See [docs/progress.md](docs/progress.md) for the full evolution timeline.

## Pipeline Overview

```
Rocket Sensor
      ↓
Signal Loading (CSV / TXT / NumPy)
      ↓
Noise Simulation (Gaussian, Impulse, Drift)
      ↓
Preprocessing / Denoising (v1: Butterworth | v2: Multi-stage)
      ↓
Compression (FFT / Wavelet / Quantization / ML)
      ↓
Transmission / Storage
      ↓
Reconstruction (Inverse Transform / Neural Decoder)
      ↓
Error Evaluation (MSE, RMSE, SNR, Compression Ratio)
      ↓
Visualization & Reports
```

## Signal Characterization (Before Compression Optimization)

**Understand the signal first.** Run the full 10-step characterization study:

```bash
# One-command dashboard — raw plot, FFT, spectrogram, events, importance, error map
python3 analyze_signal.py

# Run specific steps
python3 experiments/run_characterization.py --steps dataset,frequency,sensitivity,adaptive

# Custom input file
python3 analyze_signal.py --input data/raw/synthetic_rocket.csv
```

**Reports produced:**

| Report | Description |
|--------|-------------|
| `reports/dataset_characterization.md` | Stats for every signal dataset |
| `reports/noise_characterization.md` | Noise type detection + SNR |
| `reports/reconstruction_failures.md` | Per-spike root cause analysis |
| `reports/signal_research_answers.md` | 6 research questions with evidence |

See [docs/signal_characterization.md](docs/signal_characterization.md) for the full study guide.

## Quick Start

```bash
pip install -r requirements.txt

# v2 improved pipeline (default)
python3 src/pipeline.py --generate-data --compression adaptive_fft --compression_rate 0.90

# v1 baseline for comparison
python3 src/pipeline.py --version v1 --compression fft --compression_rate 0.1

# ML autoencoder compression
python3 src/pipeline.py --compression ml --compression_rate 32 --ml-epochs 50

# Compare v1 vs v2 benchmarks
python3 experiments/compare_v1_v2.py
```

## Project Structure

```
├── src/
│   ├── compression/          # v1 baseline (preserved)
│   ├── preprocessing/        # v1 baseline (preserved)
│   ├── reconstruction/       # Unified interface (v1 + v2 dispatch)
│   ├── v2/
│   │   ├── compression/      # Adaptive FFT, soft wavelet, μ-law, hybrid
│   │   └── denoising/        # Multi-stage: median, spectral, Wiener, drift
│   ├── ml/
│   │   └── autoencoder.py    # 1D conv autoencoder compression
│   ├── characterization/     # Signal study (dataset, FFT, events, sensitivity)
│   ├── importance.py         # Per-sample importance mask (0–1)
│   ├── pipeline.py           # CLI with --version v1|v2
│   └── ...
├── analyze_signal.py         # One-command characterization dashboard
├── experiments/
│   ├── compare_v1_v2.py      # Side-by-side benchmark script
│   └── run_characterization.py  # Step-by-step characterization runner
├── docs/
│   ├── progress.md           # v1 → v2 evolution timeline
│   ├── v1_architecture.md    # v1 API reference (preserved)
│   └── signal_characterization.md  # 10-step signal study guide
```

## Methods

| Method | Version | Status | Description |
|--------|---------|--------|-------------|
| FFT Compression | v1 | Complete | Top-N magnitude coefficient selection |
| Wavelet Compression | v1 | Complete | Hard threshold by keep-percentage |
| Quantization | v1 | Complete | Linear 8/16-bit mapping |
| **Adaptive FFT** | **v2** | **Complete** | Energy-preserving coefficient selection |
| **Soft Wavelet** | **v2** | **Complete** | Donoho-Johnstone soft thresholding |
| **μ-law Quantization** | **v2** | **Complete** | Companded quantization for spike preservation |
| **Hybrid FFT+Wavelet** | **v2** | **Complete** | Two-stage with residual encoding |
| **Multi-stage Denoising** | **v2** | **Complete** | Median → spectral → Wiener → drift → adaptive LP |
| **ML Autoencoder** | **v2** | **Complete** | Self-supervised 1D conv autoencoder |

## v2 Improvements Summary

| Area | v1 | v2 |
|------|----|----|
| Denoising | Single Butterworth (15 Hz) | 5-stage pipeline (impulse, spectral, Wiener, drift, adaptive LP) |
| FFT | Fixed keep-% by count | Energy-adaptive selection |
| Wavelet | Hard threshold | Soft threshold (universal) |
| Quantization | Linear | μ-law companding |
| ML | — | Convolutional autoencoder with latent bottleneck |

## Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root MSE (same units as signal) |
| **SNR** | Signal-to-Noise Ratio in dB |
| **Compression Ratio** | Original size / compressed size |

Results saved to `results/final_report.json`.

## Running Tests

```bash
pytest tests/ -v                    # All tests (v1 + v2)
pytest tests/test_pipeline.py -v    # v1 baseline tests
pytest tests/test_v2.py -v          # v2 improvement tests
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_signal_exploration.ipynb` | Full v1 walkthrough |
| `02_fft_analysis.ipynb` | FFT parameter sweep |
| `03_compression_testing.ipynb` | Method comparison |
| `04_noise_analysis.ipynb` | Noise impact study |
| `05_v2_ml_comparison.ipynb` | v1 vs v2 vs ML comparison |
| `research/01_signal_characterization.ipynb` | Full 10-step characterization walkthrough |

## Documentation

- [docs/progress.md](docs/progress.md) — Development timeline and v1→v2 changelog
- [docs/v1_architecture.md](docs/v1_architecture.md) — Preserved v1 API reference
- [docs/signal_characterization.md](docs/signal_characterization.md) — Signal study before compression optimization
- [docs/repository_analysis.md](docs/repository_analysis.md) — Initial repository analysis

## License

Research engineering project — Project Orion Signal Compression.
