# Project Orion: Signal Compression & Reconstruction

**Reducing telemetry bandwidth while maintaining reconstruction accuracy.**

Project Orion is a modular signal processing pipeline for rocket telemetry compression. It loads sensor data, simulates realistic noise, filters and preprocesses signals, applies compression algorithms, reconstructs the original waveform, and quantifies degradation with standard metrics.

## Pipeline Overview

```
Rocket Sensor
      ↓
Signal Loading (CSV / TXT / NumPy)
      ↓
Noise Simulation (Gaussian, Impulse, Drift)
      ↓
Preprocessing (Normalization, Filtering)
      ↓
Compression (FFT / Wavelet / Quantization)
      ↓
Transmission / Storage
      ↓
Reconstruction (Inverse Transform)
      ↓
Error Evaluation (MSE, RMSE, SNR, Compression Ratio)
      ↓
Visualization & Reports
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic rocket telemetry and run the pipeline
python src/pipeline.py --generate-data --compression fft --compression_rate 0.1

# Run with wavelet compression
python src/pipeline.py --input data/raw/synthetic_rocket.csv --compression wavelet --compression_rate 0.1 --wavelet db4

# Run with quantization (8-bit)
python src/pipeline.py --input data/raw/synthetic_rocket.csv --compression quantization --compression_rate 8
```

## Project Structure

```
├── data/
│   ├── raw/              # Input signals (CSV, TXT, NPY)
│   ├── processed/        # Intermediate processed signals
│   ├── noisy/            # Noise-corrupted signals
│   └── reconstructed/    # Reconstructed output signals
├── src/
│   ├── data_loader/      # Universal signal loader + validation
│   ├── preprocessing/    # Normalization, filtering, noise models
│   ├── compression/      # FFT, wavelet, quantization algorithms
│   ├── reconstruction/   # Unified inverse transform interface
│   ├── metrics/          # MSE, RMSE, SNR, compression ratio
│   ├── visualization/    # Time/frequency domain plots
│   ├── pipeline.py       # End-to-end experiment runner
│   └── generate_synthetic.py  # Synthetic rocket telemetry generator
├── notebooks/            # Jupyter exploration notebooks
├── experiments/          # Experiment configurations
├── results/
│   ├── plots/            # Generated visualization plots
│   └── metrics/          # JSON metric reports
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Current Methods

| Method | Status | Description |
|--------|--------|-------------|
| FFT Compression | Complete | Frequency-domain thresholding, configurable keep % |
| Wavelet Compression | Complete | Haar / Daubechies with coefficient thresholding |
| Quantization | Complete | 8-bit and 16-bit float-to-int mapping |
| Noise Filtering | Complete | Moving average, Butterworth, Savitzky-Golay |
| Noise Simulation | Complete | Gaussian, impulse, sensor drift models |
| ML Compression | Future | Neural autoencoder-based compression |

## Compression Methods

### FFT Compression
Retains the largest-magnitude frequency coefficients and zeroes the rest. Configurable keep percentages: 1%, 5%, 10%, 25%, 50%.

### Wavelet Compression
Uses PyWavelets for multi-resolution decomposition. Supports Haar, Daubechies (db4, db8), and other wavelet families. Hard-thresholds small coefficients.

### Quantization
Maps 64-bit float samples to 8-bit or 16-bit integers. Measures pure precision loss without transform coding.

## Metrics

Every experiment produces:

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error — average squared reconstruction error |
| **RMSE** | Root MSE — same units as signal amplitude |
| **SNR** | Signal-to-Noise Ratio in dB — higher is better |
| **Compression Ratio** | Original size / compressed size |

Results are saved to `results/final_report.json`.

## Running Tests

```bash
pytest tests/ -v
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_signal_exploration.ipynb` | Load, visualize, noise, filter, compress, reconstruct |
| `02_fft_analysis.ipynb` | FFT compression parameter sweep |
| `03_compression_testing.ipynb` | Compare all compression methods |
| `04_noise_analysis.ipynb` | Noise model impact on reconstruction |

## Example Output

```
Method: fft
Compression: 8.5x
MSE: 0.002341
RMSE: 0.048383
SNR: 34.2 dB
```

## Legacy Research

Previous work on QAPF (Quadrature Amplitude Phase Filtering) and TV Baseline filters is preserved in `src/compression.py` and `src/filtering.py` for future integration.

## License

Research engineering project — Project Orion Signal Compression.
