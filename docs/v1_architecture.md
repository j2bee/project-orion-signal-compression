# v1 Architecture Reference (Baseline — Preserved)

This document describes the **original v1 implementation** preserved for progress tracking. Do not modify v1 modules; extend via `src/v2/` instead.

## Module Layout

```
src/
├── data_loader/loader.py       load_signal(), save_signal()
├── data_loader/validation.py   validate_signal()
├── preprocessing/
│   ├── noise.py                add_gaussian_noise, add_impulse_noise, add_sensor_drift
│   ├── filtering.py            moving_average, butterworth_lowpass, savgol_filter
│   └── normalization.py        min_max_scale, standardize
├── compression/
│   ├── fft_compression.py      compress_fft, decompress_fft
│   ├── wavelet_compression.py  compress_wavelet, decompress_wavelet
│   └── quantization.py         compress_quantization, decompress_quantization
├── reconstruction/reconstruction.py   reconstruct(compressed, method)
├── metrics/                    mse, rmse, snr_db, compression_ratio
├── visualization/plots.py      plot_signal_overview, plot_comparison
└── pipeline.py                 run_pipeline(), CLI
```

## Compression API (v1)

All compressors return a `dict` with `"method"` key for dispatch.

### FFT
```python
compressed = compress_fft(signal, keep_percentage=0.1)
reconstructed = decompress_fft(compressed)
```

### Wavelet
```python
compressed = compress_wavelet(signal, wavelet="db4", keep_percentage=0.1)
reconstructed = decompress_wavelet(compressed)
```

### Quantization
```python
compressed = compress_quantization(signal, bits=8)
reconstructed = decompress_quantization(compressed)
```

## Pipeline Stages (v1)

1. Load signal (CSV/TXT/NPY)
2. Add composite noise (Gaussian + impulse + drift)
3. Butterworth low-pass filter (15 Hz cutoff)
4. Compress
5. Reconstruct
6. Compute metrics vs pre-compression working signal
7. Save plots and `results/final_report.json`

## CLI (v1)

```bash
python3 src/pipeline.py \
  --input data/raw/synthetic_rocket.csv \
  --compression fft \
  --compression_rate 0.1
```

Options: `--compression {fft,wavelet,quantization}`, `--wavelet`, `--no-noise`, `--no-filter`
