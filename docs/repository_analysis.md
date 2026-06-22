# Repository Analysis — Project Orion Signal Compression

## Current repository structure

```
project-orion-signal-compression/
├── .gitignore
├── README.md              # Initial QAPF/TV Baseline research notes
├── requirements.txt       # Empty placeholder
├── src/
│   ├── compression.py     # Empty placeholder (legacy location)
│   └── filtering.py         # Empty placeholder (legacy location)
└── docs/
    └── repository_analysis.md
```

## Current functionality

- **README**: Documents early research direction toward QAPF (Quadrature Amplitude Phase Filtering) and TV Baseline filters for high-ratio quantization experiments.
- **src/compression.py**: Empty file — intended for future QAPF noise-reduction work.
- **src/filtering.py**: Empty file — intended for filter implementations.
- **requirements.txt**: No dependencies declared yet.
- **No data**, **no tests**, **no notebooks**, and **no runnable pipeline** exist yet.

## Missing components

| Component | Status |
|-----------|--------|
| Data directories (`raw/`, `processed/`, `noisy/`, `reconstructed/`) | Missing |
| Universal signal loader (CSV, TXT, NumPy) | Missing |
| Signal validation and metadata extraction | Missing |
| Preprocessing (normalization, filtering, noise models) | Missing |
| Compression algorithms (FFT, wavelet, quantization) | Missing |
| Reconstruction system with unified interface | Missing |
| Evaluation metrics (MSE, RMSE, SNR, compression ratio) | Missing |
| Visualization tools (time/frequency domain plots) | Missing |
| Automated pipeline runner (`pipeline.py`) | Missing |
| Synthetic rocket telemetry dataset | Missing |
| Jupyter notebooks for exploration | Missing |
| Unit tests | Missing |
| Experiment directories and results storage | Missing |

## Recommended architecture

Modular package layout under `src/` with clear separation of concerns:

```
src/
├── data_loader/       # Load CSV/TXT/NPY, validate, extract metadata
├── preprocessing/     # Normalization, filtering, noise simulation
├── compression/       # FFT, wavelet, quantization compressors
├── reconstruction/    # Inverse transforms and unified reconstruct()
├── metrics/           # MSE, RMSE, SNR, compression ratio
├── visualization/     # Time/frequency plots, comparison charts
└── pipeline.py        # End-to-end CLI experiment runner
```

**Design principles:**

1. **Pluggable compressors** — Each algorithm exposes `compress_*` / `decompress_*` with a common metadata dict for coefficients and parameters.
2. **Unified reconstruction** — `reconstruct(compressed, method=...)` dispatches to the correct inverse transform.
3. **Observable pipeline** — Every stage saves intermediate signals and plots to `data/` and `results/`.
4. **Quantitative evaluation** — Metrics computed after every experiment and written to JSON.
5. **Preserve legacy files** — Keep `src/compression.py` and `src/filtering.py` for future QAPF work; new code lives in subpackages.

**Data flow:**

```
Raw Signal → Load → (Optional Noise) → Filter → Compress → Transmit/Store
                                                              ↓
Metrics ← Compare ← Reconstruct ← Decompress ←──────────────┘
```

**Dependencies:** NumPy, SciPy, PyWavelets, Matplotlib, Pandas (for CSV), pytest (testing).
