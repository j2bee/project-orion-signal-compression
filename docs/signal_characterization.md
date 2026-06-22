# Signal Characterization Study

**Goal:** Determine what information inside rocket telemetry is important and what can safely be removed — before optimizing compression further.

> No new compression algorithms are added in this phase. The focus is measurement, event detection, and importance scoring.

## Quick Start

```bash
# One-command dashboard (Steps 1–10)
python3 analyze_signal.py

# Run individual steps
python3 experiments/run_characterization.py --steps dataset,frequency,events

# Full study with custom input
python3 analyze_signal.py --input data/raw/synthetic_rocket.csv
```

## Pipeline Overview

```
data/ (all signal files)
      ↓
Step 1: Dataset stats → reports/dataset_characterization.md
      ↓
Step 2: FFT / band energy → results/frequency_analysis/
      ↓
Step 3: Event detection → results/characterization/*_events.json
      ↓
Step 4: Sensitivity heatmap → results/characterization/*_sensitivity_heatmap.png
      ↓
Step 5: Importance mask → src/importance.py
      ↓
Step 6: Adaptive experiment → results/characterization/adaptive_experiment.json
      ↓
Step 7: Failure analysis → reports/reconstruction_failures.md
      ↓
Step 8: Noise characterization → reports/noise_characterization.md
      ↓
Step 9: Dashboard → results/characterization/*_dashboard.png
      ↓
Step 10: Research answers → reports/signal_research_answers.md
```

## Step-by-Step Reference

### Step 1 — Dataset Investigation

**Module:** `src/characterization/dataset.py`

For every file under `data/`, computes:

| Statistic | Description |
|-----------|-------------|
| n_samples | Sample count |
| duration_s | Total duration |
| sampling_frequency_hz | Sampling rate |
| min / max / mean / variance / RMS | Amplitude statistics |
| skewness / kurtosis | Distribution shape |

**Output:** `reports/dataset_characterization.md` + per-dataset overview plots.

### Step 2 — Frequency Content

**Module:** `src/characterization/frequency_study.py`

- Dominant frequencies and harmonic content
- Band energy fractions (DC, launch dynamics, high-freq noise)
- Gini sparsity coefficient
- FFT compression suitability assessment

**Output:** `results/frequency_analysis/{name}_fft_analysis.png`, `{name}_spectrogram.png`, JSON.

### Step 3 — Event Detection

**Module:** `src/characterization/events.py`

Detects launch, ignition, vibration onset, energy bursts via:

- Derivative thresholding
- Peak detection
- Moving variance
- Local RMS bursts

**Output:** Event JSON with mission-critical region markers.

### Step 4 — Local Compression Sensitivity

**Module:** `src/characterization/sensitivity.py`

Splits signal into overlapping windows. Each window is compressed aggressively (5% FFT coefficients) and reconstructed. Produces:

- Window index vs reconstruction error heatmap
- Fragile window identification

**Question answered:** Which sections of the signal are fragile?

### Step 5 — Importance Score

**Module:** `src/importance.py`

Per-sample importance mask (0–1) from weighted combination of:

- Local variance (25%)
- Local energy (25%)
- Peak density (25%)
- Derivative magnitude (25%)

Maps to keep percentages: high importance → 50%, low → 5%.

### Step 6 — Adaptive Compression Experiment

**Module:** `src/characterization/adaptive_experiment.py`

Compares:

| Method | Coefficient retention |
|--------|----------------------|
| Uniform baseline | 10% all windows |
| Importance-weighted | 50% high / 5% low |

**Output:** `results/characterization/adaptive_experiment.json`

### Step 7 — Failure Analysis

**Module:** `src/characterization/failures.py`

For each reconstruction error spike:

- Location in signal
- Local variance and derivative
- Likely root cause (not just metrics)

**Output:** `reports/reconstruction_failures.md`

### Step 8 — Noise Characterization

**Module:** `src/characterization/noise_characterization.py`

Tests residual for: Gaussian, impulse, sensor drift, periodic interference, quantization.

Computes signal power, noise power, SNR.

**Output:** `reports/noise_characterization.md`, `results/characterization/noise_residual_diagnostics.png`

### Step 9 — Dashboard

**Script:** `analyze_signal.py`

Six-panel figure:

1. Raw signal
2. FFT magnitude spectrum
3. Spectrogram
4. Event markers + mission-critical regions
5. Importance map
6. Reconstruction error map

### Step 10 — Research Questions

**Output:** `reports/signal_research_answers.md`

Six evidence-based answers with references to JSON/plots.

## Key Modules

```
src/
├── importance.py                    # Importance mask (0–1)
└── characterization/
    ├── dataset.py                   # Step 1
    ├── frequency_study.py           # Step 2
    ├── events.py                    # Step 3
    ├── sensitivity.py             # Step 4
    ├── adaptive_experiment.py     # Step 6
    ├── failures.py                # Step 7
    └── noise_characterization.py  # Step 8

analyze_signal.py                    # Step 9 dashboard
experiments/run_characterization.py  # Step runner CLI
```

## Reports Index

| Report | Path |
|--------|------|
| Dataset stats | `reports/dataset_characterization.md` |
| Noise types + SNR | `reports/noise_characterization.md` |
| Failure root causes | `reports/reconstruction_failures.md` |
| Research answers | `reports/signal_research_answers.md` |

## Notebooks

| Notebook | Content |
|----------|---------|
| `notebooks/research/01_signal_characterization.ipynb` | Full 10-step walkthrough |
| `notebooks/research/02_noise_analysis.ipynb` | Noise impact study |
| `notebooks/research/03_compression_comparison.ipynb` | Method comparison |

## Tests

```bash
pytest tests/test_characterization.py -v
pytest tests/test_analyze_signal.py -v
```

## Next Steps (from research findings)

1. Validate importance-guided allocation on real flight data
2. Tune event detection thresholds for actual mission phases
3. Do not add new compression algorithm families until characterization is validated on real telemetry
