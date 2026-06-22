# Signal Research Answers

Evidence-based answers before optimizing compression further.

## 1. What information dominates the signal?

- **DC / gravity baseline** (~9.81 m/s²) in pre-launch accounts for large band energy in `dc_sub_hz_0_5`: **98.5%** of spectral power.
- **Launch dynamics (0.5–15 Hz)** carry **1.5%**.
- Top dominant frequency: **0.0 Hz** (45.0% of power).

## 2. What information can be removed safely?

- **High-frequency noise above 50 Hz**: only **0.0%** of energy — safe to discard.
- **Pre-launch quiet segment (0–10 s)**: near-DC, highly compressible.
- **Low-importance regions** (importance < 0.3): can use 5% FFT coefficients.

## 3. What frequencies matter most?

- **0.0 Hz** — 45.0% of total power
- **0.01 Hz** — 17.4% of total power
- **0.02 Hz** — 13.9% of total power
- **0.03 Hz** — 8.7% of total power
- **0.04 Hz** — 4.0% of total power

## 4. What causes the largest reconstruction errors?

- **Launch window (10–20 s)**: thrust spike + broadband transient content.
- **High derivative regions**: sharp changes discarded by magnitude-only FFT threshold.
- Sensitivity study: max local MSE = **2.39764596** at fragile windows (see heatmap).
- See `reports/reconstruction_failures.md` for per-spike root cause analysis.

## 5. Which compression method currently performs best?

- **Importance-weighted windowed FFT**: SNR = **40.48 dB** vs uniform **39.15 dB** (+1.33 dB gain).
- MSE reduction factor: **1.36×**.
- Hybrid v2 (full signal) still best for overall SNR but this study shows *where* to allocate bits.

## 6. What should the next experiment be?

**Importance-guided coefficient allocation on real flight data.**

Validate that launch-window SNR improves when high-importance regions get 50% coefficients
and quiet pre-launch gets 5%, without increasing total compressed size.

Detected noise types: **impulse, periodic_interference** (SNR = -6.84 dB).

Detected **43 events** — mission-critical regions confirmed at launch/flight boundaries.
