# Project Orion: Signal Compression & Reconstruction

This repository focuses on optimizing signal-to-noise ratios (SNR) for high-ratio quantization experiments using **QAPF** (Quadrature Amplitude Phase Filtering) and **TV Baseline** filters.

## Current Progress
- [x] Establish repository structure
- [x] Document initial ablation study results
- [ ] Analyze **Quantization Lab Note (1)** for bit-depth constraints
- [ ] Implement noise-reduction fixes in `src/compression.py`

## Initial Results Observation
* **TV Baseline:** High correlation but poor noise suppression (Negative CNR).
* **QAPF ($\gamma=0$):** Clear quantization "staircase" noise in scatter plots.
* **QAPF ($\gamma=1$):** Strong noise suppression (13.9dB CNR) but significant signal distortion.
