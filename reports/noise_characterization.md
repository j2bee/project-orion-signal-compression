# Noise Characterization Report

Assessment of noise between clean reference and noisy observation.

## Power and SNR

| Metric | Value |
|--------|-------|
| Signal power | 53.799922 |
| Noise power | 0.254538 |
| SNR | **23.25 dB** |
| Residual std | 0.504388 |

## Detected Noise Types

- **gaussian**

## Test Results

### Gaussian
- Is Gaussian: True
- Shapiro-Wilk p-value: 0.1602

### Impulse
- Is impulse: False
- Spike count: 0
- Spike fraction: 0.0

### Sensor Drift
- Is drift: False
- Slope: -8.47e-06
- R²: 0.0001

### Periodic Interference
- Is periodic: False
- Dominant frequency: 1.09 Hz
- Strength ratio: 3.33

### Quantization
- Is quantization: False
- Histogram peak ratio: 2.88

## Interpretation

- **Gaussian** residuals suggest additive white noise — Wiener/spectral subtraction effective.
- **Impulse** spikes require median filtering before spectral methods.
- **Sensor drift** needs detrending or high-pass filtering.
- **Periodic interference** may need notch filtering at the dominant frequency.
- **Quantization** noise limits effective bit depth — consider μ-law companding.

Diagnostic plot: `results/characterization/noise_residual_diagnostics.png`
