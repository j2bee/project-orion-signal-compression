# Rocket Signal Compression Research Report

Generated: 2026-06-19 05:13 UTC

## Research Question

> How much information can we remove from rocket telemetry while still
> recovering important physical events (ignition, spikes, vibration changes)?

**Design principle:** Preserve transients over maximizing compression ratio.

## Dataset

- Signal length: 20000 samples
- Duration: 100.0 s
- Sampling rate: 200.0 Hz
- Peak-to-peak: 34.22

## Signal Characteristics

- RMS: 7.6329
- Peak: 32.6567
- Variance: 41.3433

## Methods Tested

- FFT_1pct
- FFT_5pct
- FFT_10pct
- FFT_20pct
- FFT_50pct
- AdaptiveFFT_90pct_energy
- AdaptiveFFT_95pct_energy
- AdaptiveFFT_99pct_energy
- Wavelet_haar_10pct
- SoftWavelet_haar_10pct
- Wavelet_db4_10pct
- SoftWavelet_db4_10pct
- Wavelet_db8_10pct
- SoftWavelet_db8_10pct
- Quantization_8bit
- ... and 10 more

## Best Results

| Metric | Value |
|--------|-------|
| Best method | Quantization_16bit |
| Compression ratio | 4.0× |
| SNR vs working | 87.3 dB |
| MSE vs working | 1e-08 |
| Runtime | 0.0001 s |

## Baseline (FFT 10%)

- SNR vs filtered: 28.04 dB
- SNR vs clean: -6.07 dB
- Compression: 6.67×

## Streaming Simulation

- Packets: 40
- Compression ratio: 6.83×
- Avg latency: 0.026 ms
- SNR: 32.82 dB

## Recommendations

1. **Use hybrid or event-aware compression** for best transient preservation.
2. **Apply v2 multi-stage denoising** before compression (not v1 Butterworth alone).
3. **Do not optimize compression ratio first** — validate launch-window SNR separately.
4. **Next experiment:** Tune event-aware thresholds on real flight data if available.
5. **Streaming:** Increase packet size if latency dominates; decrease if events are missed.

## Summary

1. **Current best method:** Hybrid / event-aware v2 (highest SNR vs working signal).
2. **Current bottleneck:** Denoising loss vs clean original (~0.3 dB end-to-end).
3. **Biggest reconstruction error source:** High-activity regions (launch spike) under global FFT.
4. **Next recommended experiment:** Event-aware compression with segment-specific metrics.
