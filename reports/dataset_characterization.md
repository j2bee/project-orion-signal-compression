# Dataset Characterization

Statistical investigation of every available signal dataset.

## Summary Table

| Dataset | Samples | Duration (s) | fs (Hz) | Units | Mean | Std | RMS | Skew | Kurt | Min | Max |
|---------|---------|--------------|---------|-------|------|-----|-----|------|------|-----|-----|
| noisy_signal | 20000 | 99.995 | 200.0 | m/s² (acceleration) | 18.804754 | 8.555599 | 20.659551 | -0.9508 | 104.537 | -94.318924 | 135.259251 |
| synthetic_rocket | 20000 | 99.995 | 200.0 | m/s² (acceleration) | 4.113205 | 6.429877 | 7.632941 | 2.4138 | 5.6683 | -1.565447 | 32.65673 |
| test_char | 20000 | 99.995 | 200.0 | unknown (assumed m/s² for rocket telemetry) | 4.113205 | 6.429877 | 7.632941 | 2.4138 | 5.6683 | -1.565447 | 32.65673 |
| reconstructed_signal | 20000 | 99.995 | 200.0 | m/s² (acceleration) | 1.8e-05 | 1.695436 | 1.695436 | 0.5118 | 5.4356 | -7.56066 | 9.159942 |

## Interpretation

- **High kurtosis** (>3) indicates heavy tails or impulsive content (launch spikes, impulse noise).
- **Skewness** away from zero suggests asymmetric dynamics (e.g. thrust bias during launch).
- **RMS vs std** divergence indicates DC offset (gravity baseline ~9.81 m/s² in pre-launch).

Plots saved to `results/characterization/`.
