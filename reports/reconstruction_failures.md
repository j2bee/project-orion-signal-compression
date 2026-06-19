# Reconstruction Failure Analysis

**Method:** FFT at 10% coefficient retention
**Overall MSE:** 0.00526395

## Summary

Reconstruction errors concentrate at **transient events** (launch spike, phase transitions)
where the signal has high derivative and broadband frequency content. Global FFT compression
discards coefficients uniformly by magnitude, which removes the sharp features that define
mission-critical events.

## Error Spikes (Top 10)

### Failure 1 — t = 99.95 s (sample 19990)

- **Absolute error:** 0.651167
- **Signal value:** 0.1764
- **Local variance:** 5e-06
- **Local derivative:** 0.0001

**Likely root causes:**
- Moderate error — likely accumulation of discarded mid-frequency coefficients

### Failure 2 — t = 0.045 s (sample 9)

- **Absolute error:** 0.595717
- **Signal value:** 9.81
- **Local variance:** 0.0
- **Local derivative:** 0.0

**Likely root causes:**
- Moderate error — likely accumulation of discarded mid-frequency coefficients

### Failure 3 — t = 10.0 s (sample 2000)

- **Absolute error:** 0.51347
- **Signal value:** 9.81
- **Local variance:** 6.67498
- **Local derivative:** 0.2236

**Likely root causes:**
- Located in launch window — known fragile region for global compression

### Failure 4 — t = 0.225 s (sample 45)

- **Absolute error:** 0.459389
- **Signal value:** 9.81
- **Local variance:** 0.0
- **Local derivative:** 0.0

**Likely root causes:**
- Moderate error — likely accumulation of discarded mid-frequency coefficients

### Failure 5 — t = 20.0 s (sample 4000)

- **Absolute error:** 0.420273
- **Signal value:** 7.0772
- **Local variance:** 7.306156
- **Local derivative:** 0.1302

**Likely root causes:**
- Located in launch window — known fragile region for global compression

### Failure 6 — t = 99.685 s (sample 19937)

- **Absolute error:** 0.418718
- **Signal value:** 0.1836
- **Local variance:** 1.6e-05
- **Local derivative:** 0.0001

**Likely root causes:**
- Moderate error — likely accumulation of discarded mid-frequency coefficients

### Failure 7 — t = 19.85 s (sample 3970)

- **Absolute error:** 0.379452
- **Signal value:** 9.5984
- **Local variance:** 3.849808
- **Local derivative:** 0.0795

**Likely root causes:**
- Located in launch window — known fragile region for global compression

### Failure 8 — t = 20.31 s (sample 4062)

- **Absolute error:** 0.340008
- **Signal value:** 2.0908
- **Local variance:** 4.406893
- **Local derivative:** 0.0387

**Likely root causes:**
- Located in launch window — known fragile region for global compression

### Failure 9 — t = 19.69 s (sample 3938)

- **Absolute error:** 0.268658
- **Signal value:** 11.8313
- **Local variance:** 2.361184
- **Local derivative:** 0.058

**Likely root causes:**
- Located in launch window — known fragile region for global compression

### Failure 10 — t = 89.845 s (sample 17969)

- **Absolute error:** 0.259466
- **Signal value:** 1.9164
- **Local variance:** 0.377149
- **Local derivative:** 0.0912

**Likely root causes:**
- Moderate error — likely accumulation of discarded mid-frequency coefficients

## General Findings

1. **Pre-launch (0–10 s):** Low error — signal is near-DC; few coefficients needed.
2. **Launch (10–20 s):** Highest errors — thrust spike has broadband + transient content.
3. **Flight (20–90 s):** Moderate errors — repetitive vibration compresses well except at envelope changes.
4. **Descent (90–100 s):** Low-moderate errors — low amplitude, less critical.

## Recommendation

Do not apply uniform compression. Use **importance-weighted** or **event-aware**
compression that allocates more coefficients to high-derivative, high-variance regions.
