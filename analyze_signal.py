#!/usr/bin/env python3
"""
Signal Processing Dashboard — understand the signal before optimizing compression.

One command to characterize rocket telemetry:
    python analyze_signal.py
    python analyze_signal.py --input data/raw/synthetic_rocket.csv

Outputs:
    1. Raw signal plot
    2. FFT plot
    3. Spectrogram
    4. Event markers
    5. Importance map
    6. Reconstruction error map
    7. Summary report with research answers
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.characterization.adaptive_experiment import run_adaptive_compression_experiment
from src.characterization.dataset import characterize_all_datasets, write_dataset_report
from src.characterization.events import detect_events
from src.characterization.failures import analyze_reconstruction_failures
from src.characterization.frequency_study import analyze_frequency_content
from src.characterization.noise_characterization import characterize_noise
from src.characterization.sensitivity import local_compression_sensitivity
from src.data_loader.loader import load_signal
from src.generate_synthetic import create_synthetic_dataset
from src.importance import compute_importance_mask
from src.preprocessing.noise import apply_all_noise


def _plot_dashboard(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float,
    importance: np.ndarray,
    events: dict,
    sensitivity: dict,
    output_dir: Path,
    name: str,
):
    """Generate combined dashboard figure."""
    n = len(signal)
    error_timeline = np.zeros(n)
    if sensitivity.get("window_errors"):
        ws = sensitivity["window_size"]
        step = sensitivity.get("step", ws // 2)
        for i, err in enumerate(sensitivity["window_errors"]):
            center = i * step + ws // 2
            lo = max(0, center - ws // 2)
            hi = min(n, center + ws // 2)
            error_timeline[lo:hi] = max(error_timeline[lo:hi].max() if hi > lo else 0, err)

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

    # 1. Raw signal
    axes[0].plot(time, signal, linewidth=0.6, color="steelblue")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("1. Raw Signal")
    axes[0].grid(True, alpha=0.3)

    # 2. FFT
    spectrum = np.abs(np.fft.rfft(signal)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    axes[1].plot(freqs, spectrum, linewidth=0.6, color="darkorange")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("2. FFT Magnitude Spectrum")
    axes[1].set_xlim(0, min(100, fs / 2))
    axes[1].grid(True, alpha=0.3)

    # 3. Spectrogram
    axes[2].specgram(signal, NFFT=256, Fs=fs, noverlap=128, cmap="viridis")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_title("3. Spectrogram")
    axes[2].set_ylim(0, min(80, fs / 2))

    # 4. Event markers
    axes[3].plot(time, signal, linewidth=0.5, color="gray", alpha=0.7)
    colors = {"rapid_change": "red", "amplitude_peak": "orange", "vibration_onset": "green", "energy_burst": "purple"}
    for ev in events.get("events", []):
        c = colors.get(ev["type"], "black")
        axes[3].axvline(ev["time_s"], color=c, alpha=0.6, linestyle="--", linewidth=0.8)
    for region in events.get("mission_critical_regions", []):
        axes[3].axvspan(region["start_s"], region["end_s"], alpha=0.1, color="red")
    axes[3].set_ylabel("Amplitude")
    axes[3].set_title("4. Event Markers (vertical lines) + Mission-Critical Regions (shaded)")

    # 5. Importance map
    axes[4].plot(time, importance, linewidth=0.8, color="darkgreen")
    axes[4].fill_between(time, 0, importance, alpha=0.3, color="darkgreen")
    axes[4].set_ylabel("Importance")
    axes[4].set_ylim(0, 1.05)
    axes[4].set_title("5. Importance Map (0=compressible, 1=must preserve)")
    axes[4].grid(True, alpha=0.3)

    # 6. Error map
    axes[5].plot(time, error_timeline, linewidth=0.8, color="darkred")
    axes[5].fill_between(time, 0, error_timeline, alpha=0.3, color="darkred")
    axes[5].set_xlabel("Time (s)")
    axes[5].set_ylabel("Local MSE")
    axes[5].set_title("6. Reconstruction Error Map (aggressive FFT 5% per window)")
    axes[5].grid(True, alpha=0.3)

    fig.suptitle(f"Signal Analysis Dashboard — {name}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_research_answers(
    freq_results: dict,
    events: dict,
    sensitivity: dict,
    adaptive: dict,
    noise_char: dict,
    failures_path: str,
    output_path: str = "reports/signal_research_answers.md",
) -> str:
    """Write Step 10 research questions with evidence."""
    lines = [
        "# Signal Research Answers",
        "",
        "Evidence-based answers before optimizing compression further.",
        "",
        "## 1. What information dominates the signal?",
        "",
        f"- **DC / gravity baseline** (~9.81 m/s²) in pre-launch accounts for large "
        f"band energy in `dc_sub_hz_0_5`: **{freq_results['band_energy_fraction'].get('dc_sub_hz_0_5', 0)*100:.1f}%** of spectral power.",
        f"- **Launch dynamics (0.5–15 Hz)** carry **{freq_results['band_energy_fraction'].get('launch_dynamics_0_5_15', 0)*100:.1f}%**.",
        f"- Top dominant frequency: **{freq_results['dominant_frequencies'][0]['hz']} Hz** "
        f"({freq_results['dominant_frequencies'][0]['power_fraction']*100:.1f}% of power).",
        "",
        "## 2. What information can be removed safely?",
        "",
        f"- **High-frequency noise above 50 Hz**: only "
        f"**{freq_results['band_energy_fraction'].get('high_freq_noise_50_plus', 0)*100:.1f}%** of energy — safe to discard.",
        "- **Pre-launch quiet segment (0–10 s)**: near-DC, highly compressible.",
        "- **Low-importance regions** (importance < 0.3): can use 5% FFT coefficients.",
        "",
        "## 3. What frequencies matter most?",
        "",
    ]
    for d in freq_results["dominant_frequencies"][:5]:
        lines.append(f"- **{d['hz']} Hz** — {d['power_fraction']*100:.1f}% of total power")
    lines += [
        "",
        "## 4. What causes the largest reconstruction errors?",
        "",
        "- **Launch window (10–20 s)**: thrust spike + broadband transient content.",
        "- **High derivative regions**: sharp changes discarded by magnitude-only FFT threshold.",
        f"- Sensitivity study: max local MSE = **{sensitivity.get('max_error', 'N/A')}** "
        f"at fragile windows (see heatmap).",
        f"- See `{failures_path}` for per-spike root cause analysis.",
        "",
        "## 5. Which compression method currently performs best?",
        "",
        f"- **Importance-weighted windowed FFT**: SNR = **{adaptive['importance_weighted']['snr_db']} dB** "
        f"vs uniform **{adaptive['uniform']['snr_db']} dB** "
        f"(+{adaptive['improvement']['snr_gain_db']} dB gain).",
        f"- MSE reduction factor: **{adaptive['improvement']['mse_reduction_factor']}×**.",
        "- Hybrid v2 (full signal) still best for overall SNR but this study shows *where* to allocate bits.",
        "",
        "## 6. What should the next experiment be?",
        "",
        "**Importance-guided coefficient allocation on real flight data.**",
        "",
        "Validate that launch-window SNR improves when high-importance regions get 50% coefficients",
        "and quiet pre-launch gets 5%, without increasing total compressed size.",
        "",
        f"Detected noise types: **{', '.join(noise_char.get('detected_noise_types', []))}** "
        f"(SNR = {noise_char.get('snr_db', 'N/A')} dB).",
        "",
        f"Detected **{events.get('n_events', 0)} events** — mission-critical regions confirmed at launch/flight boundaries.",
        "",
    ]

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report)
    return report


def analyze_signal(
    input_path: str = "data/raw/synthetic_rocket.csv",
    output_dir: str = "results/characterization",
) -> dict:
    """
    Run full signal characterization dashboard.

    Parameters
    ----------
    input_path : str
        Primary signal to analyze.
    output_dir : str
        Output directory for plots and JSON.

    Returns
    -------
    dict
        All analysis results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(input_path).exists():
        create_synthetic_dataset(input_path)

    print("[1/10] Loading signal...")
    time, clean, metadata = load_signal(input_path)
    fs = metadata["sampling_frequency"]
    name = Path(input_path).stem

    print("[2/10] Dataset characterization (all files)...")
    datasets = characterize_all_datasets("data")
    write_dataset_report(datasets, "reports/dataset_characterization.md", str(output_dir))

    print("[3/10] Frequency content analysis...")
    freq_results = analyze_frequency_content(clean, fs, name, "results/frequency_analysis")

    print("[4/10] Event detection...")
    events = detect_events(time, clean, fs)
    with open(output_dir / f"{name}_events.json", "w") as f:
        json.dump(events, f, indent=2)

    print("[5/10] Importance map...")
    importance, imp_meta = compute_importance_mask(clean)

    print("[6/10] Local compression sensitivity...")
    sensitivity = local_compression_sensitivity(
        clean, window_size=512, aggressive_keep=0.05,
        output_dir=str(output_dir), name=name,
    )
    sensitivity["window_center_times_s"] = [
        round(float(time[min(int(c), len(time) - 1)]), 3)
        for c in range(0, len(clean) - 512 + 1, 256)
    ][: len(sensitivity["window_errors"])]

    print("[7/10] Adaptive compression experiment...")
    adaptive = run_adaptive_compression_experiment(
        clean, output_path=str(output_dir / "adaptive_experiment.json")
    )

    print("[8/10] Noise characterization...")
    noisy, _ = apply_all_noise(clean, seed=42)
    noise_char = characterize_noise(clean, noisy, fs, str(output_dir / "noise_characterization.json"))

    print("[9/10] Failure analysis...")
    analyze_reconstruction_failures(time, clean, keep_percentage=0.10)

    print("[10/10] Generating dashboard...")
    _plot_dashboard(time, clean, fs, importance, events, sensitivity, output_dir, name)

    research_report = _write_research_answers(
        freq_results, events, sensitivity, adaptive, noise_char,
        "reports/reconstruction_failures.md",
    )

    summary = {
        "input": input_path,
        "metadata": metadata,
        "frequency": freq_results,
        "events": {"n_events": events["n_events"]},
        "sensitivity": {"max_error": sensitivity["max_error"], "fragile_regions": len(sensitivity.get("fragile_window_centers_samples", []))},
        "adaptive_experiment": adaptive,
        "noise": noise_char,
        "outputs": {
            "dashboard": str(output_dir / f"{name}_dashboard.png"),
            "dataset_report": "reports/dataset_characterization.md",
            "failures_report": "reports/reconstruction_failures.md",
            "research_answers": "reports/signal_research_answers.md",
            "frequency_dir": "results/frequency_analysis/",
        },
    }

    with open(output_dir / f"{name}_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDashboard saved to {output_dir / f'{name}_dashboard.png'}")
    print(f"Research answers: reports/signal_research_answers.md")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Rocket signal characterization dashboard")
    parser.add_argument("--input", "-i", default="data/raw/synthetic_rocket.csv")
    parser.add_argument("--output", "-o", default="results/characterization")
    args = parser.parse_args()
    analyze_signal(args.input, args.output)


if __name__ == "__main__":
    main()
