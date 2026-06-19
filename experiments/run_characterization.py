#!/usr/bin/env python3
"""
Run the full signal characterization study (Steps 1–10).

Usage:
    python3 experiments/run_characterization.py
    python3 experiments/run_characterization.py --input data/raw/synthetic_rocket.csv
    python3 experiments/run_characterization.py --steps dataset,frequency,events
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_signal import analyze_signal
from src.characterization.adaptive_experiment import run_adaptive_compression_experiment
from src.characterization.dataset import characterize_all_datasets, write_dataset_report
from src.characterization.events import detect_events
from src.characterization.failures import analyze_reconstruction_failures
from src.characterization.frequency_study import analyze_frequency_content
from src.characterization.noise_characterization import (
    characterize_noise,
    plot_noise_diagnostics,
    write_noise_report,
)
from src.characterization.sensitivity import local_compression_sensitivity
from src.data_loader.loader import load_signal
from src.generate_synthetic import create_synthetic_dataset
from src.importance import compute_importance_mask
from src.preprocessing.noise import apply_all_noise

ALL_STEPS = [
    "dataset",
    "frequency",
    "events",
    "importance",
    "sensitivity",
    "adaptive",
    "noise",
    "failures",
    "dashboard",
]


def run_steps(
    input_path: str = "data/raw/synthetic_rocket.csv",
    output_dir: str = "results/characterization",
    steps: list = None,
) -> dict:
    """Run selected characterization steps."""
    if steps is None:
        steps = ALL_STEPS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(input_path).exists():
        create_synthetic_dataset(input_path)

    time, signal, metadata = load_signal(input_path)
    fs = metadata["sampling_frequency"]
    name = Path(input_path).stem
    results = {"input": input_path, "steps_run": steps}

    if "dataset" in steps:
        print("[dataset] Characterizing all datasets...")
        datasets = characterize_all_datasets("data")
        write_dataset_report(datasets, "reports/dataset_characterization.md", str(output_dir))
        results["datasets"] = datasets

    if "frequency" in steps:
        print("[frequency] FFT and band energy analysis...")
        results["frequency"] = analyze_frequency_content(
            signal, fs, name, "results/frequency_analysis"
        )

    if "events" in steps:
        print("[events] Detecting physical events...")
        events = detect_events(time, signal, fs)
        path = output_dir / f"{name}_events.json"
        with open(path, "w") as f:
            json.dump(events, f, indent=2)
        results["events"] = events

    if "importance" in steps:
        print("[importance] Computing importance mask...")
        importance, imp_meta = compute_importance_mask(signal)
        results["importance"] = {
            "mean": float(importance.mean()),
            "max": float(importance.max()),
            "high_fraction": float((importance >= 0.7).mean()),
            "weights": imp_meta["weights"],
        }

    if "sensitivity" in steps:
        print("[sensitivity] Local compression sensitivity heatmap...")
        results["sensitivity"] = local_compression_sensitivity(
            signal, window_size=512, aggressive_keep=0.05,
            output_dir=str(output_dir), name=name,
        )

    if "adaptive" in steps:
        print("[adaptive] Uniform vs importance-weighted experiment...")
        results["adaptive"] = run_adaptive_compression_experiment(
            signal, output_path=str(output_dir / "adaptive_experiment.json")
        )

    if "noise" in steps:
        print("[noise] Noise type detection and SNR...")
        noisy, _ = apply_all_noise(signal, seed=42)
        noise_char = characterize_noise(
            signal, noisy, fs, str(output_dir / "noise_characterization.json")
        )
        plot_noise_diagnostics(signal, noisy, noise_char)
        write_noise_report(noise_char, "reports/noise_characterization.md", str(output_dir))
        results["noise"] = noise_char

    if "failures" in steps:
        print("[failures] Reconstruction failure analysis...")
        analyze_reconstruction_failures(time, signal, keep_percentage=0.10)
        results["failures_report"] = "reports/reconstruction_failures.md"

    if "dashboard" in steps:
        print("[dashboard] Full analyze_signal dashboard...")
        results["dashboard"] = analyze_signal(input_path, str(output_dir))

    print("\nCharacterization complete.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run signal characterization study")
    parser.add_argument("--input", "-i", default="data/raw/synthetic_rocket.csv")
    parser.add_argument("--output", "-o", default="results/characterization")
    parser.add_argument(
        "--steps", "-s",
        default="all",
        help=f"Comma-separated steps or 'all'. Options: {','.join(ALL_STEPS)}",
    )
    args = parser.parse_args()

    if args.steps == "all":
        steps = ALL_STEPS
    else:
        steps = [s.strip() for s in args.steps.split(",")]
        unknown = set(steps) - set(ALL_STEPS)
        if unknown:
            parser.error(f"Unknown steps: {unknown}")

    run_steps(args.input, args.output, steps)


if __name__ == "__main__":
    main()
