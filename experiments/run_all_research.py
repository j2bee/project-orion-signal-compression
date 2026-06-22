#!/usr/bin/env python3
"""Run all research experiments in sequence."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: str):
    print(f"\n{'='*60}\n>>> {cmd}\n")
    subprocess.check_call(cmd, shell=True, cwd=ROOT)


def main():
    run("python3 experiments/baseline_run.py")
    run("python3 experiments/compression_comparison.py")
    run("python3 -c \""
        "from src.data_loader.loader import load_signal; "
        "from src.generate_synthetic import create_synthetic_dataset; "
        "from src.streaming import run_streaming_benchmark; "
        "from pathlib import Path; "
        "p='data/raw/synthetic_rocket.csv'; "
        "create_synthetic_dataset(p) if not Path(p).exists() else None; "
        "_,s,m=load_signal(p); "
        "run_streaming_benchmark(s, m['sampling_frequency'])\""
    )
    run("python3 -c \""
        "from src.data_loader.loader import load_signal; "
        "from src.generate_synthetic import create_synthetic_dataset; "
        "from src.noise_analysis import run_noise_sweep; "
        "from pathlib import Path; "
        "p='data/raw/synthetic_rocket.csv'; "
        "create_synthetic_dataset(p) if not Path(p).exists() else None; "
        "_,s,m=load_signal(p); "
        "run_noise_sweep(s, m['sampling_frequency'])\""
    )
    run("python3 generate_report.py")
    print("\nAll research experiments complete.")


if __name__ == "__main__":
    main()
