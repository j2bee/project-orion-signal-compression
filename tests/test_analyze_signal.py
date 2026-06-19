"""End-to-end tests for analyze_signal dashboard and run_characterization script."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_signal import analyze_signal
from experiments.run_characterization import run_steps
from src.characterization.noise_characterization import (
    characterize_noise,
    plot_noise_diagnostics,
    write_noise_report,
)
from src.generate_synthetic import create_synthetic_dataset


@pytest.fixture
def synthetic_path(tmp_path):
    path = tmp_path / "synthetic_rocket.csv"
    create_synthetic_dataset(str(path))
    return str(path)


class TestAnalyzeSignal:
    def test_full_dashboard(self, synthetic_path, tmp_path):
        out = tmp_path / "characterization"
        summary = analyze_signal(synthetic_path, str(out))

        assert "frequency" in summary
        assert "adaptive_experiment" in summary
        assert Path(summary["outputs"]["dashboard"]).exists()
        assert (out / "synthetic_rocket_events.json").exists()
        assert (out / "adaptive_experiment.json").exists()

    def test_summary_json_written(self, synthetic_path, tmp_path):
        out = tmp_path / "characterization"
        analyze_signal(synthetic_path, str(out))
        summary_path = out / "synthetic_rocket_analysis_summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["metadata"]["sampling_frequency"] > 0


class TestRunCharacterization:
    def test_single_step(self, synthetic_path, tmp_path):
        out = tmp_path / "char"
        result = run_steps(synthetic_path, str(out), steps=["frequency"])
        assert "frequency" in result
        assert "dominant_frequencies" in result["frequency"]

    def test_noise_step_writes_report(self, synthetic_path, tmp_path):
        out = tmp_path / "char"
        reports = tmp_path / "reports"
        reports.mkdir()
        # run_steps writes to fixed reports/ path — run noise inline instead
        from src.data_loader.loader import load_signal
        from src.preprocessing.noise import apply_all_noise

        _, signal, meta = load_signal(synthetic_path)
        noisy, _ = apply_all_noise(signal, seed=42)
        noise_char = characterize_noise(signal, noisy, meta["sampling_frequency"])
        report = write_noise_report(noise_char, str(reports / "noise.md"), str(out))
        assert "SNR" in report
        assert "detected_noise_types" in noise_char or "detected_noise_types" in str(noise_char)

    def test_noise_diagnostics_plot(self, synthetic_path, tmp_path):
        from src.data_loader.loader import load_signal
        from src.preprocessing.noise import apply_all_noise

        _, signal, meta = load_signal(synthetic_path)
        noisy, _ = apply_all_noise(signal, seed=42)
        noise_char = characterize_noise(signal, noisy, meta["sampling_frequency"])
        plot_path = plot_noise_diagnostics(
            signal, noisy, noise_char, str(tmp_path / "noise_diag.png")
        )
        assert Path(plot_path).exists()
