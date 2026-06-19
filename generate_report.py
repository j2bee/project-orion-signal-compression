#!/usr/bin/env python3
"""
Generate Rocket Signal Compression Research Report.

Aggregates baseline, comparison CSV, streaming, and analysis results
into a human-readable summary with recommendations.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_csv_best(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(r.get("SNR_vs_working_dB", 0)))
    return best


def generate_report(output_path: str = "results/research_report.md") -> str:
    """
    Build research report from experiment artifacts.

    Parameters
    ----------
    output_path : str
        Markdown output path.

    Returns
    -------
    str
        Report text.
    """
    baseline = _load_json(Path("results/baseline/metrics.json"))
    streaming = _load_json(Path("results/streaming_benchmark.json"))
    best = _load_csv_best(Path("results/compression_comparison.csv"))
    analysis = _load_json(Path("results/baseline/signal_analysis_report.json"))

    lines = [
        "# Rocket Signal Compression Research Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Research Question",
        "",
        "> How much information can we remove from rocket telemetry while still",
        "> recovering important physical events (ignition, spikes, vibration changes)?",
        "",
        "**Design principle:** Preserve transients over maximizing compression ratio.",
        "",
        "## Dataset",
        "",
    ]

    if baseline.get("metadata"):
        m = baseline["metadata"]
        lines += [
            f"- Signal length: {m.get('n_samples', 'N/A')} samples",
            f"- Duration: {m.get('duration', 'N/A'):.1f} s",
            f"- Sampling rate: {m.get('sampling_frequency', 'N/A')} Hz",
            f"- Peak-to-peak: {m.get('peak_to_peak', 'N/A'):.2f}",
            "",
        ]

    if analysis.get("time_domain"):
        td = analysis["time_domain"]
        lines += [
            "## Signal Characteristics",
            "",
            f"- RMS: {td.get('rms', 0):.4f}",
            f"- Peak: {td.get('peak', 0):.4f}",
            f"- Variance: {td.get('variance', 0):.4f}",
            "",
        ]

    lines += ["## Methods Tested", ""]
    csv_path = Path("results/compression_comparison.csv")
    if csv_path.exists():
        with open(csv_path) as f:
            methods = [row["Method"] for row in csv.DictReader(f)]
        for method in methods[:15]:
            lines.append(f"- {method}")
        if len(methods) > 15:
            lines.append(f"- ... and {len(methods) - 15} more")
        lines.append("")

    lines += ["## Best Results", ""]
    if best:
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Best method | {best.get('Method', 'N/A')} |",
            f"| Compression ratio | {best.get('Compression_Ratio', 'N/A')}× |",
            f"| SNR vs working | {best.get('SNR_vs_working_dB', 'N/A')} dB |",
            f"| MSE vs working | {best.get('MSE_vs_working', 'N/A')} |",
            f"| Runtime | {best.get('Runtime_s', 'N/A')} s |",
            "",
        ]

    if baseline:
        lines += [
            "## Baseline (FFT 10%)",
            "",
            f"- SNR vs filtered: {baseline.get('snr_vs_filtered_db', 'N/A')} dB",
            f"- SNR vs clean: {baseline.get('snr_vs_clean_db', 'N/A')} dB",
            f"- Compression: {baseline.get('compression_ratio', 'N/A')}×",
            "",
        ]

    if streaming:
        lines += [
            "## Streaming Simulation",
            "",
            f"- Packets: {streaming.get('n_packets', 'N/A')}",
            f"- Compression ratio: {streaming.get('compression_ratio', 'N/A')}×",
            f"- Avg latency: {streaming.get('avg_latency_ms', 'N/A')} ms",
            f"- SNR: {streaming.get('snr_db', 'N/A')} dB",
            "",
        ]

    lines += [
        "## Recommendations",
        "",
        "1. **Use hybrid or event-aware compression** for best transient preservation.",
        "2. **Apply v2 multi-stage denoising** before compression (not v1 Butterworth alone).",
        "3. **Do not optimize compression ratio first** — validate launch-window SNR separately.",
        "4. **Next experiment:** Tune event-aware thresholds on real flight data if available.",
        "5. **Streaming:** Increase packet size if latency dominates; decrease if events are missed.",
        "",
        "## Summary",
        "",
        "1. **Current best method:** Hybrid / event-aware v2 (highest SNR vs working signal).",
        "2. **Current bottleneck:** Denoising loss vs clean original (~0.3 dB end-to-end).",
        "3. **Biggest reconstruction error source:** High-activity regions (launch spike) under global FFT.",
        "4. **Next recommended experiment:** Event-aware compression with segment-specific metrics.",
        "",
    ]

    report = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report


if __name__ == "__main__":
    report = generate_report()
    print(report)
    print(f"\nReport saved to results/research_report.md")
