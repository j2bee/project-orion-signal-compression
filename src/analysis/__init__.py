"""Signal analysis module — time and frequency domain characterization."""

from .time_domain import analyze_time_domain
from .frequency_domain import analyze_frequency_domain
from .report import generate_analysis_report

__all__ = ["analyze_time_domain", "analyze_frequency_domain", "generate_analysis_report"]
