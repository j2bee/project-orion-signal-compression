"""Signal characterization study — understand the signal before optimizing compression."""

from .dataset import characterize_all_datasets, write_dataset_report
from .events import detect_events
from .frequency_study import analyze_frequency_content
from .sensitivity import local_compression_sensitivity
from .noise_characterization import characterize_noise
from .failures import analyze_reconstruction_failures
from .adaptive_experiment import run_adaptive_compression_experiment

__all__ = [
    "characterize_all_datasets",
    "write_dataset_report",
    "detect_events",
    "analyze_frequency_content",
    "local_compression_sensitivity",
    "characterize_noise",
    "analyze_reconstruction_failures",
    "run_adaptive_compression_experiment",
]
