"""
Pipeline utilities that power reproducible inference jobs.

This package currently hosts the offline inference workflow that was
previously implemented via notebooks. The modules are intentionally
free of Hydra-specific logic so they can be reused from scripts,
Airflow tasks, or unit tests.
"""

from .offline_inference import OfflineInferenceArtifacts, run_offline_inference

__all__ = ["OfflineInferenceArtifacts", "run_offline_inference"]
