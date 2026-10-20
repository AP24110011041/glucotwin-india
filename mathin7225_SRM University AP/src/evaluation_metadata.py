"""Build an in-memory reproducibility manifest for a future evaluation run."""

from __future__ import annotations

import hashlib
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from src.event_metrics import (
    ALERT_REFRACTORY_MINUTES,
    EVENT_FORECAST_HORIZON_MINUTES,
    MAX_EVENT_OBSERVATION_GAP_MINUTES,
    MINIMUM_IN_RANGE_MINUTES,
)
from src.feature_engineering import (
    MAX_FEATURE_FORWARD_FILL_MINUTES,
    MAX_LABEL_CGM_GAP_MINUTES,
    TARGET_HORIZON_MINUTES,
    TARGET_THRESHOLD_MG_DL,
)


PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "shap",
    "streamlit",
    "plotly",
    "matplotlib",
    "joblib",
    "requests",
    "pytest",
)


def _archive_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: '{path}'")
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for block in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def build_run_metadata(
    dataset_archive_path: str | Path,
    *,
    code_revision: str | None,
    anchor_interval_minutes: int = 10,
) -> dict[str, Any]:
    """Return a reproducibility manifest without writing a report or data.

    ``code_revision`` must be supplied explicitly; use ``None`` when source
    revision identity is unavailable rather than guessing it from the workspace.
    The dataset archive is fingerprinted only; it is never extracted or parsed.
    """
    if anchor_interval_minutes <= 0:
        raise ValueError("anchor_interval_minutes must be positive")
    archive = Path(dataset_archive_path)
    return {
        "metadata_schema_version": 1,
        "code_revision": code_revision,
        "environment": {
            "python_version": platform.python_version(),
            "packages": _installed_versions(),
        },
        "dataset": {
            "source": "CGMacros PhysioNet release 1.0.0",
            "archive_filename": archive.name,
            "archive_sha256": _archive_sha256(archive),
            "usage_rights_verified": False,
        },
        "target_configuration": {
            "sensor": "Libre GL only",
            "threshold_mg_dl": TARGET_THRESHOLD_MG_DL,
            "horizon_minutes": TARGET_HORIZON_MINUTES,
            "window": "(t, t + horizon]",
            "positive_rule": "Any observed Libre value strictly above threshold.",
            "coverage_max_gap_minutes": MAX_LABEL_CGM_GAP_MINUTES,
            "feature_forward_fill_max_age_minutes": MAX_FEATURE_FORWARD_FILL_MINUTES,
        },
        "evaluation_protocol": {
            "split": "Leave-One-Patient-Out",
            "test_patients_per_fold": 1,
            "anchor_interval_minutes": anchor_interval_minutes,
            "anchor_origin": "Each patient's first timestamp",
            "features_labels_and_eligibility_before_anchor_selection": True,
            "preprocessing_fit_scope": "Training patients within each fold",
            "probability_threshold": 0.5,
            "threshold_tuning": "None",
            "baseline_majority_tie_rule": "Predict negative",
        },
        "event_configuration": {
            "sensor": "Libre GL only",
            "threshold_mg_dl": TARGET_THRESHOLD_MG_DL,
            "minimum_observed_low_run_minutes": MINIMUM_IN_RANGE_MINUTES,
            "maximum_adjacent_observation_gap_minutes": MAX_EVENT_OBSERVATION_GAP_MINUTES,
            "alert_refractory_minutes": ALERT_REFRACTORY_MINUTES,
            "forecast_matching_horizon_minutes": EVENT_FORECAST_HORIZON_MINUTES,
            "event_matching": "One-to-one; event strictly after alert through inclusive horizon.",
            "measured_exposure": (
                "Adjacent observed Libre intervals no wider than the maximum gap, "
                "clipped to first/last eligible prediction anchors."
            ),
            "sensor_cadence_or_interpolation_verified": False,
        },
    }