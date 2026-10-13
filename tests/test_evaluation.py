from __future__ import annotations

import hashlib
from importlib.util import find_spec

import pandas as pd
import pytest

from src.evaluation import (
    EvaluationDataset,
    _current_state_alerts,
    evaluate_baselines,
    evaluate_lightgbm_lopo,
    iter_lopo_folds,
    make_lightgbm_pipeline,
    prepare_evaluation_data,
    select_prediction_anchors,
    validate_approved_predictors,
)
from src.feature_engineering import build_features, normalize_participant_frame
from src.event_metrics import detect_libre_event_onsets
from src.evaluation_metadata import build_run_metadata


def _normalized(
    patient_id: str,
    libre: list[float | None],
    start: str,
) -> pd.DataFrame:
    return normalize_participant_frame(
        pd.DataFrame(
            {
                "Timestamp": pd.date_range(start, periods=len(libre), freq="min"),
                "Libre GL": libre,
                "Dexcom GL": [None] * len(libre),
            }
        ),
        patient_id,
    )


def test_anchors_use_each_patients_own_start_time() -> None:
    features = pd.DataFrame(
        {
            "patient_id": ["002", "001", "002", "001", "002", "001"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:17",
                    "2025-01-01 00:13",
                    "2025-01-01 00:28",
                    "2025-01-01 00:03",
                    "2025-01-01 00:07",
                    "2025-01-01 00:23",
                ]
            ),
        }
    )

    anchors = select_prediction_anchors(features, interval_minutes=10)

    actual = {
        patient_id: group["timestamp"].dt.strftime("%H:%M").tolist()
        for patient_id, group in anchors.groupby("patient_id")
    }
    assert actual == {"001": ["00:03", "00:13", "00:23"], "002": ["00:07", "00:17"]}


def test_lopo_folds_hold_out_exactly_one_patient_deterministically() -> None:
    rows = pd.DataFrame(
        {
            "patient_id": ["010", "001", "002", "010", "001", "002"],
            "row": range(6),
        }
    )

    first = iter_lopo_folds(rows)
    second = iter_lopo_folds(rows)

    assert [fold.held_out_patient_id for fold in first] == ["001", "002", "010"]
    assert [fold.held_out_patient_id for fold in first] == [
        fold.held_out_patient_id for fold in second
    ]
    for fold in first:
        assert fold.test["patient_id"].astype(str).unique().tolist() == [
            fold.held_out_patient_id
        ]
        assert set(fold.train["patient_id"]).isdisjoint(set(fold.test["patient_id"]))


def test_predictor_allowlist_excludes_standard_custom_and_dexcom_columns() -> None:
    columns = [
        "patient_id",
        "timestamp",
        "glucose_current_mg_dl",
        "glucose_over_180_next_2h",
        "custom_target",
        "dexcom_gl_current_mg_dl",
    ]

    assert validate_approved_predictors(
        columns,
        ["glucose_current_mg_dl"],
        target_columns={"custom_target"},
    ) == ("glucose_current_mg_dl",)
    with pytest.raises(ValueError, match="Dexcom"):
        validate_approved_predictors(columns, ["dexcom_gl_current_mg_dl"])
    with pytest.raises(ValueError, match="IDs, timestamps, or targets"):
        validate_approved_predictors(columns, ["custom_target"], {"custom_target"})


@pytest.mark.parametrize(
    "forbidden_predictor",
    [
        "patient_id",
        "timestamp",
        "glucose_over_180_next_2h",
        "dexcom_gl_current_mg_dl",
    ],
)
def test_evaluation_entry_points_reject_directly_supplied_forbidden_predictors(
    forbidden_predictor: str,
) -> None:
    rows = pd.DataFrame(
        {
            "patient_id": ["001", "002"],
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="min"),
            "glucose_over_180_next_2h": pd.array([0, 1], dtype="Int8"),
            "glucose_current_mg_dl": [100.0, 100.0],
            "glucose_current_missing_or_stale": [0, 0],
            "glucose_observed_at_timestamp": [1, 1],
            "dexcom_gl_current_mg_dl": [300.0, 300.0],
        }
    )
    evaluation_data = EvaluationDataset(
        rows=rows,
        predictor_columns=(forbidden_predictor,),
    )

    with pytest.raises(ValueError):
        evaluate_baselines(evaluation_data)
    with pytest.raises(ValueError):
        evaluate_lightgbm_lopo(evaluation_data)


def test_features_and_labels_are_built_before_anchor_selection() -> None:
    glucose: list[float | None] = [100.0] * 122
    glucose[1] = 250.0
    normalized = _normalized("001", glucose, "2025-01-01 00:00")

    evaluation_data = prepare_evaluation_data(
        normalized,
        approved_predictors=["glucose_current_mg_dl"],
    )

    assert len(evaluation_data.rows) == 13
    assert evaluation_data.rows.iloc[0]["timestamp"] == pd.Timestamp("2025-01-01 00:00")
    assert int(evaluation_data.rows.iloc[0]["glucose_over_180_next_2h"]) == 1
    assert evaluation_data.rows["glucose_over_180_next_2h"].notna().sum() == 1


def test_baselines_use_training_majority_and_return_fold_uncertainty() -> None:
    rows = pd.DataFrame(
        {
            "patient_id": ["001", "001", "002", "002"],
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="10min"),
            "glucose_over_180_next_2h": pd.array([1, 0, 0, 0], dtype="Int8"),
            "glucose_observed_at_timestamp": [1, 1, 1, 1],
            "glucose_current_mg_dl": [190.0, 100.0, 100.0, 100.0],
            "glucose_current_missing_or_stale": [0, 0, 0, 0],
        }
    )
    evaluation_data = EvaluationDataset(
        rows=rows,
        predictor_columns=("glucose_current_mg_dl",),
    )

    report = evaluate_baselines(evaluation_data, bootstrap_replicates=50, random_state=7)

    assert len(report["fold_metrics"]) == 4
    assert report["baseline_fold_choices"][0]["majority_prediction"] == 0
    assert report["baseline_fold_choices"][1]["tie_rule"].startswith("Predict negative")
    persistence = next(
        record
        for record in report["fold_metrics"]
        if record["method"] == "training_majority"
        and record["held_out_patient_id"] == "001"
        and record["stratum"] == "all_eligible_anchors"
    )
    assert (persistence["true_positives"], persistence["true_negatives"]) == (0, 1)
    assert report["current_state_detector"]["total_alert_anchors"] == 1
    assert not any(
        metric["method"] == "libre_current_state_detector"
        for metric in report["event_fold_metrics"]
    )
    assert {item["random_state"] for item in report["participant_uncertainty"]} == {7}


def test_persistence_suppresses_missing_and_stale_values_on_irregular_timestamps() -> None:
    timestamps = pd.to_datetime(
        [
            "2025-01-01 00:00",
            "2025-01-01 00:02",
            "2025-01-01 00:04",
            "2025-01-01 00:10",
            "2025-01-01 00:11",
        ]
    )
    normalized = normalize_participant_frame(
        pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Libre GL": [None, 220.0, None, None, 220.0],
                "Dexcom GL": [300.0, None, 300.0, 300.0, None],
            }
        ),
        "001",
    )
    features = build_features(normalized)

    assert features["glucose_current_missing_or_stale"].tolist() == [1, 0, 0, 1, 0]
    assert _current_state_alerts(features).tolist() == [0, 1, 0, 0, 1]


def test_event_report_merges_many_positive_anchors_for_one_event() -> None:
    start = pd.Timestamp("2025-01-01")
    rows = pd.DataFrame(
        {
            "patient_id": ["001"] * 3 + ["002"] * 2,
            "timestamp": [
                start,
                start + pd.Timedelta(minutes=10),
                start + pd.Timedelta(minutes=20),
                start,
                start + pd.Timedelta(minutes=10),
            ],
            "glucose_over_180_next_2h": pd.array([1, 1, 1, 1, 1], dtype="Int8"),
            "glucose_observed_at_timestamp": [1] * 5,
            "glucose_current_mg_dl": [100.0] * 5,
            "glucose_current_missing_or_stale": [0] * 5,
        }
    )
    source_rows = pd.DataFrame(
        {
            "patient_id": ["001"] * 141 + ["002"] * 21,
            "timestamp": list(pd.date_range(start, periods=141, freq="min"))
            + list(pd.date_range(start, periods=21, freq="min")),
            "libre_gl_mg_dl": [100.0] * 61
            + [190.0] * 80
            + [100.0] * 21,
        }
    )
    events = detect_libre_event_onsets(source_rows)
    evaluation_data = EvaluationDataset(
        rows=rows,
        predictor_columns=("glucose_current_mg_dl",),
        observed_series=source_rows,
        event_onsets=events,
    )

    report = evaluate_baselines(evaluation_data, bootstrap_replicates=20)

    event_metrics = next(
        row
        for row in report["event_fold_metrics"]
        if row["method"] == "training_majority"
        and row["held_out_patient_id"] == "001"
    )
    assert event_metrics["alert_episodes"] == 1
    assert event_metrics["actual_events"] == 1
    assert event_metrics["matched_events"] == 1
    assert event_metrics["false_alert_episodes"] == 0
    assert event_metrics["measured_observation_minutes"] == 20.0
    assert event_metrics["false_alert_episodes_per_day"] == 0.0
    assert all(
        row["method"] != "libre_current_state_detector"
        for row in report["event_fold_metrics"]
    )
    assert "current_state_detector" in report


def test_lightgbm_factory_returns_an_unfitted_fold_preprocessing_pipeline() -> None:
    if find_spec("lightgbm") is None:
        pytest.skip("LightGBM 4.7.0 is not installed; pipeline runtime is unverified")

    pipeline = make_lightgbm_pipeline({"n_estimators": 5})

    assert pipeline.named_steps.keys() == {"preprocessor", "classifier"}
    assert not hasattr(pipeline, "classes_")


def test_run_metadata_records_archive_code_environment_and_protocol(tmp_path: Path) -> None:
    archive = tmp_path / "synthetic-dataset.zip"
    archive_bytes = b"synthetic metadata fixture only"
    archive.write_bytes(archive_bytes)

    metadata = build_run_metadata(archive, code_revision="test-revision")

    assert metadata["code_revision"] == "test-revision"
    assert metadata["dataset"]["archive_sha256"] == hashlib.sha256(archive_bytes).hexdigest()
    assert metadata["dataset"]["usage_rights_verified"] is False
    assert metadata["environment"]["python_version"]
    assert metadata["environment"]["packages"]["lightgbm"] == "4.7.0"
    assert metadata["target_configuration"]["threshold_mg_dl"] == 180.0
    assert metadata["target_configuration"]["horizon_minutes"] == 120
    assert metadata["evaluation_protocol"]["split"] == "Leave-One-Patient-Out"
    assert metadata["evaluation_protocol"]["anchor_interval_minutes"] == 10
    assert metadata["event_configuration"]["sensor"] == "Libre GL only"
    assert metadata["event_configuration"]["sensor_cadence_or_interpolation_verified"] is False
    assert {path.name for path in tmp_path.iterdir()} == {archive.name}
