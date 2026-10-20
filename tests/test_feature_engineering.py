from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.feature_engineering import (
    build_features,
    create_future_target,
    infer_patient_id,
    join_static_ehr,
    load_static_ehr,
    normalize_participant_frame,
    process_dataset,
)
from src.feature_groups import get_predictor_columns


def _raw_participant(
    glucose: list[float | None],
    start: str = "2025-01-01 00:00:00",
    dexcom: list[float | None] | None = None,
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(glucose), freq="min")
    if dexcom is None:
        dexcom = [None] * len(glucose)
    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Libre GL": glucose,
            "Dexcom GL": dexcom,
            "HR": [70.0] * len(glucose),
            "Meal Type": [None] * len(glucose),
            "Carbs": [None] * len(glucose),
        }
    )


def _normalized(
    glucose: list[float | None],
    patient_id: str = "001",
    start: str = "2025-01-01 00:00:00",
    dexcom: list[float | None] | None = None,
) -> pd.DataFrame:
    return normalize_participant_frame(
        _raw_participant(glucose, start=start, dexcom=dexcom), patient_id
    )


def test_infers_id_sorts_rows_and_drops_verified_export_index(tmp_path: Path) -> None:
    path = tmp_path / "CGMacros-007.csv"
    assert infer_patient_id(path) == "007"
    raw = _raw_participant([120.0, 121.0, 122.0]).iloc[[2, 0, 1]].reset_index(drop=True)
    raw.insert(0, "Unnamed: 0", [0, 1, 2])

    normalized = normalize_participant_frame(raw, infer_patient_id(path))

    assert normalized["timestamp"].is_monotonic_increasing
    assert normalized["patient_id"].eq("007").all()
    assert "unnamed_0" not in normalized.columns
    assert normalized["glucose_mg_dl"].tolist() == [120.0, 121.0, 122.0]


def test_duplicate_rows_are_removed_but_conflicting_duplicate_timestamps_fail() -> None:
    raw = _raw_participant([100.0, 100.0]).iloc[[0]].copy()
    duplicate = pd.concat([raw, raw], ignore_index=True)
    normalized = normalize_participant_frame(duplicate, "001")
    assert len(normalized) == 1

    conflict = pd.concat([raw, raw.assign(**{"Libre GL": 110.0})], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        normalize_participant_frame(conflict, "001")


def test_sensor_fusion_preserves_channels_and_uses_libre_then_dexcom_fallback() -> None:
    normalized = normalize_participant_frame(
        _raw_participant([100.0, None, None], dexcom=[105.0, 110.0, None]), "001"
    )

    assert normalized["libre_gl_mg_dl"].iloc[0] == 100.0
    assert pd.isna(normalized["libre_gl_mg_dl"].iloc[1])
    assert normalized["dexcom_gl_mg_dl"].iloc[:2].tolist() == [105.0, 110.0]
    assert normalized["glucose_mg_dl"].iloc[:2].tolist() == [100.0, 110.0]
    assert normalized["glucose_source"].tolist() == ["libre", "dexcom", "missing"]


def test_features_do_not_cross_patient_boundaries() -> None:
    first = _normalized([140.0], patient_id="001")
    second = _normalized([None], patient_id="002", start="2025-01-01 00:01:00")
    features = build_features(pd.concat([first, second], ignore_index=True))

    second_row = features.loc[features["patient_id"].eq("002")].iloc[0]
    assert second_row["glucose_current_mg_dl"] != 140.0
    assert second_row["glucose_current_missing_or_stale"] == 1


def test_future_feature_changes_do_not_change_earlier_features() -> None:
    baseline_values = [130.0] * 126
    changed_values = baseline_values.copy()
    changed_values[20] = 250.0
    baseline = build_features(_normalized(baseline_values))
    changed = build_features(_normalized(changed_values))
    earlier_features = [
        column
        for column in baseline.columns
        if column not in {"patient_id", "timestamp", "glucose_over_180_next_2h"}
    ]

    assert_frame_equal(
        baseline.loc[[10], earlier_features].reset_index(drop=True),
        changed.loc[[10], earlier_features].reset_index(drop=True),
    )


def test_dexcom_spikes_do_not_affect_libre_only_future_target() -> None:
    libre = [100.0] * 122
    dexcom = [None] * 122
    dexcom[1] = 250.0
    normalized = _normalized(libre, dexcom=dexcom)

    target, _ = create_future_target(normalized)

    assert int(target.iloc[0]) == 0


def test_dexcom_gaps_do_not_invalidate_complete_libre_coverage() -> None:
    libre = [100.0] * 122
    dexcom = [None if index % 20 else 110.0 for index in range(122)]
    normalized = _normalized(libre, dexcom=dexcom)

    target, summary = create_future_target(normalized)

    assert int(target.iloc[0]) == 0
    assert summary["eligible_rows"] == 2
    assert summary["incomplete_libre_coverage_rows"] == 0


def test_dexcom_cannot_rescue_a_libre_coverage_gap() -> None:
    libre: list[float | None] = [100.0] * 122
    libre[40:47] = [None] * 7
    dexcom = [120.0] * 122
    normalized = _normalized(libre, dexcom=dexcom)

    target, summary = create_future_target(normalized)

    assert pd.isna(target.iloc[0])
    assert summary["incomplete_libre_coverage_rows"] > 0


def test_current_timestamp_is_not_part_of_its_future_target() -> None:
    glucose = [181.0] + [100.0] * 120
    normalized = _normalized(glucose)

    target, _ = create_future_target(normalized)

    assert int(target.iloc[0]) == 0


def test_future_value_exactly_180_does_not_make_target_positive() -> None:
    glucose: list[float | None] = [100.0] * 121
    glucose[120] = 180.0
    normalized = _normalized(glucose)

    target, summary = create_future_target(normalized)

    assert int(target.iloc[0]) == 0
    assert summary["eligible_rows"] > 0


def test_future_value_at_horizon_endpoint_is_included() -> None:
    glucose: list[float | None] = [100.0] * 121
    glucose[120] = 181.0
    normalized = _normalized(glucose)

    target, summary = create_future_target(normalized)

    assert int(target.iloc[0]) == 1
    assert summary["eligible_rows"] > 0


def test_long_sensor_gap_makes_target_ineligible() -> None:
    glucose: list[float | None] = [120.0] * 130
    glucose[10:17] = [None] * 7
    normalized = _normalized(glucose)

    target, summary = create_future_target(normalized)

    assert pd.isna(target.iloc[0])
    assert summary["incomplete_libre_coverage_rows"] > 0


def test_five_minute_libre_gap_is_allowed_but_six_minutes_is_not() -> None:
    five_minute_gap: list[float | None] = [120.0] * 131
    five_minute_gap[1:5] = [None] * 4
    six_minute_gap: list[float | None] = [120.0] * 131
    six_minute_gap[1:6] = [None] * 5

    allowed_target, allowed_summary = create_future_target(_normalized(five_minute_gap))
    rejected_target, rejected_summary = create_future_target(_normalized(six_minute_gap))

    assert int(allowed_target.iloc[0]) == 0
    assert allowed_summary["incomplete_libre_coverage_rows"] == 0
    assert pd.isna(rejected_target.iloc[0])
    assert rejected_summary["incomplete_libre_coverage_rows"] > 0


def test_incomplete_final_two_hour_windows_are_unlabeled() -> None:
    normalized = _normalized([120.0] * 126)

    target, summary = create_future_target(normalized)

    assert target.iloc[:6].notna().all()
    assert target.iloc[6:].isna().all()
    assert summary["eligible_rows"] == 6
    assert summary["incomplete_horizon_rows"] == 120


def test_missingness_zero_values_and_bounded_past_forward_fill_are_distinct() -> None:
    normalized = _normalized([0.0, None, None, None, None, None, None])
    features = build_features(normalized)

    assert normalized.loc[0, "glucose_observed"] == 1
    assert normalized.loc[0, "glucose_mg_dl"] == 0.0
    assert normalized.loc[1, "glucose_observed"] == 0
    assert features.loc[1, "glucose_missing_at_timestamp"] == 1
    assert features.loc[1, "glucose_current_mg_dl"] == 0.0
    assert pd.isna(features.loc[6, "glucose_current_mg_dl"])
    assert features.loc[6, "glucose_current_missing_or_stale"] == 1


def test_dexcom_changes_do_not_affect_glucose_features() -> None:
    libre = [120.0] * 126
    baseline = build_features(_normalized(libre, dexcom=[None] * len(libre)))
    changed_dexcom = [None] * len(libre)
    changed_dexcom[20] = 250.0
    changed = build_features(_normalized(libre, dexcom=changed_dexcom))
    glucose_columns = [
        column
        for column in baseline.columns
        if column.startswith("glucose_") or column.startswith("libre_gl_")
    ]

    assert_frame_equal(
        baseline[glucose_columns].reset_index(drop=True),
        changed[glucose_columns].reset_index(drop=True),
    )


def test_export_schema_is_stable_for_participants_004_and_007(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    for patient_id in ("004", "007"):
        (data_dir / f"CGMacros-{patient_id}").mkdir(parents=True)
    participant_004 = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01", periods=3, freq="min"),
            "Dexcom GL": [210.0, 211.0, 212.0],
            "Libre GL": [110.0, 111.0, 112.0],
            "HR": [70.0, 71.0, 72.0],
        }
    )
    participant_007 = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-02", periods=3, freq="min"),
            "Libre GL": [130.0, 131.0, 132.0],
            "Dexcom GL": [230.0, 231.0, 232.0],
            "RecordIndex": [5, 6, 7],
        }
    )
    participant_004.to_csv(
        data_dir / "CGMacros-004" / "CGMacros-004.csv", index=False
    )
    participant_007.to_csv(
        data_dir / "CGMacros-007" / "CGMacros-007.csv", index=False
    )
    pd.DataFrame(
        {"subject": [4, 7], "Age": [44, 47], "Gender": ["F", "M"], "BMI": [25.0, 26.0]}
    ).to_csv(data_dir / "bio.csv", index=False)

    process_dataset(data_dir, tmp_path / "out")
    normalized = pd.read_csv(tmp_path / "out" / "cgmacros_normalized_timeseries.csv")

    assert normalized.columns.is_unique
    assert normalized.columns.tolist().count("record_index") == 1
    patient_ids = normalized["patient_id"].astype(str).str.zfill(3)
    assert patient_ids.value_counts().to_dict() == {"004": 3, "007": 3}
    assert normalized.loc[patient_ids.eq("004"), "libre_gl_mg_dl"].tolist() == [110, 111, 112]
    assert normalized.loc[patient_ids.eq("004"), "dexcom_gl_mg_dl"].tolist() == [210, 211, 212]
    assert normalized.loc[patient_ids.eq("007"), "libre_gl_mg_dl"].tolist() == [130, 131, 132]
    assert normalized.loc[patient_ids.eq("007"), "dexcom_gl_mg_dl"].tolist() == [230, 231, 232]
    assert normalized.loc[patient_ids.eq("007"), "record_index"].tolist() == [5, 6, 7]


def test_predictor_columns_exclude_ids_timestamps_and_targets() -> None:
    columns = [
        "patient_id",
        "timestamp",
        "glucose_current_mg_dl",
        "glucose_over_180_next_2h",
        "other_target",
    ]

    assert get_predictor_columns(columns, target_columns={"other_target"}) == [
        "glucose_current_mg_dl"
    ]


def test_meal_nutrient_missingness_is_distinct_from_measured_zero() -> None:
    raw = _raw_participant([120.0] * 3)
    raw.loc[0, "Meal Type"] = "Breakfast"
    raw.loc[0, "Carbs"] = 0.0
    raw.loc[1, "Meal Type"] = "Lunch"
    normalized = normalize_participant_frame(raw, "001")

    features = build_features(normalized)

    assert features.loc[0, "meal_carbs_g_sum_15m"] == 0.0
    assert features.loc[0, "meal_carbs_g_observation_count_15m"] == 1.0
    assert features.loc[0, "meal_type_breakfast_at_timestamp"] == 1
    assert features.loc[1, "meal_carbs_g_missing_at_timestamp"] == 1
    assert features.loc[1, "meal_event_observed_at_timestamp"] == 1


def test_explicit_meal_with_zero_nutrients_remains_a_meal_event() -> None:
    normalized = _normalized([120.0] * 3)
    nutrient_columns = [
        "meal_calories",
        "meal_carbs_g",
        "meal_protein_g",
        "meal_fat_g",
        "meal_fiber_g",
        "meal_sugar_g",
        "meal_amount_consumed_pct",
    ]
    normalized.loc[0, "meal_type"] = "Breakfast"
    normalized.loc[0, nutrient_columns] = 0.0

    features = build_features(normalized)

    assert features.loc[0, "meal_event_observed_at_timestamp"] == 1
    assert features.loc[0, "meal_type_last_120m"] == "breakfast"
    assert features.loc[1, "meal_type_last_120m"] == "breakfast"


def test_meal_type_last_120m_uses_inclusive_boundary_then_expires() -> None:
    normalized = _normalized([120.0] * 122)
    normalized.loc[0, "meal_type"] = "Lunch"

    features = build_features(normalized)

    assert features.loc[0, "meal_type_last_120m"] == "lunch"
    assert features.loc[120, "meal_type_last_120m"] == "lunch"
    assert features.loc[121, "meal_type_last_120m"] == "none"


def test_meal_type_last_120m_uses_none_when_no_meal_event_exists() -> None:
    features = build_features(_normalized([120.0] * 4))

    assert features["meal_type_last_120m"].tolist() == ["none"] * 4


def test_meal_type_last_120m_marks_missing_and_unrecognized_event_types_unknown() -> None:
    normalized = _normalized([120.0] * 4)
    normalized.loc[0, "meal_carbs_g"] = 30.0
    normalized.loc[1, "meal_type"] = "Brunch"

    features = build_features(normalized)

    assert features.loc[0, "meal_type_last_120m"] == "unknown"
    assert features.loc[1, "meal_type_last_120m"] == "unknown"
    assert features.loc[3, "meal_type_last_120m"] == "unknown"


def test_meal_type_last_120m_selects_the_most_recent_meal_type() -> None:
    normalized = _normalized([120.0] * 8)
    normalized.loc[0, "meal_type"] = "lunch"
    normalized.loc[5, "meal_type"] = " DINNER "

    features = build_features(normalized)

    assert features.loc[4, "meal_type_last_120m"] == "lunch"
    assert features.loc[5, "meal_type_last_120m"] == "dinner"
    assert features.loc[7, "meal_type_last_120m"] == "dinner"


def test_future_meal_event_does_not_change_earlier_meal_type_feature() -> None:
    baseline = _normalized([120.0] * 20)
    with_future_meal = baseline.copy()
    with_future_meal.loc[15, "meal_type"] = "dinner"

    baseline_features = build_features(baseline)
    future_features = build_features(with_future_meal)

    assert baseline_features.loc[10, "meal_type_last_120m"] == "none"
    assert future_features.loc[10, "meal_type_last_120m"] == "none"


def test_static_ehr_join_uses_normalized_patient_ids(tmp_path: Path) -> None:
    bio_path = tmp_path / "bio.csv"
    pd.DataFrame(
        {
            "subject": [1, 2],
            "Age": [45, 52],
            "Gender": ["F", "M"],
            "BMI": [25.0, 30.0],
            "A1c PDL (Lab)": [6.5, 7.0],
        }
    ).to_csv(bio_path, index=False)
    static = load_static_ehr(bio_path)
    time_series = pd.DataFrame(
        {
            "patient_id": ["002", "001", "001"],
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="min"),
        }
    )

    joined = join_static_ehr(time_series, static)

    assert joined["age_years"].tolist() == [52, 45, 45]
    assert joined["gender"].tolist() == ["M", "F", "F"]
    assert joined["a1c_pdl_lab"].tolist() == [7.0, 6.5, 6.5]


def test_static_ehr_preserves_synthetic_diagnosis_and_history_fields(
    tmp_path: Path,
) -> None:
    bio_path = tmp_path / "bio.csv"
    pd.DataFrame(
        {
            "subject": [1],
            "Condition": ["Type 2 diabetes"],
            "Diabetes Duration Years": [8],
            "Age": [45],
            "Gender": ["F"],
            "BMI": [25.0],
            "A1c PDL Lab": [7.2],
        }
    ).to_csv(bio_path, index=False)

    static = load_static_ehr(bio_path)

    assert static.loc[0, "diagnosis"] == "Type 2 diabetes"
    assert static.loc[0, "diabetes_duration_years"] == 8
    assert static.loc[0, "age_years"] == 45
    assert static.loc[0, "a1c_pdl_lab"] == 7.2


def test_repeated_processing_is_deterministic_and_schema_stable(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    (data_dir / "CGMacros-001").mkdir(parents=True)
    (data_dir / "CGMacros-002").mkdir(parents=True)
    _raw_participant([120.0] * 126).to_csv(
        data_dir / "CGMacros-001" / "CGMacros-001.csv", index=False
    )
    second = _raw_participant([130.0] * 126)
    second["Sugar"] = np.nan
    second.to_csv(data_dir / "CGMacros-002" / "CGMacros-002.csv", index=False)
    pd.DataFrame(
        {"subject": [1, 2], "Age": [40, 50], "Gender": ["F", "M"], "BMI": [24.0, 29.0]}
    ).to_csv(data_dir / "bio.csv", index=False)

    first_summary = process_dataset(data_dir, tmp_path / "out-first")
    second_summary = process_dataset(data_dir, tmp_path / "out-second")

    assert first_summary == second_summary
    for filename in (
        "cgmacros_normalized_timeseries.csv",
        "cgmacros_features.csv",
        "cgmacros_target_eligibility_summary.json",
    ):
        first = (tmp_path / "out-first" / filename).read_bytes()
        second = (tmp_path / "out-second" / filename).read_bytes()
        assert first == second
    normalized_output = pd.read_csv(tmp_path / "out-first" / "cgmacros_normalized_timeseries.csv")
    assert len(normalized_output) == 252
    assert normalized_output["patient_id"].nunique() == 2
    feature_output = pd.read_csv(tmp_path / "out-first" / "cgmacros_features.csv")
    assert int(feature_output["glucose_over_180_next_2h"].isna().sum()) == first_summary[
        "ineligible_rows"
    ]


def test_processing_failure_preserves_previous_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    for patient_id in ("001", "002"):
        (data_dir / f"CGMacros-{patient_id}").mkdir(parents=True)
    _raw_participant([120.0] * 126).to_csv(
        data_dir / "CGMacros-001" / "CGMacros-001.csv", index=False
    )
    invalid_participant = _raw_participant([120.0] * 126)
    invalid_participant["Libre GL"] = invalid_participant["Libre GL"].astype(object)
    invalid_participant.loc[0, "Libre GL"] = "invalid"
    invalid_participant.to_csv(
        data_dir / "CGMacros-002" / "CGMacros-002.csv", index=False
    )
    pd.DataFrame(
        {"subject": [1, 2], "Age": [40, 50], "Gender": ["F", "M"], "BMI": [24.0, 29.0]}
    ).to_csv(data_dir / "bio.csv", index=False)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    outputs = {
        "cgmacros_normalized_timeseries.csv": b"existing normalized output",
        "cgmacros_features.csv": b"existing feature output",
        "cgmacros_target_eligibility_summary.json": b"existing summary",
    }
    for filename, content in outputs.items():
        (output_dir / filename).write_bytes(content)

    with pytest.raises(ValueError, match="Non-numeric values"):
        process_dataset(data_dir, output_dir)

    assert {
        filename: (output_dir / filename).read_bytes()
        for filename in outputs
    } == outputs
    assert not list(output_dir.glob(".feature-engineering-*"))