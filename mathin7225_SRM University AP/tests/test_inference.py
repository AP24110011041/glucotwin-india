from __future__ import annotations

import pandas as pd
import pytest

from src.inference import build_scenario_timeseries, predict_new_scenario
from src.feature_engineering import build_features
from src.run_lopo import generate_synthetic_smoke_fixture


def _scenario(**overrides):
    values = {
        "patient_id": "MANUAL-001",
        "timestamp": "2025-02-01 12:00:00",
        "current_glucose_mg_dl": 145.0,
        "wearable_values": {
            "heart_rate_bpm": 76.0,
            "activity_mets": 1.2,
            "steps": 200.0,
        },
        "meals": [
            {"meal_type": "lunch", "carbs_g": 55.0, "minutes_since_eating": 30}
        ],
        "static_profile": {"age_years": 45, "bmi": 26.0, "a1c_pdl_lab": 7.1},
    }
    values.update(overrides)
    return build_scenario_timeseries(**values)


def test_build_scenario_timeseries_preserves_meals_and_missing_wearables() -> None:
    scenario = _scenario()

    assert scenario["patient_id"].eq("MANUAL-001").all()
    assert scenario["libre_gl_mg_dl"].notna().sum() == 1
    current = scenario.loc[scenario["timestamp"].eq(pd.Timestamp("2025-02-01 12:00:00"))].iloc[0]
    meal = scenario.loc[scenario["meal_type"].eq("lunch")].iloc[0]
    assert current["heart_rate_bpm"] == 76.0
    assert pd.isna(current["activity_calories"])
    assert meal["meal_carbs_g"] == 55.0
    assert meal["timestamp"] == pd.Timestamp("2025-02-01 11:30:00")


def test_build_scenario_timeseries_accepts_timestamped_history() -> None:
    scenario = _scenario(
        history=[
            {
                "timestamp": "2025-02-01 11:45:00",
                "glucose_mg_dl": 138.0,
                "heart_rate_bpm": 74.0,
            }
        ]
    )

    historical = scenario.loc[scenario["timestamp"].eq(pd.Timestamp("2025-02-01 11:45:00"))].iloc[0]
    assert historical["libre_gl_mg_dl"] == 138.0
    assert historical["heart_rate_bpm"] == 74.0


@pytest.mark.parametrize("meal_type", [None, "dessert", "  BRUNCH  "])
def test_manual_missing_or_unrecognized_meal_type_reaches_unknown_feature(meal_type) -> None:
    scenario = _scenario(
        meals=[
            {
                "meal_type": meal_type,
                "carbs_g": 35,
                "minutes_since_eating": 30,
            }
        ]
    )
    features = build_features(scenario)

    assert features.iloc[-1]["meal_type_last_120m"] == "unknown"


def test_manual_same_timestamp_meal_types_resolve_as_agreement_or_mixed() -> None:
    matching = _scenario(
        meals=[
            {"meal_type": "Lunch", "carbs_g": 20, "minutes_since_eating": 30},
            {"meal_type": "lunch", "carbs_g": 25, "minutes_since_eating": 30},
        ]
    )
    conflicting = _scenario(
        meals=[
            {"meal_type": "lunch", "carbs_g": 20, "minutes_since_eating": 30},
            {"meal_type": "dinner", "carbs_g": 25, "minutes_since_eating": 30},
        ]
    )

    assert build_features(matching).iloc[-1]["meal_type_last_120m"] == "lunch"
    assert build_features(conflicting).iloc[-1]["meal_type_last_120m"] == "mixed"


def test_training_and_inference_match_recent_meal_type_for_equivalent_history() -> None:
    training = generate_synthetic_smoke_fixture(seed=42, n_participants=4, days=1)
    patient = training.loc[training["patient_id"].eq("SYN-0001")].sort_values(
        "timestamp"
    )
    meal = patient.loc[patient["meal_type"].notna()].iloc[0]
    prediction_time = pd.Timestamp(meal["timestamp"]) + pd.Timedelta(minutes=7)
    training_row = patient.loc[patient["timestamp"].eq(prediction_time)].iloc[0]
    history_rows = patient.loc[
        patient["timestamp"].lt(prediction_time)
        & patient["timestamp"].ge(prediction_time - pd.Timedelta(minutes=120))
    ]
    history = [
        {
            "timestamp": row.timestamp,
            "glucose_mg_dl": row.libre_gl_mg_dl,
            **{
                column: getattr(row, column)
                for column in (
                    "heart_rate_bpm",
                    "activity_calories",
                    "activity_mets",
                    "activity_intensity",
                    "steps",
                )
            },
        }
        for row in history_rows.itertuples()
    ]
    scenario = build_scenario_timeseries(
        patient_id="EQUIVALENT-SCENARIO",
        timestamp=prediction_time,
        current_glucose_mg_dl=training_row["libre_gl_mg_dl"],
        wearable_values={
            column: training_row[column]
            for column in (
                "heart_rate_bpm",
                "activity_calories",
                "activity_mets",
                "activity_intensity",
                "steps",
            )
        },
        history=history,
        meals=[
            {
                "meal_type": meal["meal_type"],
                "carbs_g": meal["meal_carbs_g"],
                "minutes_since_eating": 7,
            }
        ],
        static_profile={
            column: training_row[column]
            for column in (
                "age_years",
                "gender",
                "diagnosis",
                "diabetes_duration_years",
                "bmi",
                "a1c_pdl_lab",
            )
        },
    )

    training_value = build_features(patient).loc[
        lambda features: features["timestamp"].eq(prediction_time),
        "meal_type_last_120m",
    ].iloc[0]
    inference_value = build_features(scenario).iloc[-1]["meal_type_last_120m"]

    assert training_value == inference_value == "breakfast"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestamp", "not-a-time", "valid date and time"),
        ("current_glucose_mg_dl", None, "required"),
        ("current_glucose_mg_dl", 601, "between 20 and 600"),
        (
            "history",
            [{"timestamp": "2025-02-01 12:01", "glucose_mg_dl": 140}],
            "cannot be after",
        ),
        (
            "history",
            [{"timestamp": "2025-02-01 09:59", "glucose_mg_dl": 140}],
            "within 120 minutes",
        ),
        ("wearable_values", {"heart_rate_bpm": 251}, "heart_rate_bpm must be between"),
        (
            "meals",
            [{"meal_type": "lunch", "carbs_g": 30, "minutes_since_eating": -1}],
            "minutes_since_eating must be between",
        ),
        (
            "meals",
            [{"meal_type": "lunch", "carbs_g": 30, "minutes_since_eating": 121}],
            "minutes_since_eating must be between",
        ),
    ],
)
def test_build_scenario_timeseries_rejects_invalid_inputs(field, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        _scenario(**{field: value})


def test_new_scenario_prediction_uses_ordered_training_schema() -> None:
    training = generate_synthetic_smoke_fixture(seed=42, n_participants=4, days=1)
    result = predict_new_scenario(training, _scenario(), seed=42)

    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["decision_threshold"] == 0.5
    assert result["prediction_horizon_minutes"] == 120
    assert result["target_threshold_mg_dl"] == 180.0
    assert result["feature_count"] == len(result["feature_columns"]) == 151
    assert not {
        "patient_id",
        "timestamp",
        "glucose_over_180_next_2h",
        "dexcom_gl_mg_dl",
    }.intersection(result["feature_columns"])
    assert result["input_summary"]["current_glucose_mg_dl"] == 145.0
    assert result["input_summary"]["heart_rate_bpm"] == 76.0
    assert result["input_summary"]["activity_mets"] == 1.2
    assert result["input_summary"]["steps"] == 200.0
    assert result["input_summary"]["meal_events_last_120_minutes"] == 1
    assert result["clinical_use_supported"] is False


def test_new_scenario_prediction_rejects_training_participant_overlap() -> None:
    training = generate_synthetic_smoke_fixture(seed=43, n_participants=4, days=1)
    scenario = _scenario(patient_id="SYN-0001")

    with pytest.raises(ValueError, match="must not appear in training_series"):
        predict_new_scenario(training, scenario)