"""New-scenario inference using the existing feature and evaluation contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import (
    TARGET_COLUMN,
    make_lightgbm_pipeline,
    prepare_evaluation_data,
    validate_approved_predictors,
)
from src.feature_engineering import (
    MEAL_TYPE_CATEGORIES,
    NUMERIC_TIME_SERIES_COLUMNS,
    STATIC_EHR_COLUMNS,
    TARGET_HORIZON_MINUTES,
    TARGET_THRESHOLD_MG_DL,
    WEARABLE_COLUMNS,
    build_features,
)
from src.feature_groups import get_predictor_columns
from src.run_lopo import _effective_lightgbm_parameters, _threshold_predictions


MEAL_TYPES = MEAL_TYPE_CATEGORIES
WEARABLE_RANGES = {
    "heart_rate_bpm": (20.0, 250.0),
    "activity_calories": (0.0, 5000.0),
    "activity_mets": (0.0, 25.0),
    "activity_intensity": (0.0, 100.0),
    "steps": (0.0, 100000.0),
}
STATIC_RANGES = {
    "age_years": (18.0, 120.0),
    "diabetes_duration_years": (0.0, 100.0),
    "bmi": (10.0, 100.0),
    "a1c_pdl_lab": (3.0, 25.0),
}


def _optional_number(value: Any, name: str, bounds: tuple[float, float]) -> float:
    if value is None or pd.isna(value):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric or missing") from error
    if not np.isfinite(number) or not bounds[0] <= number <= bounds[1]:
        raise ValueError(f"{name} must be between {bounds[0]:g} and {bounds[1]:g}")
    return number


def build_scenario_timeseries(
    *,
    patient_id: str,
    timestamp: Any,
    current_glucose_mg_dl: Any,
    wearable_values: Mapping[str, Any] | None = None,
    history: Sequence[Mapping[str, Any]] = (),
    meals: Sequence[Mapping[str, Any]] = (),
    static_profile: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a single-scenario input series in the feature builder's raw schema."""
    normalized_patient_id = str(patient_id).strip()
    if not normalized_patient_id:
        raise ValueError("patient_id cannot be empty")
    current_time = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(current_time):
        raise ValueError("timestamp must be a valid date and time")

    current_glucose = _optional_number(
        current_glucose_mg_dl, "current_glucose_mg_dl", (20.0, 600.0)
    )
    if pd.isna(current_glucose):
        raise ValueError("current_glucose_mg_dl is required for a new-scenario prediction")

    wearable_values = wearable_values or {}
    unknown_wearables = set(wearable_values).difference(WEARABLE_COLUMNS)
    if unknown_wearables:
        raise ValueError(f"Unsupported wearable fields: {sorted(unknown_wearables)}")
    static_profile = static_profile or {}
    unknown_static = set(static_profile).difference(STATIC_EHR_COLUMNS)
    if unknown_static:
        raise ValueError(f"Unsupported static profile fields: {sorted(unknown_static)}")

    row: dict[str, Any] = {
        "patient_id": normalized_patient_id,
        "timestamp": current_time,
        **{column: np.nan for column in NUMERIC_TIME_SERIES_COLUMNS},
        "meal_type": pd.NA,
        "meal_type_records_at_timestamp": [],
        **{column: np.nan for column in STATIC_EHR_COLUMNS},
    }
    row["libre_gl_mg_dl"] = current_glucose
    for column, bounds in WEARABLE_RANGES.items():
        row[column] = _optional_number(wearable_values.get(column), column, bounds)
    for column, bounds in STATIC_RANGES.items():
        row[column] = _optional_number(static_profile.get(column), column, bounds)
    for column in set(STATIC_EHR_COLUMNS).difference(STATIC_RANGES):
        row[column] = static_profile.get(column)

    rows_by_time: dict[pd.Timestamp, dict[str, Any]] = {current_time: row}
    for index, reading in enumerate(history):
        history_time = pd.to_datetime(reading.get("timestamp"), errors="coerce")
        if pd.isna(history_time):
            raise ValueError(f"history[{index}].timestamp must be valid")
        try:
            is_future = history_time > current_time
        except TypeError as error:
            raise ValueError("History and scenario timestamps must use compatible timezones") from error
        if is_future:
            raise ValueError(f"history[{index}].timestamp cannot be after the scenario timestamp")
        if current_time - history_time > pd.Timedelta(minutes=120):
            raise ValueError(f"history[{index}].timestamp must be within 120 minutes")
        if history_time in rows_by_time:
            raise ValueError(f"Duplicate scenario timestamp: {history_time}")
        history_row = {
            "patient_id": normalized_patient_id,
            "timestamp": history_time,
            **{column: np.nan for column in NUMERIC_TIME_SERIES_COLUMNS},
            "meal_type": pd.NA,
            "meal_type_records_at_timestamp": [],
            **{column: row[column] for column in STATIC_EHR_COLUMNS},
        }
        history_glucose = _optional_number(
            reading.get("glucose_mg_dl"),
            f"history[{index}].glucose_mg_dl",
            (20.0, 600.0),
        )
        history_row["libre_gl_mg_dl"] = history_glucose
        for column, bounds in WEARABLE_RANGES.items():
            history_row[column] = _optional_number(
                reading.get(column), f"history[{index}].{column}", bounds
            )
        rows_by_time[history_time] = history_row

    for index, meal in enumerate(meals):
        raw_meal_type = meal.get("meal_type")
        normalized_meal_type = (
            str(raw_meal_type).strip().casefold()
            if raw_meal_type is not None and not pd.isna(raw_meal_type)
            else ""
        )
        meal_type = (
            normalized_meal_type
            if normalized_meal_type in MEAL_TYPES
            else "unknown"
        )
        try:
            minutes_since = float(meal.get("minutes_since_eating"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"meals[{index}].minutes_since_eating must be numeric") from error
        if not np.isfinite(minutes_since) or not 0 <= minutes_since <= TARGET_HORIZON_MINUTES:
            raise ValueError(
                f"meals[{index}].minutes_since_eating must be between 0 and "
                f"{TARGET_HORIZON_MINUTES}"
            )
        carbs = _optional_number(meal.get("carbs_g"), f"meals[{index}].carbs_g", (0.0, 300.0))
        if pd.isna(carbs):
            raise ValueError(f"meals[{index}].carbs_g is required")
        meal_time = current_time - pd.Timedelta(minutes=minutes_since)
        meal_row = rows_by_time.get(meal_time)
        if meal_row is None:
            meal_row = {
                "patient_id": normalized_patient_id,
                "timestamp": meal_time,
                **{column: np.nan for column in NUMERIC_TIME_SERIES_COLUMNS},
                "meal_type": pd.NA,
                "meal_type_records_at_timestamp": [],
                **{column: row[column] for column in STATIC_EHR_COLUMNS},
            }
            rows_by_time[meal_time] = meal_row
        previous_carbs = meal_row["meal_carbs_g"]
        meal_row["meal_carbs_g"] = carbs + (0.0 if pd.isna(previous_carbs) else previous_carbs)
        meal_row["meal_type_records_at_timestamp"].append(meal_type)
        previous_type = meal_row["meal_type"]
        meal_row["meal_type"] = (
            meal_type if pd.isna(previous_type) else "mixed"
        )

    return pd.DataFrame(rows_by_time.values()).sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)


def predict_new_scenario(
    training_series: pd.DataFrame,
    scenario_series: pd.DataFrame,
    *,
    seed: int = 42,
    lightgbm_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit on eligible training anchors, then score the latest scenario row."""
    if scenario_series.empty:
        raise ValueError("scenario_series cannot be empty")
    if training_series.empty:
        raise ValueError("training_series cannot be empty")
    if "patient_id" not in scenario_series or scenario_series["patient_id"].isna().any():
        raise ValueError("scenario_series must contain non-missing patient_id values")
    if scenario_series["patient_id"].astype(str).nunique() != 1:
        raise ValueError("scenario_series must contain exactly one participant")
    training_ids = set(training_series["patient_id"].astype(str))
    scenario_id = str(scenario_series["patient_id"].iloc[0])
    if scenario_id in training_ids:
        raise ValueError("scenario participant must not appear in training_series")

    training_features = build_features(training_series)
    predictor_columns = validate_approved_predictors(
        training_features.columns,
        get_predictor_columns(training_features.columns),
    )
    evaluation_data = prepare_evaluation_data(
        training_series, approved_predictors=predictor_columns
    )
    train_rows = evaluation_data.rows.loc[
        evaluation_data.rows[TARGET_COLUMN].notna()
    ]
    if train_rows["patient_id"].astype(str).nunique() < 2:
        raise ValueError("At least two training participants are required")
    if train_rows[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training labels must contain both target classes")

    scenario_features = build_features(scenario_series)
    missing_features = [
        column for column in predictor_columns if column not in scenario_features
    ]
    if missing_features:
        raise ValueError(
            "Scenario feature schema does not match the trained model; missing: "
            f"{missing_features}"
        )
    scenario_row = scenario_features.sort_values(
        "timestamp", kind="mergesort"
    ).iloc[[-1]]
    if scenario_row["timestamp"].isna().any():
        raise ValueError("scenario timestamp must be valid")

    estimator = make_lightgbm_pipeline(
        _effective_lightgbm_parameters(seed, dict(lightgbm_parameters or {}))
    )
    estimator.fit(
        train_rows.loc[:, predictor_columns],
        train_rows[TARGET_COLUMN].astype("int8"),
    )
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not contain the positive target class")
    probability = float(
        estimator.predict_proba(scenario_row.loc[:, predictor_columns])[0, classes.index(1)]
    )
    prediction = int(_threshold_predictions(np.asarray([probability]))[0])
    missing_feature_columns = tuple(
        scenario_row.loc[:, predictor_columns].columns[
            scenario_row.loc[:, predictor_columns].iloc[0].isna()
        ].tolist()
    )
    def optional_feature_value(column: str) -> float | None:
        value = scenario_row[column].iloc[0]
        return None if pd.isna(value) else float(value)

    return {
        "probability": probability,
        "prediction": prediction,
        "decision_threshold": 0.5,
        "prediction_horizon_minutes": TARGET_HORIZON_MINUTES,
        "target_threshold_mg_dl": TARGET_THRESHOLD_MG_DL,
        "timestamp": scenario_row["timestamp"].iloc[0],
        "feature_count": len(predictor_columns),
        "feature_columns": predictor_columns,
        "missing_feature_count": len(missing_feature_columns),
        "missing_feature_columns": missing_feature_columns,
        "input_summary": {
            "current_glucose_mg_dl": optional_feature_value("glucose_current_mg_dl"),
            "heart_rate_bpm": optional_feature_value("heart_rate_bpm_current"),
            "activity_calories": optional_feature_value("activity_calories_current"),
            "activity_mets": optional_feature_value("activity_mets_current"),
            "activity_intensity": optional_feature_value("activity_intensity_current"),
            "steps": optional_feature_value("steps_current"),
            "meal_events_last_120_minutes": int(
                scenario_row["meal_event_count_120m"].iloc[0]
            ),
        },
        "training_participant_count": int(train_rows["patient_id"].nunique()),
        "synthetic_training_data": True,
        "clinical_use_supported": False,
    }