"""Reproducible patient-wise evaluation helpers; no evaluation runs on import."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import (
    TARGET_HORIZON_MINUTES,
    TARGET_THRESHOLD_MG_DL,
    build_features,
)
from src.event_metrics import (
    ALERT_REFRACTORY_MINUTES,
    EVENT_FORECAST_HORIZON_MINUTES,
    MAX_EVENT_OBSERVATION_GAP_MINUTES,
    detect_libre_event_onsets,
    merge_alert_episodes,
    score_event_alerts,
)
from src.feature_groups import TARGET_COLUMNS, get_predictor_columns


TARGET_COLUMN = "glucose_over_180_next_2h"
CURRENT_GLUCOSE_COLUMN = "glucose_current_mg_dl"
ONSET_STRATUM = "current_libre_le_180"
ANCHOR_METRIC_COLUMNS = (
    "anchor_precision",
    "anchor_recall",
    "anchor_f1",
)
EVENT_METRIC_COLUMNS = (
    "event_precision",
    "event_recall",
    "false_alert_episodes_per_day",
)


@dataclass(frozen=True)
class EvaluationDataset:
    """Anchors plus full-resolution Libre data needed for event scoring."""

    rows: pd.DataFrame
    predictor_columns: tuple[str, ...]
    anchor_interval_minutes: int = 10
    observed_series: pd.DataFrame | None = None
    event_onsets: pd.DataFrame | None = None
    max_event_observation_gap_minutes: float = MAX_EVENT_OBSERVATION_GAP_MINUTES


@dataclass(frozen=True)
class PatientFold:
    """One deterministic leave-one-patient-out split."""

    held_out_patient_id: str
    train: pd.DataFrame
    test: pd.DataFrame


def validate_approved_predictors(
    feature_columns: Iterable[str],
    approved_predictors: Iterable[str],
    target_columns: Iterable[str] = TARGET_COLUMNS,
) -> tuple[str, ...]:
    """Validate an explicit predictor allow-list against the feature-group helper."""
    approved = tuple(approved_predictors)
    if not approved:
        raise ValueError("At least one approved predictor column is required")
    if len(set(approved)) != len(approved):
        raise ValueError("Approved predictor columns must be unique")
    dexcom_columns = [
        column for column in approved if "dexcom" in column.casefold()
    ]
    if dexcom_columns:
        raise ValueError(f"Dexcom columns are not approved predictors: {dexcom_columns}")

    available_predictors = set(
        get_predictor_columns(feature_columns, target_columns=target_columns)
    )
    unapproved = [column for column in approved if column not in available_predictors]
    if unapproved:
        raise ValueError(
            "Predictors must be returned by get_predictor_columns and cannot include "
            f"IDs, timestamps, or targets: {unapproved}"
        )
    return approved


def select_prediction_anchors(
    features: pd.DataFrame, interval_minutes: int = 10
) -> pd.DataFrame:
    """Select rows on a patient-relative interval without crossing patients."""
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    required = {"patient_id", "timestamp"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    if features.empty:
        return features.copy()
    if features["patient_id"].isna().any():
        raise ValueError("patient_id cannot be missing when selecting anchors")

    timestamps = pd.to_datetime(features["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp contains invalid values")
    if pd.DataFrame(
        {"patient_id": features["patient_id"], "timestamp": timestamps}
    ).duplicated().any():
        raise ValueError("Duplicate patient timestamps are not allowed")

    first_timestamp = timestamps.groupby(features["patient_id"]).transform("min")
    elapsed = timestamps - first_timestamp
    interval = pd.Timedelta(minutes=interval_minutes)
    elapsed_ns = elapsed.array.as_unit("ns").asi8
    anchor_mask = elapsed_ns % interval.value == 0
    anchors = features.loc[anchor_mask].copy()
    anchors["timestamp"] = timestamps.loc[anchor_mask]
    return anchors.sort_values(
        ["patient_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def prepare_evaluation_data(
    normalized_time_series: pd.DataFrame,
    approved_predictors: Iterable[str],
    anchor_interval_minutes: int = 10,
    max_event_observation_gap_minutes: float = MAX_EVENT_OBSERVATION_GAP_MINUTES,
) -> EvaluationDataset:
    """Build full-series causal features/labels before selecting 10-minute anchors."""
    features = build_features(normalized_time_series)
    if TARGET_COLUMN not in features:
        raise ValueError(f"Feature table is missing target column '{TARGET_COLUMN}'")
    predictor_columns = validate_approved_predictors(
        features.columns, approved_predictors
    )
    observed_series = normalized_time_series[
        ["patient_id", "timestamp", "libre_gl_mg_dl"]
    ].copy()
    event_onsets = detect_libre_event_onsets(
        observed_series,
        max_observation_gap_minutes=max_event_observation_gap_minutes,
    )
    anchors = select_prediction_anchors(features, anchor_interval_minutes)
    return EvaluationDataset(
        rows=anchors,
        predictor_columns=predictor_columns,
        anchor_interval_minutes=anchor_interval_minutes,
        observed_series=observed_series,
        event_onsets=event_onsets,
        max_event_observation_gap_minutes=max_event_observation_gap_minutes,
    )


def iter_lopo_folds(rows: pd.DataFrame) -> list[PatientFold]:
    """Return deterministic folds with exactly one whole patient in each test set."""
    if "patient_id" not in rows:
        raise ValueError("rows must contain patient_id")
    if rows.empty or rows["patient_id"].isna().any():
        raise ValueError("rows must contain non-missing patient IDs")

    patient_ids = rows["patient_id"].astype(str)
    patients = sorted(patient_ids.unique().tolist())
    if len(patients) < 2:
        raise ValueError("LOPO evaluation requires at least two participants")

    folds: list[PatientFold] = []
    for held_out_patient_id in patients:
        test_mask = patient_ids.eq(held_out_patient_id)
        train = rows.loc[~test_mask].copy()
        test = rows.loc[test_mask].copy()
        train_patients = set(train["patient_id"].astype(str).unique())
        test_patients = set(test["patient_id"].astype(str).unique())
        if test_patients != {held_out_patient_id} or train_patients.intersection(test_patients):
            raise RuntimeError("LOPO fold contains a patient separation violation")
        folds.append(PatientFold(held_out_patient_id, train, test))
    return folds


def _binary_metrics(
    rows: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, int | float]:
    actual = rows["y_true"].astype("int8").to_numpy()
    predicted = np.asarray(predictions, dtype="int8")
    if len(actual) != len(predicted):
        raise ValueError("Prediction count does not match eligible row count")

    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "eligible_anchors": int(len(rows)),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "true_negatives": true_negative,
        "false_negatives": false_negative,
        "anchor_precision": float(precision),
        "anchor_recall": float(recall),
        "anchor_f1": float(f1),
    }


def _stratum_rows(rows: pd.DataFrame, stratum: str) -> pd.DataFrame:
    if stratum == "all_eligible_anchors":
        return rows
    if stratum == ONSET_STRATUM:
        current_observed = rows["glucose_observed_at_timestamp"].eq(1)
        current_not_high = rows[CURRENT_GLUCOSE_COLUMN].le(TARGET_THRESHOLD_MG_DL)
        return rows.loc[current_observed & current_not_high]
    raise ValueError(f"Unknown evaluation stratum: {stratum}")


def _prediction_records(
    method: str,
    patient_id: str,
    test_rows: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    eligible = test_rows.loc[test_rows[TARGET_COLUMN].notna()].copy()
    if len(eligible) != len(predictions):
        raise ValueError("Predictions must cover exactly the eligible test anchors")
    records = eligible[
        [
            "patient_id",
            "timestamp",
            TARGET_COLUMN,
            "glucose_observed_at_timestamp",
            CURRENT_GLUCOSE_COLUMN,
        ]
    ].copy()
    records["patient_id"] = patient_id
    records["y_true"] = records.pop(TARGET_COLUMN).astype("int8")
    records["y_pred"] = np.asarray(predictions, dtype="int8")
    records["method"] = method
    return records.reset_index(drop=True)


def _current_state_alerts(rows: pd.DataFrame) -> np.ndarray:
    """Alert only when an observed current Libre reading exceeds threshold."""
    fresh = rows["glucose_observed_at_timestamp"].eq(1)
    above_threshold = rows[CURRENT_GLUCOSE_COLUMN].gt(TARGET_THRESHOLD_MG_DL)
    return (fresh & above_threshold).fillna(False).to_numpy(dtype="int8")


def _validate_evaluation_predictors(evaluation_data: EvaluationDataset) -> None:
    validate_approved_predictors(
        evaluation_data.rows.columns,
        evaluation_data.predictor_columns,
    )


def _score_prediction_records(
    predictions: pd.DataFrame,
    methods: Iterable[str],
    patient_ids: Iterable[str],
    evaluation_data: EvaluationDataset,
    bootstrap_replicates: int,
    random_state: int,
) -> dict[str, Any]:
    method_names = tuple(methods)
    held_out_patients = tuple(patient_ids)
    fold_metrics: list[dict[str, Any]] = []
    strata = ("all_eligible_anchors", ONSET_STRATUM)
    for method in method_names:
        for patient_id in held_out_patients:
            patient_rows = predictions.loc[
                predictions["method"].eq(method)
                & predictions["patient_id"].astype(str).eq(str(patient_id))
            ]
            for stratum in strata:
                rows = _stratum_rows(patient_rows, stratum)
                metrics = _binary_metrics(
                    rows,
                    rows["y_pred"].to_numpy(dtype="int8"),
                )
                fold_metrics.append(
                    {
                        "method": method,
                        "held_out_patient_id": str(patient_id),
                        "stratum": stratum,
                        **metrics,
                    }
                )

    overall_metrics: list[dict[str, Any]] = []
    event_fold_metrics: list[dict[str, Any]] = []
    event_overall_metrics: list[dict[str, Any]] = []
    participant_uncertainty: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for method in method_names:
        method_rows = predictions.loc[predictions["method"].eq(method)]
        for stratum in strata:
            rows = _stratum_rows(method_rows, stratum)
            overall_metrics.append(
                {
                    "method": method,
                    "stratum": stratum,
                    **_binary_metrics(
                        rows,
                        rows["y_pred"].to_numpy(dtype="int8"),
                    ),
                }
            )
            patient_values = [
                record
                for record in fold_metrics
                if record["method"] == method
                and record["stratum"] == stratum
                and record["eligible_anchors"] > 0
            ]
            for metric in ANCHOR_METRIC_COLUMNS:
                values = np.asarray(
                    [record[metric] for record in patient_values], dtype=float
                )
                if len(values) == 0:
                    mean = lower = upper = 0.0
                else:
                    samples = rng.choice(
                        values,
                        size=(bootstrap_replicates, len(values)),
                        replace=True,
                    ).mean(axis=1)
                    mean = float(values.mean())
                    lower, upper = (float(value) for value in np.quantile(samples, [0.025, 0.975]))
                participant_uncertainty.append(
                    {
                        "method": method,
                        "stratum": stratum,
                        "metric": metric,
                        "participants": len(values),
                        "participant_mean": mean,
                        "bootstrap_95pct_ci": [lower, upper],
                        "bootstrap_replicates": bootstrap_replicates,
                        "random_state": random_state,
                    }
                )

        for patient_id in held_out_patients:
            patient_rows = predictions.loc[
                predictions["method"].eq(method)
                & predictions["patient_id"].astype(str).eq(str(patient_id))
            ]
            if patient_rows.empty:
                event_result: dict[str, Any] = {
                    "actual_events": None,
                    "alert_episodes": 0,
                    "matched_events": 0,
                    "matched_alert_episodes": 0,
                    "false_alert_episodes": 0,
                    "event_precision": None,
                    "event_recall": None,
                    "measured_observation_minutes": None,
                    "measured_observation_days": None,
                    "false_alert_episodes_per_day": None,
                    "false_alert_exposure_unavailable_reason": (
                        "No eligible prediction anchors for this patient."
                    ),
                    "event_recall_unavailable_reason": (
                        "No eligible prediction anchors for this patient."
                    ),
                    "matches": [],
                }
            else:
                start = patient_rows["timestamp"].min()
                end = patient_rows["timestamp"].max()
                if evaluation_data.event_onsets is None:
                    event_times: Iterable[pd.Timestamp] = ()
                else:
                    event_rows = evaluation_data.event_onsets.loc[
                        evaluation_data.event_onsets["patient_id"]
                        .astype(str)
                        .eq(str(patient_id))
                    ]
                    event_times = event_rows["event_onset"]
                event_result = score_event_alerts(
                    patient_rows.loc[
                        patient_rows["y_pred"].eq(1), "timestamp"
                    ].tolist(),
                    event_times,
                    evaluation_data.observed_series,
                    patient_id,
                    start,
                    end,
                    forecast_horizon_minutes=TARGET_HORIZON_MINUTES,
                    refractory_minutes=ALERT_REFRACTORY_MINUTES,
                    max_observation_gap_minutes=(
                        evaluation_data.max_event_observation_gap_minutes
                    ),
                )
                if evaluation_data.event_onsets is None:
                    event_result["actual_events"] = None
                    event_result["event_recall"] = None
                    event_result["event_recall_unavailable_reason"] = (
                        "Full-series event onsets were not supplied."
                    )
            event_fold_metrics.append(
                {
                    "method": method,
                    "held_out_patient_id": str(patient_id),
                    **event_result,
                }
            )

        evaluated_events = [
            record
            for record in event_fold_metrics
            if record["method"] == method and record["actual_events"] is not None
        ]
        actual_events = sum(record["actual_events"] for record in evaluated_events)
        alert_episodes = sum(record["alert_episodes"] for record in evaluated_events)
        matched_events = sum(record["matched_events"] for record in evaluated_events)
        matched_alert_episodes = sum(
            record["matched_alert_episodes"] for record in evaluated_events
        )
        false_alert_episodes = sum(
            record["false_alert_episodes"] for record in evaluated_events
        )
        exposure_available = bool(evaluated_events) and all(
            record["measured_observation_minutes"] is not None
            for record in evaluated_events
        )
        measured_minutes = (
            sum(record["measured_observation_minutes"] for record in evaluated_events)
            if exposure_available
            else None
        )
        measured_days = measured_minutes / (60 * 24) if measured_minutes is not None else None
        event_overall = {
            "method": method,
            "patients_with_eligible_anchors": len(evaluated_events),
            "actual_events": actual_events if evaluated_events else None,
            "alert_episodes": alert_episodes,
            "matched_events": matched_events,
            "matched_alert_episodes": matched_alert_episodes,
            "false_alert_episodes": false_alert_episodes,
            "event_precision": (
                matched_alert_episodes / alert_episodes if alert_episodes else None
            ),
            "event_recall": matched_events / actual_events if actual_events else None,
            "measured_observation_minutes": measured_minutes,
            "measured_observation_days": measured_days,
            "false_alert_episodes_per_day": (
                false_alert_episodes / measured_days
                if measured_days is not None and measured_days > 0
                else None
            ),
            "false_alert_exposure_unavailable_reason": (
                None
                if measured_days is not None and measured_days > 0
                else "No positive measured eligible observation time is available."
            ),
            "event_recall_unavailable_reason": (
                None if actual_events else "No qualifying event onset in the evaluation horizon."
            ),
        }
        event_overall_metrics.append(event_overall)

        for metric in EVENT_METRIC_COLUMNS:
            values = np.asarray(
                [
                    record[metric]
                    for record in evaluated_events
                    if record[metric] is not None
                ],
                dtype=float,
            )
            if len(values) == 0:
                mean = lower = upper = None
            else:
                samples = rng.choice(
                    values,
                    size=(bootstrap_replicates, len(values)),
                    replace=True,
                ).mean(axis=1)
                mean = float(values.mean())
                lower, upper = (
                    float(value) for value in np.quantile(samples, [0.025, 0.975])
                )
            participant_uncertainty.append(
                {
                    "method": method,
                    "stratum": "event_level",
                    "metric": metric,
                    "participants": len(values),
                    "participant_mean": mean,
                    "bootstrap_95pct_ci": (
                        [lower, upper] if lower is not None else None
                    ),
                    "bootstrap_replicates": bootstrap_replicates,
                    "random_state": random_state,
                }
            )

    return {
        "fold_metrics": fold_metrics,
        "overall_metrics": overall_metrics,
        "event_fold_metrics": event_fold_metrics,
        "event_overall_metrics": event_overall_metrics,
        "participant_uncertainty": participant_uncertainty,
        "event_definition": (
            "Libre >180 mg/dL after at least 60 minutes from first to last observed "
            "Libre readings <=180 mg/dL, with every intervening source row observed "
            "and adjacent timestamps no more than one minute apart. Missing rows and "
            "larger gaps reset the low run; high values already in progress at series "
            "start are not incident events."
        ),
        "alert_episode_definition": (
            "Repeated positive anchors at or before 60 minutes after an episode's first "
            "alert are merged. Later alerts begin a new episode."
        ),
        "event_matching_definition": (
            "One alert episode matches at most one earliest future event onset in "
            "(alert_time, alert_time + 120 minutes]; each event can match only once."
        ),
        "false_alert_exposure_definition": (
            "Measured time between adjacent original rows with observed Libre values, "
            "timestamp gap <= one minute, clipped to the first and last eligible "
            "prediction anchors. Missing rows and larger gaps add no exposure; no "
            "duration is extrapolated beyond observations."
        ),
        "onset_stratum_definition": (
            "Eligible anchors with an observed Libre value at the current timestamp "
            "and current Libre glucose <= 180 mg/dL."
        ),
    }


def evaluate_baselines(
    evaluation_data: EvaluationDataset,
    bootstrap_replicates: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Score the train-fold majority forecast and report current-state detector activity."""
    _validate_evaluation_predictors(evaluation_data)
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    fold_assignments = iter_lopo_folds(evaluation_data.rows)
    prediction_frames: list[pd.DataFrame] = []
    majority_choices: list[dict[str, Any]] = []
    methods = ("training_majority",)
    current_state_activity: list[dict[str, Any]] = []

    for fold in fold_assignments:
        train_labels = fold.train.loc[fold.train[TARGET_COLUMN].notna(), TARGET_COLUMN].astype("int8")
        if train_labels.empty:
            raise ValueError(
                f"Training fold without eligible labels when holding out {fold.held_out_patient_id}"
            )
        positive_count = int(train_labels.sum())
        negative_count = int(len(train_labels) - positive_count)
        majority_class = int(positive_count > negative_count)
        current_state_anchors = fold.test
        current_state_alerts = _current_state_alerts(current_state_anchors)
        current_state_activity.append(
            {
                "held_out_patient_id": fold.held_out_patient_id,
                "prediction_anchors": int(len(current_state_anchors)),
                "fresh_libre_anchors": int(
                    current_state_anchors["glucose_observed_at_timestamp"].eq(1).sum()
                ),
                "alert_anchors": int(current_state_alerts.sum()),
                "alert_episodes": len(
                    merge_alert_episodes(
                        current_state_anchors.loc[
                            current_state_alerts.astype(bool), "timestamp"
                        ].tolist(),
                        ALERT_REFRACTORY_MINUTES,
                    )
                ),
                "role": "Current-state detector only; excluded from future-event forecast metrics.",
            }
        )
        test_eligible = fold.test.loc[fold.test[TARGET_COLUMN].notna()]
        prediction_frames.append(
            _prediction_records(
                "training_majority",
                fold.held_out_patient_id,
                test_eligible,
                np.full(len(test_eligible), majority_class, dtype="int8"),
            )
        )
        majority_choices.append(
            {
                "held_out_patient_id": fold.held_out_patient_id,
                "training_eligible_anchors": int(len(train_labels)),
                "training_positive_labels": positive_count,
                "training_negative_labels": negative_count,
                "majority_prediction": majority_class,
                "tie_rule": "Predict negative when training classes are tied.",
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    results = _score_prediction_records(
        predictions,
        methods,
        [fold.held_out_patient_id for fold in fold_assignments],
        evaluation_data,
        bootstrap_replicates,
        random_state,
    )
    results["baseline_fold_choices"] = majority_choices
    results["current_state_detector"] = {
        "decision_rule": (
            "Alert only when the Libre value is observed at the current prediction "
            "timestamp and exceeds 180 mg/dL. Missing or stale values do not alert."
        ),
        "future_event_metrics": "Not computed for this current-state detector.",
        "fold_activity": current_state_activity,
        "total_alert_anchors": sum(
            row["alert_anchors"] for row in current_state_activity
        ),
        "total_alert_episodes": sum(
            row["alert_episodes"] for row in current_state_activity
        ),
    }
    results["majority_baseline_role"] = (
        "Future-event forecast baseline; current-state detector activity is reported separately."
    )
    return results


def make_lightgbm_pipeline(
    parameters: Mapping[str, Any] | None = None,
) -> Any:
    """Create an unfitted LightGBM pipeline with fold-fitted preprocessing."""
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
    )
    classifier_parameters: dict[str, Any] = {
        "random_state": 42,
        "verbosity": -1,
    }
    classifier_parameters.update(parameters or {})
    classifier = LGBMClassifier(**classifier_parameters)
    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )


def evaluate_lightgbm_lopo(
    evaluation_data: EvaluationDataset,
    parameters: Mapping[str, Any] | None = None,
    bootstrap_replicates: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit/evaluate one fresh LightGBM pipeline per LOPO fold when explicitly called.

    No threshold tuning is performed: event probability is converted to an alert
    at the fixed 0.5 cutoff. A new default preprocessing pipeline is constructed
    and fitted on the training patients for each fold; custom estimator factories
    are intentionally not accepted.
    """
    _validate_evaluation_predictors(evaluation_data)
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    folds = iter_lopo_folds(evaluation_data.rows)
    prediction_frames: list[pd.DataFrame] = []
    for fold in folds:
        train = fold.train.loc[fold.train[TARGET_COLUMN].notna()]
        test = fold.test.loc[fold.test[TARGET_COLUMN].notna()]
        if train[TARGET_COLUMN].nunique() < 2:
            raise ValueError(
                f"Training fold for held-out patient {fold.held_out_patient_id} "
                "must contain both event classes"
            )
        estimator = make_lightgbm_pipeline(parameters)
        estimator.fit(
            train.loc[:, evaluation_data.predictor_columns],
            train[TARGET_COLUMN].astype("int8"),
        )
        if test.empty:
            prediction = np.empty(0, dtype="int8")
        else:
            classes = list(estimator.classes_)
            if 1 not in classes:
                raise ValueError("Estimator did not produce an event class")
            event_probability = estimator.predict_proba(
                test.loc[:, evaluation_data.predictor_columns]
            )[:, classes.index(1)]
            prediction = (event_probability >= 0.5).astype("int8")
        prediction_frames.append(
            _prediction_records(
                "lightgbm_fixed_0_5",
                fold.held_out_patient_id,
                test,
                prediction,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    results = _score_prediction_records(
        predictions,
        ("lightgbm_fixed_0_5",),
        [fold.held_out_patient_id for fold in folds],
        evaluation_data,
        bootstrap_replicates,
        random_state,
    )
    results["decision_threshold"] = 0.5
    results["threshold_tuning"] = "Not implemented; no threshold tuning is performed."
    return results