"""Synthetic smoke-mode LOPO runner for GlucoTwin India.

This module intentionally avoids CGMacros and existing processed real-data files.
It exercises the committed evaluation APIs with a small deterministic synthetic
fixture and keeps all real-data claims out of scope.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation import (
    CURRENT_GLUCOSE_COLUMN,
    TARGET_COLUMN,
    EvaluationDataset,
    evaluate_baselines,
    evaluate_lightgbm_lopo,
    iter_lopo_folds,
    make_lightgbm_pipeline,
    prepare_evaluation_data,
    validate_approved_predictors,
)
from src.feature_engineering import TARGET_THRESHOLD_MG_DL, build_features
from src.feature_groups import get_predictor_columns

SUPPORTED_FEATURE_SETS = ("full", "cgm-only")
SYNTHETIC_LABEL = "SYNTHETIC"


def _effective_lightgbm_parameters(
    seed: int = 42,
    lightgbm_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact LightGBM configuration used in synthetic smoke training."""
    return {
        "n_estimators": 75,
        "learning_rate": 0.08,
        "max_depth": 3,
        "num_leaves": 15,
        "random_state": seed,
        "verbosity": -1,
        **(lightgbm_parameters or {}),
    }


def _select_feature_set(columns: Iterable[str], feature_set: str) -> list[str]:
    """Return an approved predictor list for full or CGM-only evaluation."""
    full_predictors = get_predictor_columns(columns)
    if feature_set == "full":
        approved = full_predictors
    elif feature_set == "cgm-only":
        approved = [
            column for column in full_predictors if column.startswith("glucose_")
        ]
        if not approved:
            raise ValueError("No CGM-only predictors were available for the synthetic fixture")
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set!r}. Choose from {SUPPORTED_FEATURE_SETS}.")
    validate_approved_predictors(columns, approved)
    return approved


def generate_synthetic_smoke_fixture(
    seed: int = 42,
    n_participants: int = 4,
    days: int = 2,
    start: str = "2025-01-01 00:00:00",
) -> pd.DataFrame:
    """Create a small deterministic synthetic fixture with both target classes."""
    if n_participants < 4:
        raise ValueError("n_participants must be at least 4 for LOPO smoke validation")
    if days < 1:
        raise ValueError("days must be at least 1")

    rng = np.random.default_rng(seed)
    patient_ids = [f"SYN-{index:03d}" for index in range(1, n_participants + 1)]
    samples_per_day = int(24 * 60)
    timestamps = pd.date_range(start=start, periods=days * samples_per_day, freq="1min")
    rows: list[dict[str, Any]] = []

    for patient_index, patient_id in enumerate(patient_ids):
        elapsed = np.arange(len(timestamps), dtype=float)
        phase = elapsed / 24.0
        base = 112.0 + 12.0 * np.sin(2.0 * np.pi * (phase + patient_index * 0.17))
        circadian = 10.0 * np.sin(2.0 * np.pi * ((elapsed / 12.0) - 7.0) / 24.0)
        meal_effect = np.zeros(len(timestamps), dtype=float)
        meal_hours = [8, 13, 19]
        for hour in meal_hours:
            offset = int(np.round(hour * 60.0 + rng.uniform(-5.0, 5.0)))
            offset = min(max(offset, 0), len(timestamps) - 1)
            meal_window = slice(offset, min(offset + 24, len(timestamps)))
            meal_effect[meal_window] += 25.0 * np.linspace(0.1, 1.0, len(range(*meal_window.indices(len(timestamps)))))
        high_segments = np.zeros(len(timestamps), dtype=float)
        if patient_index % 2 == 0:
            high_segments[75:135] = 70.0
            high_segments[420:500] = 78.0
        else:
            high_segments[180:240] = 95.0
            high_segments[540:610] = 60.0
        noise = rng.normal(0.0, 5.0, size=len(timestamps))
        glucose = np.clip(base + circadian + meal_effect + high_segments + noise, 60.0, 260.0)
        glucose = np.round(glucose, 1)
        target = np.full(len(timestamps), np.nan, dtype=float)
        for index in range(len(timestamps) - 24):
            target[index] = int(np.any(glucose[index + 1 : index + 25] > TARGET_THRESHOLD_MG_DL))

        heart_rate = 72.0 + 5.0 * np.sin(elapsed / 18.0 + patient_index)
        activity_calories = np.linspace(0.0, 3.0, len(timestamps)) + (patient_index * 0.5)
        activity_mets = np.clip(np.sin(elapsed / 32.0 + patient_index) + 1.0, 0.0, 2.5)
        activity_intensity = np.clip(activity_mets * 5.0, 0.0, 12.0)
        steps = np.clip(np.abs(np.sin(elapsed / 16.0 + patient_index)) * 80.0, 0.0, 120.0)
        meal_calories = np.zeros(len(timestamps), dtype=float)
        meal_carbs_g = np.zeros(len(timestamps), dtype=float)
        meal_protein_g = np.zeros(len(timestamps), dtype=float)
        meal_fat_g = np.zeros(len(timestamps), dtype=float)
        meal_fiber_g = np.zeros(len(timestamps), dtype=float)
        meal_sugar_g = np.zeros(len(timestamps), dtype=float)
        meal_amount_consumed_pct = np.zeros(len(timestamps), dtype=float)
        meal_type = [None] * len(timestamps)

        for timestamp, value in zip(timestamps, glucose, strict=True):
            index = np.where(timestamps == timestamp)[0][0]
            rows.append(
                {
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "libre_gl_mg_dl": float(value),
                    "dexcom_gl_mg_dl": np.nan,
                    "heart_rate_bpm": float(heart_rate[index]),
                    "activity_calories": float(activity_calories[index]),
                    "activity_mets": float(activity_mets[index]),
                    "activity_intensity": float(activity_intensity[index]),
                    "steps": float(steps[index]),
                    "meal_calories": float(meal_calories[index]),
                    "meal_carbs_g": float(meal_carbs_g[index]),
                    "meal_protein_g": float(meal_protein_g[index]),
                    "meal_fat_g": float(meal_fat_g[index]),
                    "meal_fiber_g": float(meal_fiber_g[index]),
                    "meal_sugar_g": float(meal_sugar_g[index]),
                    "meal_amount_consumed_pct": float(meal_amount_consumed_pct[index]),
                    "meal_type": meal_type[index],
                    "meal_image_path": None,
                    "record_index": int(index),
                    "glucose_mg_dl": float(value),
                    "glucose_source": "libre" if pd.notna(value) else "missing",
                    "glucose_observed": int(pd.notna(value)),
                    "libre_missing": int(pd.isna(value)),
                    "dexcom_missing": 1,
                    "synthetic_label": SYNTHETIC_LABEL,
                    "target": float(target[index]),
                }
            )

    frame = pd.DataFrame(rows)
    if frame["patient_id"].nunique() < n_participants:
        raise ValueError("Synthetic fixture failed to produce all requested participants")
    if frame.groupby("patient_id")["target"].nunique().min() < 2:
        raise ValueError("Synthetic fixture does not contain both target classes per patient")
    return frame


def _resolve_feature_columns(normalized_time_series: pd.DataFrame, feature_set: str) -> list[str]:
    feature_table = build_features(normalized_time_series)
    return _select_feature_set(feature_table.columns, feature_set)


def build_feature_table_for_evaluation(normalized_time_series: pd.DataFrame) -> pd.DataFrame:
    """Create the feature table expected by the current evaluator."""
    feature_table = build_features(normalized_time_series)
    return feature_table


def _build_oof_predictions(
    normalized_time_series: pd.DataFrame,
    feature_set: str = "full",
    seed: int = 42,
    lightgbm_parameters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], EvaluationDataset]:
    """Train a fresh LightGBM model per patient fold and return OOF predictions."""
    feature_table = build_features(normalized_time_series)
    feature_columns = _select_feature_set(feature_table.columns, feature_set)
    evaluation_data = prepare_evaluation_data(
        normalized_time_series,
        approved_predictors=feature_columns,
    )
    effective_parameters = _effective_lightgbm_parameters(seed=seed, lightgbm_parameters=lightgbm_parameters)

    folds = iter_lopo_folds(evaluation_data.rows)
    records: list[dict[str, Any]] = []

    for fold in folds:
        train = fold.train.loc[fold.train[TARGET_COLUMN].notna()].copy()
        test = fold.test.loc[fold.test[TARGET_COLUMN].notna()].copy()
        if train.empty:
            raise ValueError(f"Training fold for patient {fold.held_out_patient_id} has no eligible labels")
        if train[TARGET_COLUMN].nunique() < 2:
            raise ValueError(
                f"Training fold for patient {fold.held_out_patient_id} has only one target class; "
                "a valid LOPO fold needs both classes."
            )

        estimator = make_lightgbm_pipeline(effective_parameters)
        X_train = train.loc[:, evaluation_data.predictor_columns]
        y_train = train[TARGET_COLUMN].astype("int8")
        estimator.fit(X_train, y_train)
        classes = list(estimator.classes_)
        if 1 not in classes:
            raise ValueError(f"Estimator for held-out patient {fold.held_out_patient_id} produced no positive class")

        X_test = test.loc[:, evaluation_data.predictor_columns]
        probabilities = estimator.predict_proba(X_test)[:, classes.index(1)]
        predictions = (probabilities >= 0.5).astype("int8")

        output = test[["patient_id", "timestamp", TARGET_COLUMN, CURRENT_GLUCOSE_COLUMN, "glucose_observed_at_timestamp"]].copy()
        output["fold_id"] = fold.held_out_patient_id
        output["y_true"] = output.pop(TARGET_COLUMN).astype("int8")
        output["y_prob"] = probabilities.astype(float)
        output["y_pred"] = predictions.astype("int8")
        output["synthetic_label"] = SYNTHETIC_LABEL
        output["oof_key"] = output["patient_id"].astype(str) + "|" + output["timestamp"].astype(str)
        records.append(output.reset_index(drop=True))

    oof = pd.concat(records, ignore_index=True)
    if oof.empty:
        raise ValueError("No out-of-fold predictions were produced from the synthetic LOPO fixture")
    if oof["oof_key"].duplicated().any():
        raise ValueError("Out-of-fold predictions are not unique per patient/timestamp key")
    expected_keys = set(
        evaluation_data.rows.loc[evaluation_data.rows[TARGET_COLUMN].notna(), ["patient_id", "timestamp"]]
        .apply(lambda row: f"{row['patient_id']}|{row['timestamp']}", axis=1)
    )
    if set(oof["oof_key"]) != expected_keys:
        raise ValueError("Out-of-fold predictions do not cover exactly the eligible anchors for held-out patients")

    summary = {
        "synthetic_mode": True,
        "seed": seed,
        "feature_set": feature_set,
        "fold_count": len(folds),
        "lightgbm_parameters": effective_parameters,
        "eligible_anchor_count": int(len(oof)),
        "positive_anchor_count": int(oof["y_true"].sum()),
        "negative_anchor_count": int((oof["y_true"] == 0).sum()),
        "current_libre_reference": {
            "definition": "Fresh observed current Libre > 180 mg/dL and not missing at the prediction timestamp.",
            "alert_count": int(((oof["glucose_observed_at_timestamp"] == 1) & (oof[CURRENT_GLUCOSE_COLUMN].gt(TARGET_THRESHOLD_MG_DL))).sum()),
            "alert_rate": float(((oof["glucose_observed_at_timestamp"] == 1) & (oof[CURRENT_GLUCOSE_COLUMN].gt(TARGET_THRESHOLD_MG_DL))).mean()),
        },
    }
    return oof, summary, evaluation_data


def _safe_auroc(y_true: pd.Series, y_prob: pd.Series) -> float | None:
    values = y_true.to_numpy(dtype="int8")
    scores = y_prob.to_numpy(dtype=float)
    if len(values) == 0 or np.unique(values).size < 2:
        return None
    try:
        return float(roc_auc_score(values, scores))
    except ValueError:
        return None


def _safe_auprc(y_true: pd.Series, y_prob: pd.Series) -> float | None:
    values = y_true.to_numpy(dtype="int8")
    scores = y_prob.to_numpy(dtype=float)
    if len(values) == 0 or np.unique(values).size < 2:
        return None
    try:
        return float(average_precision_score(values, scores))
    except ValueError:
        return None


def _safe_brier(y_true: pd.Series, y_prob: pd.Series) -> float | None:
    values = y_true.to_numpy(dtype="int8")
    scores = y_prob.to_numpy(dtype=float)
    if len(values) == 0 or np.unique(values).size < 2:
        return None
    try:
        return float(brier_score_loss(values, scores))
    except ValueError:
        return None


def _summarize_oof_metrics(oof: pd.DataFrame) -> dict[str, Any]:
    if oof.empty:
        raise ValueError("No out-of-fold predictions were supplied for metric summarization")
    metrics: dict[str, Any] = {
        "positive_prevalence": float(oof["y_true"].mean()),
        "oof_anchor_count": int(len(oof)),
        "fold_count": int(oof["fold_id"].nunique()),
        "held_out_patients": sorted(oof["patient_id"].astype(str).unique().tolist()),
        "auroc": _safe_auroc(oof["y_true"], oof["y_prob"]),
        "auprc": _safe_auprc(oof["y_true"], oof["y_prob"]),
        "brier_score": _safe_brier(oof["y_true"], oof["y_prob"]),
    }
    metrics["baseline_prevalence_defined"] = True
    metrics["reference_rule"] = (
        "Current Libre threshold reference only: alert when the observed current value exceeds 180 mg/dL; "
        "not a future-event forecast."
    )
    return metrics


def run_smoke_test(
    output_dir: str | Path | None = None,
    seed: int = 42,
    feature_set: str = "full",
    n_participants: int = 4,
    days: int = 2,
) -> dict[str, Any]:
    """Train a synthetic LOPO smoke loop and save explicit OOF artifacts."""
    output_path = Path(output_dir) if output_dir is not None else Path("data/processed/run_lopo_smoke")
    output_path.mkdir(parents=True, exist_ok=True)

    fixture = generate_synthetic_smoke_fixture(seed=seed, n_participants=n_participants, days=days)
    normalized = fixture.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    normalized["libre_gl_mg_dl"] = pd.to_numeric(normalized["libre_gl_mg_dl"], errors="coerce")
    normalized["dexcom_gl_mg_dl"] = pd.to_numeric(normalized["dexcom_gl_mg_dl"], errors="coerce")
    normalized["heart_rate_bpm"] = pd.to_numeric(normalized["heart_rate_bpm"], errors="coerce")
    normalized["activity_calories"] = pd.to_numeric(normalized["activity_calories"], errors="coerce")
    normalized["activity_mets"] = pd.to_numeric(normalized["activity_mets"], errors="coerce")
    normalized["activity_intensity"] = pd.to_numeric(normalized["activity_intensity"], errors="coerce")
    normalized["steps"] = pd.to_numeric(normalized["steps"], errors="coerce")
    normalized["meal_calories"] = pd.to_numeric(normalized["meal_calories"], errors="coerce")
    normalized["meal_carbs_g"] = pd.to_numeric(normalized["meal_carbs_g"], errors="coerce")
    normalized["meal_protein_g"] = pd.to_numeric(normalized["meal_protein_g"], errors="coerce")
    normalized["meal_fat_g"] = pd.to_numeric(normalized["meal_fat_g"], errors="coerce")
    normalized["meal_fiber_g"] = pd.to_numeric(normalized["meal_fiber_g"], errors="coerce")
    normalized["meal_sugar_g"] = pd.to_numeric(normalized["meal_sugar_g"], errors="coerce")
    normalized["meal_amount_consumed_pct"] = pd.to_numeric(normalized["meal_amount_consumed_pct"], errors="coerce")
    normalized["glucose_mg_dl"] = pd.to_numeric(normalized["glucose_mg_dl"], errors="coerce")
    normalized["record_index"] = pd.to_numeric(normalized["record_index"], errors="coerce")

    oof, summary, evaluation_data = _build_oof_predictions(
        normalized,
        feature_set=feature_set,
        seed=seed,
    )
    oof_output_path = output_path / "synthetic_oof_predictions.csv"
    oof.to_csv(oof_output_path, index=False)

    baseline_report = evaluate_baselines(evaluation_data, bootstrap_replicates=25, random_state=seed)
    lightgbm_report = evaluate_lightgbm_lopo(
        evaluation_data,
        parameters={"n_estimators": 75, "learning_rate": 0.08, "max_depth": 3, "num_leaves": 15, "random_state": seed, "verbosity": -1},
        bootstrap_replicates=25,
        random_state=seed,
    )

    metrics = _summarize_oof_metrics(oof)
    result = {
        "synthetic_data": True,
        "synthetic_label": SYNTHETIC_LABEL,
        "seed": seed,
        "feature_set": feature_set,
        "fold_count": summary["fold_count"],
        "lightgbm_parameters": summary["lightgbm_parameters"],
        "eligible_anchor_count": summary["eligible_anchor_count"],
        "out_of_fold_predictions_path": str(oof_output_path),
        "report": {
            "prevalence_baseline": metrics["positive_prevalence"],
            "current_libre_threshold_reference": summary["current_libre_reference"],
            "oof_metrics": {
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier_score": metrics["brier_score"],
                "positive_prevalence": metrics["positive_prevalence"],
            },
            "baseline_summary": {
                "fold_count": len(baseline_report["baseline_fold_choices"]),
                "majority_baseline_role": baseline_report["majority_baseline_role"],
                "current_state_detector": baseline_report["current_state_detector"],
            },
            "lightgbm_lopo_summary": {
                "decision_threshold": lightgbm_report.get("decision_threshold"),
                "threshold_tuning": lightgbm_report.get("threshold_tuning"),
                "overall_metrics": lightgbm_report.get("overall_metrics"),
                "event_overall_metrics": lightgbm_report.get("event_overall_metrics"),
            },
        },
    }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run the synthetic LOPO smoke test only")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the synthetic smoke fixture")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/run_lopo_smoke"), help="output directory for synthetic smoke artifacts")
    parser.add_argument(
        "--feature-set",
        choices=SUPPORTED_FEATURE_SETS,
        default="full",
        help="approved feature set: full predictor list or CGM-only subset",
    )
    parser.add_argument("--n-participants", type=int, default=4, help="number of synthetic participants for the smoke fixture")
    parser.add_argument("--days", type=int, default=2, help="synthetic days per participant")
    arguments = parser.parse_args()

    if arguments.smoke_test:
        result = run_smoke_test(
            output_dir=arguments.output_dir,
            seed=arguments.seed,
            feature_set=arguments.feature_set,
            n_participants=arguments.n_participants,
            days=arguments.days,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
