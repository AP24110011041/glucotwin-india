"""Normalize CGMacros data and build causal two-hour glucose features and labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_THRESHOLD_MG_DL = 180.0
TARGET_HORIZON_MINUTES = 120
MAX_FEATURE_FORWARD_FILL_MINUTES = 5
MAX_LABEL_CGM_GAP_MINUTES = 5

PARTICIPANT_FILE_PATTERN = re.compile(r"^CGMacros-(\d+)\.csv$", re.IGNORECASE)
UNNAMED_INDEX_PATTERN = re.compile(r"^unnamed_(\d+)$", re.IGNORECASE)

HEADER_ALIASES = {
    "timestamp": "timestamp",
    "subject": "subject",
    "age": "age_years",
    "age_years": "age_years",
    "gender": "gender",
    "bmi": "bmi",
    "body_weight": "body_weight_lb",
    "height": "height_in",
    "self_identify": "self_identify",
    "a1c_pdl_lab": "a1c_pdl_lab",
    "fasting_glu_pdl_lab": "fasting_glucose_pdl_lab",
    "insulin": "insulin_pdl_lab",
    "triglycerides": "triglycerides_pdl_lab",
    "cholesterol": "cholesterol_pdl_lab",
    "hdl": "hdl_pdl_lab",
    "non_hdl": "non_hdl_pdl_lab",
    "ldl_cal": "ldl_calculated_pdl_lab",
    "vldl_cal": "vldl_calculated_pdl_lab",
    "cho_hdl_ratio": "cholesterol_hdl_ratio_pdl_lab",
    "libre_gl": "libre_gl_mg_dl",
    "dexcom_gl": "dexcom_gl_mg_dl",
    "hr": "heart_rate_bpm",
    "calories_activity": "activity_calories",
    "mets": "activity_mets",
    "intensity": "activity_intensity",
    "steps": "steps",
    "meal_type": "meal_type",
    "calories": "meal_calories",
    "carbs": "meal_carbs_g",
    "protein": "meal_protein_g",
    "fat": "meal_fat_g",
    "fiber": "meal_fiber_g",
    "sugar": "meal_sugar_g",
    "amount_consumed": "meal_amount_consumed_pct",
    "image_path": "meal_image_path",
    "record_index": "record_index",
    "recordindex": "record_index",
}

STATIC_EHR_COLUMNS = (
    "age_years",
    "gender",
    "bmi",
    "body_weight_lb",
    "height_in",
    "self_identify",
    "a1c_pdl_lab",
    "fasting_glucose_pdl_lab",
    "insulin_pdl_lab",
    "triglycerides_pdl_lab",
    "cholesterol_pdl_lab",
    "hdl_pdl_lab",
    "non_hdl_pdl_lab",
    "ldl_calculated_pdl_lab",
    "vldl_calculated_pdl_lab",
    "cholesterol_hdl_ratio_pdl_lab",
)

NUMERIC_TIME_SERIES_COLUMNS = (
    "libre_gl_mg_dl",
    "dexcom_gl_mg_dl",
    "heart_rate_bpm",
    "activity_calories",
    "activity_mets",
    "activity_intensity",
    "steps",
    "meal_calories",
    "meal_carbs_g",
    "meal_protein_g",
    "meal_fat_g",
    "meal_fiber_g",
    "meal_sugar_g",
    "meal_amount_consumed_pct",
)

MEAL_NUMERIC_COLUMNS = (
    "meal_calories",
    "meal_carbs_g",
    "meal_protein_g",
    "meal_fat_g",
    "meal_fiber_g",
    "meal_sugar_g",
    "meal_amount_consumed_pct",
)

WEARABLE_COLUMNS = (
    "heart_rate_bpm",
    "activity_calories",
    "activity_mets",
    "activity_intensity",
    "steps",
)

NORMALIZED_TIME_SERIES_COLUMNS = (
    "patient_id",
    "timestamp",
    *NUMERIC_TIME_SERIES_COLUMNS,
    "meal_type",
    "meal_image_path",
    "record_index",
    "glucose_mg_dl",
    "glucose_source",
    "glucose_observed",
    "libre_missing",
    "dexcom_missing",
)


def _header_key(column: object) -> str:
    """Normalize header spelling without changing source values."""
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().casefold()).strip("_")


def _canonical_header(column: object) -> str:
    key = _header_key(column)
    return HEADER_ALIASES.get(key, key or "unnamed_column")


def normalize_patient_id(value: object) -> str:
    """Return a zero-padded CGMacros subject ID suitable for file/EHR joins."""
    if pd.isna(value):
        raise ValueError("Patient ID cannot be missing")
    text = str(value).strip()
    match = re.fullmatch(r"(?:CGMacros[-_ ]*)?0*(\d+)(?:\.0+)?", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot normalize CGMacros patient ID: {value!r}")
    return f"{int(match.group(1)):03d}"


def infer_patient_id(path: Path) -> str:
    """Infer a participant ID from a CGMacros-NNN.csv filename."""
    match = PARTICIPANT_FILE_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(
            f"Participant filename must match 'CGMacros-NNN.csv': '{path.name}'"
        )
    return normalize_patient_id(match.group(1))


def _is_sequential_exported_index(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy()
    return bool(
        len(values) > 0
        and np.isfinite(values).all()
        and np.array_equal(values, np.arange(len(values), dtype=float))
    )


def normalize_participant_frame(frame: pd.DataFrame, patient_id: object) -> pd.DataFrame:
    """Canonicalize one participant, preserving both source sensor channels.

    Verified sequential zero-based ``Unnamed: N`` columns are discarded as CSV
    export indexes. Exact duplicate rows are removed; conflicting rows at one
    timestamp raise instead of silently choosing or averaging measurements.
    """
    if frame.empty:
        raise ValueError("Participant CSV contains no data rows")

    source = frame.copy()
    rename: dict[object, str] = {}
    drop_columns: list[object] = []
    for column in source.columns:
        key = _header_key(column)
        index_match = UNNAMED_INDEX_PATTERN.fullmatch(key)
        if index_match and _is_sequential_exported_index(source[column]):
            drop_columns.append(column)
            continue
        rename[column] = _canonical_header(column)

    source = source.drop(columns=drop_columns).rename(columns=rename)
    if source.columns.duplicated().any():
        duplicate_names = sorted(
            set(source.columns[source.columns.duplicated(keep=False)].astype(str))
        )
        raise ValueError(f"Columns collide after normalization: {duplicate_names}")
    if "timestamp" not in source.columns:
        raise ValueError("Participant CSV is missing the required Timestamp column")

    parsed_timestamps = pd.to_datetime(source["timestamp"], errors="coerce")
    invalid_timestamps = int(parsed_timestamps.isna().sum())
    if invalid_timestamps:
        raise ValueError(
            f"Participant {normalize_patient_id(patient_id)} has "
            f"{invalid_timestamps} invalid Timestamp value(s)"
        )
    source["timestamp"] = parsed_timestamps

    for column in NUMERIC_TIME_SERIES_COLUMNS:
        if column not in source:
            source[column] = np.nan
        else:
            converted = pd.to_numeric(source[column], errors="coerce")
            invalid = source[column].notna() & converted.isna()
            if invalid.any():
                raise ValueError(f"Non-numeric values found in numeric column '{column}'")
            source[column] = converted
    for column in ("meal_type", "meal_image_path", "record_index"):
        if column not in source:
            source[column] = np.nan

    source = source.drop_duplicates().sort_values("timestamp", kind="mergesort")
    duplicate_timestamps = source["timestamp"].duplicated(keep=False)
    if duplicate_timestamps.any():
        duplicate_count = int(duplicate_timestamps.sum())
        raise ValueError(
            f"Participant {normalize_patient_id(patient_id)} has "
            f"{duplicate_count} conflicting row(s) at duplicate timestamps"
        )

    source["patient_id"] = normalize_patient_id(patient_id)
    libre = source["libre_gl_mg_dl"]
    dexcom = source["dexcom_gl_mg_dl"]
    source["glucose_mg_dl"] = libre.combine_first(dexcom)
    source["glucose_source"] = np.select(
        [libre.notna(), dexcom.notna()], ["libre", "dexcom"], default="missing"
    )
    source["glucose_observed"] = source["glucose_mg_dl"].notna().astype("int8")
    source["libre_missing"] = libre.isna().astype("int8")
    source["dexcom_missing"] = dexcom.isna().astype("int8")

    ordered_columns = [
        column for column in NORMALIZED_TIME_SERIES_COLUMNS if column in source.columns
    ]
    ordered_columns.extend(sorted(set(source.columns).difference(ordered_columns)))
    return source.loc[:, ordered_columns].reset_index(drop=True)


def load_static_ehr(path: Path) -> pd.DataFrame:
    """Load and validate baseline static fields from the inspected bio.csv schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Static EHR file not found: '{path}'")
    bio = pd.read_csv(path)
    subject_columns = [
        column for column in bio.columns if _canonical_header(column) == "subject"
    ]
    if len(subject_columns) != 1:
        raise ValueError("bio.csv must contain exactly one subject column")

    bio = bio.rename(columns={column: _canonical_header(column) for column in bio.columns})
    if bio.columns.duplicated().any():
        raise ValueError("bio.csv has columns that collide after normalization")
    subject = subject_columns[0]
    # Resolve the subject column by its canonical name after normalization.
    subject_name = _canonical_header(subject)
    bio["patient_id"] = bio[subject_name].map(normalize_patient_id)
    if bio["patient_id"].duplicated().any():
        raise ValueError("bio.csv contains duplicate normalized subject IDs")

    available_static = [column for column in STATIC_EHR_COLUMNS if column in bio.columns]
    if not available_static:
        raise ValueError("bio.csv contains no recognized baseline static clinical fields")
    static = bio[["patient_id", *available_static]].copy()
    for column in available_static:
        if column not in {"gender", "self_identify"}:
            static[column] = pd.to_numeric(static[column], errors="coerce")
    return static


def join_static_ehr(
    time_series: pd.DataFrame, static_ehr: pd.DataFrame
) -> pd.DataFrame:
    """Join baseline EHR fields by normalized patient ID, never by row order."""
    if "patient_id" not in time_series or "patient_id" not in static_ehr:
        raise ValueError("Both time series and static EHR require patient_id")
    if static_ehr["patient_id"].duplicated().any():
        raise ValueError("Static EHR patient_id values must be unique")
    left = time_series.copy()
    left["patient_id"] = left["patient_id"].map(normalize_patient_id)
    right = static_ehr.copy()
    right["patient_id"] = right["patient_id"].map(normalize_patient_id)
    static_columns = [column for column in right.columns if column != "patient_id"]
    joined = left.merge(right, on="patient_id", how="left", validate="many_to_one")
    if static_columns and joined[static_columns].isna().all(axis=1).any():
        missing_ids = sorted(joined.loc[joined[static_columns].isna().all(axis=1), "patient_id"].unique())
        raise ValueError(f"No baseline EHR record found for patient ID(s): {missing_ids}")
    return joined


def create_future_target(
    patient_frame: pd.DataFrame,
    horizon_minutes: int = TARGET_HORIZON_MINUTES,
    threshold_mg_dl: float = TARGET_THRESHOLD_MG_DL,
    max_cgm_gap_minutes: int = MAX_LABEL_CGM_GAP_MINUTES,
) -> tuple[pd.Series, dict[str, int]]:
    """Label observed Libre excursions in (t, t+horizon], requiring Libre coverage.

    A window is eligible only if the source time series reaches its end and
    observed Libre timestamps cover the interval with no gap longer than
    ``max_cgm_gap_minutes``. Dexcom readings do not affect labels or eligibility.
    Readings are never imputed for labels.
    """
    if horizon_minutes <= 0 or max_cgm_gap_minutes <= 0:
        raise ValueError("horizon_minutes and max_cgm_gap_minutes must be positive")
    if patient_frame.empty:
        return pd.Series(dtype="Int8"), {
            "rows": 0,
            "eligible_rows": 0,
            "positive_rows": 0,
            "negative_rows": 0,
            "incomplete_horizon_rows": 0,
            "incomplete_libre_coverage_rows": 0,
        }
    if patient_frame["patient_id"].nunique() > 1:
        raise ValueError("create_future_target must receive one patient at a time")

    frame = patient_frame.sort_values("timestamp", kind="mergesort")
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    times_ns = timestamps.as_unit("ns").asi8
    horizon_ns = int(pd.Timedelta(minutes=horizon_minutes).value)
    gap_ns = int(pd.Timedelta(minutes=max_cgm_gap_minutes).value)
    end_ns = times_ns + horizon_ns

    libre = pd.to_numeric(frame.get("libre_gl_mg_dl", pd.Series(np.nan, index=frame.index)), errors="coerce")
    observed = libre.notna().to_numpy()
    exceeds = libre.gt(threshold_mg_dl).to_numpy()
    prefix_exceeds = np.concatenate(([0], np.cumsum(exceeds, dtype=np.int64)))
    future_row_end = np.searchsorted(times_ns, end_ns, side="right")
    positive_counts = prefix_exceeds[future_row_end] - prefix_exceeds[np.arange(len(frame)) + 1]

    observed_times = times_ns[observed]
    if len(observed_times):
        first_observed_index = np.searchsorted(observed_times, times_ns, side="right")
        observed_end_index = np.searchsorted(observed_times, end_ns, side="right")
        has_observed_glucose = observed_end_index > first_observed_index
        first_future = np.full(len(frame), np.iinfo(np.int64).max, dtype=np.int64)
        last_future = np.full(len(frame), np.iinfo(np.int64).min, dtype=np.int64)
        first_future[has_observed_glucose] = observed_times[first_observed_index[has_observed_glucose]]
        last_future[has_observed_glucose] = observed_times[observed_end_index[has_observed_glucose] - 1]
        boundary_coverage = (
            (first_future - times_ns <= gap_ns)
            & (end_ns - last_future <= gap_ns)
        )

        observed_gaps = np.diff(observed_times)
        long_gap_mask = observed_gaps > gap_ns
        long_gap_left = observed_times[:-1][long_gap_mask]
        long_gap_right = observed_times[1:][long_gap_mask]
        gap_start = np.searchsorted(long_gap_left, times_ns, side="left")
        gap_end = np.searchsorted(long_gap_right, end_ns, side="right")
        no_internal_long_gap = gap_end <= gap_start
        cgm_coverage = has_observed_glucose & boundary_coverage & no_internal_long_gap
    else:
        cgm_coverage = np.zeros(len(frame), dtype=bool)

    horizon_available = end_ns <= times_ns[-1]
    eligible = horizon_available & cgm_coverage
    target_values = np.full(len(frame), -1, dtype=np.int8)
    target_values[eligible] = (positive_counts[eligible] > 0).astype(np.int8)
    target = pd.Series(
        pd.array([pd.NA if value < 0 else value for value in target_values], dtype="Int8"),
        index=frame.index,
        name="glucose_over_180_next_2h",
    )
    summary = {
        "rows": int(len(frame)),
        "eligible_rows": int(eligible.sum()),
        "positive_rows": int(((target_values == 1) & eligible).sum()),
        "negative_rows": int(((target_values == 0) & eligible).sum()),
        "incomplete_horizon_rows": int((~horizon_available).sum()),
        "incomplete_libre_coverage_rows": int((horizon_available & ~cgm_coverage).sum()),
    }
    return target, summary


def _limited_forward_fill(
    values: pd.Series, timestamps: pd.DatetimeIndex, max_age_minutes: int
) -> tuple[pd.Series, pd.Series]:
    """Use only the latest past value and mask it once its elapsed age exceeds a limit."""
    observed_at = pd.Series(timestamps, index=values.index).where(values.notna()).ffill()
    age_minutes = (
        (pd.Series(timestamps, index=values.index) - observed_at).dt.total_seconds() / 60
    )
    current = values.ffill().where(age_minutes <= max_age_minutes)
    return current, age_minutes


def _features_for_patient(
    patient_frame: pd.DataFrame,
    horizon_minutes: int,
    threshold_mg_dl: float,
    max_feature_forward_fill_minutes: int,
    max_label_cgm_gap_minutes: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = patient_frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    feature_columns: dict[str, Any] = {
        "patient_id": frame["patient_id"].astype(str).to_numpy(),
        "timestamp": timestamps,
    }

    glucose = pd.to_numeric(frame["libre_gl_mg_dl"], errors="coerce")
    feature_columns["glucose_observed_at_timestamp"] = glucose.notna().to_numpy(dtype="int8")
    feature_columns["glucose_missing_at_timestamp"] = glucose.isna().to_numpy(dtype="int8")
    feature_columns["libre_missing_at_timestamp"] = frame["libre_gl_mg_dl"].isna().to_numpy(dtype="int8")
    glucose_current, glucose_age = _limited_forward_fill(
        glucose, timestamps, max_feature_forward_fill_minutes
    )
    feature_columns["glucose_current_mg_dl"] = glucose_current.to_numpy()
    feature_columns["glucose_current_age_minutes"] = glucose_age.to_numpy()
    feature_columns["glucose_current_missing_or_stale"] = glucose_current.isna().to_numpy(dtype="int8")

    glucose_indexed = pd.Series(glucose.to_numpy(), index=timestamps)
    for lag_minutes in (1, 5, 15, 30, 60, 120):
        lag_index = timestamps - pd.Timedelta(minutes=lag_minutes)
        feature_columns[f"glucose_lag_{lag_minutes}m_mg_dl"] = glucose_indexed.reindex(lag_index).to_numpy()

    glucose_rolling = glucose.copy()
    glucose_rolling.index = timestamps
    for window_minutes in (15, 30, 60, 120):
        rolling = glucose_rolling.rolling(
            f"{window_minutes}min", closed="both", min_periods=1
        )
        feature_columns[f"glucose_mean_{window_minutes}m_mg_dl"] = rolling.mean().to_numpy()
        feature_columns[f"glucose_min_{window_minutes}m_mg_dl"] = rolling.min().to_numpy()
        feature_columns[f"glucose_max_{window_minutes}m_mg_dl"] = rolling.max().to_numpy()
        feature_columns[f"glucose_std_{window_minutes}m_mg_dl"] = rolling.std(ddof=0).to_numpy()
        feature_columns[f"glucose_observation_count_{window_minutes}m"] = rolling.count().to_numpy()
    feature_columns["glucose_trend_15m_mg_dl_per_min"] = (
        glucose.to_numpy() - feature_columns["glucose_lag_15m_mg_dl"]
    ) / 15.0
    feature_columns["glucose_trend_30m_mg_dl_per_min"] = (
        glucose.to_numpy() - feature_columns["glucose_lag_30m_mg_dl"]
    ) / 30.0

    meal_columns_present = [column for column in MEAL_NUMERIC_COLUMNS if column in frame]
    meal_type = frame.get("meal_type", pd.Series(pd.NA, index=frame.index))
    meal_observed = meal_type.notna()
    if meal_columns_present:
        meal_observed |= frame[meal_columns_present].notna().any(axis=1)
    feature_columns["meal_event_observed_at_timestamp"] = meal_observed.astype("int8")
    feature_columns["meal_data_not_recorded_at_timestamp"] = (~meal_observed).to_numpy(dtype="int8")
    normalized_meal_type = meal_type.astype("string").str.strip().str.casefold()
    for meal_name in ("breakfast", "lunch", "dinner"):
        feature_columns[f"meal_type_{meal_name}_at_timestamp"] = (
            normalized_meal_type.eq(meal_name).fillna(False).to_numpy(dtype="int8")
        )

    event_index = pd.Series(meal_observed.to_numpy(dtype=float), index=timestamps)
    for window_minutes in (15, 30, 60, 120):
        feature_columns[f"meal_event_count_{window_minutes}m"] = event_index.rolling(
            f"{window_minutes}min", closed="both", min_periods=1
        ).sum().to_numpy()
    for column in MEAL_NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)
        feature_columns[f"{column}_missing_at_timestamp"] = values.isna().to_numpy(dtype="int8")
        event_indexed = pd.Series(values.to_numpy(), index=timestamps)
        for window_minutes in (15, 60, 120):
            rolling = event_indexed.rolling(
                f"{window_minutes}min", closed="both", min_periods=1
            )
            feature_columns[f"{column}_sum_{window_minutes}m"] = rolling.sum().to_numpy()
            feature_columns[f"{column}_observation_count_{window_minutes}m"] = (
                rolling.count().to_numpy()
            )

    meal_event_time = pd.Series(timestamps, index=frame.index).where(meal_observed).ffill()
    feature_columns["meal_time_since_last_observed_event_minutes"] = (
        (pd.Series(timestamps, index=frame.index) - meal_event_time).dt.total_seconds() / 60
    ).to_numpy()

    for column in WEARABLE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        current, age = _limited_forward_fill(
            values, timestamps, max_feature_forward_fill_minutes
        )
        feature_columns[f"{column}_current"] = current.to_numpy()
        feature_columns[f"{column}_age_minutes"] = age.to_numpy()
        feature_columns[f"{column}_missing_at_timestamp"] = values.isna().to_numpy(dtype="int8")
        indexed = pd.Series(values.to_numpy(), index=timestamps)
        for window_minutes in (15, 60):
            rolling = indexed.rolling(f"{window_minutes}min", closed="both", min_periods=1)
            feature_columns[f"{column}_mean_{window_minutes}m"] = rolling.mean().to_numpy()
            feature_columns[f"{column}_max_{window_minutes}m"] = rolling.max().to_numpy()
            feature_columns[f"{column}_observation_count_{window_minutes}m"] = rolling.count().to_numpy()

    for column in STATIC_EHR_COLUMNS:
        if column in frame:
            feature_columns[column] = frame[column].to_numpy()
            feature_columns[f"{column}_missing"] = frame[column].isna().to_numpy(dtype="int8")

    target, target_summary = create_future_target(
        frame,
        horizon_minutes=horizon_minutes,
        threshold_mg_dl=threshold_mg_dl,
        max_cgm_gap_minutes=max_label_cgm_gap_minutes,
    )
    feature_columns["glucose_over_180_next_2h"] = target.array
    return pd.DataFrame(feature_columns), target_summary


def build_features(
    normalized_time_series: pd.DataFrame,
    horizon_minutes: int = TARGET_HORIZON_MINUTES,
    threshold_mg_dl: float = TARGET_THRESHOLD_MG_DL,
    max_feature_forward_fill_minutes: int = MAX_FEATURE_FORWARD_FILL_MINUTES,
    max_label_cgm_gap_minutes: int = MAX_LABEL_CGM_GAP_MINUTES,
) -> pd.DataFrame:
    """Build past-only features and a separately computed future target per patient."""
    required = {"patient_id", "timestamp", "libre_gl_mg_dl"}
    missing = required.difference(normalized_time_series.columns)
    if missing:
        raise ValueError(f"Normalized time series is missing required columns: {sorted(missing)}")
    if normalized_time_series["timestamp"].isna().any():
        raise ValueError("Normalized time series contains invalid timestamps")
    if max_feature_forward_fill_minutes < 0:
        raise ValueError("max_feature_forward_fill_minutes cannot be negative")

    parts: list[pd.DataFrame] = []
    for _, patient_frame in normalized_time_series.groupby("patient_id", sort=True):
        if patient_frame["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps must be resolved before feature generation")
        features, _ = _features_for_patient(
            patient_frame,
            horizon_minutes,
            threshold_mg_dl,
            max_feature_forward_fill_minutes,
            max_label_cgm_gap_minutes,
        )
        parts.append(features)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(
        ["patient_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def process_dataset(
    data_dir: Path,
    output_dir: Path = Path("data/processed"),
    bio_path: Path | None = None,
    horizon_minutes: int = TARGET_HORIZON_MINUTES,
    threshold_mg_dl: float = TARGET_THRESHOLD_MG_DL,
    max_feature_forward_fill_minutes: int = MAX_FEATURE_FORWARD_FILL_MINUTES,
    max_label_cgm_gap_minutes: int = MAX_LABEL_CGM_GAP_MINUTES,
) -> dict[str, Any]:
    """Process recursively discovered participant files and save local outputs."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: '{data_dir}'")
    participant_paths = sorted(
        path
        for path in data_dir.rglob("*.csv")
        if PARTICIPANT_FILE_PATTERN.fullmatch(path.name)
    )
    if not participant_paths:
        raise FileNotFoundError(f"No CGMacros-NNN.csv participant files found under '{data_dir}'")

    patient_ids = [infer_patient_id(path) for path in participant_paths]
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Multiple participant CSV files resolve to the same patient ID")
    static_path = bio_path or data_dir / "bio.csv"
    static_ehr = load_static_ehr(static_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "cgmacros_normalized_timeseries.csv"
    features_path = output_dir / "cgmacros_features.csv"
    summary_path = output_dir / "cgmacros_target_eligibility_summary.json"
    normalized_path.unlink(missing_ok=True)
    features_path.unlink(missing_ok=True)

    totals: dict[str, int] = {
        "rows": 0,
        "eligible_rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "incomplete_horizon_rows": 0,
        "incomplete_libre_coverage_rows": 0,
    }
    for participant_path, patient_id in zip(participant_paths, patient_ids, strict=True):
        raw = pd.read_csv(participant_path)
        normalized = normalize_participant_frame(raw, patient_id)
        normalized = join_static_ehr(normalized, static_ehr)
        features, patient_summary = _features_for_patient(
            normalized,
            horizon_minutes,
            threshold_mg_dl,
            max_feature_forward_fill_minutes,
            max_label_cgm_gap_minutes,
        )
        normalized.to_csv(
            normalized_path,
            index=False,
            mode="a",
            header=not normalized_path.exists(),
            date_format="%Y-%m-%d %H:%M:%S",
        )
        features.to_csv(
            features_path,
            index=False,
            mode="a",
            header=not features_path.exists(),
            date_format="%Y-%m-%d %H:%M:%S",
        )
        for key in totals:
            totals[key] += patient_summary[key]

    eligibility_summary: dict[str, Any] = {
        "participant_count": len(participant_paths),
        "time_series_rows": totals["rows"],
        "target_threshold_mg_dl": threshold_mg_dl,
        "target_horizon_minutes": horizon_minutes,
        "max_cgm_gap_for_label_eligibility_minutes": max_label_cgm_gap_minutes,
        "max_feature_forward_fill_age_minutes": max_feature_forward_fill_minutes,
        "eligible_rows": totals["eligible_rows"],
        "positive_rows": totals["positive_rows"],
        "negative_rows": totals["negative_rows"],
        "ineligible_rows": totals["rows"] - totals["eligible_rows"],
        "incomplete_horizon_rows": totals["incomplete_horizon_rows"],
        "incomplete_libre_coverage_rows": totals["incomplete_libre_coverage_rows"],
        "label_rule": "Any observed Libre value > threshold in (t, t+horizon].",
        "eligibility_rule": (
            "The participant time series must reach the horizon end and observed Libre "
            "timestamps must cover the full window with no gap above the configured "
            "maximum. Dexcom readings do not affect eligibility. No glucose imputation "
            "is used for labels."
        ),
        "feature_sensor": "Libre GL only; Dexcom is preserved but not used for glucose features or labels.",
        "sensor_fusion_rule": (
            "Preserve Libre and Dexcom separately; fused glucose uses Libre when present, "
            "otherwise Dexcom. Values are never averaged or forward-filled in the "
            "normalized series."
        ),
        "meal_missingness_note": (
            "No meal event recorded at a timestamp is marked explicitly; the source "
            "cannot distinguish no meal from an unrecorded meal."
        ),
    }
    summary_path.write_text(
        json.dumps(eligibility_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return eligibility_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"C:\cgmacros\data\CGMacros"),
        help="root directory containing extracted participant CSV files",
    )
    parser.add_argument(
        "--bio-file",
        type=Path,
        default=None,
        help="bio.csv path (default: <data-dir>/bio.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="local directory for normalized data, features, and summary",
    )
    arguments = parser.parse_args()
    try:
        summary = process_dataset(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            bio_path=arguments.bio_file,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())