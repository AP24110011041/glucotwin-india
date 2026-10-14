"""Create reproducible synthetic data for software development, not clinical use."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SYNTHETIC_LABEL = "SYNTHETIC"
SAMPLE_INTERVAL = pd.Timedelta(minutes=5)
TARGET_HORIZON_STEPS = 24


def _make_patient_records(
    participant_ids: list[str], generator: np.random.Generator
) -> pd.DataFrame:
    count = len(participant_ids)
    return pd.DataFrame(
        {
            "participant_id": participant_ids,
            "condition": ["Type 2 diabetes"] * count,
            "age_years": generator.integers(35, 76, size=count),
            "sex": generator.choice(["female", "male", "other"], size=count),
            "diabetes_duration_years": generator.integers(1, 26, size=count),
            "hba1c_percent": np.round(generator.uniform(6.5, 10.5, size=count), 1),
            "bmi_kg_m2": np.round(generator.uniform(20.0, 40.0, size=count), 1),
            "synthetic_label": [SYNTHETIC_LABEL] * count,
        }
    )


def generate_synthetic_patient_records(
    n_participants: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Return deterministic synthetic baseline demographics, history, and labs."""
    if n_participants < 1:
        raise ValueError("n_participants must be at least 1")
    generator = np.random.default_rng(seed)
    participant_ids = [f"SYN-{index:04d}" for index in range(1, n_participants + 1)]
    return _make_patient_records(participant_ids, generator)


def _make_participant_timeseries(
    participant_id: str,
    timestamps: pd.DatetimeIndex,
    generator: np.random.Generator,
) -> pd.DataFrame:
    sample_count = len(timestamps)
    elapsed_hours = np.arange(sample_count) / 12
    baseline = generator.uniform(115, 165)
    circadian = 10 * np.sin(2 * np.pi * (elapsed_hours - 7) / 24)

    meal_carbs = np.zeros(sample_count, dtype=float)
    meal_indices = [
        index
        for hour in (8, 13, 19)
        if (index := int(round((hour + generator.uniform(-0.5, 0.5)) * 12))) < sample_count
    ]
    for index in meal_indices:
        meal_carbs[index] = generator.uniform(25, 90)

    meal_effect = np.zeros(sample_count, dtype=float)
    for index in meal_indices:
        response_length = min(sample_count - index, 25)
        response = np.arange(response_length)
        meal_effect[index : index + response_length] += (
            meal_carbs[index] * 0.45 * (response / 4) * np.exp(-response / 8)
        )

    noise = generator.normal(0, 5, size=sample_count)
    glucose = np.clip(baseline + circadian + meal_effect + noise, 65, 320)
    heart_rate = np.clip(68 + 5 * np.sin(2 * np.pi * elapsed_hours / 24) + generator.normal(0, 4, sample_count), 45, 150)
    steps = generator.poisson(8, size=sample_count)
    steps[(elapsed_hours % 24 < 7) | (elapsed_hours % 24 > 22)] = generator.poisson(
        2, size=int(np.count_nonzero((elapsed_hours % 24 < 7) | (elapsed_hours % 24 > 22)))
    )

    target = np.full(sample_count, np.nan)
    for index in range(sample_count - TARGET_HORIZON_STEPS):
        future_glucose = glucose[index + 1 : index + TARGET_HORIZON_STEPS + 1]
        target[index] = float(np.any(future_glucose > 180))

    return pd.DataFrame(
        {
            "participant_id": participant_id,
            "timestamp": timestamps,
            "glucose_mg_dl": np.round(glucose, 1),
            "meal_event": meal_carbs > 0,
            "meal_carbs_g": np.round(meal_carbs, 1),
            "heart_rate_bpm": np.round(heart_rate, 1),
            "steps_5min": steps,
            "glucose_over_180_next_2h": pd.array(target, dtype="Int8"),
            "synthetic_label": SYNTHETIC_LABEL,
        }
    )


def generate_synthetic_data(
    n_participants: int = 5,
    days: int = 2,
    seed: int = 42,
    start: str = "2026-01-01T00:00:00",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return synthetic EHR records and five-minute time series.

    This simulator uses arbitrary rules for software testing. It is not a model of
    human physiology, has not been clinically validated, and must not inform care.
    """
    if n_participants < 1:
        raise ValueError("n_participants must be at least 1")
    if days < 1:
        raise ValueError("days must be at least 1")
    try:
        start_timestamp = pd.Timestamp(start)
    except (TypeError, ValueError) as error:
        raise ValueError(f"start must be a valid timestamp, received {start!r}") from error

    generator = np.random.default_rng(seed)
    participant_ids = [f"SYN-{index:04d}" for index in range(1, n_participants + 1)]
    patient_records = _make_patient_records(participant_ids, generator)
    samples_per_day = 24 * 60 // 5
    timestamps = pd.date_range(
        start=start_timestamp,
        periods=days * samples_per_day,
        freq=SAMPLE_INTERVAL,
    )
    time_series = pd.concat(
        [
            _make_participant_timeseries(participant_id, timestamps, generator)
            for participant_id in participant_ids
        ],
        ignore_index=True,
    )
    return patient_records, time_series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=int, default=5, help="number of synthetic participants")
    parser.add_argument("--days", type=int, default=2, help="number of synthetic days per participant")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/synthetic"),
        help="directory for generated CSV files",
    )
    arguments = parser.parse_args()

    try:
        patients, time_series = generate_synthetic_data(
            n_participants=arguments.participants,
            days=arguments.days,
            seed=arguments.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    patients.to_csv(arguments.output_dir / "patients.csv", index=False)
    time_series.to_csv(arguments.output_dir / "timeseries.csv", index=False)
    print(f"Wrote synthetic-only data to '{arguments.output_dir}'.")
    print("These simulated records are not clinically validated and are not patient data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
