import pandas as pd
from pandas.testing import assert_frame_equal

from src.synth_data import (
    SYNTHETIC_LABEL,
    generate_synthetic_data,
    generate_synthetic_patient_records,
)


def test_generation_is_reproducible() -> None:
    first_patients, first_series = generate_synthetic_data(n_participants=3, days=1, seed=17)
    second_patients, second_series = generate_synthetic_data(n_participants=3, days=1, seed=17)

    assert_frame_equal(first_patients, second_patients)
    assert_frame_equal(first_series, second_series)


def test_synthetic_patient_records_include_baseline_history_and_labs() -> None:
    first = generate_synthetic_patient_records(n_participants=3, seed=17)
    second = generate_synthetic_patient_records(n_participants=3, seed=17)

    assert_frame_equal(first, second)
    assert {
        "participant_id",
        "condition",
        "age_years",
        "sex",
        "diabetes_duration_years",
        "hba1c_percent",
        "bmi_kg_m2",
        "synthetic_label",
    }.issubset(first.columns)
    assert set(first["synthetic_label"]) == {SYNTHETIC_LABEL}


def test_expected_schema_multiple_participants_and_synthetic_labels() -> None:
    patients, series = generate_synthetic_data(n_participants=3, days=1)

    assert {
        "participant_id",
        "condition",
        "age_years",
        "sex",
        "diabetes_duration_years",
        "hba1c_percent",
        "bmi_kg_m2",
        "synthetic_label",
    }.issubset(patients.columns)
    assert {
        "participant_id",
        "timestamp",
        "glucose_mg_dl",
        "meal_event",
        "meal_carbs_g",
        "heart_rate_bpm",
        "steps_5min",
        "glucose_over_180_next_2h",
        "synthetic_label",
    }.issubset(series.columns)
    assert patients["participant_id"].nunique() == 3
    assert series["participant_id"].nunique() == 3
    assert set(patients["synthetic_label"]) == {SYNTHETIC_LABEL}
    assert set(series["synthetic_label"]) == {SYNTHETIC_LABEL}


def test_timestamps_are_valid_and_five_minute_spaced() -> None:
    _, series = generate_synthetic_data(n_participants=2, days=1)

    timestamps = pd.to_datetime(series["timestamp"])
    assert timestamps.notna().all()
    for _, participant_series in series.groupby("participant_id"):
        intervals = pd.to_datetime(participant_series["timestamp"]).diff().dropna()
        assert (intervals == pd.Timedelta(minutes=5)).all()


def test_participant_integrity_and_incomplete_future_labels() -> None:
    patients, series = generate_synthetic_data(n_participants=3, days=2, seed=29)

    assert patients["participant_id"].is_unique
    assert set(series["participant_id"]) == set(patients["participant_id"])
    for _, participant_series in series.groupby("participant_id", sort=False):
        timestamps = pd.to_datetime(participant_series["timestamp"])
        assert timestamps.is_monotonic_increasing
        target = participant_series["glucose_over_180_next_2h"]
        glucose = participant_series["glucose_mg_dl"]
        assert target.iloc[-24:].isna().all()
        assert target.iloc[:-24].notna().all()
        for index in range(len(participant_series) - 24):
            expected = int(glucose.iloc[index + 1 : index + 25].gt(180).any())
            assert int(target.iloc[index]) == expected


def test_invalid_generation_arguments_have_helpful_errors() -> None:
    try:
        generate_synthetic_data(n_participants=0)
    except ValueError as error:
        assert "n_participants must be at least 1" in str(error)
    else:
        raise AssertionError("Expected invalid participant count to be rejected")
