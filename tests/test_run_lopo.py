from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_features
from src.synth_data import generate_synthetic_patient_records
from src.run_lopo import (
    _safe_brier,
    _threshold_predictions,
    generate_synthetic_smoke_fixture,
    main,
    run_smoke_test,
    _build_oof_predictions,
)


def test_threshold_predictions_uses_inclusive_half_cutoff() -> None:
    probabilities = np.array([0.49, 0.50, 0.51], dtype=float)

    predictions = _threshold_predictions(probabilities)

    np.testing.assert_array_equal(predictions, np.array([0, 1, 1], dtype="int8"))
    assert predictions.dtype == np.dtype("int8")


def test_synthetic_fixture_has_patient_wise_targets_and_synthetic_tags() -> None:
    fixture = generate_synthetic_smoke_fixture(seed=11, n_participants=4, days=2)

    assert fixture["patient_id"].nunique() == 4
    assert set(fixture["synthetic_label"]) == {"SYNTHETIC"}
    assert fixture["target"].notna().any()
    assert fixture.groupby("patient_id")["target"].nunique().min() >= 2


def test_synthetic_fixture_targets_match_production_120_minute_labels() -> None:
    fixture = generate_synthetic_smoke_fixture(seed=23, n_participants=4, days=2)
    expected = build_features(fixture)["glucose_over_180_next_2h"]
    eligible = fixture["target"].notna() & expected.notna()

    assert fixture["target"].notna().equals(expected.notna())
    np.testing.assert_array_equal(
        fixture.loc[eligible, "target"].astype("int8").to_numpy(),
        expected.loc[eligible].astype("int8").to_numpy(),
    )
    for _, patient_rows in fixture.groupby("patient_id", sort=False):
        np.testing.assert_array_equal(
            patient_rows["record_index"].to_numpy(),
            np.arange(len(patient_rows)),
        )


def test_static_and_dynamic_synthetic_streams_reach_lopo_predictors() -> None:
    fixture = generate_synthetic_smoke_fixture(seed=31, n_participants=4, days=2)
    profiles = generate_synthetic_patient_records(
        n_participants=4, seed=31
    ).rename(
        columns={
            "participant_id": "patient_id",
            "condition": "diagnosis",
            "sex": "gender",
            "bmi_kg_m2": "bmi",
            "hba1c_percent": "a1c_pdl_lab",
        }
    ).set_index("patient_id")
    _, summary, evaluation_data = _build_oof_predictions(
        fixture, feature_set="full", seed=31
    )
    static_features = {
        "age_years",
        "gender",
        "diagnosis",
        "diabetes_duration_years",
        "bmi",
        "a1c_pdl_lab",
    }
    dynamic_features = {
        "glucose_current_mg_dl",
        "heart_rate_bpm_current",
        "meal_carbs_g_sum_15m",
    }

    assert static_features.issubset(evaluation_data.predictor_columns)
    assert dynamic_features.issubset(evaluation_data.predictor_columns)
    assert set(summary["static_predictor_columns"]).issuperset(static_features)
    assert set(summary["dynamic_predictor_columns"]).issuperset(dynamic_features)
    assert fixture.groupby("patient_id")["diagnosis"].nunique().eq(1).all()
    for column in static_features:
        assert fixture.groupby("patient_id")[column].nunique().eq(1).all()
    for patient_id, patient_rows in fixture.groupby("patient_id"):
        assert patient_id in profiles.index
        for feature, profile_column in (
            ("age_years", "age_years"),
            ("gender", "gender"),
            ("diagnosis", "diagnosis"),
            ("diabetes_duration_years", "diabetes_duration_years"),
            ("bmi", "bmi"),
            ("a1c_pdl_lab", "a1c_pdl_lab"),
        ):
            assert patient_rows[feature].iloc[0] == profiles.loc[
                patient_id, profile_column
            ]
    assert fixture["meal_type"].notna().any()
    assert fixture.loc[fixture["meal_type"].notna(), "meal_carbs_g"].gt(0).all()


def test_oof_predictions_are_unique_and_cover_eligible_anchors() -> None:
    fixture = generate_synthetic_smoke_fixture(seed=17, n_participants=4, days=2)
    oof, _, _ = _build_oof_predictions(fixture, feature_set="full", seed=17)

    assert not oof.empty
    assert oof["oof_key"].is_unique
    assert oof["patient_id"].astype(str).nunique() == 4
    fold_by_patient = oof.groupby("patient_id")["fold_id"].first().astype(str)
    assert fold_by_patient.to_dict() == {
        patient_id: patient_id for patient_id in fold_by_patient.index.astype(str)
    }
    assert set(oof.columns).issuperset({"patient_id", "timestamp", "y_true", "y_prob", "y_pred", "fold_id"})


def test_brier_score_is_defined_for_single_class_labels() -> None:
    assert _safe_brier(pd.Series([0, 0]), pd.Series([0.1, 0.2])) == pytest.approx(0.025)


def test_smoke_cli_reports_invalid_participant_count_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lopo",
            "--smoke-test",
            "--n-participants",
            "2",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "n_participants must be at least 4" in capsys.readouterr().err


def test_synthetic_fixture_has_usable_1_minute_lag_feature() -> None:
    fixture = generate_synthetic_smoke_fixture(seed=19, n_participants=4, days=1)
    feature_frame = build_features(fixture)

    assert "glucose_lag_1m_mg_dl" in feature_frame.columns
    assert feature_frame["glucose_lag_1m_mg_dl"].notna().any()


def test_repeated_smoke_runs_are_deterministic() -> None:
    fixture = generate_synthetic_smoke_fixture(seed=5, n_participants=4, days=2)
    first_oof, _, _ = _build_oof_predictions(fixture, feature_set="full", seed=5)
    second_oof, _, _ = _build_oof_predictions(fixture, feature_set="full", seed=5)

    assert first_oof["oof_key"].tolist() == second_oof["oof_key"].tolist()
    assert first_oof["y_true"].tolist() == second_oof["y_true"].tolist()
    assert first_oof["y_pred"].tolist() == second_oof["y_pred"].tolist()
    np.testing.assert_allclose(
        first_oof["y_prob"].to_numpy(),
        second_oof["y_prob"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_cli_smoke_mode_writes_csv_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke-output"
    summary = run_smoke_test(output_dir=output_dir, seed=9, feature_set="full", n_participants=4, days=2)

    predictions_path = output_dir / "synthetic_oof_predictions.csv"
    assert predictions_path.exists()
    assert summary["synthetic_data"] is True
    assert summary["seed"] == 9
    assert summary["feature_set"] == "full"
    assert summary["fold_count"] == 4
    assert summary["model_input_streams"]["join_key"] == "patient_id"
    assert "diagnosis" in summary["model_input_streams"]["static_predictor_columns"]
    assert summary["model_input_streams"]["dynamic_predictor_count"] > 0
    assert summary["lightgbm_parameters"]["random_state"] == 9
    assert summary["lightgbm_parameters"]["n_estimators"] == 75
    assert summary["eligible_anchor_count"] > 0
    assert summary["report"]["oof_metrics"]["auroc"] is not None
    assert summary["report"]["baseline_summary"]["overall_metrics"]
    assert summary["report"]["baseline_summary"]["event_overall_metrics"]
    assert summary["report"]["baseline_summary"]["participant_uncertainty"]
    assert summary["report"]["lightgbm_lopo_summary"]["participant_uncertainty"]
    assert summary["report"]["current_libre_threshold_reference"]["alert_count"] >= 0

    saved = pd.read_csv(predictions_path)
    assert {"patient_id", "timestamp", "y_true", "y_prob", "y_pred", "fold_id"}.issubset(saved.columns)
