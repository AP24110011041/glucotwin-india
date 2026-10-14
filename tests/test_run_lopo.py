from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import build_features
from src.run_lopo import (
    generate_synthetic_smoke_fixture,
    run_smoke_test,
    _build_oof_predictions,
)


def test_synthetic_fixture_has_patient_wise_targets_and_synthetic_tags() -> None:
    fixture = generate_synthetic_smoke_fixture(seed=11, n_participants=4, days=2)

    assert fixture["patient_id"].nunique() == 4
    assert set(fixture["synthetic_label"]) == {"SYNTHETIC"}
    assert fixture["target"].notna().any()
    assert fixture.groupby("patient_id")["target"].nunique().min() >= 2


def test_oof_predictions_are_unique_and_cover_eligible_anchors() -> None:
    fixture = generate_synthetic_smoke_fixture(seed=17, n_participants=4, days=2)
    oof, _, _ = _build_oof_predictions(fixture, feature_set="full", seed=17)

    assert not oof.empty
    assert oof["oof_key"].is_unique
    assert oof["patient_id"].astype(str).nunique() == 4
    assert set(oof.columns).issuperset({"patient_id", "timestamp", "y_true", "y_prob", "y_pred", "fold_id"})


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
    assert summary["lightgbm_parameters"]["random_state"] == 9
    assert summary["lightgbm_parameters"]["n_estimators"] == 75
    assert summary["eligible_anchor_count"] > 0
    assert summary["report"]["oof_metrics"]["auroc"] is not None
    assert summary["report"]["current_libre_threshold_reference"]["alert_count"] >= 0

    saved = pd.read_csv(predictions_path)
    assert {"patient_id", "timestamp", "y_true", "y_prob", "y_pred", "fold_id"}.issubset(saved.columns)
