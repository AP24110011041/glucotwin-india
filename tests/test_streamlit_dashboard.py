from __future__ import annotations

import pandas as pd

from app.app import load_demo_bundle


def test_synthetic_demo_bundle_has_expected_contract() -> None:
    bundle = load_demo_bundle(seed=7, n_participants=4, days=2)
    oof = bundle["oof"]
    fixture = bundle["fixture"]

    assert bundle["synthetic_label"] == "SYNTHETIC"
    assert fixture["patient_id"].nunique() == 4
    assert {"patient_id", "timestamp", "libre_gl_mg_dl"}.issubset(fixture.columns)
    assert {"patient_id", "timestamp", "y_true", "y_prob", "y_pred", "fold_id"}.issubset(oof.columns)
    assert oof["oof_key"].is_unique if "oof_key" in oof.columns else True
    assert oof["y_true"].notna().all()
    assert oof["y_prob"].notna().all()


def test_synthetic_demo_bundle_is_deterministic() -> None:
    first = load_demo_bundle(seed=11, n_participants=4, days=2)
    second = load_demo_bundle(seed=11, n_participants=4, days=2)

    assert first["summary"]["seed"] == second["summary"]["seed"] == 11
    assert first["oof"]["oof_key"].tolist() == second["oof"]["oof_key"].tolist()
    assert first["oof"]["y_true"].tolist() == second["oof"]["y_true"].tolist()
    assert first["oof"]["y_pred"].tolist() == second["oof"]["y_pred"].tolist()
    pd.testing.assert_series_equal(
        first["oof"]["y_prob"].reset_index(drop=True),
        second["oof"]["y_prob"].reset_index(drop=True),
        check_names=False,
    )
