from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.app import _calibration_bin_metadata, _table_value, load_demo_bundle
from src.run_lopo import generate_synthetic_smoke_fixture



@pytest.fixture(autouse=True)
def keep_synthetic_oof_csv_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    protected_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "synthetic_demo"
        / "synthetic_oof_predictions.csv"
    ).resolve()
    captured: dict[str, pd.DataFrame] = {}
    original_to_csv = pd.DataFrame.to_csv
    original_read_csv = pd.read_csv

    def is_protected_path(value) -> bool:
        return isinstance(value, (str, Path)) and Path(value).resolve() == protected_path

    def to_csv_in_memory(frame, path_or_buf=None, *args, **kwargs):
        if is_protected_path(path_or_buf):
            captured["oof"] = frame.copy()
            return None
        return original_to_csv(frame, path_or_buf, *args, **kwargs)

    def read_csv_in_memory(filepath_or_buffer, *args, **kwargs):
        if is_protected_path(filepath_or_buffer):
            return captured["oof"].copy()
        return original_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv_in_memory)
    monkeypatch.setattr(pd, "read_csv", read_csv_in_memory)


def _prediction_expander(app: AppTest):
    return next(
        element
        for element in app.expander
        if element.label == "Inspect synthetic predictions (optional)"
    )


def _oof_table(app: AppTest) -> pd.DataFrame:
    prediction_expander = _prediction_expander(app)
    return next(
        element.value
        for element in prediction_expander.dataframe
        if "Participant" in element.value.columns
    )


def _prediction_selectbox(app: AppTest, label: str):
    prediction_expander = _prediction_expander(app)
    return next(element for element in prediction_expander.selectbox if element.label == label)


def test_project_streamlit_theme_pins_high_contrast_light_mode() -> None:
    config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    theme_section = config_path.read_text(encoding="utf-8").split("[theme]", 1)[1]

    assert 'base = "light"' in theme_section
    assert 'backgroundColor = "#F5F7F2"' in theme_section
    assert 'textColor = "#173A36"' in theme_section


def test_dashboard_renders_sections_warning_calibration_and_valid_bounds() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)

    assert not app.exception
    assert [element.value for element in app.subheader] == [
        "Data & Model Overview",
        "Model Diagnostics",
        "Synthetic Glucose Timeline",
    ]
    prediction_expander = _prediction_expander(app)
    assert prediction_expander.proto.HasField("expanded")
    assert prediction_expander.proto.expanded is False
    advanced_expander = next(
        element for element in app.expander if element.label == "Advanced Technical Details"
    )
    assert advanced_expander.proto.expanded is False
    assert any(
        "inspect model outputs against generated outcomes" in element.value
        for element in prediction_expander.caption
    )
    assert any(
        "Synthetic research prototype" in element.value
        and "not for medical decision-making" in element.value
        for element in app.markdown
    )
    assert "**Synthetic research prototype — not clinically validated and not for medical decision-making.**" in [
        element.value for element in app.markdown
    ]

    main_markdown = "\n".join(element.value for element in app.markdown)
    assert "Synthetic data status" not in main_markdown
    assert "Combined synthetic model inputs" not in main_markdown
    assert "Static predictors:" not in main_markdown
    assert "Dynamic predictors:" not in main_markdown
    assert "age_years_missing" not in main_markdown
    assert "glucose_lag_1m_mg_dl" not in main_markdown
    assert "learning_rate" not in main_markdown

    summary = {
        "model_input_streams": next(
            json.loads(element.value)
            for element in advanced_expander.json
            if "dynamic_predictor_count" in element.value
        )
    }
    assert next(metric.value for metric in app.metric if metric.label == "Data Source") == "Synthetic"
    assert next(metric.value for metric in app.metric if metric.label == "Dynamic Features") == str(
        summary["model_input_streams"]["dynamic_predictor_count"]
    )
    assert len(app.get("plotly_chart")) == 5
    comparison_tables = [
        element for element in app.dataframe if "Forecast" in element.value.columns
    ]
    anchor_table = next(
        element for element in comparison_tables if "Precision" in element.value.columns
    )
    event_table = next(
        element for element in comparison_tables if "False alerts/hr" in element.value.columns
    )
    assert json.loads(anchor_table.proto.columns)["_index"]["hidden"] is True
    assert json.loads(event_table.proto.columns)["_index"]["hidden"] is True
    assert anchor_table.value.iloc[0]["Forecast"] == "Training-fold majority"
    model_event = event_table.value.loc[
        event_table.value["Forecast"].eq("LightGBM at 0.5")
    ].iloc[0]
    assert any(
        f"At the 0.5 cutoff the model caught {model_event['Matched events']} of "
        f"{model_event['Events']} events and raised "
        f"{model_event['False alert episodes']} false-alert episodes." in element.value
        for element in app.markdown
    )
    assert any(
        f"Only {model_event['Events']} events across 4 synthetic participants: too few for reliable rates."
        in element.value
        for element in app.markdown
    )

    figures = [json.loads(element.proto.spec) for element in app.get("plotly_chart")]
    roc_figure = figures[0]
    assert "AUROC:" in roc_figure["layout"]["title"]["text"]
    assert roc_figure["layout"]["shapes"][0]["line"]["dash"] == "dash"
    pr_figure = figures[1]
    assert "AUPRC:" in pr_figure["layout"]["title"]["text"]
    assert "prevalence:" in pr_figure["layout"]["title"]["text"]
    assert any(
        shape["type"] == "line" and shape["line"]["dash"] == "dash"
        for shape in pr_figure["layout"]["shapes"]
    )
    reliability_figure = figures[2]
    assert "20+ anchors" in str(reliability_figure["data"][0]["customdata"])
    assert "Anchors per bin" in reliability_figure["data"][0]["hovertemplate"]
    assert any(
        element.value == "Few anchors per bin; this does not establish calibration."
        for element in app.caption
    )
    timeline = next(
        figure
        for figure in figures
        if figure["layout"]["title"]["text"].startswith("Synthetic glucose history")
    )
    assert any(
        annotation.get("text") == "180 mg/dL threshold"
        for annotation in timeline["layout"].get("annotations", [])
    )
    assert any(
        element.label == "Show model risk (out-of-fold)" and element.value is False
        for element in app.checkbox
    )
    participants = next(
        element for element in app.number_input if element.label == "Participants"
    )
    assert participants.min == 4
def test_undefined_table_metrics_show_a_reason() -> None:
    assert _table_value(None, "no monitored hours") == "n/a (no monitored hours)"
    assert _table_value(np.nan, "no positive events") == "n/a (no positive events)"


def test_reliability_bins_below_twenty_anchors_are_marked_low_sample() -> None:
    bin_counts, sample_labels = _calibration_bin_metadata(pd.Series(np.linspace(0, 1, 19)))

    assert (bin_counts < 20).all()
    assert set(sample_labels) == {"Low sample (<20 anchors)"}


def test_timeline_optional_oof_risk_uses_secondary_probability_axis() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)

    checkbox = next(
        element
        for element in app.checkbox
        if element.label == "Show model risk (out-of-fold)"
    )
    checkbox.set_value(True).run(timeout=60)

    timeline = next(
        json.loads(element.proto.spec)
        for element in app.get("plotly_chart")
        if json.loads(element.proto.spec)["layout"]["title"]["text"].startswith(
            "Synthetic glucose history"
        )
    )
    risk_trace = next(
        trace
        for trace in timeline["data"]
        if trace.get("name") == "Model risk (OOF probability)"
    )
    assert risk_trace["yaxis"] == "y2"
    assert timeline["layout"]["yaxis2"]["range"] == [0, 1]


def test_run_form_updates_evaluation_and_oof_filters_predictions() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)

    seed = next(element for element in app.number_input if element.label == "Seed")
    seed.set_value(43)
    app.run(timeout=60)
    assert next(
        element.value for element in app.metric if element.label == "Evaluation Folds"
    ) == "4"

    seed = next(element for element in app.number_input if element.label == "Seed")
    participants = next(
        element for element in app.number_input if element.label == "Participants"
    )
    days = next(element for element in app.number_input if element.label == "Days")
    seed.set_value(43)
    participants.set_value(5)
    days.set_value(1)
    submit = next(
        element
        for element in app.button
        if element.label == "Run synthetic evaluation"
    )
    submit.click().run(timeout=60)

    assert not app.exception
    expected_config = {"seed": 43, "n_participants": 5, "days": 1}
    assert app.session_state["demo_config"] == expected_config
    assert next(
        element.value for element in app.metric if element.label == "Data Source"
    ) == "Synthetic"
    assert next(
        element.value for element in app.metric if element.label == "Evaluation Folds"
    ) == "5"
    assert next(element.value for element in app.number_input if element.label == "Days") == 1

    patient_filter = _prediction_selectbox(app, "Participant filter")
    patient_filter.set_value("SYN-0002").run(timeout=60)
    rows_to_inspect = _prediction_selectbox(app, "Rows to inspect")
    assert list(rows_to_inspect.options) == [
        "First 10 rows",
        "First 100 rows",
        "All matching rows",
    ]
    assert rows_to_inspect.value == 10
    oof_table = _oof_table(app)
    assert len(oof_table) == 10
    assert oof_table["Participant"].unique().tolist() == ["SYN-0002"]
    np.testing.assert_array_equal(
        (oof_table["Model probability"] >= 0.5).astype(int).to_numpy(),
        oof_table["Prediction (0.5)"].astype(int).to_numpy(),
    )

    rows_to_inspect = _prediction_selectbox(app, "Rows to inspect")
    rows_to_inspect.set_value(100).run(timeout=60)
    hundred_patient_rows = _oof_table(app)
    assert len(hundred_patient_rows) == 100
    assert hundred_patient_rows["Participant"].unique().tolist() == ["SYN-0002"]

    fold_filter = _prediction_selectbox(app, "Held-out fold filter")
    fold_filter.set_value("SYN-0002").run(timeout=60)
    expected_fixture = generate_synthetic_smoke_fixture(**expected_config)
    expected_patient_rows = expected_fixture.loc[
        expected_fixture["patient_id"].astype(str).eq("SYN-0002")
    ]
    elapsed_seconds = (
        expected_patient_rows["timestamp"] - expected_patient_rows["timestamp"].min()
    ).dt.total_seconds()
    expected_row_count = int(
        (
            expected_patient_rows["target"].notna()
            & elapsed_seconds.mod(600).eq(0)
        ).sum()
    )
    rows_to_inspect = _prediction_selectbox(app, "Rows to inspect")
    rows_to_inspect.set_value("All matching rows").run(timeout=60)
    all_patient_rows = _oof_table(app)
    assert len(all_patient_rows) == expected_row_count
    assert all_patient_rows["Participant"].unique().tolist() == ["SYN-0002"]
    assert all_patient_rows["Held-out fold"].astype(str).unique().tolist() == [
        "SYN-0002"
    ]

    patient_filter = _prediction_selectbox(app, "Participant filter")
    patient_filter.set_value("All").run(timeout=60)
    fold_filter = _prediction_selectbox(app, "Held-out fold filter")
    fold_filter.set_value("SYN-0003").run(timeout=60)
    oof_table = _oof_table(app)
    assert oof_table["Held-out fold"].astype(str).unique().tolist() == ["SYN-0003"]


def test_dashboard_script_imports_without_repository_on_pythonpath(tmp_path: Path) -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    import_script = f"""
import importlib.util
import sys
from pathlib import Path

app_path = Path({str(app_path)!r})
repository_root = app_path.parent.parent
sys.path = [
    entry for entry in sys.path
    if Path(entry or ".").resolve() != repository_root
]
spec = importlib.util.spec_from_file_location("dashboard_app", app_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""
    subprocess.run(
        [sys.executable, "-c", import_script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


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
