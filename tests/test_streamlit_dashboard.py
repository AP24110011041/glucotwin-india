from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.app import (
    _anchor_precision_display,
    _calibration_bin_metadata,
    _render_patient_view,
    _table_value,
    load_demo_bundle,
)
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


def test_demo_bundle_keeps_oof_artifact_out_of_protected_demo_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.app as dashboard

    output_directories: list[Path] = []

    def fake_run_smoke_test(output_dir, **kwargs):
        output_path = Path(output_dir)
        output_directories.append(output_path.resolve())
        pd.DataFrame(
            {
                "patient_id": ["SYN-0001"],
                "timestamp": [pd.Timestamp("2025-01-01")],
                "y_true": [0],
                "y_prob": [0.2],
                "y_pred": [0],
                "fold_id": ["SYN-0001"],
            }
        ).to_csv(output_path / "synthetic_oof_predictions.csv", index=False)
        return {"synthetic_data": True}

    monkeypatch.setattr(dashboard, "run_smoke_test", fake_run_smoke_test)
    dashboard.load_demo_bundle.clear()

    bundle = dashboard.load_demo_bundle(seed=42, n_participants=4, days=1)

    assert len(output_directories) == 1
    assert output_directories[0] != dashboard.DEMO_DIR.resolve()
    assert not output_directories[0].exists()
    assert bundle["oof"]["patient_id"].tolist() == ["SYN-0001"]


def test_dashboard_renders_sections_warning_calibration_and_valid_bounds() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)

    assert not app.exception
    assert [element.value for element in app.subheader] == [
        "Data & Model Overview",
        "Model Diagnostics",
        "Synthetic Glucose Timeline",
        "New Synthetic Scenario",
        "Synthetic patient view (conceptual)",
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
    assert any(
        element.value == "Synthetic patient. Concept demonstration only. Not for clinical use."
        for element in app.caption
    )
    assert "Hypothetical" not in "\n".join(
        [
            *(element.value for element in app.subheader),
            *(element.value for element in app.markdown),
            *(element.value for element in app.caption),
            *(element.label for element in app.selectbox),
        ]
    )

    main_markdown = "\n".join(element.value for element in app.markdown)
    assert "Synthetic data status" not in main_markdown
    assert "Combined synthetic model inputs" not in main_markdown
    assert any(
        element.label == "Run synthetic scenario prediction"
        for element in app.button
    )
    assert any(element.label == "Current glucose (mg/dL)" for element in app.number_input)
    assert all(
        next(element for element in app.checkbox if element.label == label).value is False
        for label in (
            "Heart rate measured",
            "METs measured",
            "Calories measured",
            "Intensity measured",
            "Steps measured",
        )
    )
    assert any(
        "Recent synthetic meal history" in element.value for element in app.markdown
    )
    assert any(
        "latest meal type within" in element.value
        for element in app.caption
    )
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
    assert len(app.get("plotly_chart")) == 6
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
    assert any(
        "Retrospective replay of out-of-fold predictions on synthetic data. Not a live prediction."
        in element.value
        for element in app.caption
    )
    participants = next(
        element for element in app.number_input if element.label == "Participants"
    )
    assert participants.min == 4


def test_new_scenario_form_reports_invalid_timestamp() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)

    next(
        element
        for element in app.text_input
        if element.label == "Scenario timestamp (YYYY-MM-DD HH:MM)"
    ).set_value("not-a-timestamp")
    next(
        element
        for element in app.button
        if element.label == "Run synthetic scenario prediction"
    ).click().run(timeout=60)

    assert not app.exception
    assert any("Scenario input is invalid" in element.value for element in app.error)


def test_new_scenario_form_displays_prediction_output() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)

    next(
        element
        for element in app.button
        if element.label == "Run synthetic scenario prediction"
    ).click().run(timeout=60)

    assert not app.exception
    assert any(
        element.label == "New-scenario model probability" for element in app.metric
    )
    assert any(
        "Prediction horizon: 120 minutes" in element.value for element in app.caption
    )
    assert any(
        "Inputs: glucose 140.0 mg/dL | heart rate missing" in element.value
        for element in app.caption
    )
    assert any(
        "no calibrated uncertainty interval is available" in element.value
        for element in app.warning
    )


def test_patient_view_selectors_chart_and_probability_match_oof() -> None:
    bundle = load_demo_bundle(seed=42, n_participants=4, days=2)
    oof = bundle["oof"].copy()
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)
    participant_selector = next(
        element for element in app.selectbox if element.label == "Synthetic participant"
    )
    default_patient = participant_selector.value
    time_selector = next(
        element
        for element in app.selectbox
        if element.label == "Predicted time (out-of-fold anchor)"
    )
    default_time = pd.Timestamp(time_selector.value)
    patient_glucose = bundle["fixture"].loc[
        bundle["fixture"]["patient_id"].astype(str).eq(default_patient)
        & pd.to_numeric(bundle["fixture"]["libre_gl_mg_dl"], errors="coerce").notna(),
        "timestamp",
    ]
    assert (default_time - pd.Timestamp(patient_glucose.min())).total_seconds() >= 360 * 60

    selected_patient = sorted(oof["patient_id"].astype(str).unique())[-1]
    selected_row = (
        oof.loc[oof["patient_id"].astype(str).eq(selected_patient)]
        .sort_values("timestamp")
        .iloc[-1]
    )
    selected_time = pd.Timestamp(selected_row["timestamp"]).strftime("%Y-%m-%d %H:%M")
    participant_selector.set_value(selected_patient).run(timeout=60)
    time_selector = next(
        element
        for element in app.selectbox
        if element.label == "Predicted time (out-of-fold anchor)"
    )
    assert selected_time in time_selector.options
    time_selector.set_value(selected_time).run(timeout=60)

    assert not app.exception
    assert next(
        metric.value for metric in app.metric if metric.label == "Out-of-fold probability"
    ) == f"{float(selected_row['y_prob']):.1%}"
    alert_metric = next(
        metric
        for metric in app.metric
        if metric.label == "Alert at the fixed 50% cutoff"
    )
    assert alert_metric.value in {"yes", "no"}
    patient_chart = json.loads(app.get("plotly_chart")[-1].proto.spec)
    traces = {trace["name"]: trace for trace in patient_chart["data"]}
    assert set(traces) >= {
        "Observed",
        "Actual outcome (retrospective)",
    }
    assert traces["Observed"]["line"]["color"] != traces[
        "Actual outcome (retrospective)"
    ]["line"]["color"]
    annotations = patient_chart["layout"].get("annotations", [])
    assert any(item.get("text") == "180 mg/dL threshold" for item in annotations)
    assert any(item.get("text") == "Prediction time" for item in annotations)
    assert any(
        item.get("text") == "Actual outcome (retrospective)"
        for item in annotations
    )
    x_range = patient_chart["layout"]["xaxis"]["range"]
    range_start, range_end = map(pd.Timestamp, x_range)
    selected_timestamp = pd.Timestamp(selected_time)
    assert range_start == selected_timestamp - pd.Timedelta(hours=6)
    assert range_end == selected_timestamp + pd.Timedelta(hours=2)
    assert range_end - range_start == pd.Timedelta(hours=8)
    assert patient_chart["layout"]["legend"]["orientation"] == "h"
    assert patient_chart["layout"]["legend"]["y"] < 0
    quality_table = next(
        table
        for table in app.dataframe
        if "Observed data-quality measure" in table.value.columns
    ).value
    latest_age = quality_table.loc[
        quality_table["Observed data-quality measure"].eq(
            "Age of latest glucose reading"
        ),
        "Value",
    ].iloc[0]
    assert latest_age == "1 minute"

    context_table = next(
        table for table in app.dataframe if "Observed signal" in table.value.columns
    ).value
    assert "Glucose data quality" not in context_table["Observed signal"].tolist()
    activity_summary = context_table.loc[
        context_table["Observed signal"].eq("Activity"), "Observed value"
    ].iloc[0]
    steps_display = next(
        value
        for value in activity_summary.split("; ")
        if value.endswith(" steps (observed total)")
    )
    steps_count = steps_display.removesuffix(" steps (observed total)").replace(",", "")
    assert steps_count.isdigit()
    assert any(
        element.value
        == "Synthetic signals are generated independently and are not physiologically coupled (for example, steps and METs do not constrain each other)."
        for element in app.caption
    )


def test_patient_view_first_anchor_reports_no_glucose_history() -> None:
    bundle = load_demo_bundle(seed=42, n_participants=4, days=2)
    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)
    participant_selector = next(
        element for element in app.selectbox if element.label == "Synthetic participant"
    )
    selected_patient = participant_selector.value
    first_anchor = (
        bundle["oof"].loc[
            bundle["oof"]["patient_id"].astype(str).eq(selected_patient)
        ]
        .sort_values("timestamp")
        .iloc[0]
    )
    time_selector = next(
        element
        for element in app.selectbox
        if element.label == "Predicted time (out-of-fold anchor)"
    )
    time_selector.set_value(
        pd.Timestamp(first_anchor["timestamp"]).strftime("%Y-%m-%d %H:%M")
    ).run(timeout=60)

    assert not app.exception
    quality_table = next(
        table
        for table in app.dataframe
        if "Observed data-quality measure" in table.value.columns
    ).value
    quality_values = quality_table["Value"].astype(str).tolist()
    assert "No glucose history before this time" in quality_values
    assert "n/a" in quality_values
    assert not any("0.0%" in value for value in quality_values)
    context_table = next(
        table for table in app.dataframe if "Observed signal" in table.value.columns
    ).value
    assert "Glucose data quality" not in context_table["Observed signal"].tolist()


def test_patient_view_missing_glucose_quality_for_participant_with_gaps() -> None:
    def render_patient_view_with_gap() -> None:
        import numpy as np
        import pandas as pd

        from app.app import _render_patient_view, load_demo_bundle

        bundle = load_demo_bundle(seed=42, n_participants=4, days=2)
        fixture = bundle["fixture"].copy()
        oof = bundle["oof"].copy()
        selected_patient = sorted(oof["patient_id"].astype(str).unique())[0]
        patient_oof = oof.loc[
            oof["patient_id"].astype(str).eq(selected_patient)
        ].copy()
        patient_oof["timestamp"] = pd.to_datetime(patient_oof["timestamp"])
        patient_oof = patient_oof.sort_values("timestamp")
        patient_fixture = fixture.loc[
            fixture["patient_id"].astype(str).eq(selected_patient)
        ].copy()
        patient_fixture["timestamp"] = pd.to_datetime(patient_fixture["timestamp"])
        first_reading = patient_fixture.loc[
            pd.to_numeric(patient_fixture["libre_gl_mg_dl"], errors="coerce").notna(),
            "timestamp",
        ].min()
        selected_time = next(
            timestamp
            for timestamp in patient_oof["timestamp"]
            if (timestamp - first_reading).total_seconds() >= 360 * 60
        )
        gap_start = selected_time - pd.Timedelta(minutes=20)
        gap_end = selected_time - pd.Timedelta(minutes=11)
        fixture_timestamps = pd.to_datetime(fixture["timestamp"])
        gap = (
            fixture["patient_id"].astype(str).eq(selected_patient)
            & fixture_timestamps.between(gap_start, gap_end, inclusive="both")
        )
        gap_count = int(gap.sum())
        assert gap_count == 10
        fixture.loc[gap, "libre_gl_mg_dl"] = np.nan
        _render_patient_view(fixture, oof)

    app = AppTest.from_function(render_patient_view_with_gap).run(timeout=60)
    assert not app.exception
    quality_table = next(
        table
        for table in app.dataframe
        if "Observed data-quality measure" in table.value.columns
    ).value
    missing_display = quality_table.loc[
        quality_table["Observed data-quality measure"].eq("Missing glucose readings"),
        "Value",
    ].iloc[0]
    missing_share_text, count_text = missing_display.split("%", maxsplit=1)
    count_text = count_text.strip().strip("()")
    observed_expected = count_text.split(" ", maxsplit=1)[0]
    observed_count, expected_count = map(int, observed_expected.split("/"))
    assert float(missing_share_text) > 0
    assert observed_count + 10 == expected_count == 360

    context_table = next(
        table for table in app.dataframe if "Observed signal" in table.value.columns
    ).value
    assert "Glucose data quality" not in context_table["Observed signal"].tolist()
    assert not any(
        "0.0%" in str(value) for value in context_table["Observed value"].tolist()
    )


def test_no_alert_precision_uses_existing_report_counts() -> None:
    assert _anchor_precision_display(
        {"true_positives": 0, "false_positives": 0, "anchor_precision": 0.0}
    ) == "n/a (no alerts)"
    assert _anchor_precision_display(
        {"true_positives": 1, "false_positives": 0, "anchor_precision": 1.0}
    ) == 1.0


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
