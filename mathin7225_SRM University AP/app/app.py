from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_lopo import generate_synthetic_smoke_fixture, run_smoke_test
from src.inference import build_scenario_timeseries, predict_new_scenario
from src.feature_engineering import TARGET_HORIZON_MINUTES

APP_TITLE = "GlucoTwin India"
SYNTHETIC_BANNER = "Synthetic research prototype — not clinically validated and not for medical decision-making."
DEMO_DIR = Path(__file__).resolve().parent / "synthetic_demo"
DEFAULT_DEMO_CONFIG = {"seed": 42, "n_participants": 4, "days": 2}


@st.cache_data(show_spinner=False)
def load_demo_bundle(seed: int = 42, n_participants: int = 4, days: int = 2) -> dict[str, Any]:
    """Build the deterministic synthetic demo bundle used by the dashboard."""
    with tempfile.TemporaryDirectory(prefix="glucotwin-dashboard-") as output_dir:
        summary = run_smoke_test(
            output_dir=output_dir,
            seed=seed,
            feature_set="full",
            n_participants=n_participants,
            days=days,
        )
        oof = pd.read_csv(Path(output_dir) / "synthetic_oof_predictions.csv")
    fixture = generate_synthetic_smoke_fixture(seed=seed, n_participants=n_participants, days=days)
    return {
        "summary": summary,
        "oof": oof,
        "fixture": fixture,
        "seed": seed,
        "n_participants": n_participants,
        "days": days,
        "synthetic_label": "SYNTHETIC",
    }


def _safe_metric_value(value: Any) -> Any:
    if value is None:
        return "Not available"
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    return value


def _table_value(value: Any, reason: str) -> Any:
    if value is None or pd.isna(value):
        return f"n/a ({reason})"
    return value


def _anchor_precision_display(metric: dict[str, Any]) -> Any:
    true_positives = metric.get("true_positives")
    false_positives = metric.get("false_positives")
    if (
        true_positives is not None
        and false_positives is not None
        and true_positives + false_positives == 0
    ):
        return "n/a (no alerts)"
    return _table_value(metric.get("anchor_precision"), "no predicted positives")


def _calibration_bin_metadata(y_prob: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    quantile_edges = np.percentile(y_prob, np.linspace(0, 100, 11))
    bin_ids = np.searchsorted(quantile_edges[1:-1], y_prob, side="left")
    all_bin_counts = np.bincount(bin_ids, minlength=10)
    bin_counts = all_bin_counts[all_bin_counts > 0]
    sample_labels = np.where(bin_counts < 20, "Low sample (<20 anchors)", "20+ anchors")
    return bin_counts, sample_labels


def _render_overview(summary: dict[str, Any]) -> None:
    st.subheader("Data & Model Overview")

    metrics = summary.get("report", {}).get("oof_metrics", {})
    metadata = summary.get("lightgbm_parameters", {})
    input_streams = summary.get("model_input_streams", {})
    fold_count = summary.get("fold_count", "Not available")
    dynamic_predictor_count = input_streams.get(
        "dynamic_predictor_count", "Not available"
    )

    data_column, folds_column, features_column = st.columns(3)
    data_column.metric("Data Source", "Synthetic")
    folds_column.metric("Evaluation Folds", fold_count)
    features_column.metric("Dynamic Features", dynamic_predictor_count)

    st.markdown("**Data Used by the Model**")
    profile_column, glucose_column, lifestyle_column = st.columns(3)
    profile_column.markdown("**Participant Profile**")
    profile_column.write("Age, gender, diagnosis, diabetes duration, BMI and HbA1c")
    glucose_column.markdown("**Glucose History**")
    glucose_column.write("Current and previous glucose readings")
    lifestyle_column.markdown("**Lifestyle Signals**")
    lifestyle_column.write("Meals, heart rate, physical activity and steps")

    st.caption(
        "All inputs are synthetically generated. No real patient records or live "
        "glucose-monitoring hardware are connected. Synthetic evaluation results "
        "do not establish clinical accuracy."
    )

    with st.expander("Advanced Technical Details", expanded=False):
        st.write(f"Feature set: {summary.get('feature_set', 'Not available')}")
        st.write(f"Evaluation fold count: {fold_count}")
        st.write(f"Dynamic predictor count: {dynamic_predictor_count}")
        if input_streams:
            st.write("Model input metadata:")
            st.json(input_streams)
        st.write("LightGBM parameters:")
        st.json(metadata)

        oof_metrics = {
            "AUROC": metrics.get("auroc"),
            "AUPRC": metrics.get("auprc"),
            "Brier score": metrics.get("brier_score"),
        }
        st.write("Synthetic out-of-fold metrics:")
        for name, value in oof_metrics.items():
            st.write(f"- {name}: {_safe_metric_value(value)}")
        st.caption(
            "Pooled synthetic OOF point estimates; AUROC, AUPRC, and Brier have no "
            "participant-bootstrap intervals in this demo."
        )

    report = summary.get("report", {})
    anchor_comparisons = []
    for label, group, method in (
        (
            "Training-fold majority",
            report.get("baseline_summary", {}).get("overall_metrics", []),
            "training_majority",
        ),
        (
            "LightGBM at 0.5",
            report.get("lightgbm_lopo_summary", {}).get("overall_metrics", []),
            "lightgbm_fixed_0_5",
        ),
    ):
        metric = next(
            (
                row
                for row in group
                if row.get("method") == method
                and row.get("stratum") == "all_eligible_anchors"
            ),
            None,
        )
        if metric is not None:
            anchor_comparisons.append(
                {
                    "Forecast": label,
                    "Eligible anchors": metric["eligible_anchors"],
                    "Precision": _anchor_precision_display(metric),
                    "Recall": _table_value(
                        metric.get("anchor_recall"), "no positive events"
                    ),
                    "F1": _table_value(
                        metric.get("anchor_f1"), "precision or recall unavailable"
                    ),
                }
            )
    if anchor_comparisons:
        st.write("Synthetic anchor-level comparison")
        st.dataframe(
            pd.DataFrame(anchor_comparisons), width="stretch", hide_index=True
        )

    event_comparisons = []
    lightgbm_event_metric = None
    for label, group in (
        (
            "Training-fold majority",
            report.get("baseline_summary", {}).get("event_overall_metrics", []),
        ),
        (
            "LightGBM at 0.5",
            report.get("lightgbm_lopo_summary", {}).get("event_overall_metrics", []),
        ),
    ):
        if group:
            method = "training_majority" if label == "Training-fold majority" else "lightgbm_fixed_0_5"
            metric = next(
                (row for row in group if row.get("method") == method), group[0]
            )
            if label == "LightGBM at 0.5":
                lightgbm_event_metric = metric
            monitored_minutes = metric.get("measured_observation_minutes")
            false_alert_rate = (
                metric.get("false_alert_episodes") / (monitored_minutes / 60)
                if monitored_minutes is not None and monitored_minutes > 0
                else None
            )
            event_comparisons.append(
                {
                    "Forecast": label,
                    "Events": _table_value(
                        metric.get("actual_events"), "event count unavailable"
                    ),
                    "Alert episodes": _table_value(
                        metric.get("alert_episodes"), "alert count unavailable"
                    ),
                    "Matched events": _table_value(
                        metric.get("matched_events"), "match count unavailable"
                    ),
                    "False alert episodes": _table_value(
                        metric.get("false_alert_episodes"), "alert count unavailable"
                    ),
                    "False alerts/hr": _table_value(
                        false_alert_rate, "no monitored hours"
                    ),
                }
            )
    if event_comparisons:
        if lightgbm_event_metric is not None:
            event_count = lightgbm_event_metric.get("actual_events")
            participant_count = summary.get("fold_count")
            if event_count is not None and participant_count is not None:
                st.write(
                    "At the 0.5 cutoff the model caught "
                    f"{lightgbm_event_metric['matched_events']} of {event_count} events "
                    "and raised "
                    f"{lightgbm_event_metric['false_alert_episodes']} false-alert episodes."
                )
                st.write(
                    f"Only {event_count} events across {participant_count} synthetic "
                    "participants: too few for reliable rates."
                )
        st.write("Synthetic event-level comparison")
        st.dataframe(
            pd.DataFrame(event_comparisons), width="stretch", hide_index=True
        )


def _render_oof_explorer(oof: pd.DataFrame) -> None:
    with st.expander("Inspect synthetic predictions (optional)", expanded=False):
        st.caption(
            "Each row is one eligible synthetic prediction anchor. The held-out "
            "participant was excluded from that fold's model training. Use this view "
            "to inspect model outputs against generated outcomes by participant/fold; "
            "these are not patient records."
        )
        if oof.empty:
            st.info("No OOF rows are available for the current synthetic demo.")
            return

        patient_options = ["All"] + sorted(oof["patient_id"].astype(str).unique().tolist())
        fold_options = ["All"] + sorted(oof["fold_id"].astype(str).unique().tolist())
        patient_column, fold_column, row_count_column = st.columns(3)
        selected_patient = patient_column.selectbox(
            "Participant filter", patient_options
        )
        selected_fold = fold_column.selectbox("Held-out fold filter", fold_options)
        row_limit = row_count_column.selectbox(
            "Rows to inspect",
            options=(10, 100, "All matching rows"),
            format_func=lambda value: (
                value if isinstance(value, str) else f"First {value} rows"
            ),
        )

        filtered = oof.copy()
        if selected_patient != "All":
            filtered = filtered[filtered["patient_id"].astype(str) == selected_patient]
        if selected_fold != "All":
            filtered = filtered[filtered["fold_id"].astype(str) == selected_fold]

        if filtered.empty:
            st.info("No rows match the current selection.")
            return

        display = filtered[
            ["patient_id", "timestamp", "y_true", "y_prob", "y_pred", "fold_id"]
        ].copy()
        display = display.rename(columns={
            "patient_id": "Participant",
            "timestamp": "Anchor time",
            "y_true": "Observed event",
            "y_prob": "Model probability",
            "y_pred": "Prediction (0.5)",
            "fold_id": "Held-out fold",
        })
        if row_limit != "All matching rows":
            display = display.head(int(row_limit))
        st.caption(
            "Columns: participant ID; prediction-anchor time; observed future event "
            "(any synthetic Libre >180 mg/dL within 120 minutes); model probability; "
            "model prediction at the fixed 0.5 cutoff; held-out LOPO fold. "
            f"Showing {len(display):,} of {len(filtered):,} matching synthetic anchors."
        )
        st.dataframe(display, width="stretch", height=460)


def _render_model_diagnostics(
    oof: pd.DataFrame, metrics: dict[str, Any]
) -> None:
    st.subheader("Model Diagnostics")
    if oof.empty:
        st.info("No synthetic OOF rows are available.")
        return

    y_true = oof["y_true"].astype(int)
    y_prob = pd.to_numeric(oof["y_prob"], errors="coerce")

    if y_true.nunique() < 2:
        st.warning("Not enough class diversity in this synthetic sample for ROC or PR curves.")
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_fig = px.line(
            x=fpr,
            y=tpr,
            title=f"Synthetic ROC curve (AUROC: {_safe_metric_value(metrics.get('auroc'))})",
            labels={"x": "False positive rate", "y": "True positive rate"},
        )
        roc_fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"dash": "dash", "color": "gray"},
        )
        st.plotly_chart(roc_fig, width="stretch")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        prevalence = metrics.get("positive_prevalence")
        prevalence_label = (
            f"{prevalence:.1%}" if prevalence is not None else "Not available"
        )
        pr_fig = px.line(
            x=recall,
            y=precision,
            title=(
                f"Synthetic precision-recall curve (AUPRC: "
                f"{_safe_metric_value(metrics.get('auprc'))}; "
                f"prevalence: {prevalence_label})"
            ),
            labels={"x": "Recall", "y": "Precision"},
        )
        if prevalence is not None:
            pr_fig.add_hline(
                y=prevalence,
                line_dash="dash",
                line_color="gray",
                annotation_text="Prevalence",
                annotation_position="top left",
            )
        st.plotly_chart(pr_fig, width="stretch")

        observed_frequency, mean_probability = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )
        bin_counts, sample_labels = _calibration_bin_metadata(y_prob)
        low_sample = bin_counts < 20
        calibration_fig = go.Figure(
            go.Scatter(
                x=mean_probability,
                y=observed_frequency,
                mode="lines+markers",
                name="Observed by bin",
                marker={
                    "color": ["#B5483A" if low else "#267C68" for low in low_sample],
                    "symbol": ["x" if low else "circle" for low in low_sample],
                    "size": 10,
                },
                customdata=np.column_stack((bin_counts, sample_labels)),
                hovertemplate=(
                    "Mean predicted probability: %{x:.3f}<br>"
                    "Observed event frequency: %{y:.3f}<br>"
                    "Anchors per bin: %{customdata[0]}<br>"
                    "%{customdata[1]}<extra></extra>"
                ),
            )
        )
        calibration_fig.update_layout(
            title="Synthetic reliability diagram",
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed event frequency",
        )
        calibration_fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"dash": "dash", "color": "gray"},
        )
        st.plotly_chart(calibration_fig, width="stretch")
        st.caption("Few anchors per bin; this does not establish calibration.")

    thresholded = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, thresholded, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.write("Confusion matrix at threshold 0.5")
    st.dataframe(cm_df, width="stretch")

    if y_prob.notna().any():
        dist = px.histogram(
            y_prob.dropna().to_frame(name="Probability"),
            x="Probability",
            nbins=20,
            title="Synthetic OOF probability distribution",
        )
        st.plotly_chart(dist, width="stretch")


def _render_synthetic_timeline(fixture: pd.DataFrame, oof: pd.DataFrame) -> None:
    st.subheader("Synthetic Glucose Timeline")
    patients = sorted(fixture["patient_id"].astype(str).unique().tolist())
    if not patients:
        st.info("No synthetic patient rows are available.")
        return

    selected_patient = st.selectbox("Participant", patients)
    patient_frame = fixture[fixture["patient_id"].astype(str) == selected_patient].sort_values("timestamp")
    show_model_risk = st.checkbox("Show model risk (out-of-fold)")

    chart = px.line(
        patient_frame,
        x="timestamp",
        y="libre_gl_mg_dl",
        title=f"Synthetic glucose history for {selected_patient}",
        labels={"libre_gl_mg_dl": "Libre glucose (mg/dL)", "timestamp": "Time"},
    )
    chart.add_hline(
        y=180,
        line_dash="dash",
        annotation_text="180 mg/dL threshold",
        annotation_position="top right",
    )
    if show_model_risk:
        risk_frame = oof.loc[
            oof["patient_id"].astype(str).eq(selected_patient)
        ].sort_values("timestamp")
        chart.add_trace(
            go.Scatter(
                x=risk_frame["timestamp"],
                y=risk_frame["y_prob"],
                name="Model risk (OOF probability)",
                mode="lines",
                yaxis="y2",
            )
        )
        chart.update_layout(
            yaxis2={
                "title": "Model risk (OOF probability)",
                "overlaying": "y",
                "side": "right",
                "range": [0, 1],
            }
        )
    st.plotly_chart(chart, width="stretch")
    st.caption("This is a synthetic trend only; it is not patient data and not a clinical signal.")


def _median_sampling_interval_minutes(patient_fixture: pd.DataFrame) -> float | None:
    timestamps = (
        pd.to_datetime(patient_fixture["timestamp"])
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    intervals = timestamps.diff().dt.total_seconds().div(60)
    intervals = intervals.loc[intervals > 0]
    return float(intervals.median()) if not intervals.empty else None


def _format_minutes(minutes: float) -> str:
    unit = "minute" if minutes == 1 else "minutes"
    return f"{minutes:g} {unit}"


def _glucose_window_quality(
    patient_fixture: pd.DataFrame,
    selected_time: pd.Timestamp,
    window_hours: int,
    sampling_interval_minutes: float | None,
) -> dict[str, Any]:
    window_start = selected_time - pd.Timedelta(hours=window_hours)
    window = patient_fixture.loc[
        patient_fixture["timestamp"].ge(window_start)
        & patient_fixture["timestamp"].lt(selected_time)
    ].sort_values("timestamp")
    glucose = pd.to_numeric(window["libre_gl_mg_dl"], errors="coerce")
    observed_count = int(glucose.notna().sum())
    expected_count = (
        int(round(window_hours * 60 / sampling_interval_minutes))
        if sampling_interval_minutes is not None and sampling_interval_minutes > 0
        else None
    )
    if observed_count == 0:
        missing_share = None
        missing_display = "No glucose history before this time"
        latest_reading_age = "n/a"
    else:
        missing_share = (
            1 - observed_count / expected_count
            if expected_count is not None and expected_count > 0
            else None
        )
        missing_display = (
            f"{missing_share:.1%} ({observed_count}/{expected_count} observed/expected)"
            if missing_share is not None
            else "n/a (sampling interval unavailable)"
        )
        latest_timestamp = window.loc[glucose.notna(), "timestamp"].max()
        latest_age_minutes = (selected_time - latest_timestamp).total_seconds() / 60
        latest_reading_age = _format_minutes(latest_age_minutes)
    return {
        "expected_count": expected_count,
        "observed_count": observed_count,
        "missing_share": missing_share,
        "missing_display": missing_display,
        "latest_reading_age": latest_reading_age,
    }


def _render_patient_view(fixture: pd.DataFrame, oof: pd.DataFrame) -> None:
    st.subheader("Synthetic patient view (conceptual)")
    st.caption("Synthetic patient. Concept demonstration only. Not for clinical use.")

    patient_ids = sorted(
        set(fixture["patient_id"].astype(str))
        & set(oof["patient_id"].astype(str))
    )
    if not patient_ids:
        st.info("No synthetic participants with eligible out-of-fold anchors are available.")
        return

    selected_patient = st.selectbox(
        "Synthetic participant", patient_ids, key="patient_view_participant"
    )
    patient_fixture = fixture.loc[
        fixture["patient_id"].astype(str).eq(selected_patient)
    ].copy()
    patient_fixture["timestamp"] = pd.to_datetime(patient_fixture["timestamp"])
    patient_oof = oof.loc[
        oof["patient_id"].astype(str).eq(selected_patient)
    ].copy()
    patient_oof["timestamp"] = pd.to_datetime(patient_oof["timestamp"])
    patient_oof = patient_oof.sort_values("timestamp")
    if patient_oof.empty:
        st.info("No eligible out-of-fold anchors are available for this participant.")
        return

    st.markdown("#### Observed synthetic profile")
    profile = patient_fixture.iloc[0]
    profile_table = pd.DataFrame(
        [
            {"Synthetic profile field": "Age (synthetic)", "Value": f"{profile['age_years']} years"},
            {"Synthetic profile field": "Gender (synthetic)", "Value": str(profile["gender"])},
            {"Synthetic profile field": "Diagnosis (synthetic)", "Value": str(profile["diagnosis"])},
            {
                "Synthetic profile field": "Diabetes duration (synthetic)",
                "Value": f"{profile['diabetes_duration_years']} years",
            },
            {"Synthetic profile field": "BMI (synthetic)", "Value": f"{profile['bmi']:.1f} kg/m²"},
            {"Synthetic profile field": "HbA1c (synthetic)", "Value": f"{profile['a1c_pdl_lab']:.1f}%"},
        ]
    )
    st.dataframe(profile_table, width="stretch", hide_index=True)

    first_glucose_time = patient_fixture.loc[
        pd.to_numeric(patient_fixture["libre_gl_mg_dl"], errors="coerce").notna(),
        "timestamp",
    ].min()
    available_history = (
        patient_oof["timestamp"] - first_glucose_time
    ).dt.total_seconds().div(60)
    default_anchor_index = next(
        (index for index, minutes in enumerate(available_history) if minutes >= 360),
        0,
    )
    time_options = patient_oof["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    selected_time_label = st.selectbox(
        "Predicted time (out-of-fold anchor)",
        time_options,
        index=default_anchor_index,
        key="patient_view_time",
    )
    selected_time = pd.Timestamp(selected_time_label)
    history_minutes = max(
        0.0, (selected_time - first_glucose_time).total_seconds() / 60
    )
    if history_minutes < 360:
        st.info(f"Only {_format_minutes(history_minutes)} of history before this time")
    prediction = patient_oof.loc[
        patient_oof["timestamp"].eq(selected_time)
    ].iloc[0]

    st.markdown("#### Observed glucose and retrospective outcome")
    observed_start = selected_time - pd.Timedelta(hours=6)
    outcome_end = selected_time + pd.Timedelta(hours=2)
    chart_window = patient_fixture.loc[
        patient_fixture["timestamp"].between(
            observed_start, outcome_end, inclusive="both"
        )
    ].sort_values("timestamp")
    observed = chart_window.loc[chart_window["timestamp"].le(selected_time)]
    actual_outcome = chart_window.loc[
        chart_window["timestamp"].gt(selected_time)
        & chart_window["timestamp"].le(outcome_end)
    ]
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=observed["timestamp"],
            y=observed["libre_gl_mg_dl"],
            mode="lines",
            name="Observed",
            line={"color": "#267C68", "width": 2},
            connectgaps=False,
        )
    )
    chart.add_trace(
        go.Scatter(
            x=actual_outcome["timestamp"],
            y=actual_outcome["libre_gl_mg_dl"],
            mode="lines",
            name="Actual outcome (retrospective)",
            line={"color": "#B5483A", "width": 2},
            connectgaps=False,
        )
    )
    chart.add_vrect(
        x0=selected_time,
        x1=outcome_end,
        fillcolor="rgba(245, 191, 175, 0.30)",
        line_width=0,
        annotation_text="Actual outcome (retrospective)",
        annotation_position="top right",
    )
    chart.add_vline(
        x=selected_time,
        line_dash="dash",
        line_color="#173A36",
        annotation_text="Prediction time",
        annotation_position="bottom",
    )
    chart.add_hline(
        y=180,
        line_dash="dash",
        line_color="gray",
        annotation_text="180 mg/dL threshold",
        annotation_position="top right",
    )
    chart.update_layout(
        title="Observed glucose and retrospective outcome",
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        xaxis={"range": [observed_start, outcome_end]},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.28,
            "yanchor": "top",
        },
        margin={"b": 100},
    )
    st.plotly_chart(chart, width="stretch")

    st.markdown("#### Predicted: model output")
    st.metric("Out-of-fold probability", f"{float(prediction['y_prob']):.1%}")
    st.metric(
        "Alert at the fixed 50% cutoff",
        "yes" if int(prediction["y_pred"]) == 1 else "no",
    )
    st.caption(
        "Retrospective replay of out-of-fold predictions on synthetic data. "
        "Not a live prediction."
    )

    st.markdown("#### Observed context (previous 2 hours)")
    context_start = selected_time - pd.Timedelta(hours=2)
    context = patient_fixture.loc[
        patient_fixture["timestamp"].ge(context_start)
        & patient_fixture["timestamp"].lt(selected_time)
    ]
    meal_rows = context.loc[context["meal_type"].notna()]
    if meal_rows.empty:
        meal_summary = "not recorded"
    else:
        meal_summary = "; ".join(
            f"{row.meal_type} at {row.timestamp:%H:%M}"
            + (
                f" ({row.meal_carbs_g:g} g carbs)"
                if pd.notna(row.meal_carbs_g)
                else ""
            )
            for row in meal_rows.itertuples()
        )

    heart_rate = pd.to_numeric(context["heart_rate_bpm"], errors="coerce").dropna()
    heart_rate_summary = (
        f"{heart_rate.mean():.1f} bpm (mean)" if not heart_rate.empty else "not recorded"
    )
    activity_mets = pd.to_numeric(context["activity_mets"], errors="coerce").dropna()
    activity_steps = pd.to_numeric(context["steps"], errors="coerce").dropna()
    activity_summary = []
    if not activity_mets.empty:
        activity_summary.append(f"{activity_mets.mean():.2f} METs (mean)")
    else:
        activity_summary.append("METs: not recorded")
    if not activity_steps.empty:
        activity_summary.append(f"{activity_steps.sum():,.0f} steps (observed total)")
    else:
        activity_summary.append("steps: not recorded")
    sampling_interval_minutes = _median_sampling_interval_minutes(patient_fixture)
    context_table = pd.DataFrame(
        [
            {"Observed signal": "Meals", "Observed value": meal_summary},
            {"Observed signal": "Heart rate", "Observed value": heart_rate_summary},
            {"Observed signal": "Activity", "Observed value": "; ".join(activity_summary)},
        ]
    )
    st.dataframe(context_table, width="stretch", hide_index=True)
    st.caption(
        "Synthetic signals are generated independently and are not physiologically "
        "coupled (for example, steps and METs do not constrain each other)."
    )

    st.markdown("#### Observed data quality (6-hour window)")
    quality = _glucose_window_quality(
        patient_fixture, selected_time, 6, sampling_interval_minutes
    )
    quality_table = pd.DataFrame(
        [
            {
                "Observed data-quality measure": "Missing glucose readings",
                "Value": quality["missing_display"],
            },
            {
                "Observed data-quality measure": "Age of latest glucose reading",
                "Value": quality["latest_reading_age"],
            },
        ]
    )
    st.dataframe(quality_table, width="stretch", hide_index=True)


def _render_new_scenario(fixture: pd.DataFrame) -> None:
    st.subheader("New Synthetic Scenario")
    st.caption(
        "This is a newly fitted synthetic-model scenario, separate from historical "
        "out-of-fold predictions. It is exploratory only, not clinically validated, "
        "and not for medical decision-making."
    )

    recent_synthetic_meals = fixture.loc[
        fixture["meal_type"].notna(),
        ["patient_id", "timestamp", "meal_type", "meal_carbs_g"],
    ].sort_values("timestamp").tail(5).copy()
    if not recent_synthetic_meals.empty:
        recent_synthetic_meals.insert(0, "Source", "Synthetic reference (not used)")
        st.markdown("**Recent synthetic meal history**")
        st.dataframe(recent_synthetic_meals, width="stretch", hide_index=True)

    st.markdown("**Manual meals for this scenario**")
    st.caption(
        "Add rows to record meals; edit cells or remove rows directly. These manual "
        "entries are used only for this scenario and are not the synthetic meals above."
    )
    st.caption(
        "Carbohydrate totals, time since a meal, and the latest meal type within "
        "120 minutes enter model features. Missing or unrecognized types become "
        "unknown; no recent meal becomes none; conflicting same-time manual types "
        "become mixed. Existing breakfast/lunch/dinner indicators remain timestamp-local."
    )
    empty_meals = pd.DataFrame(
        {
            "meal_type": pd.Series(dtype="str"),
            "carbs_g": pd.Series(dtype="float"),
            "minutes_since_eating": pd.Series(dtype="int"),
        }
    )
    empty_history = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="str"),
            "glucose_mg_dl": pd.Series(dtype="float"),
            "heart_rate_bpm": pd.Series(dtype="float"),
            "activity_mets": pd.Series(dtype="float"),
        }
    )
    with st.form("new_scenario_prediction"):
        timestamp = st.text_input(
            "Scenario timestamp (YYYY-MM-DD HH:MM)", value="2025-01-03 12:00"
        )
        current_glucose, heart_rate, activity_mets = st.columns(3)
        glucose_value = current_glucose.number_input(
            "Current glucose (mg/dL)", min_value=20.0, max_value=600.0, value=140.0
        )
        heart_rate_available = heart_rate.checkbox(
            "Heart rate measured", value=False
        )
        heart_rate_value = heart_rate.number_input(
            "Heart rate (bpm)",
            min_value=20.0,
            max_value=250.0,
            value=72.0,
            disabled=not heart_rate_available,
        )
        activity_mets_available = activity_mets.checkbox("METs measured", value=False)
        mets_value = activity_mets.number_input(
            "Activity (METs)",
            min_value=0.0,
            max_value=25.0,
            value=1.0,
            disabled=not activity_mets_available,
        )
        activity_calories, activity_intensity, steps = st.columns(3)
        calories_available = activity_calories.checkbox(
            "Calories measured", value=False
        )
        calories_value = activity_calories.number_input(
            "Activity calories",
            min_value=0.0,
            max_value=5000.0,
            value=0.0,
            disabled=not calories_available,
        )
        intensity_available = activity_intensity.checkbox(
            "Intensity measured", value=False
        )
        intensity_value = activity_intensity.number_input(
            "Activity intensity",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            disabled=not intensity_available,
        )
        steps_available = steps.checkbox("Steps measured", value=False)
        steps_value = steps.number_input(
            "Steps",
            min_value=0.0,
            max_value=100000.0,
            value=0.0,
            disabled=not steps_available,
        )
        st.markdown("**Recent measurements (optional)**")
        st.caption(
            "Add timestamped glucose or wearable readings from the prior 120 minutes. "
            "Unentered values remain missing and are imputed by the trained pipeline."
        )
        history_table = st.data_editor(
            empty_history,
            num_rows="dynamic",
            column_config={
                "timestamp": st.column_config.TextColumn(
                    "Timestamp (YYYY-MM-DD HH:MM)", required=True
                ),
                "glucose_mg_dl": st.column_config.NumberColumn(
                    "Libre glucose (mg/dL)", min_value=20.0, max_value=600.0
                ),
                "heart_rate_bpm": st.column_config.NumberColumn(
                    "Heart rate (bpm)", min_value=20.0, max_value=250.0
                ),
                "activity_mets": st.column_config.NumberColumn(
                    "Activity (METs)", min_value=0.0, max_value=25.0
                ),
            },
            width="stretch",
            hide_index=True,
            key="scenario_measurement_history",
        )
        profile_columns = st.columns(4)
        age = profile_columns[0].number_input(
            "Age (synthetic profile)", min_value=18.0, max_value=120.0, value=45.0
        )
        diabetes_duration = profile_columns[1].number_input(
            "Diabetes duration (years)", min_value=0.0, max_value=100.0, value=5.0
        )
        bmi = profile_columns[2].number_input(
            "BMI (synthetic profile)", min_value=10.0, max_value=100.0, value=25.0
        )
        a1c = profile_columns[3].number_input(
            "HbA1c (synthetic profile)", min_value=3.0, max_value=25.0, value=7.0
        )
        genders = sorted(fixture["gender"].dropna().astype(str).unique().tolist())
        diagnoses = sorted(fixture["diagnosis"].dropna().astype(str).unique().tolist())
        gender_column, diagnosis_column = st.columns(2)
        gender = gender_column.selectbox("Gender (synthetic model input)", genders)
        diagnosis = diagnosis_column.selectbox(
            "Diagnosis (synthetic model input)", diagnoses
        )
        meal_table = st.data_editor(
            empty_meals,
            num_rows="dynamic",
            column_config={
                "meal_type": st.column_config.TextColumn(
                    "Meal type (optional)",
                    required=False,
                    help="Known values: breakfast, lunch, dinner, snack, other. Blank or unrecognized values become unknown.",
                ),
                "carbs_g": st.column_config.NumberColumn(
                    "Carbohydrate quantity (g)", min_value=0.0, max_value=300.0
                ),
                "minutes_since_eating": st.column_config.NumberColumn(
                    "Time since eating (minutes)",
                    min_value=0,
                    max_value=TARGET_HORIZON_MINUTES,
                ),
            },
            width="stretch",
            hide_index=True,
            key="manual_scenario_meals",
        )
        submitted = st.form_submit_button(
            "Run synthetic scenario prediction", type="primary"
        )

    if submitted:
        try:
            scenario = build_scenario_timeseries(
                patient_id="MANUAL-SCENARIO",
                timestamp=timestamp,
                current_glucose_mg_dl=glucose_value,
                wearable_values={
                    "heart_rate_bpm": heart_rate_value if heart_rate_available else None,
                    "activity_calories": calories_value if calories_available else None,
                    "activity_mets": mets_value if activity_mets_available else None,
                    "activity_intensity": intensity_value if intensity_available else None,
                    "steps": steps_value if steps_available else None,
                },
                history=history_table.dropna(how="all").to_dict("records"),
                meals=meal_table.dropna(how="all").to_dict("records"),
                static_profile={
                    "age_years": age,
                    "gender": gender,
                    "diagnosis": diagnosis,
                    "diabetes_duration_years": diabetes_duration,
                    "bmi": bmi,
                    "a1c_pdl_lab": a1c,
                },
            )
            result = predict_new_scenario(fixture, scenario)
        except (TypeError, ValueError) as error:
            st.session_state.pop("new_scenario_result", None)
            st.error(f"Scenario input is invalid: {error}")
        else:
            st.session_state["new_scenario_result"] = result

    result = st.session_state.get("new_scenario_result")
    if result:
        probability_column, output_column = st.columns(2)
        probability_column.metric(
            "New-scenario model probability", f"{result['probability']:.1%}"
        )
        output_column.metric(
            "Fixed 50% model output",
            "threshold event predicted" if result["prediction"] else "not predicted",
        )
        st.progress(result["probability"], text="Synthetic model output; not calibrated for clinical use")
        st.caption(
            f"Prediction horizon: {result['prediction_horizon_minutes']} minutes | "
            f"Libre target: >{result['target_threshold_mg_dl']:.0f} mg/dL | "
            f"Input time: {result['timestamp']} | Features: {result['feature_count']} | "
            f"Synthetic training participants: {result['training_participant_count']}"
        )
        input_summary = result["input_summary"]
        heart_rate_display = (
            f"{input_summary['heart_rate_bpm']:.0f} bpm"
            if input_summary["heart_rate_bpm"] is not None
            else "missing"
        )
        activity_display = (
            f"{input_summary['activity_mets']:.1f} METs"
            if input_summary["activity_mets"] is not None
            else "missing"
        )
        calories_display = (
            f"{input_summary['activity_calories']:.1f} kcal"
            if input_summary["activity_calories"] is not None
            else "missing"
        )
        intensity_display = (
            f"{input_summary['activity_intensity']:.1f}"
            if input_summary["activity_intensity"] is not None
            else "missing"
        )
        steps_display = (
            f"{input_summary['steps']:.0f}"
            if input_summary["steps"] is not None
            else "missing"
        )
        st.caption(
            f"Inputs: glucose {input_summary['current_glucose_mg_dl']:.1f} mg/dL | "
            f"heart rate {heart_rate_display} | activity {activity_display} | "
            f"calories {calories_display} | intensity {intensity_display} | "
            f"steps {steps_display} | "
            f"manual meal events in 120 minutes: "
            f"{input_summary['meal_events_last_120_minutes']}"
        )
        if result["missing_feature_count"]:
            st.warning(
                f"{result['missing_feature_count']} of {result['feature_count']} model "
                "inputs are missing at this scenario timestamp and are handled by "
                "training-fitted imputation. Review the measurement history before "
                "interpreting this exploratory output."
            )
        st.warning(
            "Synthetic-only exploratory output. This probability is not clinically "
            "validated or calibrated; no calibrated uncertainty interval is available. "
            "It is not a medical recommendation."
        )


def main() -> None:
    st.set_page_config(page_title="GlucoTwin India Demo", page_icon="🧪")
    st.title("GlucoTwin India")
    st.markdown(f"**{SYNTHETIC_BANNER}**")

    with st.sidebar:
        st.header("Synthetic demo controls")
        active_config = st.session_state.get("demo_config", DEFAULT_DEMO_CONFIG)
        with st.form("synthetic_run_controls"):
            seed = st.number_input(
                "Seed", min_value=0, value=active_config["seed"], step=1
            )
            participants = st.number_input(
                "Participants",
                min_value=4,
                max_value=12,
                value=active_config["n_participants"],
                step=1,
            )
            days = st.number_input(
                "Days", min_value=1, max_value=7, value=active_config["days"], step=1
            )
            run_evaluation = st.form_submit_button(
                "Run synthetic evaluation", type="primary"
            )

    if run_evaluation:
        st.session_state["demo_config"] = {
            "seed": int(seed),
            "n_participants": int(participants),
            "days": int(days),
        }
    elif "demo_config" not in st.session_state:
        st.session_state["demo_config"] = DEFAULT_DEMO_CONFIG.copy()

    with st.spinner("Building the synthetic LOPO results..."):
        bundle = load_demo_bundle(**st.session_state["demo_config"])
    summary = bundle["summary"]
    fixture = bundle["fixture"]
    oof = bundle["oof"]

    _render_overview(summary)
    _render_oof_explorer(oof)
    _render_model_diagnostics(oof, summary.get("report", {}).get("oof_metrics", {}))
    _render_synthetic_timeline(fixture, oof)
    _render_new_scenario(fixture)
    _render_patient_view(fixture, oof)


if __name__ == "__main__":
    main()
