from __future__ import annotations

import sys
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

APP_TITLE = "GlucoTwin India"
SYNTHETIC_BANNER = "Synthetic research prototype — not clinically validated and not for medical decision-making."
DEMO_DIR = Path(__file__).resolve().parent / "synthetic_demo"
DEFAULT_DEMO_CONFIG = {"seed": 42, "n_participants": 4, "days": 2}


@st.cache_data(show_spinner=False)
def load_demo_bundle(seed: int = 42, n_participants: int = 4, days: int = 2) -> dict[str, Any]:
    """Build the deterministic synthetic demo bundle used by the dashboard."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_smoke_test(
        output_dir=DEMO_DIR,
        seed=seed,
        feature_set="full",
        n_participants=n_participants,
        days=days,
    )
    oof_path = DEMO_DIR / "synthetic_oof_predictions.csv"
    oof = pd.read_csv(oof_path)
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
                    "Precision": _table_value(
                        metric.get("anchor_precision"), "no predicted positives"
                    ),
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
        annotation_position="top left",
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


if __name__ == "__main__":
    main()
