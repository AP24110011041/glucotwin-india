from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_lopo import generate_synthetic_smoke_fixture, run_smoke_test

APP_TITLE = "GlucoTwin India"
SYNTHETIC_BANNER = "SYNTHETIC RESEARCH DEMO — NOT FOR CLINICAL USE."
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


def _render_overview(summary: dict[str, Any], fixture: pd.DataFrame) -> None:
    st.subheader("Overview")
    st.markdown(f"**{SYNTHETIC_BANNER}**")

    metrics = summary.get("report", {}).get("oof_metrics", {})
    metadata = summary.get("lightgbm_parameters", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Seed", summary.get("seed", "n/a"))
    col2.metric("Participants", fixture["patient_id"].nunique())
    col3.metric("Eligible anchors", summary.get("eligible_anchor_count", 0))
    col4.metric("Prevalence", _safe_metric_value(metrics.get("positive_prevalence")))

    st.write("Synthetic data status")
    st.write(f"- synthetic_label: {summary.get('synthetic_label', 'SYNTHETIC')}")
    st.write(f"- feature_set: {summary.get('feature_set', 'full')}")
    st.write(f"- patient folds: {summary.get('fold_count', 0)}")
    st.write(f"- LightGBM parameters: {metadata}")

    input_streams = summary.get("model_input_streams", {})
    if input_streams:
        st.markdown("**Combined synthetic model inputs**")
        st.write(f"- Joined by: {input_streams['join_key']}")
        st.write(
            "- Static predictors: "
            + ", ".join(input_streams["static_predictor_columns"])
        )
        st.write(
            "- Dynamic predictors: "
            f"{input_streams['dynamic_predictor_count']} features, including "
            + ", ".join(input_streams["dynamic_predictor_examples"])
        )
        st.caption(
            "Both streams are generated synthetic inputs. No real patient records "
            "or real-time hardware are connected."
        )

    oof_metrics = {
        "AUROC": metrics.get("auroc"),
        "AUPRC": metrics.get("auprc"),
        "Brier score": metrics.get("brier_score"),
    }
    st.write("Available OOF metrics:")
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
                    "Precision": metric["anchor_precision"],
                    "Recall": metric["anchor_recall"],
                    "F1": metric["anchor_f1"],
                }
            )
    if anchor_comparisons:
        st.write("Synthetic anchor-level comparison")
        st.dataframe(pd.DataFrame(anchor_comparisons), width="stretch")

    event_comparisons = []
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
            metric = group[0]
            event_comparisons.append(
                {
                    "Forecast": label,
                    "Events": metric["actual_events"],
                    "Alert episodes": metric["alert_episodes"],
                    "Matched events": metric["matched_events"],
                    "False alert episodes": metric["false_alert_episodes"],
                    "False alerts per measured day": metric[
                        "false_alert_episodes_per_day"
                    ],
                }
            )
    if event_comparisons:
        st.write("Synthetic event-level comparison")
        st.dataframe(pd.DataFrame(event_comparisons), width="stretch")


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


def _render_model_diagnostics(oof: pd.DataFrame) -> None:
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
            title="Synthetic ROC curve",
            labels={"x": "False positive rate", "y": "True positive rate"},
        )
        st.plotly_chart(roc_fig, width="stretch")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_fig = px.line(
            x=recall,
            y=precision,
            title="Synthetic precision-recall curve",
            labels={"x": "Recall", "y": "Precision"},
        )
        st.plotly_chart(pr_fig, width="stretch")

        observed_frequency, mean_probability = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )
        calibration_fig = px.line(
            x=mean_probability,
            y=observed_frequency,
            markers=True,
            title="Synthetic reliability diagram",
            labels={
                "x": "Mean predicted probability",
                "y": "Observed event frequency",
            },
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


def _render_synthetic_timeline(fixture: pd.DataFrame) -> None:
    st.subheader("Synthetic Glucose Timeline")
    patients = sorted(fixture["patient_id"].astype(str).unique().tolist())
    if not patients:
        st.info("No synthetic patient rows are available.")
        return

    selected_patient = st.selectbox("Participant", patients)
    patient_frame = fixture[fixture["patient_id"].astype(str) == selected_patient].sort_values("timestamp")

    chart = px.line(
        patient_frame,
        x="timestamp",
        y="libre_gl_mg_dl",
        title=f"Synthetic glucose history for {selected_patient}",
        labels={"libre_gl_mg_dl": "Libre glucose (mg/dL)", "timestamp": "Time"},
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

    _render_overview(summary, fixture)
    _render_oof_explorer(oof)
    _render_model_diagnostics(oof)
    _render_synthetic_timeline(fixture)


if __name__ == "__main__":
    main()
