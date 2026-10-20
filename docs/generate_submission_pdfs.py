from __future__ import annotations

import sys
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent
ARCHITECTURE_PATH = OUTPUT_DIR / "architecture_diagram.pdf"
PRESENTATION_PATH = OUTPUT_DIR / "presentation.pdf"
DISCLAIMER = "SYNTHETIC RESEARCH DEMO \u2014 NOT FOR CLINICAL USE"
COLORS = {
    "ink": "#173A36",
    "green": "#287A66",
    "mint": "#DDEDE4",
    "lime": "#C9DB75",
    "cyan": "#65AEB0",
    "amber": "#E5A557",
    "coral": "#D96D58",
    "paper": "#F5F7F2",
    "white": "#FFFFFF",
    "muted": "#58706A",
    "line": "#D2DDD5",
}


def _new_page(title: str, subtitle: str, page: int, total: int) -> tuple[Any, Any]:
    figure = plt.figure(figsize=(13.333, 7.5), facecolor=COLORS["paper"])
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(Rectangle((0, 0.91), 1, 0.09, color=COLORS["ink"], zorder=0))
    axis.text(0.055, 0.955, title, color=COLORS["white"], fontsize=21, weight="bold", va="center")
    axis.text(0.055, 0.875, subtitle, color=COLORS["muted"], fontsize=10.5, va="center")
    axis.text(0.055, 0.025, DISCLAIMER, color=COLORS["coral"], fontsize=8, weight="bold", va="center")
    axis.text(0.95, 0.025, f"GLUCOTWIN INDIA  |  {page:02d}/{total:02d}", color=COLORS["muted"], fontsize=8, ha="right", va="center")
    return figure, axis


def _box(axis: Any, x: float, y: float, width: float, height: float, title: str, body: str, color: str = "green", title_size: int = 11, body_size: int = 9, radius: float = 0.018) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=COLORS["white"], edgecolor=COLORS[color], linewidth=1.6,
        )
    )
    axis.text(x + 0.018, y + height - 0.027, title, color=COLORS["ink"], fontsize=title_size, weight="bold", va="top")
    axis.text(x + 0.018, y + height - 0.064, body, color=COLORS["muted"], fontsize=body_size, va="top", linespacing=1.35, wrap=True)


def _arrow(axis: Any, start: tuple[float, float], end: tuple[float, float], color: str = "green", style: str = "-|>", width: float = 1.8, connection: str = "arc3") -> None:
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=13, linewidth=width,
            color=COLORS[color], connectionstyle=connection, zorder=1,
        )
    )


def _architecture_diagram() -> None:
    figure, axis = _new_page(
        "GlucoTwin India | Architecture",
        "Synthetic EHR profiles join a generated 1-minute dynamic fixture by participant ID.",
        1, 1,
    )
    axis.text(0.055, 0.82, "LOCAL DATA ENGINEERING PATH (OPTIONAL; NOT USED BY THE DEMO)", color=COLORS["green"], fontsize=9, weight="bold")
    _box(axis, 0.055, 0.62, 0.20, 0.14, "Optional local CSVs", "CGMacros is not loaded by the dashboard.", color="amber")
    _box(axis, 0.34, 0.62, 0.23, 0.14, "Inspect + engineer", "inspect_data.py\nfeature_engineering.py", color="amber")
    _box(axis, 0.67, 0.62, 0.27, 0.14, "Local processed outputs", "Normalized series, features, eligibility summary", color="amber")
    _arrow(axis, (0.255, 0.69), (0.34, 0.69), color="amber")
    _arrow(axis, (0.57, 0.69), (0.67, 0.69), color="amber")

    axis.text(0.055, 0.53, "SYNTHETIC DASHBOARD PATH", color=COLORS["green"], fontsize=9, weight="bold")
    _box(axis, 0.055, 0.31, 0.20, 0.16, "Joined synthetic streams", "Static EHR\n+ 1-min CGM, meals\n+ wearable signals", color="cyan", title_size=9, body_size=7.5)
    _box(axis, 0.34, 0.31, 0.20, 0.16, "Causal features", "Past-only predictors\n120-min Libre target", color="green", title_size=10, body_size=8)
    _box(axis, 0.62, 0.31, 0.20, 0.16, "Patient-wise LOPO", "LightGBM\n10-min anchors\nTrain-fold preprocessing", color="green", title_size=9.5, body_size=7.5)
    _box(axis, 0.88, 0.31, 0.095, 0.16, "Results", "OOF CSV\nJSON stdout", color="cyan", title_size=9, body_size=8)
    _arrow(axis, (0.255, 0.39), (0.34, 0.39), color="cyan")
    _arrow(axis, (0.54, 0.39), (0.62, 0.39), color="green")
    _arrow(axis, (0.82, 0.39), (0.88, 0.39), color="cyan")
    _box(axis, 0.62, 0.12, 0.355, 0.11, "Streamlit + Plotly", "Overview | OOF Explorer | Model Diagnostics | Synthetic Timeline", color="green", title_size=10, body_size=8)
    _arrow(axis, (0.925, 0.31), (0.83, 0.23), color="cyan", connection="arc3,rad=-0.25")
    axis.text(0.055, 0.065, "New scenario: validated current/history inputs + manual meals -> same 151 ordered predictors -> fresh synthetic-only fit -> separate output.", fontsize=7.2, color=COLORS["green"])
    axis.text(0.055, 0.045, "src/synth_data.py writes a separate 5-minute CSV series; the runner reuses its profile helper, not that dynamic CSV.", fontsize=7.0, color=COLORS["muted"])
    figure.savefig(ARCHITECTURE_PATH, format="pdf", bbox_inches="tight")
    plt.close(figure)


def _paragraph(axis: Any, x: float, y: float, text: str, width: int = 63, size: int = 13, color: str = "ink", line_height: float = 0.038, weight: str = "normal") -> float:
    lines = textwrap.wrap(text, width=width)
    axis.text(x, y, "\n".join(lines), color=COLORS[color], fontsize=size, va="top", linespacing=1.3, weight=weight)
    return y - max(1, len(lines)) * line_height


def _bullet_list(axis: Any, items: list[str], x: float = 0.08, y: float = 0.76, width: int = 60, size: int = 13, gap: float = 0.03) -> None:
    for item in items:
        axis.text(x, y, "•", color=COLORS["green"], fontsize=size + 3, weight="bold", va="top")
        lines = textwrap.wrap(item, width=width)
        axis.text(x + 0.035, y, "\n".join(lines), color=COLORS["ink"], fontsize=size, va="top", linespacing=1.3)
        y -= len(lines) * 0.041 + gap


def _presentation(
    summary: dict[str, Any], ablation_summary: dict[str, Any] | None = None
) -> None:
    total = 9
    with PdfPages(PRESENTATION_PATH) as pdf:
        figure, axis = _new_page("GlucoTwin India", "Type 2 diabetes glucose-risk prediction prototype | SRM University AP", 1, total)
        axis.text(0.08, 0.70, "Exploratory prediction of future glucose threshold events", fontsize=28, weight="bold", color=COLORS["ink"], wrap=True)
        axis.text(0.08, 0.52, "Competition prototype overview", fontsize=17, color=COLORS["green"])
        axis.add_patch(FancyBboxPatch((0.08, 0.32), 0.54, 0.10, boxstyle="round,pad=0.015,rounding_size=0.015", facecolor=COLORS["mint"], edgecolor="none"))
        axis.text(0.105, 0.37, DISCLAIMER, fontsize=13, weight="bold", color=COLORS["coral"], va="center")
        axis.text(0.08, 0.255, "Team: mathin7225 | 2 members", fontsize=11, color=COLORS["muted"])
        axis.text(0.08, 0.205, "Leader: Shaik Mahammad Mathin (SRM University AP) | Teammate: Shaik Sameera Kowsar (Tirumala Engineering College)", fontsize=8.7, color=COLORS["muted"])
        axis.text(0.08, 0.155, "Incubator: None | Video: PENDING genuine unlisted URL (20+ minutes)", fontsize=8.7, color=COLORS["muted"])
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

        figure, axis = _new_page("Problem and healthcare use case", "A software research objective, not a clinical intervention.", 2, total)
        axis.text(0.08, 0.73, "Question", fontsize=15, color=COLORS["green"], weight="bold")
        _paragraph(axis, 0.08, 0.67, "Can information available at a prediction time help identify whether Libre glucose will exceed 180 mg/dL in the following two hours?", width=66, size=20, line_height=0.05, weight="bold")
        _bullet_list(axis, ["Potential use case: explore a future-risk alert concept for research and education.", "Target: any observed Libre value strictly above 180 mg/dL in (t, t + 120 minutes].", "The endpoint requires observed Libre coverage and a maximum five-minute gap; Dexcom does not fill gaps or define labels.", "No care recommendation, diagnosis, or clinical action is implemented."], y=0.43, width=76, size=12, gap=0.014)
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

        figure, axis = _new_page("Synthetic data streams", "Static profiles and dynamic signals join by participant ID before modeling.", 3, total)
        _box(axis, 0.08, 0.53, 0.38, 0.25, "A | Static synthetic profile", "Per participant:\nage, sex, diagnosis,\ndiabetes duration, BMI,\nHbA1c", color="cyan", title_size=10, body_size=9.5)
        _box(axis, 0.54, 0.53, 0.38, 0.25, "B | Dynamic synthetic fixture", "One-minute fixture:\nLibre glucose, meals,\nheart rate, activity", color="lime", title_size=10, body_size=9.5)
        axis.text(0.08, 0.43, "Static and dynamic rows are joined by synthetic participant ID before causal feature generation and LOPO training.", fontsize=12, weight="bold", color=COLORS["green"], wrap=True)
        axis.text(0.08, 0.32, "The separate five-minute CSV series from src/synth_data.py is not used as model input; its static profile helper is reused. Neither stream is patient data or real-time hardware.", fontsize=11, color=COLORS["coral"], wrap=True)
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

        figure, axis = _new_page("Architecture and data flow", "The architecture PDF is a separate vector diagram in this folder.", 4, total)
        _box(axis, 0.07, 0.57, 0.24, 0.20, "Synthetic streams", "Static profiles + 1-minute dynamic history joined by participant ID", color="cyan", body_size=10)
        _box(axis, 0.38, 0.57, 0.24, 0.20, "Features + endpoint", "Past-only predictors; Libre label; coverage eligibility", color="green", body_size=10)
        _box(axis, 0.69, 0.57, 0.24, 0.20, "LOPO + LightGBM", "One test participant per fold; preprocessing fitted on training rows", color="lime", body_size=10)
        _arrow(axis, (0.31, 0.67), (0.38, 0.67)); _arrow(axis, (0.62, 0.67), (0.69, 0.67))
        _box(axis, 0.20, 0.25, 0.25, 0.16, "OOF and summaries", "Synthetic predictions + metrics", color="cyan")
        _box(axis, 0.55, 0.25, 0.25, 0.16, "Dashboard", "Streamlit views + Plotly diagnostics", color="green")
        _arrow(axis, (0.81, 0.57), (0.43, 0.41), color="cyan", connection="arc3,rad=0.12")
        _arrow(axis, (0.45, 0.33), (0.55, 0.33))
        axis.text(0.08, 0.12, "New-scenario inputs use the same ordered 151-feature schema, including recent meal type, and a fresh synthetic-only fit; output is separate from historical OOF replay. No real data or hardware stream enters this demo.", fontsize=10.5, color=COLORS["muted"], wrap=True)
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

        figure, axis = _new_page("Model and validation", "Configuration shown here is the actual deterministic synthetic smoke configuration.", 5, total)
        _bullet_list(axis, ["Model: LightGBM classifier, 75 estimators, learning rate 0.08, max depth 3, 15 leaves.", "Preprocessing: numeric median imputation; categorical most-frequent imputation plus one-hot encoding, fitted within each training fold.", "Split: leave-one-patient-out; one complete synthetic participant held out per fold; anchors every 10 elapsed minutes.", "Prediction cutoff: fixed 0.5; no threshold tuning.", "Metrics: pooled OOF AUROC, average precision (AUPRC), Brier score, prevalence; anchor and event summaries; majority baseline.", "Uncertainty: participant bootstrap is implemented for per-participant anchor/event metrics; pooled probability metrics shown next are point estimates without participant-bootstrap intervals."], y=0.76, width=88, size=12, gap=0.012)
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

        figure, axis = _new_page("Observed synthetic smoke results", "Fixed run: seed 42 | 4 synthetic participants | 2 synthetic days | 1,104 eligible anchors", 6, total)
        oof = summary["report"]["oof_metrics"]
        values = [("AUROC", oof["auroc"]), ("AUPRC", oof["auprc"]), ("Brier", oof["brier_score"]), ("Positive prevalence", oof["positive_prevalence"])]
        for index, (label, value) in enumerate(values):
            x = 0.07 + index * 0.225
            axis.add_patch(FancyBboxPatch((x, 0.62), 0.19, 0.16, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor=COLORS["white"], edgecolor=COLORS["line"], linewidth=1.2))
            axis.text(x + 0.015, 0.735, label, fontsize=10, color=COLORS["muted"])
            axis.text(x + 0.015, 0.665, f"{value:.3f}", fontsize=22, weight="bold", color=COLORS["ink"])
        anchor = summary["report"]["lightgbm_lopo_summary"]["overall_metrics"]
        baseline = next(row for row in summary["report"]["baseline_summary"]["overall_metrics"] if row["stratum"] == "all_eligible_anchors")
        model = next(row for row in anchor if row["stratum"] == "all_eligible_anchors")
        axis.text(0.08, 0.53, "Anchor classification at fixed 0.5", fontsize=13, weight="bold", color=COLORS["green"])
        rows = [("Training-fold majority", baseline), ("LightGBM", model)]
        for row_index, (label, result) in enumerate(rows):
            y = 0.44 - row_index * 0.09
            axis.text(0.09, y, label, fontsize=11, color=COLORS["ink"])
            axis.text(0.40, y, f"Precision {result['anchor_precision']:.3f}   Recall {result['anchor_recall']:.3f}   F1 {result['anchor_f1']:.3f}", fontsize=11, color=COLORS["ink"])
        events = summary["report"]["lightgbm_lopo_summary"]["event_overall_metrics"][0]
        axis.text(0.08, 0.25, f"LightGBM event summary: {events['actual_events']} synthetic events; {events['alert_episodes']} alert episodes; {events['matched_events']} matched; {events['false_alert_episodes']} false alert episodes; {events['false_alert_episodes_per_day']:.2f} false alerts per measured day.", fontsize=9.5, color=COLORS["muted"], wrap=True)
        ablation_metrics = (
            ablation_summary["report"]["oof_metrics"] if ablation_summary else None
        )
        ablation_text = (
            "Same-data ablation without meal_type_last_120m: "
            f"AUROC {ablation_metrics['auroc']:.3f}, "
            f"AUPRC {ablation_metrics['auprc']:.3f}, "
            f"Brier {ablation_metrics['brier_score']:.3f}. Mixed results; "
            "no overall improvement claimed."
            if ablation_metrics
            else "Same-data feature-ablation report unavailable; no feature-benefit claim is made."
        )
        axis.text(0.08, 0.15, ablation_text, fontsize=8.5, color=COLORS["muted"], wrap=True)
        axis.text(0.08, 0.065, "All figures are synthetic software checks. They do not establish clinical validity or utility.", fontsize=9.5, color=COLORS["coral"], weight="bold", wrap=True)
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

        figure, axis = _new_page("Dashboard demonstration", "All views below are rendered from the synthetic smoke bundle.", 7, total)
        sections = [
            ("Overview", "Seed, synthetic participant count, eligible anchors, prevalence, model settings, metrics, and baseline comparisons."),
            ("OOF Explorer", "Filter synthetic out-of-fold predictions by participant and held-out fold."),
            ("Model Diagnostics", "ROC and precision-recall curves, reliability diagram, fixed-cutoff confusion matrix, probability distribution."),
            ("Synthetic Glucose Timeline", "Plot the generated synthetic CGM history for a selected synthetic participant."),
            ("New Synthetic Scenario", "Enter current and timestamped history measurements, add/edit/remove manual meals, then run a separate 120-minute scenario prediction with missing-input disclosure."),
        ]
        for index, (title, body) in enumerate(sections):
            y = 0.75 - index * 0.13
            _box(axis, 0.08, y, 0.84, 0.095, title, body, color=["green", "cyan", "lime", "amber", "coral"][index], title_size=10.5, body_size=8.5)
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

        figure, axis = _new_page("Limitations and future work", "The prototype demonstrates software behavior, not clinical validation.", 8, total)
        _bullet_list(axis, ["No real CGMacros evaluation is evidenced; current metrics and new-scenario training are synthetic only.", "The new-scenario probability is not calibrated or clinically validated; incomplete history leaves predictors missing and the pipeline imputes them.", "The separate five-minute CSV time series is not consumed by the dashboard; the shared static profile helper and its separate one-minute dynamic fixture are combined by participant ID.", "Dataset use rights, citation, challenge rules, and redistribution requirements need current authoritative verification.", "Source signal acquisition cadence versus upstream interpolation and sensor delivery latency remain unknown; no real-time hardware is connected.", "Pooled AUROC/AUPRC/Brier do not have participant-bootstrap intervals; no fitted calibration or tuned operating point is provided.", "Future: verify rights and cadence; wire versioned data provenance/run manifests; evaluate per-patient eligibility and missingness; prespecify external validation and calibration; follow official submission format."], y=0.76, width=91, size=10.5, gap=0.008)
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

        figure, axis = _new_page("Submission actions", "Complete personal, legal, and publication details before sharing the final package.", 9, total)
        _bullet_list(axis, ["Team: mathin7225, 2 members. Leader: Shaik Mahammad Mathin (SRM University AP); teammate: Shaik Sameera Kowsar (Tirumala Engineering College). Incubator: None, per current project information.", "Video: PENDING genuine unlisted URL; confirm duration is at least 20 minutes and link access works without requesting permission.", "License: MIT names Shaik Mahammad Mathin. The named holder must confirm authority to license all included contributions; third-party data/dependencies retain their own terms.", "Required folder convention: mathin7225_SRM University AP. Do not rename this Git repository. The package is incomplete until the video is supplied.", "Public release: verify the repository and PDF links open in a signed-out browser; do not publish dataset contents or private identifiers."], y=0.76, width=93, size=11.5, gap=0.012)
        pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)


def main() -> None:
    corrected_report_path = (
        ROOT / "data" / "processed" / "meal_type_corrected" / "summary.json"
    )
    ablation_report_path = (
        ROOT
        / "data"
        / "processed"
        / "meal_type_corrected_no_recent_type"
        / "summary.json"
    )
    if not corrected_report_path.is_file():
        raise FileNotFoundError(
            "The corrected LOPO report is required to generate current submission PDFs."
        )
    if not ablation_report_path.is_file():
        raise FileNotFoundError(
            "The same-fixture feature-ablation report is required for a supported comparison."
        )
    summary = json.loads(corrected_report_path.read_text(encoding="utf-8"))
    ablation_summary = json.loads(
        ablation_report_path.read_text(encoding="utf-8")
    )
    if summary.get("synthetic_data") is not True:
        raise RuntimeError("Submission PDF metrics must come from the synthetic smoke path")
    if (
        summary.get("seed") != 42
        or summary.get("feature_set") != "full"
        or summary.get("fold_count") != 4
        or summary.get("eligible_anchor_count") != 1104
    ):
        raise RuntimeError("Corrected report does not match the documented synthetic LOPO configuration")
    if (
        ablation_summary.get("synthetic_data") is not True
        or ablation_summary.get("seed") != summary.get("seed")
        or ablation_summary.get("feature_set") != summary.get("feature_set")
        or ablation_summary.get("fold_count") != summary.get("fold_count")
        or ablation_summary.get("eligible_anchor_count")
        != summary.get("eligible_anchor_count")
    ):
        raise RuntimeError("Feature-ablation report is not comparable to the corrected LOPO report")
    _architecture_diagram()
    _presentation(summary, ablation_summary)
    print(f"Wrote {ARCHITECTURE_PATH}")
    print(f"Wrote {PRESENTATION_PATH}")
    print("All presentation metrics are from seed-42 synthetic smoke data.")


if __name__ == "__main__":
    main()
