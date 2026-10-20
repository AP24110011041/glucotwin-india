# GlucoTwin India

An early research prototype for a Type 2 diabetes digital twin, prepared for the Happiest Health Digital Twin Challenge 2026.

**SYNTHETIC RESEARCH DEMO — NOT FOR CLINICAL USE**

## Team and Institutional Details

- Project: GlucoTwin India
- Team name: mathin7225
- Team size: 2 members
- Team leader: Shaik Mahammad Mathin — SRM University AP
- Teammate: Shaik Sameera Kowsar — Tirumala Engineering College
- Incubator: None

## Healthcare Use Case

Explore, for research and education only, whether timestamped glucose and contextual signals could support forecasting an observed Libre glucose value above 180 mg/dL in the next two hours. This is not a care workflow, medical device, diagnosis, treatment recommendation, or clinically validated system.

## Problem Statement

Explore whether longitudinal Libre glucose, meal, wearable, and clinical data can support prediction of whether Libre glucose will exceed 180 mg/dL during the next two hours. This repository is a research software prototype; its dashboard trains on synthetic data only, and no real-participant predictive performance is claimed.

## Intended Use

Research and educational prototyping only. The project is not a medical device, does not provide diagnosis or treatment recommendations, and must not be used for clinical decisions or patient care.

## Potential Dataset and Terms

The potential dataset is [CGMacros on PhysioNet](https://physionet.org/content/cgmacros/1.0.0/), release 1.0.0. Previous repository notes record a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license and DOI [10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658); these details were not independently re-verified during this review. Treat the rights and citation details as unconfirmed until checked against the current official release page. Confirm that the intended challenge and research use complies with the current terms before downloading, processing, or sharing any data or derived results.

To obtain it, open the official release page, review and accept the applicable terms, then use the page's download option (or its documented command-line instructions) to download the files you need. Extract the files manually under `data/raw/cgmacros/`. Nothing in this repository downloads the dataset automatically, and dataset files are excluded from Git. Follow the current citation instructions on PhysioNet; treat the DOI recorded above as provisional until verified against the release page.

On Windows PowerShell, after downloading the archive from that page to your Downloads folder, locate and extract it, then inspect the extracted CSV files:

```powershell
Get-ChildItem "$HOME\Downloads" -Filter "CGMacros_dateshifted365.zip"
$datasetZip = Join-Path $HOME "Downloads\CGMacros_dateshifted365.zip"
Expand-Archive -LiteralPath $datasetZip -DestinationPath "data/raw/cgmacros"
Get-ChildItem "data/raw/cgmacros" -Recurse -File
python src/inspect_data.py data/raw/cgmacros
```

The inspector scans extracted CSV files recursively; it does not open ZIP archives. If the archive is stored elsewhere, update `$datasetZip` to its local path.

If access, license terms, or local data availability prevent use, use the included reproducible synthetic generator for development. Synthetic data is not a substitute for real-world validation.

## CGMacros Day 2 Feature Engineering

With the extracted CGMacros files outside this repository, build local normalized time series, causal features, labels, and an aggregate eligibility summary:

```powershell
python src/feature_engineering.py --data-dir "C:\cgmacros\data\CGMacros" --output-dir data/processed
```

The pipeline writes `cgmacros_normalized_timeseries.csv`, `cgmacros_features.csv`, and `cgmacros_target_eligibility_summary.json` under `data/processed/`, which is excluded from Git. The normalized export preserves the Libre and Dexcom source columns and also includes a derived `glucose_mg_dl` column that selects Libre when present and otherwise Dexcom. Predictive glucose features, future labels, and coverage eligibility use Libre only; Dexcom does not fill Libre gaps or determine labels. Feature-only forward-fills use past Libre values for at most five minutes and include age/missingness indicators. A target is eligible only when the next 120 minutes are available and observed Libre timestamps cover the window with no gap over five minutes. The label is positive when Libre exceeds 180 mg/dL in `(t, t+120 minutes]`. Use `src/feature_groups.py` to select predictors without patient IDs, timestamps, or target columns. The source dataset contains smooth fractional minute-level sequences, but the available files do not establish whether those values were interpolated upstream or represent device acquisition cadence; do not interpret them as verified one-minute device measurements. No model is trained by this command.

## Evaluation Framework

`src/evaluation.py` provides a callable LOPO evaluation pipeline. `prepare_evaluation_data` builds features, labels, and eligibility on each complete chronological patient series first, then selects anchors every ten elapsed minutes from that patient's first timestamp. It keeps ineligible anchors for fold assignment; metric calculations use only eligible labels. Provide an explicit approved predictor list derived from `get_predictor_columns`; the evaluator rejects IDs, timestamps, targets, duplicates, and Dexcom columns.

`iter_lopo_folds` creates deterministic folds with exactly one whole participant held out. `evaluate_baselines` scores the training-fold majority class as a future-event forecast; ties predict negative. It reports the Libre current-state detector separately as activity only: an alert occurs only when a fresh Libre value is observed at the prediction timestamp and exceeds 180 mg/dL. That detector is not scored against future-event labels or included in future-event recall. The LightGBM interface creates a fresh pipeline per fold with numeric imputation and categorical encoding fit during that fold's training step. Its probability cutoff is fixed at 0.5; threshold tuning is not implemented. The interface does not accept a custom estimator factory. These functions do not run automatically, and no model fitting or evaluation is performed by feature generation.

Anchor-level precision, recall, F1, and TP/FP/TN/FN counts score each eligible sampled prediction anchor independently; overlapping two-hour target windows are not independent events. Separate event metrics use Libre only. An event onset is the first observed Libre reading >180 mg/dL after a run from its first to last observed reading at or below 180 mg/dL spans at least 60 minutes, every intervening source row is observed, and each adjacent timestamp gap is at most one minute. Missing readings or larger gaps reset the run; a high value at the start of a patient's series or immediately after a gap is not declared a new onset. This sampled-data definition does not establish continuous time-in-range between device readings.

Alert episodes start at the first positive forecast anchor. Further positive anchors at or before 60 minutes after that first alert are suppressed; a later alert starts another episode. Each episode matches at most one earliest event onset strictly after the alert and no later than 120 minutes after it, and each event can match at most one alert. Future-event recall and false alert episodes/day are reported separately from anchor-level classification metrics. Measured exposure sums only intervals between adjacent rows with observed Libre values whose timestamp gap is at most one minute, clipped to the first and last eligible prediction anchors. Missing or wider-gap intervals contribute no exposure, and no duration is extrapolated outside observations. This denominator represents observed Libre monitoring time, not continuous alert-decision opportunity time: the model only emits decisions at sampled prediction anchors. If positive measured exposure cannot be calculated, false-alert episodes/day is unavailable with a reason.

Uncertainty uses a deterministic participant-level bootstrap of per-patient anchor classification and event metrics; participants, not overlapping windows, are resampled. It describes between-participant variation, not uncertainty from independent windows or a population-level clinical validation. The current AUROC, AUPRC, and Brier summaries are pooled point estimates without participant-bootstrap intervals. Keep all learned preprocessing and any future feature selection inside each training fold. The CGMacros source signal's interpolation and device-acquisition cadence remain unresolved; report competition results only as exploratory results on the dataset-provided signal, not as device-cadence or clinical evidence.

`src/evaluation_metadata.py` provides an in-memory run-manifest helper for a future report. It records a caller-supplied code revision, Python/package versions, the supplied dataset archive's SHA-256, target/anchor/LOPO settings, and event/exposure definitions. It writes no report and marks usage-rights and sensor-cadence verification as false until independently confirmed.

## Architecture and Data Flow

- `data/raw/` is reserved for manually acquired local source data; dataset files are excluded from Git.
- `src/inspect_data.py` reads local CSVs and reports schema, missingness, participant-like identifiers, domains, and timestamp intervals.
- `src/feature_engineering.py` normalizes participant CSVs, joins static EHR fields by participant ID, creates past-only features and Libre-only future labels, and writes local processed outputs.
- `src/evaluation.py` selects patient-relative anchors, validates an explicit predictor allow-list, makes one whole patient the test set per LOPO fold, and fits preprocessing inside each training fold.
- `src/run_lopo.py` joins synthetic participant-level demographics, diagnosis, diabetes duration, BMI, and HbA1c with its generated one-minute Libre, meal, and wearable/activity fixture by `patient_id`. The combined synthetic features are passed to each fold's LightGBM predictor. It writes synthetic OOF predictions and prints baseline and metric summaries as JSON to stdout; it does not load CGMacros or processed real data.
- `src/inference.py` builds validated timestamped scenario inputs, applies the same causal feature builder and ordered 151-feature predictor schema, then fits the existing LightGBM pipeline on synthetic training participants. The scenario participant is kept separate from training. Manual meal and measurement history are inputs, not clinical evidence.
- `app/app.py` is a working Streamlit dashboard with Overview, OOF Explorer, Model Diagnostics, Synthetic Glucose Timeline, and New Synthetic Scenario sections. OOF replay and new-scenario outputs are labeled separately. Dashboard-generated predictions are held in memory; the app does not read or write a CSV artifact.
- `data/processed/`, `models/`, and `app/synthetic_demo/synthetic_oof_predictions.csv` are local outputs and are excluded from Git. `notebooks/` and `reports/` currently contain placeholders only.

The current implementation uses Python, NumPy, pandas, scikit-learn pipelines/metrics, LightGBM, Streamlit, and Plotly. `requirements.txt` also declares packages reserved for broader work; the current source does not directly import SHAP, SciPy, Matplotlib, or Requests. See the editable [architecture and submission notes](docs/competition_submission.md) for the data-flow diagram and evidence checklist.

## Technical Stack

Python; pandas and NumPy for data handling; scikit-learn for fold-fitted preprocessing and metrics; LightGBM for classification; Streamlit and Plotly for the synthetic research dashboard; Matplotlib for the generated submission PDFs; pytest for regression tests. The project requirements also include optional packages for other repository tools.

## AI/ML Model

The synthetic dashboard smoke path uses a LightGBM classifier (75 estimators, learning rate 0.08, max depth 3, 15 leaves). The full model schema contains 151 ordered predictors generated by `build_features`: glucose history, meal event/type/nutrient summaries, wearable/activity summaries, and static profile values with missingness indicators. The synthetic training fixture marks meal nutrient values missing between explicit meal events; zero-valued nutrients on an explicit event remain measured zeros. `meal_type_last_120m` is the most recent observed meal type in the inclusive prior 120-minute window; it is `none` without a meal event and `unknown` for an event with a missing or unrecognized type. Manual same-time meal records resolve to the shared normalized type or `mixed` when types differ. Existing timestamp-local meal-type indicators remain unchanged. IDs, timestamps, Dexcom fields, and the future target are excluded. Each LOPO fold holds out one complete synthetic participant. Numeric median imputation and categorical imputation/one-hot encoding are fitted within the training fold. The forecast cutoff is fixed at 0.5; there is no threshold tuning or fitted calibration. New-scenario inference fits a fresh model on the synthetic fixture and applies that schema to a separate scenario.

Corrected synthetic LOPO snapshot (seed 42, 4 synthetic participants, 2 days, 4 folds, 1,104 eligible anchors): AUROC 0.8519, AUPRC 0.6012, positive prevalence 0.1159, and Brier score 0.0726. Event summary: 8 synthetic events, 17 forecast episodes, 4 matched events, 13 false episodes, and 1.70 false-alert episodes per measured day. A same-fixture ablation without `meal_type_last_120m` had AUROC 0.8451, AUPRC 0.6077, and Brier score 0.0720. These results are mixed and do not establish an overall improvement. All figures are synthetic-only diagnostics, not real-patient evidence or clinical validation.

## Demo Video

YouTube unlisted demo, minimum 20 minutes: **PENDING — add the actual unlisted video URL after recording and verifying the video is at least 20 minutes.** No video or URL is included in this checkout.

## License

The [MIT License](LICENSE) names Shaik Mahammad Mathin as copyright holder and is scoped to original project code and materials only. Confirm the named holder's authority to license all covered material before publication. The project license does not grant rights to CGMacros or any other third-party data, trademarks, or dependencies; those remain under their respective terms. Confirm dataset rights and intended-use restrictions independently before use or redistribution.

## Competition Deliverables

- Architecture diagram: [PDF](docs/architecture_diagram.pdf)
- Project presentation: [PDF](docs/presentation.pdf)
- Requirement-by-requirement audit and remaining actions: [submission requirements audit](docs/submission_requirements_audit.md)
- Source and reproducible PDF generator: [PDF generator](docs/generate_submission_pdfs.py)

The presentation's numerical results use the corrected deterministic synthetic LOPO report and a same-fixture ablation. The comparison is mixed and is not real-patient evidence. The architecture and presentation PDFs are tracked locally, but public accessibility has not been verified. After owner review and approved publication, verify every link from a signed-out browser. The package remains incomplete until the genuine 20+ minute demo video is supplied.

## Submission Folder Name

The required submission folder name is `mathin7225_SRM University AP`. The package is incomplete pending the genuine 20+ minute demo video and remaining owner/legal review. This instruction does not rename the existing Git repository.

## Setup

From the repository root, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run a small synthetic-only LOPO smoke test. This command writes generated output under the ignored `data/processed/` directory; it does not target the dashboard's protected demo artifact:

```powershell
python -m src.run_lopo --smoke-test --output-dir data/processed/synthetic-smoke
```

This trains one model per synthetic held-out participant, writes `synthetic_oof_predictions.csv` under the chosen output directory, and prints a JSON summary to stdout. It is a software-path check only, not evidence about real participants.

## Synthetic Dashboard

The Streamlit app is a synthetic research demo only. It does not read CGMacros data, does not require external services or credentials, and must not be used for clinical performance claims.

From the repository root, launch the demo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py --server.headless true --server.port 8501
```

The dashboard builds a deterministic synthetic bundle at runtime and keeps generated OOF results in memory. The new-scenario view accepts current glucose, optional timestamped history and wearable measurements, static synthetic-profile fields, and editable meal entries; it retrains on the synthetic fixture and reports the fixed 120-minute Libre >180 mg/dL exploratory target. Missing meal types are permitted and become `unknown`; recent meal type is carried forward causally for at most 120 minutes. Missing features are imputed by the fitted pipeline and counted in the output. The dashboard labels historical OOF replay separately from new-scenario output and distinguishes manual meals from synthetic reference meals. The static profile generator is shared with `src/synth_data.py`; that script's separate five-minute CSV time series is not the dashboard model input. No real-time hardware or CGMacros stream is connected. This is for software validation, UI prototyping, and demonstration only.
Generate a small, deterministic synthetic example (the command overwrites the two named output files):

```powershell
python src/synth_data.py --participants 5 --days 2 --seed 42
```

The generator writes `patients.csv` and `timeseries.csv` under `data/processed/synthetic/`. Inspect local CSV files with:

```powershell
python src/inspect_data.py data/raw
```

Run the repository test suite from the project virtual environment with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

## Data Provenance and Privacy

Use only anonymized data with appropriate permission, open-source data whose license permits the intended use, or data generated by the included synthetic simulator. Do not add direct identifiers, secrets, or source datasets to version control. Keep downloaded and processed data local; verify source terms and provenance before any analysis or sharing.

Every row emitted by the synthetic generator is labeled `SYNTHETIC`. The simulator uses simplified, arbitrary rules to create test fixtures and is not clinically validated. Its values and derived labels must not be presented as observed patient data or evidence of predictive performance.

## Validation Policy

The implemented model evaluator uses leave-one-patient-out validation: one whole participant is held out per fold, and the LightGBM preprocessing pipeline is fit only on training participants. It does not perform random row splits or threshold tuning. Features are computed per patient from values timestamped at or before each prediction anchor; this assumes timestamped measurements are available at prediction time, while real sensor delivery latency has not been verified. No real-data run is evidenced in this repository review. Do not report performance until the protocol has been run on appropriately licensed real data; synthetic data is for software checks only.

The runner reports pooled out-of-fold AUROC, average precision (AUPRC), positive prevalence, and Brier score. AUROC and average precision are unavailable when the evaluated labels contain only one class; Brier score remains defined for a single class. The dashboard adds a binned reliability diagram as a descriptive calibration check, not a fitted calibration model or proof of calibration. The AUPRC should be interpreted alongside the positive prevalence; a training-fold majority baseline and current-state detector are reported separately. Anchor classification uses a fixed 0.5 probability cutoff with no threshold tuning. Synthetic OOF observations are temporally overlapping and are not independent clinical evidence.

## Research Prototype Disclaimer

**SYNTHETIC RESEARCH DEMO — NOT FOR CLINICAL USE**

This is an early research prototype, not a clinically validated system or medical device. No safety, efficacy, generalizability, or clinical utility has been established. Predictions, synthetic examples, and future model outputs are not medical advice and must not guide care.
