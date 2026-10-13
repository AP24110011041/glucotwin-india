# GlucoTwin India

An early research prototype for a Type 2 diabetes digital twin, prepared for the Happiest Health Digital Twin Challenge 2026.

## Problem Statement

Explore whether longitudinal Libre glucose, meal, wearable, and clinical data can support prediction of whether Libre glucose will exceed 180 mg/dL during the next two hours. This repository is a starting point for data inspection and experimentation; no model or clinical performance is claimed.

## Intended Use

Research and educational prototyping only. The project is not a medical device, does not provide diagnosis or treatment recommendations, and must not be used for clinical decisions or patient care.

## Planned Dataset

The primary planned dataset is [CGMacros on PhysioNet](https://physionet.org/content/cgmacros/1.0.0/), release 1.0.0. Review the dataset description, license, documentation, and citation requirements on the official release page before use. The release lists a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. Confirm that the intended challenge and research use complies with the current terms.

To obtain it, open the official release page, review and accept the applicable terms, then use the page's download option (or its documented command-line instructions) to download the files you need. Extract the files manually under `data/raw/cgmacros/`. Nothing in this repository downloads the dataset automatically, and dataset files are excluded from Git. Cite the dataset as instructed by PhysioNet; the release DOI is [10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658).

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

Uncertainty uses a deterministic participant-level bootstrap of per-patient metrics; participants, not overlapping windows, are resampled. It describes between-participant variation, not uncertainty from independent windows or a population-level clinical validation. Keep all learned preprocessing and any future feature selection inside each training fold. The CGMacros source signal's interpolation and device-acquisition cadence remain unresolved; report competition results only as exploratory results on the dataset-provided signal, not as device-cadence or clinical evidence.

`src/evaluation_metadata.py` provides an in-memory run-manifest helper for a future report. It records a caller-supplied code revision, Python/package versions, the supplied dataset archive's SHA-256, target/anchor/LOPO settings, and event/exposure definitions. It writes no report and marks usage-rights and sensor-cadence verification as false until independently confirmed.

## Architecture Overview

- `data/raw/`: manually acquired source files; never commit datasets.
- `data/processed/`: generated or transformed local data; excluded from Git.
- `src/inspect_data.py`: recursively reports CSV schemas, missingness, participant identifiers, likely data domains, and timestamp sampling intervals.
- `src/synth_data.py`: creates synthetic participant EHR records and five-minute glucose, meal, and wearable time series.
- `app/`: planned Streamlit application.
- `notebooks/`: planned exploratory analysis.
- `models/`: local model artifacts; excluded from Git.
- `reports/`: planned experiment reports.

Planned analysis stack: Python, pandas, LightGBM, SHAP, Streamlit, and Plotly. The dependency list also includes NumPy, SciPy, scikit-learn, Matplotlib, joblib, requests, and pytest.

## Setup

From the repository root, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Generate a small, deterministic synthetic example (the command overwrites the two named output files):

```powershell
python src/synth_data.py --participants 5 --days 2 --seed 42
```

The generator writes `patients.csv` and `timeseries.csv` under `data/processed/synthetic/`. Inspect local CSV files with:

```powershell
python src/inspect_data.py data/raw
```

Run the repository test suite with:

```powershell
python -m pytest tests/
```

## Data Provenance and Privacy

Use only anonymized data with appropriate permission, open-source data whose license permits the intended use, or data generated by the included synthetic simulator. Do not add direct identifiers, secrets, or source datasets to version control. Keep downloaded and processed data local; verify source terms and provenance before any analysis or sharing.

Every row emitted by the synthetic generator is labeled `SYNTHETIC`. The simulator uses simplified, arbitrary rules to create test fixtures and is not clinically validated. Its values and derived labels must not be presented as observed patient data or evidence of predictive performance.

## Validation Policy

When model development begins, evaluation must use leave-one-patient-out validation only. Never split rows randomly across training and validation. Fit preprocessing and feature selection within each training fold, and build each prediction row using only information available at that timestamp. Do not report performance until the protocol has been executed on appropriately licensed real data; synthetic data is for software checks only.

## Research Prototype Disclaimer

This is an early research prototype, not a clinically validated system or medical device. No safety, efficacy, generalizability, or clinical utility has been established. Predictions, synthetic examples, and future model outputs are not medical advice and must not guide care.
