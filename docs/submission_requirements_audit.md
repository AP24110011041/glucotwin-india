# Happiest Health Digital Twin Challenge 2026: Requirements Audit

Source of truth: the mandatory requirements supplied by the project owner in the task prompt. This audit does not infer other competition rules.

| Requirement | Status | Evidence | Remaining action |
|---|---|---|---|
| Team details, college, and incubator information | PRESENT | [README.md](../README.md#team-and-institutional-details) | Team `mathin7225` has 2 members: leader Shaik Mahammad Mathin (SRM University AP) and teammate Shaik Sameera Kowsar (Tirumala Engineering College). Incubator is reported as `None`. No private phone/contact details are included. |
| Project title | PRESENT | [README.md](../README.md#glucotwin-india); [presentation.pdf](presentation.pdf) | None. |
| Problem statement | PRESENT | [README.md](../README.md#problem-statement); [presentation.pdf](presentation.pdf) | None. |
| Healthcare use case | PRESENT | [README.md](../README.md#healthcare-use-case) | The described use is exploratory research only, not clinical care. |
| Technical stack | PRESENT | [README.md](../README.md#technical-stack) | None for the implemented prototype. |
| AI/ML model and framework details | PRESENT | [README.md](../README.md#ai-ml-model); [src/run_lopo.py](../src/run_lopo.py); [src/evaluation.py](../src/evaluation.py) | Results in the presentation are synthetic only. |
| Link to a minimum 20-minute unlisted YouTube demo | MISSING | [README.md](../README.md#demo-video) marks the URL pending; no video or URL exists in this checkout. | Record/upload a genuine video of at least 20 minutes, set access to unlisted, add its real URL, and verify it signed out. |
| Open-source license details | NEEDS USER INPUT | [LICENSE](../LICENSE); [README.md](../README.md#license) | MIT text names Shaik Mahammad Mathin and is scoped to project-authored code/materials. Confirm authority to license all covered material before publication. Third-party data and dependencies are excluded. |
| Architecture diagram in PDF/PPT format | PRESENT | [architecture_diagram.pdf](architecture_diagram.pdf); editable generator: [generate_submission_pdfs.py](generate_submission_pdfs.py) | Verify the linked PDF after publication. |
| Project presentation in PDF/PPT format | PRESENT | [presentation.pdf](presentation.pdf); editable generator: [generate_submission_pdfs.py](generate_submission_pdfs.py) | Corrected metrics are from synthetic seed-42 LOPO output and ablation; results are mixed, not real-patient outcomes or clinical validation. |
| Publicly accessible files and links without extra permissions | PARTIALLY VERIFIED | The [GitHub repository](https://github.com/AP24110011041/glucotwin-india) is publicly viewable; local artifacts are listed in [README.md](../README.md#competition-deliverables) | PDF endpoint checks timed out. The changed PDFs and docs in this checkout are uncommitted and not published. Recheck public artifact links from a signed-out browser after owner-approved publication. |
| Folder naming convention `Team Name_College Name` | PRESENT | [README.md](../README.md#submission-folder-name) | Required package folder name is `mathin7225_SRM University AP`. This does not rename the existing Git repository. |

## Verified Deliverable Facts

- Architecture and presentation PDFs are regenerated from the source and the saved corrected synthetic LOPO reports. The source and PDFs show the current 151-predictor schema, confirmed incubator value `None`, and the video as pending.
- Corrected synthetic LOPO report (`seed=42`, 4 synthetic participants, 2 synthetic days, 4 folds, 1,104 eligible anchors): AUROC 0.8519, AUPRC 0.6012, prevalence 0.1159, Brier 0.0726. It records 8 synthetic events, 17 alert episodes, 4 matched, 13 false episodes, and 1.70 false-alert episodes per measured day. Same-fixture ablation without `meal_type_last_120m`: AUROC 0.8451, AUPRC 0.6077, Brier 0.0720. Results are mixed and synthetic-only, not clinical validation.
- The dashboard LOPO runner joins a synthetic static profile table (demographics, diagnosis, duration, BMI, HbA1c) to its generated one-minute dynamic fixture by `patient_id`; both appear in the model predictor list. `src/synth_data.py` also writes a separate five-minute dynamic CSV series, which is not the dashboard model input.
- The new-scenario flow uses a separate scenario participant, the same shared 151 ordered predictors used in training, the existing 120-minute Libre >180 mg/dL target, and fresh synthetic-only training. `meal_type_last_120m` uses only observed meal events at or before each feature timestamp and maps absent/unrecognized types explicitly. It displays missing-feature/imputation context and does not make a clinical accuracy claim.
- The GitHub repository page is publicly viewable. Public access to its PDF files could not be verified because the endpoint check timed out.
- This checkout contains no evidenced real-data model run, real-patient results, clinical validation, or demo video. The named license holder's authority remains unconfirmed.
- Dataset terms and the recorded DOI require confirmation against current authoritative source terms before use or redistribution.
