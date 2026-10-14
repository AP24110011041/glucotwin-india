# Happiest Health Digital Twin Challenge 2026: Requirements Audit

Source of truth: the mandatory requirements supplied by the project owner in the task prompt. This audit does not infer other competition rules.

| Requirement | Status | Evidence | Remaining action |
|---|---|---|---|
| Team details, college, and incubator information | NEEDS USER INPUT | [README.md](../README.md#team-and-institutional-details) | Team `mathin7225` has 2 members: leader Shaik Mahammad Mathin (SRM University AP) and teammate Shaik Sameera Kowsar (Tirumala Engineering College). Incubator information remains unresolved; add a verified name or `None`. No private phone/contact details are included. |
| Project title | PRESENT | [README.md](../README.md#glucotwin-india); [presentation.pdf](presentation.pdf) | None. |
| Problem statement | PRESENT | [README.md](../README.md#problem-statement); [presentation.pdf](presentation.pdf) | None. |
| Healthcare use case | PRESENT | [README.md](../README.md#healthcare-use-case) | The described use is exploratory research only, not clinical care. |
| Technical stack | PRESENT | [README.md](../README.md#technical-stack) | None for the implemented prototype. |
| AI/ML model and framework details | PRESENT | [README.md](../README.md#ai-ml-model); [src/run_lopo.py](../src/run_lopo.py); [src/evaluation.py](../src/evaluation.py) | Results in the presentation are synthetic only. |
| Link to a minimum 20-minute unlisted YouTube demo | MISSING | [README.md](../README.md#demo-video) contains a placeholder; no video or URL exists in this checkout. | Record/upload a video of at least 20 minutes, set access to unlisted, add the real URL, and verify it signed out. |
| Open-source license details | NEEDS USER INPUT | [LICENSE](../LICENSE); [README.md](../README.md#license) | MIT text is prepared for project-authored code only. Confirm the copyright holder/authority and replace the placeholder before publication. Third-party data and dependencies are excluded. |
| Architecture diagram in PDF/PPT format | PRESENT | [architecture_diagram.pdf](architecture_diagram.pdf); editable generator: [generate_submission_pdfs.py](generate_submission_pdfs.py) | Verify the linked PDF after publication. |
| Project presentation in PDF/PPT format | PRESENT | [presentation.pdf](presentation.pdf); editable generator: [generate_submission_pdfs.py](generate_submission_pdfs.py) | Metrics are seed-42 synthetic smoke results, not real-patient outcomes. |
| Publicly accessible files and links without extra permissions | NEEDS USER INPUT | Relative artifact links are in [README.md](../README.md#competition-deliverables) | The architecture and presentation PDFs are tracked in this local checkout, but public and signed-out access remain unverified. After owner review and approved publication, test all links in a signed-out browser. Do not commit/push until owner review. |
| Folder naming convention `Team Name_College Name` | PRESENT | [README.md](../README.md#submission-folder-name) | Required package folder name is `mathin7225_SRM University AP`. This does not rename the existing Git repository. |

## Verified Deliverable Facts

- The architecture diagram and presentation PDFs were generated from the current source and a fresh deterministic synthetic run (`seed=42`, 4 synthetic participants, 2 synthetic days, 1,104 eligible anchors).
- The dashboard LOPO runner joins a synthetic static profile table (demographics, diagnosis, duration, BMI, HbA1c) to its generated one-minute dynamic fixture by `patient_id`; both appear in the model predictor list. `src/synth_data.py` also writes a separate five-minute dynamic CSV series, which is not the dashboard model input.
- This checkout contains no evidenced real-data model run, real-patient results, clinical validation, video, or confirmed copyright holder.
- Dataset terms and the recorded DOI require confirmation against current authoritative source terms before use or redistribution.
