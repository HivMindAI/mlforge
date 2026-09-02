# Changelog

All notable MLForge changes are recorded here. The project follows semantic versioning during its
`0.y.z` development series as described in [the compatibility policy](docs/compatibility.md).

## [Unreleased]

No changes yet.

## [0.5.0] - 2026-09-02

### Added

- Add deterministic shuffled K-fold regression comparison with Ridge Regression and Random Forest
  Regressor, complete MAE, R-squared, and RMSE fold evidence, and lower-is-better RMSE ranking.
- Extend selection-driven all-row final fitting, trusted artifacts, CLI comparison, the local web
  workflow, and browser verification through regression prediction.
- Add a small house-price regression example and prediction input for local and browser workflows.

### Compatibility

- Write cross-validation and final-model manifest schema version 2 while continuing to read
  immutable version-1 classification manifests.
- Upgrade the SQLite web workspace to schema version 2 with a transactional migration that
  preserves existing experiment, job, finalization, and prediction lineage.

## [0.4.0] - 2026-08-31

### Added

- Add a local single-user FastAPI and Next.js application covering CSV upload, explicit target
  selection, dataset profiling, persisted classification experiments, deterministic
  cross-validation, rank-one final fitting, model inspection, schema-validated batch prediction,
  bounded previews, and complete CSV downloads.
- Add SQLite-backed dataset, experiment, job, finalization, and prediction metadata with one
  bounded in-process worker and honest restart recovery for interrupted jobs.
- Add a private two-container deployment profile with an internal API network, loopback-only browser
  port, unprivileged containers, health probes, and one durable workspace volume.
- Add explicit web schema version 1, transactional adoption of existing unversioned workspaces,
  structural and foreign-key validation, newer-schema rejection, and backup/restore coverage.
- Add a Playwright golden-path test for the complete supported browser workflow.

### Changed

- Distribute the frontend and private-deployment sources in the source archive while keeping the
  Python wheel focused on the importable core and FastAPI adapter.
- Align the Python package and private frontend on version 0.4.0 and update the roadmap,
  compatibility policy, release validation, and deployment guidance to describe the product that
  is actually shipped.

### Security and operations

- Keep the supported deployment limited to one trusted operator behind an SSH tunnel or reviewed
  private access gateway. Authentication, public exposure, multi-tenancy, distributed execution,
  and online model serving remain unsupported.

## [0.3.0] - 2026-08-22

### Added

- Add an explicit `fit_selected_model` application service and `mlforge finalize` command that
  accept persisted successful cross-validation selection evidence, refit its winner on every exact
  selected dataset row, and save the prediction-ready artifact through the existing artifact store.
- Add strict create-only final-model manifests recording selection UUID/digest, fold evidence,
  dataset identity, reconstructed preprocessing and estimator parameters, `fit_scope=all_rows`,
  artifact identity/payload hash, environment, warnings, terminal status, and failure evidence.
- Add artifact manifest version 2 with explicit final-model lineage while retaining version-1
  training-run artifact inspection and loading compatibility.
- Add a complete finalization example and extend installed-wheel smoke validation through final
  fitting, safe artifact inspection, trusted loading, and prediction.

### Reliability

- Revalidate the selected CSV and in-memory dataframe before final fitting, refuse changed
  selection manifests or estimator parameters, and never present all-row training values as
  evaluation metrics.
- Keep Milestone 9 service infrastructure out of scope; this remains a local, single-process,
  classification-only selection and fitting workflow.

### Project status

- Complete the intended local MLForge workflow from validated data through benchmarking,
  cross-validation selection, verified all-row fitting, artifact persistence, trusted loading,
  and prediction.
- Move the feature-complete portfolio project into maintenance mode. Future releases are limited
  to justified bug, security, compatibility, and documentation fixes; Milestone 9 remains
  postponed and conditional.

## [0.2.1] - 2026-08-20

### Changed

- Redesign the README around a concise value proposition, verified quick start, real benchmark
  output, reliability guarantees, architecture, current capabilities, and explicit limitations.
- Add a repository-owned terminal visual based on the bundled cross-validation example and keep
  the Python/CLI examples aligned with the public API.
- Expand CI from Ubuntu Python 3.11/3.12 to include Windows Python 3.12, including clean built-wheel
  smoke testing on both operating systems.
- Enforce an 80% statement-coverage floor through the canonical pytest configuration without
  adding coverage tooling to runtime dependencies.
- Refresh package metadata, contributor commands, support-version language, compatibility notes,
  and release-validation guidance for the public `0.2.x` line.
- Keep reusable installed-wheel validation in `scripts/wheel_smoke.py`, include the repository
  visual in source archives, and avoid one-off release tooling in the repository root.

This maintenance release does not change ML behavior, serialized schemas, or public APIs. It does
not start Milestone 9 or implement final-model fitting.

## [0.2.0] - 2026-08-18

### Added

- Public `mlforge.benchmarks` Python APIs and the `mlforge benchmark` CLI workflow for comparing a
  dummy baseline, logistic regression, and random forest on one shared classification protocol.
- Versioned, create-only holdout benchmark manifests containing the declared metric, exact split
  fingerprint, complete leaderboard, estimator timing, warnings, and terminal failure evidence.
- Deterministic stratified K-fold classification benchmarking with a shared fold plan and stable
  partition fingerprints for every estimator.
- Per-fold classification metrics plus arithmetic means and population standard deviations, with
  deterministic ranking by primary-metric mean, lower variability, and estimator identifier.
- Macro F1, macro precision, and macro recall classification metrics for training and benchmarking.

### Reliability and documentation

- Fit a fresh preprocessing and estimator pipeline inside each training fold so validation rows
  cannot influence imputation, scaling, categorical encoding, or model fitting.
- Preserve partial and complete estimator failures in strict immutable manifests, with the exact
  failing fold and partition when cross-validation cannot complete.
- Harden semantic manifest validation so recorded ranks, split dimensions, fold identities, metric
  direction, and aggregate evidence must agree rather than merely satisfy the JSON shape.
- Expand installed-package smoke testing to cover training, artifact save/inspect/load, prediction,
  holdout benchmarking, cross-validation, leaderboard ranking, and strict manifest readback.
- Improve the quick start, API reference, architecture, tutorial, compatibility policy, release
  validation record, runnable examples, and release procedure for the local benchmarking workflow.

This release remains a local, single-process tabular-ML toolkit. Cross-validation provides model
selection evidence; it does not fit a final deployment model or provide a nested-tuning estimate.

## [0.1.0] - 2026-08-17

### Added

- Strict local CSV ingestion, deterministic profiling, train/validation splitting, and leakage-safe
  numeric/categorical preprocessing.
- Classification and regression baselines with held-out metrics and immutable local run manifests.
- Versioned, integrity-checked trusted-local model artifacts and schema-validated batch inference.
- Prediction export to a create-only UTF-8 CSV with `row_number` and `prediction` columns.
- Python 3.11/3.12 CI, strict type checking, packaging validation, installed-wheel smoke testing,
  offline real-dataset validation, and a PyPI Trusted Publishing workflow.

### Distribution

- PyPI distribution: `hivmind-mlforge`
- Python import package: `mlforge`
- Console command: `mlforge`

[Unreleased]: https://github.com/HivMindAI/mlforge/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/HivMindAI/mlforge/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/HivMindAI/mlforge/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/HivMindAI/mlforge/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/HivMindAI/mlforge/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/HivMindAI/mlforge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/HivMindAI/mlforge/releases/tag/v0.1.0
