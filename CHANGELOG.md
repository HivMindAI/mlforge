# Changelog

All notable MLForge changes are recorded here. The project follows semantic versioning during its
`0.y.z` development series as described in [the compatibility policy](docs/compatibility.md).

## [Unreleased]

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

[Unreleased]: https://github.com/HivMindAI/mlforge/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/HivMindAI/mlforge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/HivMindAI/mlforge/releases/tag/v0.1.0
