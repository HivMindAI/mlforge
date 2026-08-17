# Changelog

All notable MLForge changes are recorded here. The project follows semantic versioning during its
`0.y.z` development series as described in [the compatibility policy](docs/compatibility.md).

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

[0.1.0]: https://github.com/HivMindAI/mlforge/releases/tag/v0.1.0
