# Release Validation

MLForge release validation is deterministic and offline. Large or third-party datasets are not
committed to the repository or included in distributions.

## Automated quality gate

The default `python -m pytest` command measures the installed `mlforge` package, reports missing
statements, and fails below 80% coverage. The threshold prevents material regressions without
encouraging tests that merely chase lines. Release validation for v0.6.0 recorded 258 passing tests
at 85.36% statement coverage, including regression cross-validation, finalization, version-3
web-workspace migration, Settings and backup behavior, HTTP workflow, and browser coverage. These
figures describe the tagged v0.6.0 release; later source checkouts must be measured independently.
The portfolio-evidence source validation on 2026-10-03 recorded 260 passing tests at 85.36%
statement coverage after adding two package-metadata assertions; it does not revise the tagged
release result.

GitHub Actions runs Ruff linting, Ruff formatting checks, strict mypy, pytest with the coverage
floor, package builds, `pip check`, and the installed-wheel smoke workflow on this matrix:

| Operating system | Python |
| --- | --- |
| Ubuntu | 3.11, 3.12 |
| Windows | 3.12 |

The release workflow independently rebuilds and validates distributions from the published tag
before the protected PyPI deployment.

### Published v0.6.0 metadata boundary

PyPI release files are immutable. The artifacts published on 2026-09-12 still display their
original `Author: HivMindAI` metadata, and their embedded long description calls v0.6.0 a release
candidate. This repository update does not alter or republish those files. Source metadata now
identifies Asadullah Hussaini as author and HivMindAI as maintainer so the distinction is correct in
the next legitimate release.

The private-deployment CI job builds both containers, which runs frontend lint and the Next.js
production build, starts the Compose profile, and waits for browser-facing readiness through the
frontend-to-API proxy. The frontend Playwright job separately exercises complete classification
and regression upload-to-prediction browser journeys against temporary application state.

## Real datasets

`tests/test_real_datasets.py` exercises two datasets bundled with the installed scikit-learn
dependency:

| Workflow | Source | Rows | Validation exercised |
| --- | --- | ---: | --- |
| Classification | [Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | Numeric and derived categorical features, injected missing values, semicolon-delimited UTF-8 CSV, training, artifact loading, 80-row prediction, and CSV export |
| Regression | [Diabetes progression dataset](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) | 442 | Numeric and derived categorical features, injected missing values, pipe-delimited Latin-1 CSV with a non-ASCII column name, training, artifact loading, 60-row prediction, and CSV export |

The tests call `sklearn.datasets.load_breast_cancer` and `sklearn.datasets.load_diabetes`; they do
not download data and do not redistribute dataset files in the MLForge wheel or source archive.
The derived categorical columns and missing values are deterministic test transformations used to
exercise supported input behavior.

## Portfolio screenshot workflow

The README screenshots record one real Core 0.6.0 application run. The ignored CSV files were
generated with `scripts/generate_portfolio_demo_data.py` using seed `20260817`; the generation
recipe is distributed, while the generated synthetic data remains local.

| Evidence | Verified value |
| --- | --- |
| Training CSV | `data/large_customer_churn_train.csv` — 25,000 rows |
| Training SHA-256 | `26a62f0ab5e320f29b192bfbeaa7b0b3c5343b25a3d25c6fb0d9e40e372bf412` |
| Target | `churn` — binary classification |
| Models | Logistic Regression, Random Forest Classifier, Dummy Classifier |
| Validation | 5 shared stratified folds; 20,000 train and 5,000 validation rows per fold |
| Ranking metric | Balanced Accuracy |
| Rank-one result | Logistic Regression — 65.54% mean, 0.50% population standard deviation |
| Prediction CSV | `data/large_customer_churn_predict.csv` — 2,500 target-free rows |
| Prediction SHA-256 | `0f20400813adb4da9bf07c5a73ebbda4418691d1f66fb1b55ed622dca02f9131` |
| Prediction result | 2,500 rows processed; 0 invalid rows |

The screenshots are application output, not mockups. The underlying generated CSVs, temporary web
workspace, benchmark manifest, finalized artifact, and prediction output are intentionally ignored
and excluded from distributions.

## v0.2.0 benchmark release-candidate matrix

The v0.2.0 Milestone 7-8 release-candidate review also ran the installed wheel, outside the source
tree, against three materially different scikit-learn classification datasets. Every estimator
completed every shared three-fold partition, each immutable manifest survived strict readback, and
the Iris workflow produced identical fold fingerprints, ranks, parameters, and metric evidence
when repeated with the same seed.

| Dataset | Rows | Features | Input characteristics | Winning default estimator |
| --- | ---: | ---: | --- | --- |
| Iris | 150 | 4 | Numeric, three classes | Logistic regression |
| Wine | 178 | 14 | Numeric plus a derived categorical band, three classes | Logistic regression |
| Wisconsin diagnostic breast cancer | 569 | 31 | Binary imbalance, derived categorical band, 59 injected missing cells | Logistic regression |

These checks use deterministic local transformations and do not download or redistribute the
datasets. The observed winners are validation evidence for these specific partitions, not a claim
that one estimator is universally best.

## Large output

The prediction-output test writes and verifies 25,000 structured prediction rows. This validates
the file-output path without printing an impractical result payload to the terminal. Generated
CSVs, run records, and model artifacts remain temporary or ignored.

## Clean package workflow

Before a release, build the wheel and source archive, install each archive into its own newly
created environment outside the source tree, run `pip check`, and execute `scripts/wheel_smoke.py`
without relying on repository package imports. The smoke script validates package metadata,
`import mlforge`, the `mlforge` module entrypoint, validated ingestion, training, artifact
persistence and inspection, explicit trusted loading, prediction, holdout benchmarking,
cross-validated benchmarking, ranking, explicit all-row final fitting, final-model artifact
lineage, immutable manifest persistence, and strict readback.
