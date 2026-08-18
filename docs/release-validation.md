# Release Validation

MLForge release validation is deterministic and offline. Large or third-party datasets are not
committed to the repository or included in distributions.

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
cross-validated benchmarking, ranking, immutable manifest persistence, and strict readback.
