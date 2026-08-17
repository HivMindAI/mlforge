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

## Large output

The prediction-output test writes and verifies 25,000 structured prediction rows. This validates
the file-output path without printing an impractical result payload to the terminal. Generated
CSVs, run records, and model artifacts remain temporary or ignored.

## Clean package workflow

Before a release, build the wheel and source archive, install the wheel into a newly created
environment outside the source tree, run `pip check`, and execute `scripts/wheel_smoke.py` from a
different working directory. The smoke script validates package metadata, `import mlforge`, the
`mlforge` module entrypoint, profiling, training, artifact persistence and inspection, explicit
trusted loading, and prediction without relying on repository imports.
