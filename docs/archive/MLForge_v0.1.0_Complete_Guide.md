<!-- Maintainer source. Release v0.1.0 at commit c30dbf55. Build with tmp/pdfs/build_mlforge_guide.py. -->

> **Historical documentation:** This guide describes MLForge v0.1.0 as it existed at commit
> `c30dbf55`. It is preserved for reference and learning, but it does not describe the current
> v0.6.0 product. Use the repository README and current files under `docs/` for supported behavior.

# Chapter 1 - MLForge in One Page

## What it is

MLForge v0.1.0 is a local, security-conscious Python framework for building reproducible supervised machine-learning baselines from tabular CSV data. It gives one coherent path from validated bytes to a trained scikit-learn pipeline, an immutable experiment record, a portable artifact, and schema-checked batch predictions.

Its most important design decision is scope. MLForge is not a hosted platform, notebook environment, workflow scheduler, online prediction server, distributed trainer, or general model marketplace. It is a small framework with strict boundaries around a practical local workflow. That makes it understandable enough to learn from and dependable enough to use for controlled baseline experiments.

## Who it is for

- Python developers and data scientists who want a repeatable tabular baseline without assembling the same ingestion, split, preprocessing, evaluation, and persistence code each time.
- Learners who want to see how production concerns such as lineage, atomic writes, trust boundaries, and package releases fit around scikit-learn.
- Small teams that need inspectable local experiment records and artifacts before they adopt a larger tracking or serving platform.
- Open-source contributors who value explicit APIs, deterministic metadata, actionable errors, and a tested release process.

## The problem it solves

A short notebook can fit a model, but it often leaves critical questions unanswered: Which exact file was used? Was the validation set allowed to influence preprocessing? Which seed and split produced the score? Can two runs be compared fairly? Does a saved model match the current dependency environment? Can prediction input silently drift? MLForge makes those concerns part of the workflow rather than optional afterthoughts.

<!-- diagram:workflow -->

## Implemented capabilities

| Area | v0.1.0 behavior |
| --- | --- |
| Data | Strict local CSV ingestion, target selection, metadata, SHA-256, profiling |
| Split | Classification or regression, deterministic holdout, optional stratification |
| Preprocessing | Numeric imputation and scaling, categorical imputation and one-hot encoding |
| Models | Logistic regression, ridge regression, random-forest classifier and regressor |
| Evaluation | Three task-appropriate validation metrics with direction metadata |
| Runs | Immutable local JSON manifests with configuration, lineage, environment, and status |
| Comparison | Fair comparison only when dataset, split, target, task, and partition match |
| Artifacts | Versioned `.mlforge` ZIP, integrity checks, explicit trust before pickle loading |
| Inference | Exact feature-schema validation and atomic batch prediction CSV output |
| Interfaces | `mlforge` CLI and deliberately explicit domain-level Python APIs |
| Engineering | 177 tests, strict lint/type checks, wheel smoke test, trusted PyPI publishing |

## What it intentionally does not do

- Automated task selection, feature engineering, hyperparameter search, cross-validation, or ensembling.
- Database, cloud object storage, remote experiment server, access control, or multi-user concurrency service.
- Real-time serving, REST API, monitoring, model promotion stages, or deployment orchestration.
- Safe loading of artifacts from untrusted sources. The payload is Python pickle and can execute code.
- Compatibility across arbitrary dependency versions. Artifact loading requires the recorded environment exactly.

> Note: The correct mental model is a dependable local baseline lifecycle, not a miniature cloud ML platform.

## Release identity

The Python distribution is named `hivmind-mlforge`; the import package and console command are named `mlforge`. Version `0.1.0` requires Python 3.11 or newer, is licensed under Apache-2.0, and declares only pandas and scikit-learn as production dependencies.

# Chapter 2 - The Actual Workflow, End to End

This chapter traces one supported classification run. The regression path is identical except for task validation, estimator choice, stratification, and metrics.

## Step 1: load validated CSV bytes

```python
from mlforge.datasets import load_csv

dataset = load_csv("training.csv", target="churn")
```

`load_csv` resolves a real regular `.csv` file, checks its configured size limit, hashes its raw bytes, validates row structure with Python's strict CSV parser, and then loads it with pandas. It verifies that the file did not change during reading and returns a `LoadedDataset` containing the frame and immutable metadata.

The target remains in `dataset.frame` at this point. Missing values are intentionally preserved so profiling and preprocessing can account for them. A missing target column, blank or duplicate header, malformed row, null byte, empty data section, wrong extension, or size violation becomes a domain error.

## Step 2: inspect quality before choosing configuration

```python
from mlforge.datasets import profile_dataset

profile = profile_dataset(dataset)
print(profile.to_json())
```

The profile reports row and column counts, inferred column kinds, missingness, uniqueness, finite numeric summaries, likely identifiers, high cardinality, target frequencies, imbalance, and a conservative task hint. The hint informs the human; it does not choose the training task.

## Step 3: define the experiment explicitly

```python
from mlforge.pipelines import TaskType
from mlforge.training import LOGISTIC_REGRESSION, TrainingConfig

config = TrainingConfig(
    task=TaskType.CLASSIFICATION,
    estimator=LOGISTIC_REGRESSION,
)
```

The configuration combines task, estimator, split settings, preprocessing settings, and feature overrides. Incompatible task-estimator pairs fail before training.

## Step 4: train and record the run

```python
from pathlib import Path
from mlforge.runs import LocalRunStore
from mlforge.training import train

runs = LocalRunStore(Path(".mlforge/runs"))
result = train(dataset, config, run_store=runs)
```

<!-- diagram:training -->

`train` creates a UUID and UTC timestamps, records a running manifest, validates the target, performs the deterministic split, infers feature roles from training features, constructs an unfitted pipeline, fits only on training rows, predicts validation rows, calculates metrics, captures warnings, and persists a successful terminal manifest. Expected failures are also persisted with structured failure information before `TrainingFailedError` is raised.

The returned `TrainingResult` holds both the fitted in-memory pipeline and its successful `RunManifest`. The manifest does not contain the model object.

## Step 5: save a versioned model artifact

```python
from mlforge.artifacts import LocalArtifactStore

artifacts = LocalArtifactStore(Path(".mlforge/artifacts"))
saved = artifacts.save(result)
print(saved.path)  # .../<run-id>.mlforge
```

Saving requires a successful result whose exact manifest is already present in the run store. MLForge verifies the fitted pipeline schema, serializes with pickle protocol 5, hashes the payload, embeds a canonical manifest and environment contract, and atomically publishes a create-only `.mlforge` archive.

## Step 6: inspect, then explicitly trust

```python
from mlforge.artifacts import inspect_artifact, load_artifact

manifest = inspect_artifact(saved.path)  # no pickle execution
loaded = load_artifact(saved.path, trusted=True)
```

Inspection validates archive structure and payload integrity without deserializing the model. Loading is a separate operation because pickle is executable. It requires `trusted=True`, verifies exact dependency versions, then deserializes and checks that the result is a fitted scikit-learn pipeline with the recorded feature names.

> Warning: `trusted=True` is not a security scanner. It is the caller's explicit statement that the artifact source has been verified and is trusted to execute code.

## Step 7: predict with an exact schema

```python
from mlforge.inference import predict_csv, write_predictions_csv

predictions = predict_csv(loaded, "prediction.csv")
output = write_predictions_csv(predictions, "predictions.csv")
```

Prediction CSV loading reuses strict ingestion rules but has no target. MLForge requires exactly the recorded feature set, restores training column order, checks numeric and categorical values against their roles, normalizes outputs to finite JSON-safe scalars, and writes `row_number,prediction` to a new file atomically.

## The state produced by one run

| State | Contents | Mutability |
| --- | --- | --- |
| Source CSV | Original training bytes | External to MLForge |
| `LoadedDataset` | pandas frame plus content and schema metadata | Frozen wrapper; frame is validated again |
| `DatasetSplit` | Train and validation partitions with original indices | Frozen wrapper around copies |
| `TrainingResult` | Fitted pipeline plus terminal manifest | In memory |
| Run JSON | Reproducibility and lineage record | Create-only |
| `.mlforge` file | Manifest plus serialized fitted pipeline | Create-only |
| Prediction CSV | Ordered row numbers and predictions | Create-only |

# Chapter 3 - Repository Structure and File Responsibilities

## Top-level map

```text
mlforge/
|-- src/mlforge/          Importable production package
|-- tests/                Unit, integration, contract, and release tests
|-- examples/             Four runnable workflows plus small CSV fixtures
|-- docs/                 Architecture, API, tutorial, security, release docs
|-- scripts/              Release-tag and installed-wheel validation
|-- .github/workflows/    Continuous integration and trusted publishing
|-- pyproject.toml        Build, dependency, lint, type, and test configuration
|-- README.md             Product entry point and quick start
|-- ROADMAP.md            Implemented v0.1 scope and future direction
|-- CONTRIBUTING.md       Contributor workflow
|-- SECURITY.md           Vulnerability reporting and artifact warning
|-- CHANGELOG.md          Release history
|-- LICENSE               Apache License 2.0
|-- MANIFEST.in           Additional source-distribution files
`-- .env.example          Documented local environment variables
```

## Production package

| Path | Responsibility |
| --- | --- |
| `src/mlforge/__init__.py` | Package version and minimal top-level export |
| `src/mlforge/__main__.py` | `python -m mlforge` entry point |
| `src/mlforge/cli.py` | Argument parsing, human/JSON rendering, exit-code boundary |
| `src/mlforge/config.py` | Frozen application configuration and environment parsing |
| `src/mlforge/errors.py` | Stable domain exception hierarchy |
| `src/mlforge/logging_config.py` | Explicit, idempotent package logging setup |
| `src/mlforge/datasets/` | CSV ingestion, loaded-data types, validation, profiling |
| `src/mlforge/pipelines/` | Task/split/preprocessing types and leakage-safe construction |
| `src/mlforge/training/` | Estimator registry, evaluation, training orchestration |
| `src/mlforge/runs/` | Versioned run manifests, local store, fair comparison |
| `src/mlforge/artifacts/` | Artifact manifest, archive validation, save/inspect/load |
| `src/mlforge/inference.py` | Frame/CSV prediction and atomic output writing |
| `src/mlforge/py.typed` | PEP 561 marker for inline type information |

Each domain package exposes an explicit `__all__`. The top-level `mlforge` package deliberately exports only `__version__`; users import domain concepts from `mlforge.datasets`, `mlforge.training`, and related modules.

## Tests by concern

| Test file | Contract protected |
| --- | --- |
| `test_dataset_ingestion.py` | Paths, encodings, delimiters, size, malformed CSV, target handling |
| `test_dataset_profiling.py` | Quality signals, finite summaries, deterministic JSON, mutation checks |
| `test_splitting.py` | Determinism, stratification feasibility, target validation, disjointness |
| `test_preprocessing.py` | Role inference, leakage prevention, missing/unseen values, drift |
| `test_training.py` | All estimators, metrics, reproducibility, failure persistence |
| `test_runs.py` | Schema invariants, atomic storage, corruption, comparison fairness |
| `test_artifacts.py` | Trust, integrity, archive limits, compatibility, immutability |
| `test_inference.py` | Exact schemas, numeric safety, atomic output, 25,000-row result |
| `test_cli.py` | Command behavior, JSON/human output, exit codes, full workflow |
| `test_public_api.py` | Exact exports and stable keyword-capable parameter names |
| `test_real_datasets.py` | Breast-cancer classification and diabetes regression workflows |
| `test_examples.py` | All four documented source examples run |
| `test_package.py` | Metadata, entry point, dependencies, license, typing, side effects |
| `test_documentation.py` | Required docs, local links, Python-block syntax |
| `test_config.py`, `test_logging_config.py` | Foundation behavior |

## Documentation and release files

`docs/architecture.md` is the concise architectural authority. `docs/api.md` records the public compatibility surface. `docs/tutorial.md` teaches the supported workflow. `docs/security.md` and root `SECURITY.md` explain operational and disclosure concerns. `docs/compatibility.md` defines version expectations. `docs/releasing.md` and `docs/release-validation.md` describe how tagged source becomes an installed, tested wheel and PyPI release.

> Note: This guide adds depth but does not replace those focused maintainer documents. When behavior changes, update the source, tests, focused docs, and this guide together.

# Chapter 4 - Architecture

## Design goals

The architecture optimizes for four qualities: simple flow, explicit boundaries, testability, and safe failure. It uses plain frozen data classes and small functions rather than a dependency-injection framework or large object graph. The filesystem is a local persistence adapter, pandas is the tabular boundary, and scikit-learn is the estimator/pipeline boundary.

<!-- diagram:architecture -->

## Layers and dependency direction

The CLI depends on domain APIs; domain APIs do not depend on CLI parsing or output formatting. Training orchestrates datasets, pipelines, runs, and scikit-learn. Inference depends on a loaded artifact and dataset validation. Foundation modules provide errors, configuration, and logging without importing the higher layers.

There are no intentional circular dependencies. Data crosses package boundaries as typed values: `LoadedDataset`, `DatasetSplit`, `TrainingConfig`, `TrainingResult`, `RunManifest`, `LoadedArtifact`, and `PredictionResult`.

## Major interfaces

| Interface | Input | Output | Boundary enforced |
| --- | --- | --- | --- |
| `load_csv` | Path, target, CSV options | `LoadedDataset` | Filesystem and CSV structure |
| `profile_dataset` | `LoadedDataset` | `DatasetProfile` | Loaded-frame integrity |
| `split_dataset` | Dataset, explicit task, split config | `DatasetSplit` | Target semantics and holdout |
| `build_preprocessor` | Split and role configuration | `ColumnTransformer` | Training-derived feature roles |
| `train` | Dataset, config, run store | `TrainingResult` | Full training lifecycle |
| `compare_runs` | Manifests and metric | `RunComparison` | Fair-comparison identity |
| `LocalArtifactStore.save` | Successful training result | `SavedArtifact` | Lineage and fitted-schema consistency |
| `inspect_artifact` | Artifact path | `ArtifactManifest` | Safe structural/integrity inspection |
| `load_artifact` | Path and explicit trust | `LoadedArtifact` | Executable trust and environment match |
| `predict_frame` | Loaded artifact and DataFrame | `PredictionResult` | Exact feature schema and values |

## Data flow versus control flow

Data flow carries bytes, frames, indices, schemas, model state, manifests, and predictions. Control flow carries explicit user choices such as task, estimator, seed, validation fraction, feature overrides, trust, and output location. MLForge does not hide these choices in module globals.

## Extension points

The current extension points are deliberately narrow:

- Add an estimator constant and factory in `training/estimators.py`, then add it to the task-specific registry and compatibility tests.
- Add a metric in `training/evaluation.py`, including its direction in the serialized `MetricValue`.
- Extend preprocessing through `PreprocessingConfig` or `FeatureOverrides` without examining validation values.
- Add a persistence backend behind a new explicit store class, preserving manifest invariants and create-only semantics.
- Add a new artifact schema version with an explicit reader/migration policy; never silently reinterpret version 1.

## Why there is no generic plugin system

A plugin loader would add discovery rules, version negotiation, security concerns, and a new compatibility surface. Version 0.1.0 has only four estimators and one local lifecycle, so ordinary Python modules and explicit registries are simpler. A plugin architecture becomes justified only when real independent extensions cannot be maintained cleanly through these existing seams.

# Chapter 5 - Dataset Management

## `CsvLoadOptions`

CSV behavior is explicit and validated. Defaults are UTF-8 with optional byte-order mark (`utf-8-sig`), comma delimiter, and a 100 MiB maximum. Encoding and delimiter must be nonblank strings; the delimiter must be one character; the size limit must be a positive integer.

```python
from mlforge.datasets import CsvLoadOptions, load_csv

options = CsvLoadOptions(
    encoding="latin-1",
    delimiter="|",
    max_file_size_bytes=20 * 1024 * 1024,
)
dataset = load_csv("diabetes.csv", target="progression", options=options)
```

## Filesystem validation

The path is expanded and strictly resolved. The target must exist, be a regular file, have the exact `.csv` suffix case-insensitively, and remain within the configured size bounds. The loader rejects directories and unusable paths with `DatasetPathError`.

MLForge hashes raw bytes with SHA-256. Hashing establishes a stable identity for comparison and lineage; it does not prove who created the file.

## Structural CSV validation

Before pandas interprets types, the standard-library CSV parser checks:

- The decoded file is nonempty and contains no null bytes.
- There is one nonblank header row.
- Header names are nonblank and unique.
- Every data row has exactly the header width.
- At least one data row exists.
- Decoding and CSV parsing complete strictly.

Pandas then loads the frame. MLForge compares the resulting columns with the validated header and checks file size and modification time before and after the load. This reduces time-of-check/time-of-use ambiguity for a file being modified concurrently.

## Training CSV versus feature CSV

`load_csv(path, target=...)` requires one named target present exactly once. `load_feature_csv(path, options=...)` applies the same structural protections but records no target; it is used for inference. A target-looking column in prediction input is simply an extra feature and will be rejected by the artifact schema.

## Dataset metadata

`DatasetMetadata` records the resolved path, file SHA-256, byte size, encoding, delimiter, target, row count, column count, and column metadata including pandas dtype. This metadata is later snapshotted into the run manifest.

## Mutation defense

`LoadedDataset` contains a mutable pandas object because useful pandas operations require it. Before downstream behavior, `validate_loaded_dataset` checks that shape, columns, dtypes, and target still match the immutable metadata. It catches accidental mutation of the loaded object; it is not a cryptographic proof of every cell value in memory.

## Profiling semantics

Numeric columns receive finite-value minimum, maximum, mean, median, and standard deviation summaries. String, boolean, and categorical columns receive type, missingness, distinct count, and quality flags. A column is high-cardinality when it has at least 50 distinct values and at least half as many distinct values as rows. A complete, unique, ID-named column can be flagged as a likely identifier.

Target profiling records frequencies and missingness. An imbalance flag is raised when the smallest nonzero class count divided by the largest is below 0.2. The task hint is conservative and can be `classification`, `regression`, or `undetermined`.

> Warning: Profiling describes data; it does not clean it, select a task, remove identifiers, or decide whether an apparent class imbalance is acceptable.

## Common failures

| Symptom | Likely cause | Correct response |
| --- | --- | --- |
| `DatasetPathError` | Missing, directory, non-CSV, oversize | Correct the path/options; do not bypass validation |
| `DatasetFormatError` | Bad encoding, row width, header, null byte | Repair or export the source CSV consistently |
| `DatasetValidationError` | Target invalid or loaded frame mutated | Use the correct target or reload clean data |
| Unexpected string dtype | Mixed values in one CSV column | Clean source values or use an explicit categorical override |
| Task hint undetermined | Ambiguous target semantics | Choose task from domain knowledge |

# Chapter 6 - Preprocessing and Leakage Prevention

## The central rule

Any learned preprocessing state must come only from training rows. Validation rows may be transformed by learned state, but they must not influence imputation values, scales, category vocabularies, or feature-role decisions.

MLForge enforces the sequence: split first, infer schema from training features, build an unfitted transformer, fit the whole model pipeline on training rows, then transform and predict validation rows.

## Feature roles

`infer_feature_schema` maps physical pandas dtypes to two modeling roles:

- Numeric: real numeric dtypes, excluding boolean and complex values.
- Categorical: boolean, string, object, or pandas categorical dtypes.

Datetime features are rejected because silently converting dates or treating timestamps as categories would encode an unreviewed feature-engineering policy. The user must transform them before MLForge or extend the framework explicitly.

`FeatureOverrides` can force listed columns to numeric or categorical roles. Unknown columns, duplicate assignments, and incompatible forced-numeric columns fail with actionable errors. An all-missing object column may need a categorical override because its observed dtype is ambiguous.

## Numeric pipeline

Numeric features use `SimpleImputer` with mean or median strategy and `keep_empty_features=True`. Optional standardization uses `StandardScaler`. Infinite numeric inputs are rejected before fit and at prediction; missing numeric values are allowed for imputation.

## Categorical pipeline

Categorical values are normalized to object form. Missing values are filled with the reserved marker `__mlforge_missing__`, but only after MLForge verifies that the marker does not already occur in split data. `OneHotEncoder(handle_unknown="ignore")` lets unseen validation or prediction categories produce all-zero columns for that feature instead of crashing.

## Column composition

`ColumnTransformer` applies the numeric and categorical pipelines to their recorded columns and drops all remainder columns. `build_model_pipeline` clones the estimator and returns an unfitted `Pipeline` with two steps: the preprocessor and the estimator.

```python
from mlforge.pipelines import (
    FeatureOverrides,
    NumericImputationStrategy,
    PreprocessingConfig,
    build_preprocessor,
)

preprocessor = build_preprocessor(
    split,
    config=PreprocessingConfig(
        numeric_imputation=NumericImputationStrategy.MEDIAN,
        scale_numeric=True,
    ),
    overrides=FeatureOverrides(categorical=("postal_code",)),
)
```

## Evidence that leakage is prevented

`tests/test_preprocessing.py` constructs validation values that would noticeably change an imputation statistic if they were included. The test fits the transformer only on training rows and confirms the learned statistic matches training data. Other tests verify unseen categories, all-missing columns, marker collisions, unsupported datetimes, mutated split alignment, infinite numerics, and non-scikit estimators.

## Common mistakes

- Calling `fit_transform` on the full dataset before splitting.
- Inferring feature roles using combined train and validation values.
- Treating integer-coded categories as automatically categorical without an override.
- Reusing a transformer already fitted on unrelated data.
- Adding a new imputer that drops all-missing columns and breaks the recorded schema.
- Allowing prediction code to accept extra columns because the estimator ignores them.

# Chapter 7 - Model Management

## Supported estimators

| Constant | Task | scikit-learn estimator | Key defaults |
| --- | --- | --- | --- |
| `LOGISTIC_REGRESSION` | Classification | `LogisticRegression` | `max_iter=1000`, seeded |
| `RANDOM_FOREST_CLASSIFIER` | Classification | `RandomForestClassifier` | 100 trees, `n_jobs=1`, seeded |
| `RIDGE_REGRESSION` | Regression | `Ridge` | `alpha=1.0` |
| `RANDOM_FOREST_REGRESSOR` | Regression | `RandomForestRegressor` | 100 trees, `n_jobs=1`, seeded |

The constants are stable string identifiers stored in run manifests. Estimator objects are created by factories so every run gets fresh model state. Random-state-bearing estimators receive the split seed. Forests use one job to reduce nondeterminism and avoid surprising resource use.

## Configuration contract

`TrainingConfig` requires an explicit `TaskType` and estimator identifier. It also carries `SplitConfig`, `PreprocessingConfig`, and `FeatureOverrides`. Construction validates the task-estimator pairing; `ridge-regression` cannot silently train a classification target, and a classifier cannot be used for regression.

## Why only four models

The release proves the lifecycle for a linear and nonlinear baseline in each task. More estimators would expand hyperparameter schema, dependency support, serialization compatibility, documentation, and tests. A model belongs in core only when its behavior can be configured explicitly and supported across training, manifest capture, artifact round-trip, CLI choices, examples, and compatibility tests.

## Estimator parameters and lineage

After the estimator is created, its parameters are normalized into ordered `RunParameter` values and recorded in the manifest. This makes defaults observable. It does not yet expose every parameter through the CLI; changing internal defaults is therefore a behavior and reproducibility decision that should be called out in release notes.

## Adding an estimator safely

1. Add a stable identifier and factory in `src/mlforge/training/estimators.py`.
2. Put it in exactly one task registry and update `ALL_ESTIMATORS`.
3. Decide deterministic seed and resource-use defaults.
4. Ensure the preprocessing output is compatible with the estimator.
5. Extend CLI choices if it should be a CLI feature.
6. Add training, public API, serialization, installed-wheel, and documentation coverage.
7. Run the full quality gate and verify an artifact can be loaded in a clean environment.

# Chapter 8 - Training and Evaluation

## Target validation

Classification targets must be complete and contain at least two classes. Numeric classification values must be finite. Regression targets must be real numeric, non-boolean, complete, and finite. A supervised dataset must contain at least one feature column.

## Split behavior

`SplitConfig` contains validation fraction, random seed, and optional stratification choice. Classification stratifies by default. Regression is unstratified and rejects a request for target stratification. MLForge checks whether every class has enough observations and whether train and validation partitions have room for each class before calling scikit-learn.

Original DataFrame indices are preserved and recorded. Partitions must be disjoint and cover the source rows. A classification training partition with one class is rejected even when an explicitly unstratified tiny split technically succeeds.

## Evaluation metrics

| Task | Metric | Direction | Interpretation |
| --- | --- | --- | --- |
| Classification | `accuracy` | Higher is better | Fraction of all labels predicted correctly |
| Classification | `balanced_accuracy` | Higher is better | Mean recall across classes |
| Classification | `f1_weighted` | Higher is better | Class-frequency-weighted F1 |
| Regression | `mean_absolute_error` | Lower is better | Average absolute prediction error |
| Regression | `root_mean_squared_error` | Lower is better | Square-root average squared error |
| Regression | `r2` | Higher is better | Variance explained relative to a mean baseline |

Metrics must be finite. Regression evaluation requires at least two validation rows because meaningful `r2` needs more than one observation. Metric records are sorted and carry direction so run comparison does not guess whether smaller is better.

## Reproducibility guarantee

With the same source bytes, target, task, configuration, dependency environment, and code version, seeded workflows are designed to reproduce the split, metrics, and predictions. The test suite verifies this for fixed seeds. This is a controlled local reproducibility promise, not a claim of bit-for-bit equality across arbitrary hardware, operating systems, BLAS implementations, or different dependency versions.

## Failure lifecycle

Training creates a run identity before model work. Expected domain, value, type, or overflow failures after that point produce a failed terminal manifest with timestamps and a structured failure record, then raise `TrainingFailedError`. A run-store failure is not hidden as a model failure because losing the audit record is itself a storage problem the caller must see.

> Verified: A failed run is observable. The framework does not return a fake result or silently skip evaluation to make the command appear successful.

# Chapter 9 - Runs, Reproducibility, and Lineage

## `RunManifest`

The version 1 manifest is a strict, frozen record. It includes run UUID, schema version, status, ordered UTC timestamps, complete run configuration, dataset snapshot, environment snapshot, split snapshot, estimator parameters, metrics, warnings, and failure details when applicable.

Status and fields obey terminal invariants. A successful run must have metrics and no failure. A failed run must have failure information and no success metrics. Warnings are unique and nonblank; metrics are uniquely named and sorted.

## Dataset identity

The dataset snapshot includes the raw source SHA-256, resolved source path, target, shape, and dtype-level schema. SHA-256 identifies bytes, not semantic equivalence: two CSV files containing the same table with different line endings are different inputs by design.

## Partition identity

The split snapshot records requested fraction and seed, actual stratification, row counts, and a partition SHA-256. The partition hash is computed from canonical JSON containing the exact original source indices assigned to train and validation. This prevents two runs with the same fraction and seed but different row order or split behavior from being treated as directly comparable.

## Environment identity

The manifest records exact versions of Python, MLForge, pandas, NumPy, SciPy, and scikit-learn. This information supports diagnosis and artifact compatibility. It is not a complete lockfile for every transitive package or native library.

## Local storage

`LocalRunStore` uses one canonical UUID filename per manifest. JSON is encoded canonically in UTF-8 and limited to 1 MiB on read. The store rejects symlinks, unexpected paths, malformed data, unsupported schema versions, and invalid run IDs.

Publishing is create-only and atomic: a temporary file is written and flushed, then a hard link creates the destination only if it does not exist. The temporary file is cleaned. An existing run can never be silently overwritten.

## Fair comparison

```python
from mlforge.runs import LocalRunStore, compare_runs

store = LocalRunStore(".mlforge/runs")
manifests = [store.read(run_id) for run_id in selected_ids]
ranking = compare_runs(manifests, metric="balanced_accuracy")
```

Comparison requires at least two distinct successful runs and rejects mismatches in task, dataset hash, target, validation fraction, seed, actual stratification, or exact partition hash. The requested metric must exist with consistent direction. Ties are broken by run UUID for deterministic output.

## User responsibility

Preserve the source CSV, record upstream preparation, and keep the release and environment available. Matching metrics do not prove matching model state, and repeatedly choosing models against one holdout introduces selection bias even when every individual run is reproducible.

# Chapter 10 - Artifacts and Serialization

## Artifact format

An MLForge v0.1 artifact is a regular file named `<run-id>.mlforge`. It is a ZIP archive with exactly two uncompressed members:

```text
manifest.json
pipeline.pkl
```

The manifest schema version is 1 and the serialization label is `pickle-protocol-5`. The manifest is limited to 1 MiB and the pipeline payload to 1 GiB. Directory entries, symlinks, encrypted members, compression, extra members, duplicates, and wrong member names are rejected.

## Manifest contents

`ArtifactManifest` binds the artifact to run UUID, task, target, exact ordered feature schema and roles, categorical missing marker, estimator context, exact environment, canonical run-manifest hash, pipeline byte size, and pipeline SHA-256.

## Saving

Saving requires a successful `TrainingResult`. MLForge reads the persisted run manifest and requires it to match the in-memory manifest exactly. It verifies the fitted scikit-learn pipeline and its feature names before pickling. The destination is create-only and published atomically using the same temporary-file and hard-link pattern as runs.

## Inspection versus loading

<!-- diagram:artifact-trust -->

`inspect_artifact` performs path, ZIP, member, size, manifest, and payload-hash validation without unpickling. Use it for metadata display and triage.

`load_artifact(path, trusted=True)` repeats structural and integrity validation, requires exact versions for Python and the scientific stack, deserializes the payload, and verifies it is a fitted scikit-learn `Pipeline` with matching feature names. A scikit-learn inconsistent-version warning is elevated to an artifact compatibility error.

## Integrity is not authenticity

The embedded SHA-256 detects accidental or malicious payload changes only when the attacker cannot also rewrite the manifest. Because both are in the same file, it does not prove publisher identity. GitHub/PyPI provenance, a trusted transport channel, a separately distributed signature, controlled storage, or an organization-specific hash allowlist is needed for authenticity.

## Threat model

| Threat | v0.1.0 response | Residual responsibility |
| --- | --- | --- |
| Accidental truncation | ZIP and size validation | Preserve backups |
| Payload modification | SHA-256 mismatch | Protect manifest and distribution channel |
| ZIP bombs/layout tricks | Exact members and bounded stored entries | Keep configured limits appropriate |
| Path traversal in run IDs | UUID canonicalization | Do not disable path checks |
| Untrusted pickle | Explicit trust gate | Never trust unknown artifacts |
| Dependency drift | Exact environment check | Rebuild/retrain under supported versions |
| Concurrent overwrite | Create-only atomic publication | Use suitable local filesystem semantics |

# Chapter 11 - Inference and Prediction

## Accepted artifact

Inference accepts only `LoadedArtifact`, not a path, raw pipeline, or merely inspected manifest. This makes the trust decision and environment validation a precondition that is visible in the type-level workflow.

## Schema contract

The prediction frame must have unique columns and exactly the recorded feature names. Missing and extra columns are both errors. MLForge restores the recorded training order before invoking the pipeline.

Numeric roles require real numeric, non-boolean, non-complex values; finite nonmissing values are mandatory. Categorical roles reject complex or nonscalar cell values and reject collision with the reserved missing marker. Missing values remain supported for the recorded imputers.

## Output normalization

Estimator outputs are converted to Python string, integer, float, or boolean scalars suitable for strict JSON. Non-finite floating predictions fail instead of leaking `NaN` or infinity into JSON or CSV.

`PredictionResult` records the artifact run ID, optional resolved source path, row count, and ordered `PredictionRecord` values. Row numbers are stable zero-based positions in inference input, not original pandas labels.

## CSV output guarantees

`write_predictions_csv` accepts only a `.csv` output path, refuses symlinks and existing destinations, creates missing parent directories, writes UTF-8 with columns `row_number,prediction`, flushes the data, and publishes atomically. A racing destination is preserved rather than overwritten.

## Example

```python
from mlforge.artifacts import load_artifact
from mlforge.inference import predict_csv, write_predictions_csv

artifact = load_artifact("artifacts/<run-id>.mlforge", trusted=True)
result = predict_csv(artifact, "new_customers.csv")
path = write_predictions_csv(result, "outputs/churn_predictions.csv")
print(result.to_json())
```

## Operational limitations

This is batch inference in the current process. There is no request server, batching scheduler, concurrency controller, latency objective, feature store, monitoring sink, or model rollout mechanism. A production service should wrap the public inference API while adding authentication, resource limits, observability, deployment controls, and a separately reviewed artifact trust policy.

# Chapter 12 - Command-Line Interface

## Installation and discovery

```powershell
py -m pip install hivmind-mlforge==0.1.0
mlforge --version
mlforge --help
```

The console script maps `mlforge` to `mlforge.cli:main`. `python -m mlforge` executes the same main function. With no subcommand, help is printed and the command exits successfully.

## Command tree

```text
mlforge
|-- dataset profile PATH --target TARGET [CSV options] [--json]
|-- train PATH --target TARGET --task TASK --estimator NAME
|   [split and preprocessing options] [--runs-dir DIR]
|   [--artifacts-dir DIR] [--json]
|-- runs list [--runs-dir DIR] [--json]
|-- runs show RUN_ID [--runs-dir DIR] [--json]
|-- runs compare RUN_ID RUN_ID [...] --metric NAME [--json]
|-- artifacts inspect ARTIFACT [--json]
`-- predict ARTIFACT PATH [CSV options] --trust-artifact
    [--output PATH] [--json]
```

## Profile data

```powershell
mlforge dataset profile examples/customer_churn.csv `
  --target churn --json
```

## Train and create an artifact

```powershell
mlforge train examples/customer_churn.csv `
  --target churn `
  --task classification `
  --estimator logistic-regression `
  --runs-dir .mlforge/runs `
  --artifacts-dir .mlforge/artifacts `
  --json
```

Supplying `--artifacts-dir` requests artifact saving after successful training. The output contains the run identity, terminal manifest, and artifact metadata in JSON mode.

## Inspect and predict

```powershell
mlforge artifacts inspect .mlforge/artifacts/<run-id>.mlforge --json

mlforge predict .mlforge/artifacts/<run-id>.mlforge `
  examples/prediction_customers.csv `
  --trust-artifact `
  --output .mlforge/predictions.csv `
  --json
```

The trust flag is intentionally prominent. Without it, model deserialization is refused.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Command completed, version/help printed, or no command requested |
| 1 | Expected MLForge domain/configuration/runtime workflow error |
| 2 | Argument parsing or CLI usage error |

Domain errors are rendered as `mlforge: error: ...` on standard error. The CLI does not broadly catch unexpected programming errors; a traceback remains visible for diagnosis.

## JSON versus human output

Use `--json` for scripts and machine parsing. JSON serialization is deterministic where the underlying domain value provides canonical JSON and rejects nonstandard numeric values. Human output is intended for terminals and may evolve without being the Python compatibility contract.

# Chapter 13 - Python API

## Import philosophy

Import from the domain that owns a concept:

```python
from mlforge.datasets import load_csv, profile_dataset
from mlforge.pipelines import SplitConfig, TaskType
from mlforge.runs import LocalRunStore, compare_runs
from mlforge.training import TrainingConfig, train
```

Do not expect `from mlforge import train`. The minimal top-level API reduces accidental coupling and makes ownership clear.

## Primary stable function signatures

| Function | Parameters in v0.1.0 |
| --- | --- |
| `load_csv` | `path, target, options` |
| `load_feature_csv` | `path, options` |
| `profile_dataset` | `dataset` |
| `split_dataset` | `dataset, task, config` |
| `infer_feature_schema` | `features, overrides` |
| `build_preprocessor` | `split, config, overrides` |
| `build_model_pipeline` | `split, estimator, config, overrides` |
| `train` | `dataset, config, run_store` |
| `evaluate_predictions` | `task, actual, predicted` |
| `compare_runs` | `manifests, metric` |
| `inspect_artifact` | `path` |
| `load_artifact` | `path, trusted` |
| `predict_frame` | `artifact, frame` |
| `predict_csv` | `artifact, path, options` |
| `write_predictions_csv` | `result, path` |

`tests/test_public_api.py` pins exact domain `__all__` exports and these parameter names. A new export or renamed keyword-capable parameter is an intentional compatibility change, not an incidental refactor.

## Complete API workflow

```python
from pathlib import Path

from mlforge.artifacts import LocalArtifactStore
from mlforge.datasets import load_csv
from mlforge.inference import predict_csv
from mlforge.pipelines import SplitConfig, TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import LOGISTIC_REGRESSION, TrainingConfig, train

workspace = Path(".mlforge")
dataset = load_csv("training.csv", target="churn")
config = TrainingConfig(
    task=TaskType.CLASSIFICATION,
    estimator=LOGISTIC_REGRESSION,
    split=SplitConfig(validation_fraction=0.2, random_seed=42),
)
trained = train(
    dataset,
    config,
    run_store=LocalRunStore(workspace / "runs"),
)
store = LocalArtifactStore(workspace / "artifacts")
saved = store.save(trained)
loaded = store.load(trained.manifest.run_id, trusted=True)
predictions = predict_csv(loaded, "prediction.csv")
print(saved.path)
print(predictions.to_json())
```

## Errors as an API

Catch the narrowest domain exception you can handle. `DatasetFormatError` may justify asking for a corrected export. `ArtifactTrustError` requires a trust decision. `ArtifactCompatibilityError` generally requires a matching environment or retraining. Catching every `Exception` and continuing would hide programming errors and can corrupt workflow assumptions.

```python
from mlforge.errors import ArtifactTrustError, DatasetError

try:
    dataset = load_csv("training.csv", target="churn")
except DatasetError as exc:
    raise SystemExit(f"Cannot use training data: {exc}") from exc
```

# Chapter 14 - Dependencies

<!-- diagram:dependency-map -->

## Production dependencies

| Dependency | Declared range | Role |
| --- | --- | --- |
| Python | `>=3.11` | Runtime, typing, dataclasses, filesystem, hashing, JSON, ZIP |
| pandas | `>=3.0,<4` | CSV-to-DataFrame boundary, dtype and tabular operations |
| scikit-learn | `>=1.9,<2` | Splitting, preprocessing, pipelines, estimators, metrics |

NumPy and SciPy are transitive scientific dependencies but are recorded explicitly in run and artifact environments because they influence model behavior and serialization compatibility.

## Development dependencies

The `dev` extra contains `build`, `mypy`, NumPy with a development cap, `pandas-stubs`, `pytest`, `pytest-cov`, `ruff`, and `scikit-learn-stubs`. These are not imported by production code.

## Dependency policy

A production dependency is justified only when it supplies substantial domain behavior that should not be reimplemented. Report generation, web serving, cloud clients, notebook tools, and release utilities do not belong in the core runtime merely because a future idea could use them.

Upper bounds on major versions communicate the compatibility range actually tested. Artifact loading is stricter than package installation: it requires exact recorded versions because pickle and learned scikit-learn objects are not a stable cross-version interchange format.

## Common dependency mistakes

Avoid importing dev-only tools from production code, expanding version ranges without CI and artifact round-trip tests, assuming a lockfile guarantees cross-machine model identity, suppressing unpickle version warnings, or adding a library for convenience already supplied safely by the standard library.

# Chapter 15 - Packaging and Configuration

## Distribution versus import name

Install `hivmind-mlforge`; import `mlforge`. The distinction avoids a namespace collision on PyPI while keeping user code concise.

`pyproject.toml` uses the setuptools build backend and a `src/` layout. The version is read dynamically from `mlforge.__version__`, ensuring runtime and distribution metadata share one source. The `mlforge` console script calls `mlforge.cli:main`.

## Why the `src/` layout matters

Tests run against the installed package rather than accidentally importing a repository-root directory. This catches missing package data, bad entry points, and build configuration errors earlier.

## Typed package

`src/mlforge/py.typed` is included in the distribution, declaring that inline annotations are intended for downstream type checkers. The test suite confirms the marker exists in installed package resources.

## License and source distribution

Package metadata declares the SPDX expression Apache-2.0, and the built distribution includes `LICENSE`. `MANIFEST.in` adds documentation, examples, scripts, contributor files, and environment examples to the source distribution where appropriate.

## Runtime configuration

`ApplicationConfig` currently governs log level. The environment variable is `MLFORGE_LOG_LEVEL`; values are case-insensitive members of `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. An explicit override takes precedence over the environment. Invalid values raise `ConfigurationError`.

```powershell
$env:MLFORGE_LOG_LEVEL = "DEBUG"
mlforge dataset profile data.csv --target label
```

Logging is not configured at import time. `configure_logging` explicitly adds one named handler for the `mlforge` logger, avoids duplicate handlers across repeated calls, and prevents propagation. Library users retain control of their application's logging policy.

## Build locally

```powershell
py -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src tests
python -m pytest
python -m build
```

The wheel must also be installed into a clean environment and exercised outside the repository so source-tree imports cannot hide packaging mistakes.

# Chapter 16 - Testing Strategy

## Test pyramid for this project

MLForge does not pursue coverage as a vanity number. Its suite concentrates on boundaries where a plausible implementation can appear to work while producing unsafe, irreproducible, or misleading results.

| Level | Examples | Confidence gained |
| --- | --- | --- |
| Unit | Config parsing, metric values, type invariants | Small policies are exact |
| Component | CSV validation, preprocessing, stores, artifacts | Each boundary fails safely |
| Integration | Train, save, load, predict, CLI workflows | Modules communicate correctly |
| Contract | Public exports, parameter names, package metadata | Users receive a stable surface |
| Real-data | Breast cancer and diabetes datasets | Mixed realistic data completes the lifecycle |
| Distribution | Clean-wheel smoke script | The built package works outside source checkout |
| Documentation | Link resolution and Python syntax | Instructions remain executable and navigable |

## Release test inventory

The v0.1.0 suite contains 177 passing tests after parameter expansion. The two real-data workflows cover 569-row classification and 442-row regression data with custom delimiters, missing numeric and categorical values, and alternate encoding. Inference tests include a 25,000-row prediction output. Tests simulate atomic publication failures, destination races, manifest corruption, archive tampering, schema drift, invalid targets, and untrusted artifact loading.

## Important behavioral tests

### Data and split

- Byte-order-mark UTF-8, Latin-1, custom delimiters, size limits, invalid encoding, empty files, blank and duplicate headers, inconsistent row width, and retained missing values.
- Deterministic, disjoint classification split with stratification.
- Regression split without stratification and explicit rejection of invalid stratification.
- Class-frequency feasibility and target finiteness before scikit-learn produces obscure errors.

### Preprocessing and training

- Validation values cannot change fitted imputation state.
- Unknown categorical validation values are encoded without failure.
- All-missing numeric and forced-categorical columns retain stable output shape.
- Every supported estimator trains, evaluates, and records parameters.
- Fixed seeds reproduce metrics and predictions.
- Expected training failures are persisted before the exception reaches the caller.

### Persistence and inference

- Run and artifact destinations are immutable and atomic.
- Partial files are absent after simulated publication failure.
- Artifact checksum, schema version, layout, dependency version, and trust failures close safely before useful model execution.
- Prediction reorders columns but rejects missing, extra, duplicate, complex, or non-finite data.
- Output cannot overwrite an existing or racing destination.

## Quality commands

```powershell
ruff check .
ruff format --check .
mypy src tests
python -m pytest
python -m build
```

If formatting is wrong, run `ruff format .` and repeat the entire gate. Do not weaken a failing test merely to get green output. First decide whether the implementation violates the contract, the test encodes an outdated contract, or the environment differs from the supported matrix.

## Writing a valuable test

A useful test states a user-visible or architecture-visible invariant, controls irrelevant variables, and fails for the right reason. Prefer a known metric example, mutation attempt, invalid archive, or end-to-end output over checking that a private helper was called.

Bad test: "the factory function ran once." Good test: "two runs with the same source and seed produce the same partition hash, metrics, and predictions."

## Development versus production validation

Local editable installs optimize iteration. Release validation builds immutable wheel and source archives, checks their metadata, installs the wheel into an isolated environment, runs a full workflow outside the repository, and publishes exactly those checked files. Both layers are required; local tests alone cannot prove packaging correctness.

# Chapter 17 - Security Model

## Security principles

MLForge treats external bytes, paths, manifests, archives, and prediction frames as untrusted at their boundaries. It favors explicit validation, bounded reads, canonical identifiers, create-only outputs, and domain errors. It does not claim to sandbox Python execution.

## Trust boundaries

| Boundary | Potential issue | Control |
| --- | --- | --- |
| CSV path | Wrong type, directory, oversized file, changing file | Strict resolution, regular-file and size checks, before/after metadata |
| CSV bytes | Invalid encoding, nulls, malformed rows | Strict decoder and CSV structure pass before pandas |
| Loaded frame | Caller mutates schema after load | Metadata consistency validation |
| Split | Leakage, overlap, invalid target | Split-first sequence, original indices, target checks |
| Run path | Traversal or overwrite | UUID canonicalization and create-only atomic write |
| Run JSON | Corruption or oversized input | UTF-8, 1 MiB bound, strict versioned schema |
| Artifact ZIP | Extra entries, compressed bomb, symlink, tampering | Exact stored layout, member bounds, checksum |
| Pickle | Arbitrary code execution | Separate inspection and explicit trusted load |
| Prediction frame | Schema drift, invalid values | Exact columns, roles, order, finite values |
| Output path | Symlink or overwrite | `.csv`, regular parent, create-only atomic publish |

## Pickle warning

Python pickle can invoke arbitrary functions while loading. MLForge cannot make an untrusted pickle safe by checking its SHA-256 when the same untrusted party controls the manifest. Inspecting metadata is safe only because `inspect_artifact` does not deserialize `pipeline.pkl`.

Before loading, establish provenance outside the artifact: obtain it from a controlled build, compare with a trusted hash, verify signed release provenance, or use organization-managed storage. If provenance is uncertain, do not load it on a developer machine, CI runner, or production server.

## Filesystem assumptions

Atomic create-only publication depends on local filesystem hard-link and flush semantics. Network shares, synchronization clients, unusual filesystems, and permission models can behave differently. The v0.1.0 design is for local storage. Teams requiring multi-host writers should implement and test a backend with transactional conditional creation.

## Secrets and sensitive data

Core MLForge requires no credentials. Do not commit `.env`, access tokens, private datasets, model artifacts, run directories, prediction outputs, caches, or virtual environments. Run manifests include resolved dataset paths, schema, metrics, and environment versions; artifacts contain learned model state. Treat both as potentially sensitive operational data.

## Input validation is not data governance

Valid CSV structure does not establish consent, lawful use, representative sampling, label correctness, fairness, or absence of sensitive attributes. Dataset profiling is a technical aid. Human review and organization policy remain necessary.

## Vulnerability reporting

Follow root `SECURITY.md` for private reporting. A security fix should include a minimal reproduction, risk analysis, regression test at the affected boundary, compatible documentation, and release notes. Avoid publishing exploitation details before a coordinated fix is available.

## Security review checklist

- Does new code deserialize, execute, import, or evaluate user-controlled content?
- Are file size, member count, nesting, and output bounds explicit?
- Can a path escape its configured root or traverse through a symlink?
- Can an existing file be replaced, truncated, or raced?
- Does a checksum establish integrity only, or is authenticity also required?
- Are error messages useful without leaking secrets or entire records?
- Does failure stop before state mutation, or leave a clearly terminal record?

# Chapter 18 - GitHub and CI/CD

## Continuous integration

`.github/workflows/ci.yml` runs on pushes and pull requests with read-only repository contents permission. Python 3.11 and 3.12 are tested. Each matrix job installs the development extra and runs lint, formatting check, strict mypy, pytest, and package build.

The workflow then creates a clean virtual environment, installs the built wheel, and runs `scripts/wheel_smoke.py` from outside the repository. This guards against missing files and accidental `src/` imports.

## Wheel smoke workflow

The smoke script verifies distribution version and `python -m mlforge --version`, writes a temporary training CSV, trains logistic regression, persists a run, saves and inspects an artifact, explicitly loads it as trusted, predicts two DataFrame rows, writes a prediction CSV, and checks its columns. Everything occurs in a temporary directory.

## Release workflow

`.github/workflows/release.yml` starts only when a GitHub Release is published. The build job checks out the exact release tag, uses Python 3.12, verifies the tag equals `v<package-version>`, builds wheel and source distribution, runs strict Twine checks, and repeats the clean-wheel smoke test.

The checked distribution files are uploaded as a workflow artifact. A separate publish job downloads exactly that artifact and publishes with PyPA's trusted-publishing action. The publish job has only `id-token: write`; it does not need a long-lived PyPI API token.

<!-- diagram:release -->

## Why two release jobs

Separating build from publish confines OIDC permission to the smallest job and guarantees that the bytes tested by the build job are the bytes sent to PyPI. Rebuilding in the publish job would create an avoidable gap between validation and release.

## Repository and pull-request practices

Public visibility permits reading, cloning, forking, and proposing changes, but only authorized collaborators can push upstream. Protect `main`, require pull requests and CI, disallow force pushes, use least privilege, and review release-workflow edits. A pull request should state problem, scope, API impact, tests, security implications, and documentation changes without unrelated refactoring.

# Chapter 19 - PyPI Publication

## Published identity

The package is installed as:

```powershell
py -m pip install hivmind-mlforge==0.1.0
```

The project page, wheel, and source distribution correspond to Git tag `v0.1.0`. The release workflow binds the publication to repository `HivMindAI/mlforge`, workflow `release.yml`, the configured PyPI environment, and the released commit through trusted-publisher attestations.

## One-time trusted publisher setup

In PyPI project settings, configure the GitHub owner, repository, workflow filename, and optional environment exactly as documented in `docs/releasing.md`. The workflow's `environment` value must match the PyPI publisher entry. A mismatch causes OIDC authorization failure even when the GitHub job is otherwise correct.

## Release checklist

1. Ensure `main` is clean and CI is green.
2. Choose a semantic version and update `src/mlforge/__init__.py`.
3. Update `CHANGELOG.md` and any compatibility or migration notes.
4. Run the full local quality gate and build.
5. Install the wheel into a clean environment and run the smoke script outside the repository.
6. Commit and push the reviewed release changes.
7. Create tag `v<version>` at the intended commit.
8. Publish a GitHub Release for that tag.
9. Watch both release jobs and verify the published PyPI files.
10. Install the exact version from PyPI into another clean environment and run a small workflow.

## Immutability and corrections

PyPI releases are immutable in practice. Do not replace version `0.1.0` with different bytes. If a published release is defective, prepare a new version such as `0.1.1`, document the correction, and publish from a new reviewed tag.

## Verification commands

```powershell
py -m venv release-check
.\release-check\Scripts\python.exe -m pip install hivmind-mlforge==0.1.0
.\release-check\Scripts\python.exe -m mlforge --version
.\release-check\Scripts\mlforge.exe --help
```

Use `importlib.metadata.version("hivmind-mlforge")` to compare installed distribution metadata with `mlforge.__version__`.

# Chapter 20 - Guided Code Walkthrough

## Start at the entry point

`src/mlforge/__main__.py` is intentionally tiny: it imports and calls `cli.main`. In `cli.py`, find `build_parser` to understand commands and `main` to see the boundary between argparse, application configuration, logging, domain calls, rendering, and exit codes.

The CLI should remain thin. If a feature is implemented only inside an argparse branch, Python users cannot reuse it and tests must simulate terminal parsing. Put behavior in the owning domain module, then call it from the CLI.

## Follow a CSV

Open `datasets/ingestion.py` and trace `load_csv`. Notice the two-stage read: structural CSV validation and pandas loading. Observe how the resolved path, size, modification time, raw hash, header, and target become `DatasetMetadata`. Then inspect `datasets/validation.py` to see how downstream modules defend against schema mutation.

In `datasets/profiling.py`, separate descriptive quality signals from decisions. The code can identify likely ID/high-cardinality columns and offer a task hint, but does not alter data or configure training.

## Follow the split and preprocessor

In `pipelines/splitting.py`, start with target validation, then class feasibility, then `train_test_split`. Check how original indices are preserved and how one-class training is rejected after an unstratified split.

In `pipelines/preprocessing.py`, find role inference and the categorical missing-marker collision check. The important architectural point is that the split, not the full dataset, is passed into builder functions. Learned state still begins only when `Pipeline.fit` receives training rows.

## Follow a training run

`training/service.py` is the orchestrator. Read it as a transaction-like sequence:

1. Allocate identity and capture starting context.
2. Persist a running manifest.
3. Split, build, fit, predict, and evaluate.
4. Capture parameters, warnings, dataset/environment/split snapshots.
5. Persist exactly one successful or failed terminal manifest.
6. Return fitted state only for success.

`training/estimators.py` owns the supported-model registry. `training/evaluation.py` owns metric definition and direction. `training/types.py` prevents invalid task-estimator configuration.

## Follow persistence

`runs/types.py` contains schema invariants; `runs/store.py` contains filesystem mechanics. This separation lets a manifest be validated independently of storage. `runs/comparison.py` defines what makes two scores fair to compare.

`artifacts/types.py` defines the archive contract. `artifacts/store.py` is intentionally defensive because it handles executable model payloads. Read `inspect_artifact` before `load_artifact`; the ordering mirrors the trust model.

## Finish at inference and verify with tests

`inference.py` validates against `ArtifactFeature`, restores order, invokes the loaded pipeline, normalizes scalars, and publishes optional CSV without refitting. Read each module beside its focused tests: artifact tests prove compatibility checks precede deserialization, preprocessing tests prove validation values do not affect imputation, and run tests prove atomic failure cleanup.

# Chapter 21 - Critical Code and Why It Matters

This chapter uses shortened conceptual excerpts. Consult the repository for exact implementation.

## Canonical dataset identity

```python
digest = hashlib.sha256()
with path.open("rb") as source:
    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
        digest.update(chunk)
sha256 = digest.hexdigest()
```

Chunked hashing bounds memory while binding a run to source bytes. Common mistake: hashing a pandas serialization instead, which can vary with dtype formatting or version.

## Split before fit

```python
split = split_dataset(dataset, task=config.task, config=config.split)
pipeline = build_model_pipeline(
    split,
    estimator,
    config=config.preprocessing,
    overrides=config.feature_overrides,
)
pipeline.fit(split.train_features, split.train_target)
predicted = pipeline.predict(split.validation_features)
```

The order makes leakage resistance reviewable. Common mistake: fitting a preprocessor before the split or passing the combined frame into schema inference.

## Exact partition fingerprint

```python
payload = {
    "train": split.train_features.index.tolist(),
    "validation": split.validation_features.index.tolist(),
}
fingerprint = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
```

The fraction and seed alone are insufficient evidence of the same holdout. Original source index sequences provide a stronger comparison identity.

## Atomic create-only publication

```python
write_and_fsync(temporary_path, payload)
os.link(temporary_path, destination)
temporary_path.unlink()
```

Creating a hard link fails if the destination already exists, so another process cannot be silently overwritten. Common mistake: `Path.replace`, which is atomic but intentionally replaces an existing file.

## Safe artifact inspection

```python
def inspect_without_loading(archive):
    manifest = read_and_validate_manifest(archive)
    actual_hash = stream_member_sha256(archive, "pipeline.pkl")
    if actual_hash != manifest.pipeline_sha256:
        raise ArtifactIntegrityError(...)
    return manifest
```

The crucial property is what is absent: there is no `pickle.loads`. Inspection can show metadata before a trust decision.

## Explicit executable boundary

```python
if trusted is not True:
    raise ArtifactTrustError(...)
validate_environment(manifest.environment)
pipeline = pickle.loads(payload)
```

The identity check uses `is not True`, not general truthiness. A string such as `"yes"` cannot accidentally grant trust. Environment checks happen before deserialization.

## Exact prediction schema

```python
missing = expected_columns - actual_columns
extra = actual_columns - expected_columns
if missing or extra:
    raise PredictionSchemaError(...)
ordered = frame.loc[:, expected_order]
```

Reordering is safe because names establish identity. Ignoring extras is unsafe because it can hide a wrong export or accidental target leakage.

## Versioned strict records

Run and artifact schemas reject unsupported versions and invalid combinations rather than filling missing fields with guesses. Versioned readers make future migrations explicit and prevent old data from acquiring new meaning silently.

# Chapter 22 - Learning Curriculum

## Outcome

After this curriculum, a learner should be able to run MLForge, explain every lifecycle boundary, diagnose common failures, add a small compatible feature, and defend the architecture in a technical review.

## Stage 1: user workflow

**Goal:** understand input, output, and the difference between run metadata and model state.

1. Install the released wheel in a clean environment.
2. Run `mlforge --help` and the profile example.
3. Train the bundled churn dataset through CLI.
4. Locate the run JSON and `.mlforge` artifact.
5. Inspect the artifact without trusting it.
6. Explicitly load the local artifact and predict the bundled feature CSV.

**Exercise:** explain why a run manifest can be read safely while a model artifact cannot be loaded from an unknown source.

## Stage 2: data and leakage

**Goal:** understand structural validation, task semantics, and split-before-fit.

1. Read `datasets/ingestion.py` with its tests.
2. Create malformed CSVs: duplicate header, wrong row width, invalid encoding.
3. Profile a numeric class label and discuss why a task hint is not ground truth.
4. Read splitting and preprocessing code.
5. Reproduce the leakage-prevention imputation test by hand.

**Exercise:** add one unseen prediction category and predict successfully; then add one extra column and explain the rejection.

## Stage 3: reproducibility and evaluation

**Goal:** understand what can and cannot be reproduced.

1. Train twice with the same seed and source; compare partition hashes and metrics.
2. Change only the seed; observe comparison rejection.
3. Change line endings in the CSV; observe the dataset hash change.
4. Compare `accuracy` and `balanced_accuracy` on imbalanced data.
5. Read how metric direction controls ordering.

**Exercise:** construct two regression manifests and reason about why lower RMSE ranks first while higher R2 ranks first.

## Stage 4: persistence and security

**Goal:** reason about durable state and executable artifacts.

1. Read `runs/store.py` and atomic failure tests.
2. Inspect the two members in an artifact ZIP without loading it.
3. Modify one payload byte and observe integrity failure.
4. Read the dependency mismatch test and verify it fails before pickle loading.
5. Explain the difference between checksum, signature, provenance, and trust.

**Exercise:** write a threat model for accepting artifacts uploaded by external users. The correct conclusion for v0.1.0 is to reject loading them.

## Stage 5: packaging and release

**Goal:** connect source code to a trustworthy install.

1. Read `pyproject.toml` and identify runtime versus dev dependencies.
2. Build wheel and source distribution.
3. Install the wheel into a temporary clean environment.
4. Run `scripts/wheel_smoke.py` outside the repository.
5. Trace GitHub release jobs and PyPI OIDC permissions.

**Exercise:** explain why publishing should download the tested build artifact instead of rebuilding.

## Stage 6: contributor change

**Goal:** make one small reviewable improvement.

1. Choose one domain behavior, such as a new metric.
2. State its public contract and failure behavior before coding.
3. Add implementation and focused tests.
4. Update exports, CLI, docs, and compatibility statements only if affected.
5. Run the full quality gate and inspect the diff for unrelated changes.

## Suggested schedule

| Week | Focus | Deliverable |
| --- | --- | --- |
| 1 | Chapters 1-6 | Diagram the lifecycle and demonstrate leakage prevention |
| 2 | Chapters 7-13 | Run classification/regression through CLI and API |
| 3 | Chapters 14-19 | Build a wheel and explain the release chain |
| 4 | Chapters 20-24 | Complete a small tested documentation or metric change |
| 5 | Chapters 25-29 | Practice interview answers and score the self-test |

# Chapter 23 - Debugging Scenarios

## Virtual environment activation fails

**Symptom:** PowerShell says `.venv\Scripts\Activate.ps1` is not recognized.

**Cause:** the environment was never created, is in another directory, or has a different name. Activation scripts cannot exist before `py -m venv .venv` succeeds.

```powershell
Set-Location -LiteralPath "C:\path\to\mlforge"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m mlforge --version
```

Activation is optional. Calling the environment's Python directly is often clearer and avoids execution-policy issues.

## `mlforge` command is not found

Confirm which Python received the package:

```powershell
py -m pip show hivmind-mlforge
py -m mlforge --version
Get-Command mlforge -ErrorAction SilentlyContinue
```

Use `python -m mlforge` when the console-scripts directory is not on `PATH`.

## CSV profile fails

Read the domain error literally. Confirm suffix, delimiter, encoding, target spelling, header uniqueness, and consistent rows. Use `CsvLoadOptions` or matching CLI flags rather than editing loader internals. If Excel produced the file, export one delimiter and encoding explicitly.

## Classification split cannot stratify

The smallest class may have only one row, or the selected validation fraction may not leave room for every class in both partitions. Collect more observations, revise labels, or explicitly disable stratification only when you understand that the training or validation set may lose a class.

## Regression target is rejected

Inspect the target dtype and values. A numeric-looking target may contain currency symbols, commas, blank strings, booleans, infinity, or mixed text. Clean the source upstream and reload it; do not coerce invalid target values silently inside training.

## Datetime feature is unsupported

Create explicit domain features such as year, month, elapsed days, weekday, or cyclical encodings before load, taking care that reference dates do not use future information. Document that upstream transformation because it is outside the run manifest.

## Artifact loading says trust is required

First call `inspect_artifact`. Establish its source and independent provenance. If it is the artifact you just created locally in this workflow, use `trusted=True`. If it arrived from an unknown person or public upload, do not load it.

## Artifact dependency mismatch

Create an environment with the exact versions in `manifest.environment` or retrain and resave under the current supported environment. Do not edit the manifest or suppress the compatibility check; the underlying pickle can still be incompatible.

## Prediction has missing or extra features

Compare `artifact.manifest.features` with the CSV header. Rename or select input columns at the data-export boundary. Do not insert fake values for unknown features or drop an unexpected target column without verifying that the export is conceptually correct.

## Prediction output already exists

MLForge never overwrites. Choose a new versioned filename, move the existing result deliberately, or remove it through a separate user-controlled process after verifying the path. This behavior protects prior predictions and concurrent writers.

## Run comparison is rejected

Inspect task, dataset SHA-256, target, fraction, seed, actual stratification, and partition hash. The error is evidence that the scores are not from the same holdout. Compare them narratively or rerun experiments on an identical partition; do not relax the fairness check.

## Tests pass locally but wheel smoke fails

The source checkout may expose files absent from the wheel. Inspect wheel contents, `pyproject.toml`, package discovery, `MANIFEST.in`, and `py.typed`. Run the smoke script from a directory outside the repository with only the wheel installed.

## Release publishing fails at OIDC

Compare PyPI trusted-publisher owner, repository, workflow filename, and environment with the GitHub job. Confirm the job has `id-token: write`, originated from the released tag, and uses the configured environment. Do not fall back to committing an API token.

# Chapter 24 - Safe Modification Guide

## First classify the change

| Change | Primary owner | Contracts to inspect |
| --- | --- | --- |
| CSV option or validation | `datasets` | CLI flags, metadata schema, ingestion tests |
| Feature role or transformer | `pipelines` | Leakage, artifact features, inference validation |
| Estimator or metric | `training` | Config registry, CLI choice, run schema, comparison |
| Run field | `runs` | Schema version, deterministic JSON, stores, artifacts |
| Artifact field/format | `artifacts` | Schema version, compatibility, trust, docs |
| Prediction behavior | `inference.py` | Artifact schema, CSV output, CLI and API |
| Command | `cli.py` plus domain module | Exit codes, JSON/human output, docs |
| Dependency/version | `pyproject.toml` | CI matrix, wheel smoke, artifacts, changelog |

## Change protocol

1. State the user-visible behavior and what remains unchanged.
2. Locate the owning module and focused tests.
3. Decide whether the public API or serialized schema changes.
4. Add or update a failing behavioral test.
5. Make the smallest production change.
6. Add domain errors rather than leaking low-level exceptions when the caller can act.
7. Update CLI, docs, examples, and exports only where the behavior is public.
8. Run lint, format, mypy, pytest, build, and relevant clean-wheel validation.
9. Review security, determinism, leakage, and filesystem effects.
10. Keep the commit and pull request limited to the active milestone.

## Serialized schema rule

Do not add a required field to manifest version 1 and pretend old files are version 1-compatible. Either make a truly optional field with unambiguous default semantics or create a new schema version and an explicit reader/migration path. Test both accepted and rejected versions.

## Public API rule

The explicit `__all__` sets and primary parameter names are compatibility contracts. A helper can remain private until it has a stable purpose. Removing or renaming a public value requires justification, documentation, and usually a deprecation plan.

## Leakage review

For any data or preprocessing change, answer:

- Which rows determine schema and learned parameters?
- Can validation or future information influence the transformation?
- Are target-derived features possible?
- Does prediction reuse exactly the training transform?
- Does the run record enough information to explain the behavior?

## Persistence review

For any file-writing change, answer:

- Is the destination exact, bounded, and free of traversal?
- Can it overwrite or follow a symlink?
- Is partial output visible after failure?
- Is publication atomic on the supported filesystem?
- Can a racing writer be preserved?
- Is the serialized input bounded and versioned on read?

## When to redesign

Redesign is justified when an invariant cannot be expressed cleanly through current interfaces, multiple modules duplicate the same policy, or a real use case proves the local adapter insufficient. It is not justified by a desire to add layers, generic repositories, plugin loaders, or cloud abstractions without an implemented consumer.

## Definition of done

A change is done when its real behavior works through the appropriate public interface, important edge cases are tested, the full configured gate passes, documentation matches, packaging still works, security assumptions are explicit, and no placeholder or unrelated refactor is hidden in the diff.

# Chapter 25 - Interview Preparation

## Beginner questions

### 1. What is MLForge?

**Intent:** test whether the candidate can state the product without exaggerating it.

**Model answer:** MLForge v0.1.0 is a local Python framework for reproducible supervised tabular baselines. It validates CSV data, profiles it, performs leakage-safe train/validation splitting and preprocessing, trains one of four scikit-learn estimators, records an immutable run manifest, saves a versioned artifact, and performs schema-checked batch predictions. It is not a hosted platform or online serving system.

### 2. Why does MLForge split before fitting preprocessing?

**Intent:** test basic understanding of data leakage.

**Model answer:** Imputation values, scaling statistics, and category vocabularies are learned parameters. If validation rows influence them, the validation score is no longer independent. MLForge splits first, derives schema from training features, fits the pipeline on training rows, and only transforms validation rows.

### 3. What is the difference between a run manifest and an artifact?

**Intent:** test understanding of metadata versus executable model state.

**Model answer:** A run manifest is bounded JSON describing configuration, lineage, environment, split, status, and metrics. An artifact is a `.mlforge` ZIP containing a manifest plus a pickled fitted pipeline. The run is safe to parse as data; artifact loading crosses an executable pickle trust boundary.

### 4. Why are both distribution and import names different?

**Intent:** test Python packaging literacy.

**Model answer:** PyPI installs the uniquely named distribution `hivmind-mlforge`, while Python imports the package `mlforge` and exposes the `mlforge` console script. Distribution metadata and import namespace are related but separate packaging concepts.

### 5. What happens when prediction columns are reordered?

**Intent:** test schema behavior.

**Model answer:** If the set of unique columns is exact, MLForge restores the artifact's recorded training order before prediction. Missing, extra, or duplicate columns are rejected.

## Intermediate questions

### 6. How does MLForge decide whether two runs are fairly comparable?

**Intent:** test experiment-design reasoning.

**Model answer:** Runs must be distinct and successful, have the same task, source-byte SHA-256, target, validation fraction, seed, actual stratification, and exact partition fingerprint. The metric must exist with consistent direction. This ensures both scores use the same holdout rather than merely similar configuration labels.

### 7. Why does artifact loading require exact dependency versions while package installation uses ranges?

**Intent:** test serialization compatibility reasoning.

**Model answer:** Source code can support a tested range through normal APIs, but pickle contains internal learned Python objects whose layout can change even within that range. Exact Python and scientific-stack versions reduce the chance of unsafe or incorrect deserialization. A mismatch requires a matching environment or retraining.

### 8. How does create-only atomic storage work and what does it protect?

**Intent:** test durable filesystem reasoning.

**Model answer:** MLForge writes and flushes a temporary file, then creates a hard link at the final path. Link creation fails if the destination exists, so no prior record is overwritten and readers do not see partial content. Cleanup removes the temporary path. It protects local single-filesystem publication, not arbitrary distributed filesystems.

### 9. Why is `inspect_artifact` a separate function from `load_artifact`?

**Intent:** test trust-boundary design.

**Model answer:** Inspection validates path, archive layout, bounded members, manifest, and payload checksum without executing pickle. Loading requires explicit trust, checks the environment, then deserializes. Separating them lets tools display metadata without crossing the executable boundary.

### 10. How are expected training failures represented?

**Intent:** test observability and error handling.

**Model answer:** Once a run starts, expected domain and common value/type failures produce a failed terminal manifest with timestamps and structured failure details. The manifest is persisted before `TrainingFailedError` is raised. Storage failures propagate separately because the audit record itself could not be guaranteed.

## Advanced questions

### 11. Does a pipeline checksum authenticate an artifact? Why or why not?

**Intent:** test security precision.

**Model answer:** No. The checksum proves the payload matches the manifest, but an attacker controlling the artifact can replace both. Authentication needs an independent trust anchor such as a signed digest, trusted build provenance, controlled storage, or a separately distributed allowlist.

### 12. What architectural change is needed for a remote multi-writer run store?

**Intent:** test ability to extend without breaking semantics.

**Model answer:** Add an explicit store adapter whose backend offers transactional conditional create, bounded versioned reads, canonical IDs, and consistent list/read semantics. Keep `RunManifest` invariants independent. Do not reuse local hard-link assumptions over a network filesystem without testing them.

### 13. How would you add cross-validation without violating the current mental model?

**Intent:** test design discipline.

**Model answer:** First define a versioned evaluation record that distinguishes fold-level partitions from the final holdout. Every fold must fit preprocessing independently on its training portion. Comparison identity must bind all fold indices and configuration. CLI/API and manifests must make cross-validation explicit; it should not silently replace the current single holdout.

### 14. What is the hardest compatibility issue in adding an artifact field?

**Intent:** test serialized-schema evolution.

**Model answer:** Existing version 1 files cannot be reinterpreted silently. If the field changes meaning or is required, introduce a new schema version and explicit reader or migration logic. Maintain bounded parsing, fail closed on unsupported versions, and test old/new behavior and trust ordering.

### 15. Where could nondeterminism remain even with a fixed seed?

**Intent:** test nuanced reproducibility understanding.

**Model answer:** Dependency versions, native numerical libraries, CPU instruction order, operating system, thread scheduling, upstream CSV generation, row order, and estimator implementation can matter. MLForge records key versions and uses seeded, single-job forest settings, but does not promise universal bit-for-bit identity across arbitrary environments.

## Interview technique

Answer from the lifecycle outward: state the user problem, identify the responsible module, name the invariant, cite the test or failure behavior, and acknowledge the limit. Strong answers distinguish implemented controls from claims the project does not make.

# Chapter 26 - Explaining the Project

## 30-second explanation

MLForge is a local Python framework for reproducible tabular machine-learning baselines. Give it a validated CSV, explicit target and task, and it builds a leakage-safe scikit-learn pipeline, evaluates a deterministic holdout, records an immutable experiment manifest, saves a versioned model artifact, and performs exact-schema batch prediction. Version 0.1.0 focuses on a small dependable lifecycle rather than cloud deployment or automated modeling.

## 2-minute explanation

Many ML examples stop at `model.fit`, but a reusable workflow also needs to identify input bytes, prevent preprocessing leakage, preserve the exact split, record environment and parameters, save model state safely, and validate inference schemas. MLForge turns those concerns into one local workflow.

CSV ingestion validates the real file, encoding, header, row widths, size, and target, then records a SHA-256 and schema. Profiling reports quality without making modeling decisions. Training requires classification or regression explicitly, performs a seeded holdout, infers roles from training data, and fits a scikit-learn preprocessing/model pipeline only on training rows. It evaluates task-appropriate metrics and writes a strict run manifest, including failures.

A successful run can become a `.mlforge` archive. Metadata and checksum can be inspected without deserializing; actual loading requires explicit trust because the pipeline is pickle. Dependency versions and feature schema are checked before batch prediction. Runs and outputs use create-only atomic local writes.

The release is a typed Python 3.11+ package with pandas and scikit-learn as its only direct runtime dependencies. CI covers Python 3.11 and 3.12, runs lint, formatting, strict types, 177 tests, builds the package, and smoke-tests the wheel in isolation. GitHub trusted publishing sends the tested artifacts to PyPI without a stored token.

## 5-minute explanation

Start with the problem: a notebook baseline is easy to create but hard to trust later. You need to know which source bytes, target, split, preprocessing state, estimator defaults, dependency versions, and validation rows created a result. You also need a safe failure story and an inference contract. MLForge v0.1.0 solves that narrowly for local supervised tabular work.

The first boundary is datasets. The CSV loader uses strict path and structure checks before pandas type inference. It detects malformed rows, duplicate headers, invalid encoding, oversize input, and files that change while being loaded. It records the raw file hash and schema. Profiling is descriptive: missingness, finite numeric summaries, cardinality, likely identifiers, imbalance, and a conservative task hint.

Next is splitting and preprocessing. The user chooses classification or regression. Target semantics and class feasibility are checked before scikit-learn. Classification stratifies by default; regression does not. Original indices identify the exact train and validation partition. Feature roles come from training features. Numeric columns are imputed and optionally scaled; categorical columns use collision-checked missing values and one-hot encoding that tolerates unseen categories. The transformer and estimator live in one pipeline fitted only on training rows.

Training is the orchestrator. It creates a UUID, UTC timestamps, and a running record; builds and fits the pipeline; predicts validation rows; computes three task-specific finite metrics; captures warnings and parameters; snapshots the dataset, split, and environment; then persists a successful or failed terminal manifest. A partition hash binds comparison to exact source indices.

Run storage uses strict versioned JSON, bounded reads, canonical UUID names, and create-only atomic writes. Comparison refuses scores from different source bytes or holdouts. That is a deliberate product opinion: a ranking is useful only when the experimental basis is fair.

Model persistence is a separate concern. The `.mlforge` ZIP has exactly a JSON manifest and pickle-protocol-5 pipeline. It binds run lineage, features, environment, size, and SHA-256. Inspection never unpickles. Loading needs `trusted=True`, exact versions, and post-load fitted-pipeline/schema checks. Checksums detect change but do not authenticate the publisher, so unknown artifacts remain unsafe.

Inference accepts a loaded artifact, demands the exact feature set, reorders columns, checks numeric and categorical values, normalizes finite predictions, and can write a new CSV atomically. It is local batch inference, not a server.

Architecturally, CLI and Python interfaces call small domain modules. Typed frozen records carry state across boundaries. There is no generic plugin system or cloud adapter because v0.1.0 does not need them. The framework supports four baseline estimators, one holdout strategy, local stores, and two direct dependencies.

Engineering quality closes the loop: exact public exports and parameter names are tested; realistic classification and regression datasets exercise the complete lifecycle; build metadata, license, typing marker, examples, and documentation are validated; and a clean installed-wheel smoke test prevents source-tree success from hiding packaging errors. The release workflow builds once, tests those files, and publishes the same files to PyPI through OIDC.

The best summary is that MLForge is intentionally smaller than a platform but more disciplined than a notebook. Its value is the coherence of the lifecycle and the explicitness of its boundaries.

# Chapter 27 - Final Mental Models

## Lifecycle map

<!-- diagram:mental-model -->

Think of the lifecycle as seven contracts, not seven loose functions. Each arrow validates that the previous state is suitable for the next stage. Re-entering the lifecycle midway requires the corresponding typed state: inference needs a `LoadedArtifact`, not a filename pretending to be trusted.

## Three planes

| Plane | Contents | Question answered |
| --- | --- | --- |
| Data plane | CSV bytes, DataFrames, partitions, transformed matrices, predictions | What values flowed through the model? |
| Control plane | Task, estimator, seed, split fraction, preprocessing, trust | What decisions controlled the flow? |
| Evidence plane | Hashes, manifests, environment, metrics, warnings, artifacts | How can the result be explained later? |

The pipeline operates in the data plane. Configuration operates in the control plane. Runs and artifact manifests form the evidence plane. A mature change considers all three.

## Four identities

1. **Dataset identity:** SHA-256 of exact CSV bytes plus target/schema context.
2. **Partition identity:** SHA-256 of exact original indices assigned to train and validation.
3. **Run identity:** UUID for one attempted configuration and lifecycle, successful or failed.
4. **Artifact identity:** run identity plus manifest and payload hashes under one environment contract.

Confusing these identities leads to weak claims. The same dataset can have many partitions and runs. The same score does not establish the same run. An artifact with the same run ID but altered bytes fails integrity checks.

## Two kinds of safety

**Data safety** means validation, leakage prevention, fair comparison, finite values, and exact schemas. **Execution safety** means controlled paths, bounded archives, atomic writes, and refusing untrusted pickle. One does not replace the other.

## The failure funnel

```text
Bad path/bytes       -> dataset domain error before modeling
Bad target/split     -> pipeline domain error before fit
Expected fit failure -> failed terminal run + TrainingFailedError
Bad run/artifact     -> store/format/integrity error before use
Untrusted artifact   -> trust error before deserialization
Bad prediction frame -> schema error before pipeline.predict
Bad output path      -> inference error before publication
```

Failures occur as early as the relevant information permits. Later layers do not silently reinterpret invalid earlier state.

## The extension test

Before adding a feature, ask: Does it serve the local reproducible tabular lifecycle? Can its configuration and result be recorded? Can it preserve split-before-fit? Can its input be bounded and validated? Can it be tested through public behavior? Does it require a new schema version or dependency? If the answers are vague, the feature is not ready for core.

# Chapter 28 - Glossary

| Term | Meaning in MLForge |
| --- | --- |
| Artifact | Versioned `.mlforge` ZIP containing manifest and fitted pickled pipeline |
| Artifact manifest | JSON metadata binding run, features, environment, size, and payload hash |
| Atomic publication | Making a complete new file visible in one operation without partial destination |
| Balanced accuracy | Mean recall across classes; useful when class frequencies differ |
| Canonical JSON | Deterministically ordered/encoded JSON used for stable hashes and records |
| Categorical feature | Boolean/string/object/category input modeled through imputation and one-hot encoding |
| Checksum | Digest used to detect changes; not proof of publisher identity |
| Classification | Supervised task predicting one of at least two discrete target classes |
| CLI | Command-line interface exposed through `mlforge` and `python -m mlforge` |
| Comparison identity | Fields that must match before metrics are ranked fairly |
| Create-only | A write that fails rather than replacing an existing destination |
| Data leakage | Validation or future information influencing training or preprocessing state |
| Dataset hash | SHA-256 of exact source CSV bytes |
| Dataset profile | Deterministic descriptive report of schema, missingness, distributions, and flags |
| Domain error | MLForge-specific exception expressing an actionable boundary failure |
| Estimator | scikit-learn classifier or regressor at the end of the pipeline |
| Evidence plane | Hashes, manifests, metrics, warnings, and artifacts explaining a result |
| Feature override | Explicit user assignment of selected columns to numeric/categorical roles |
| Feature schema | Ordered names and modeling roles inferred from training features |
| Finite | A numeric value that is neither positive/negative infinity nor NaN when required |
| Holdout | Validation partition not used to fit preprocessing or estimator state |
| Imputation | Replacing missing features using a learned numeric strategy or categorical marker |
| Inference | Applying a trusted fitted artifact to schema-compatible feature rows |
| Integrity | Evidence that bytes match a recorded digest and structure |
| Lineage | Recorded relationship among source, configuration, split, environment, run, and artifact |
| LoadedArtifact | Trusted, compatibility-checked, deserialized fitted pipeline plus manifest |
| LoadedDataset | Validated pandas frame plus immutable source/schema metadata |
| Manifest | Strict versioned metadata record serialized as JSON |
| Metric direction | Whether higher or lower values rank as better |
| Numeric feature | Real non-boolean input modeled through numeric imputation and optional scaling |
| OIDC | Short-lived identity mechanism used by GitHub Actions trusted publishing |
| One-hot encoding | Mapping each learned category to indicator columns |
| Partition fingerprint | SHA-256 of exact train/validation original index sequences |
| Pickle | Python object serialization format that can execute code while loading |
| Pipeline | scikit-learn composition of preprocessor followed by estimator |
| Prediction record | Zero-based input row number paired with one JSON-safe prediction |
| Profiling | Describing data quality without mutating data or selecting a model |
| Provenance | Independently verifiable origin and build context of an artifact/package |
| Public API | Explicit domain exports and stable function parameter contracts |
| Regression | Supervised task predicting a finite real numeric target |
| Reproducibility | Ability to recreate results under recorded source, configuration, code, and environment |
| Run | One attempted training lifecycle with a UUID and terminal status |
| Run manifest | JSON evidence for configuration, source, split, environment, metrics/failure |
| Schema drift | Prediction columns or value roles no longer matching training expectations |
| Stratification | Split preserving class proportions when feasible |
| Target | Supervised label/outcome column excluded from input features |
| Task hint | Conservative profiling suggestion; never automatic task selection |
| Trusted load | Explicit caller opt-in to execute a verified artifact's pickle |
| Trusted publishing | PyPI publication authorized by GitHub OIDC rather than stored API token |
| Validation fraction | Requested portion of rows held out from training |
| Wheel | Built Python distribution installed by pip |

# Chapter 29 - Self-Test and Answer Key

## Beginner questions (20)

1. What type of machine-learning problem does MLForge v0.1.0 support?
2. What is the PyPI distribution name?
3. What is the Python import package name?
4. Which Python versions are supported at minimum?
5. What two direct production dependencies are declared?
6. What file type does the dataset loader accept?
7. Why does `load_csv` require a target?
8. What does dataset profiling change in the DataFrame?
9. Which two task types are supported?
10. Name one supported classification estimator.
11. Name one supported regression estimator.
12. What is data leakage?
13. When is the preprocessor fitted?
14. What does a successful `TrainingResult` contain?
15. What file suffix is used for saved model artifacts?
16. Can `inspect_artifact` execute the pickle payload?
17. Which argument must be explicit before artifact loading?
18. What happens when prediction input has an extra column?
19. Does prediction CSV output overwrite an existing file?
20. What license covers MLForge?

## Intermediate questions (20)

21. Why does ingestion validate CSV structure before pandas loading?
22. What information makes the dataset SHA-256 useful, and what does it not prove?
23. When does classification stratify by default?
24. Why is regression target stratification rejected?
25. How are boolean features modeled by default?
26. How does MLForge handle a new categorical value at prediction time?
27. Why is the categorical missing marker collision-checked?
28. Name all classification metrics recorded by v0.1.0.
29. Name all regression metrics and their preferred directions.
30. What does the partition fingerprint hash?
31. Which fields must match for fair run comparison?
32. How does a failed training attempt remain observable?
33. Why is the run store create-only?
34. What are the exact members of a `.mlforge` archive?
35. Why are artifact members stored uncompressed?
36. Which environment versions are recorded?
37. Why does prediction restore feature order after checking the set?
38. What are CLI exit codes 0, 1, and 2 used for?
39. What does `tests/test_public_api.py` protect?
40. Why is an installed-wheel smoke test run outside the repository?

## Advanced questions (10)

41. Explain why an embedded checksum provides integrity but not authenticity.
42. What ordering must an artifact loader preserve to minimize unsafe execution?
43. How would you evolve the run manifest if a new required field changes semantics?
44. Which parts of MLForge's reproducibility claim can still vary across machines?
45. Why is a partition hash stronger than recording only seed and validation fraction?
46. What contract must a remote run-store adapter preserve?
47. How would you add cross-validation without leaking fold validation data?
48. Why should unexpected programming exceptions not be broadly converted to CLI success or generic domain output?
49. What is the security implication of accepting user-uploaded `.mlforge` artifacts in a web service?
50. Give a principled reason to reject a proposed core feature even if it is useful in some ML systems.

<!-- pagebreak -->

## Answer key

### Beginner answers

1. Local supervised tabular classification and regression baselines.
2. `hivmind-mlforge`.
3. `mlforge`.
4. Python 3.11 or newer.
5. pandas and scikit-learn.
6. A local regular `.csv` file within the configured size limit.
7. Training data needs one explicit supervised outcome column and its name becomes lineage.
8. Nothing; profiling is descriptive and checks that the loaded schema was not mutated.
9. `TaskType.CLASSIFICATION` and `TaskType.REGRESSION`.
10. Logistic regression or random-forest classifier.
11. Ridge regression or random-forest regressor.
12. Information outside training rows influencing learned model or preprocessing state.
13. After splitting, on training features and target only, as part of `Pipeline.fit`.
14. The fitted pipeline and a successful terminal `RunManifest`.
15. `.mlforge`.
16. No. It streams and hashes the payload without deserializing it.
17. `trusted=True`, after independently establishing provenance.
18. `PredictionSchemaError` is raised before model prediction.
19. No. The writer is create-only and preserves existing or racing destinations.
20. Apache-2.0.

### Intermediate answers

21. It detects encoding, header, null-byte, and row-width problems with predictable domain errors before pandas type inference can obscure them.
22. It identifies exact source bytes for lineage and comparison; it does not prove semantic equivalence or publisher authenticity.
23. Classification stratifies by default when class counts and partition sizes make it feasible.
24. Regression has a continuous numeric target and v0.1.0 does not define bins or a regression-stratification policy.
25. As categorical features.
26. One-hot encoding uses `handle_unknown="ignore"`, producing no active learned category for that feature.
27. A real category equal to the reserved marker would become indistinguishable from a missing value.
28. Accuracy, balanced accuracy, and weighted F1.
29. Mean absolute error lower, root mean squared error lower, and R2 higher.
30. Canonical ordered original source indices in the train and validation partitions.
31. Successful distinct runs with the same task, dataset hash, target, validation fraction, seed, actual stratification, exact partition hash, and consistently directed metric.
32. A failed terminal manifest is persisted with timestamps and structured failure information before `TrainingFailedError` is raised.
33. Experiment history should be immutable; reuse or races must not replace evidence.
34. Exactly `manifest.json` and `pipeline.pkl`.
35. The reader can enforce clear bounded sizes and reject compression-based expansion tricks; the model payload is already a binary serialization where predictable validation matters more than space savings.
36. Python, MLForge, pandas, NumPy, SciPy, and scikit-learn.
37. Scikit-learn pipelines are order-sensitive; matching names permit safe restoration to training order.
38. Zero is success/help, one is an expected MLForge workflow error, and two is argument/usage error.
39. Exact domain `__all__` exports and stable primary function parameter names.
40. To prove the built wheel contains everything and does not rely on accidental source-checkout imports.

### Advanced answers

41. The digest detects payload changes relative to the manifest, but an attacker controlling both can recompute it. Authentication needs an independent trust anchor such as signature, provenance, or trusted storage.
42. Validate path and bounded archive structure, parse strict manifest, stream and verify payload size/hash, require explicit trust, verify exact environment, then deserialize and validate the fitted pipeline/schema.
43. Introduce a new schema version with explicit parsing or migration semantics, retain or deliberately reject old versions, and test both paths. Do not reinterpret version 1 silently.
44. Native numerical libraries, hardware instruction order, operating system, thread scheduling, upstream data generation and row order, and implementation changes beyond recorded versions can matter.
45. Identical seed/fraction can select different rows when source order or splitting behavior changes; the hash binds the exact holdout identity.
46. Canonical identifiers, strict versioned bounded reads, transactional conditional create, no overwrite, deterministic serialization, clear failures, and consistent read/list behavior.
47. Each fold must build and fit preprocessing only on that fold's training rows. Record every fold partition and metric, define aggregation and comparison identity explicitly, and keep a separate untouched final holdout if used.
48. Broad conversion can hide defects, produce misleading automation output, and make corrupted assumptions look like expected user errors. Unexpected exceptions should remain diagnosable and nonzero.
49. Loading permits arbitrary Python code execution in the service process. V0.1.0 cannot safely accept unknown artifacts; use a non-executable format or an isolated, reviewed trust and sandbox design before enabling uploads.
50. Reject it when it does not serve the implemented local tabular lifecycle, cannot be recorded/tested safely, creates an unused abstraction, or adds dependency/security/compatibility cost without a concrete maintained use case.

## Scoring guide

| Score | Interpretation | Next action |
| --- | --- | --- |
| 45-50 | Maintainer-level mental model | Review an architectural change or release |
| 38-44 | Strong contributor understanding | Practice security and schema-evolution cases |
| 28-37 | Working user understanding | Repeat code walkthrough and debugging labs |
| 18-27 | Partial lifecycle understanding | Complete curriculum Stages 1-3 |
| 0-17 | Foundations need reinforcement | Run bundled examples and reread Chapters 1-13 |

## Final review checklist

- I can distinguish the distribution, package, command, run, and artifact identities.
- I can explain split-before-fit and prove where validation is excluded.
- I can trace a CSV row through training and prediction.
- I can explain fair comparison and the partition fingerprint.
- I can inspect an artifact without crossing the pickle trust boundary.
- I can state what checksums do and do not guarantee.
- I can run the quality gate and clean-wheel smoke test.
- I can evolve a public API or serialized schema deliberately.
- I can describe both implemented capabilities and intentional non-goals.
- I can explain MLForge accurately in 30 seconds, 2 minutes, or 5 minutes.
