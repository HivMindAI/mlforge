# Python API Reference

MLForge is library-first. Import from the domain modules below; only the explicitly listed names
are public. Configuration values are immutable dataclasses, named modes are enums or constants,
and expected operational failures use the hierarchy in `mlforge.errors`.

For a guided workflow, start with the [tutorial](tutorial.md). Compatibility rules are in
[compatibility.md](compatibility.md).

## Package metadata

`mlforge.__version__` is the single runtime version and matches installed distribution metadata.
The root package intentionally contains no convenience re-exports.

## Datasets

Import from `mlforge.datasets`.

| Public name | Purpose |
| --- | --- |
| `CsvLoadOptions` | Explicit encoding, delimiter, and maximum-size policy for CSV reads. |
| `load_csv(path, *, target, options=None)` | Validate a local training CSV and return `LoadedDataset`. |
| `load_feature_csv(path, *, options=None)` | Validate a target-free CSV and return a dataframe for inference. |
| `profile_dataset(dataset)` | Return deterministic quality and target summaries. |
| `LoadedDataset` | Dataframe plus immutable `DatasetMetadata`. |
| `DatasetMetadata`, `ColumnMetadata` | Source identity, parser settings, dimensions, and physical schema. |
| `DatasetProfile`, `ColumnProfile`, `NumericSummary` | JSON-safe dataset and feature summaries. |
| `TargetProfile`, `ValueFrequency` | Target-task hint and class-frequency summaries. |
| `ColumnKind`, `TaskHint` | Stable profiling classifications. |

`load_csv` requires a named target and never mutates the source. `load_feature_csv` deliberately
does not invent one. The profiling task hint is advisory and never selects training behavior.

## Pipelines

Import from `mlforge.pipelines`.

| Public name | Purpose |
| --- | --- |
| `TaskType` | Explicit `classification` or `regression` selection. |
| `SplitConfig` | Validation fraction, seed, and optional stratification policy. |
| `split_dataset(dataset, *, task, config=None)` | Create copied train/validation features and targets before fitting. |
| `DatasetSplit` | Index-preserving split result with copied partitions and actual stratification state. |
| `FeatureOverrides` | Explicit numeric/categorical role overrides for ambiguous columns. |
| `FeatureSchema` | Ordered numeric and categorical training-feature contract. |
| `NumericImputationStrategy` | Supported `mean` or `median` numeric imputation. |
| `PreprocessingConfig` | Numeric imputation, scaling, and categorical missing-marker settings. |
| `infer_feature_schema(features, *, overrides=None)` | Infer roles from the training dataframe only. |
| `build_preprocessor(split, *, config=None, overrides=None)` | Return an unfitted column transformer. |
| `build_model_pipeline(split, estimator, *, config=None, overrides=None)` | Clone an estimator and return an unfitted preprocessing/model pipeline. |

The builders never fit. Callers using these lower-level extension points must fit only with
`split.train_features` and `split.train_target`.

## Training and evaluation

Import from `mlforge.training`.

| Public name | Purpose |
| --- | --- |
| `TrainingConfig` | Task, estimator, split, preprocessing, and feature-role configuration. |
| `train(dataset, config, *, run_store=None)` | Fit one baseline, evaluate held-out rows, and persist a terminal run. |
| `TrainingResult` | Fitted pipeline, successful manifest/path, schema, and raw feature dtypes. |
| `evaluate_predictions(*, task, actual, predicted)` | Calculate the stable held-out metric set. |
| `LOGISTIC_REGRESSION`, `RANDOM_FOREST_CLASSIFIER` | Classification estimator identifiers. |
| `RIDGE_REGRESSION`, `RANDOM_FOREST_REGRESSOR` | Regression estimator identifiers. |
| `CLASSIFICATION_ESTIMATORS`, `REGRESSION_ESTIMATORS`, `ALL_ESTIMATORS` | Supported estimator collections. |

If `run_store` is omitted, `train` writes to `mlruns/`. Expected failures after a run begins are
recorded and raised as `TrainingFailedError`; invalid configuration fails before a run starts.

## Runs

Import from `mlforge.runs`.

| Public name | Purpose |
| --- | --- |
| `LocalRunStore` | Create-only local JSON manifest storage with validated reads. |
| `compare_runs(manifests, *, metric)` | Rank compatible successful runs using recorded metric direction. |
| `RunComparison`, `RunComparisonEntry` | Immutable comparison result and ranked entries. |
| `RunManifest`, `RunStatus`, `RUN_MANIFEST_SCHEMA_VERSION` | Versioned terminal run record and status. |
| `RunConfiguration`, `RunParameter` | Effective user and estimator configuration. |
| `DatasetSnapshot`, `EnvironmentSnapshot`, `SplitSnapshot` | Recorded provenance and split identity. |
| `MetricValue`, `RunFailure` | Metric semantics and safe terminal failure details. |

Comparison is intentionally strict: runs must share the same data fingerprint, task, target, split
contract, and exact row partition.

## Artifacts

Import from `mlforge.artifacts`.

| Public name | Purpose |
| --- | --- |
| `LocalArtifactStore` | Create-only local model-artifact storage keyed by successful run UUID. |
| `inspect_artifact(path)` | Validate archive structure, manifest, size, and checksum without deserializing. |
| `load_artifact(path, *, trusted=False)` | Load a compatible pickle pipeline only with explicit trust. |
| `verify_run_manifest(artifact, run)` | Verify artifact lineage against a canonical run manifest. |
| `ArtifactManifest`, `ArtifactEnvironment`, `ArtifactFeature`, `FeatureRole` | Safe metadata and ordered feature contract. |
| `SavedArtifact`, `LoadedArtifact` | Persisted and explicitly loaded artifact results. |
| `ARTIFACT_MANIFEST_SCHEMA_VERSION`, `ARTIFACT_SERIALIZATION_FORMAT`, `ARTIFACT_SUFFIX` | Versioned format constants. |

Never pass `trusted=True` merely to bypass the default error. Read [artifact security](security.md)
and establish the artifact's source and custody first.

## Inference

Import from `mlforge.inference`.

| Public name | Purpose |
| --- | --- |
| `predict_frame(artifact, frame)` | Validate exact input schema and predict an in-memory batch. |
| `predict_csv(artifact, path, *, options=None)` | Strictly load a target-free CSV and predict it. |
| `PredictionResult`, `PredictionRecord`, `PredictionValue` | JSON-safe result, one-based row records, and scalar output type. |

Prediction requires a `LoadedArtifact`, every recorded feature exactly once, and no extra columns.
The input is safely reordered to training order after validation.

## Configuration and logging

Import `ApplicationConfig`, `LogLevel`, and `LOG_LEVEL_ENVIRONMENT_VARIABLE` from
`mlforge.config`. `ApplicationConfig.from_environment()` resolves the optional
`MLFORGE_LOG_LEVEL`; `with_overrides()` applies an explicit entrypoint value.

`mlforge.logging_config` exports `configure_logging(level, *, stream=None)` and `LOGGER_NAME` for
application adapters. Importing the package never configures handlers. Library code should log to
the `mlforge` logger and leave handler policy to the calling application.

## Errors

All expected domain exceptions inherit from `mlforge.errors.MLForgeError`:

```text
MLForgeError
|- ConfigurationError
|- DatasetError
|  |- DatasetPathError
|  |- DatasetFormatError
|  `- DatasetValidationError
|- PipelineError
|  |- DatasetSplitError
|  `- PreprocessingError
|- TrainingError
|  `- TrainingFailedError
|- RunError
|  |- RunStoreError
|  `- RunComparisonError
|- ArtifactError
|  |- ArtifactPathError
|  |- ArtifactFormatError
|  |- ArtifactIntegrityError
|  |- ArtifactTrustError
|  `- ArtifactCompatibilityError
`- InferenceError
   `- PredictionSchemaError
```

Catch a specific subclass when recovery differs. Catch `MLForgeError` at application boundaries
that need one user-facing error path. Programming errors and unexpected system failures are not
silently converted into domain errors.
