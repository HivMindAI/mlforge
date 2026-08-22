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
| `CrossValidationSplitConfig` | Validated 2-10 fold count and reproducible shuffle seed. |
| `split_classification_folds(dataset, *, config=None)` | Create deterministic, stratified, index-preserving classification folds. |
| `split_partition_sha256(split)` | Fingerprint one exact train/validation row partition. |
| `DatasetSplit` | Index-preserving split result with copied partitions and actual stratification state. |
| `FeatureOverrides` | Explicit numeric/categorical role overrides for ambiguous columns. |
| `FeatureSchema` | Ordered numeric and categorical training-feature contract. |
| `NumericImputationStrategy` | Supported `mean` or `median` numeric imputation. |
| `PreprocessingConfig` | Numeric imputation, scaling, and categorical missing-marker settings. |
| `infer_feature_schema(features, *, overrides=None)` | Infer roles from the training dataframe only. |
| `build_preprocessor(split, *, config=None, overrides=None)` | Return an unfitted column transformer. |
| `build_model_pipeline(split, estimator, *, config=None, overrides=None)` | Clone an estimator and return an unfitted preprocessing/model pipeline. |
| `build_final_preprocessor(features, *, config=None, overrides=None)` | Return an unfitted all-row transformer for explicit final fitting. |
| `build_final_model_pipeline(features, estimator, *, config=None, overrides=None)` | Clone an estimator and return an unfitted all-row final pipeline. |

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
| `DUMMY_CLASSIFIER`, `LOGISTIC_REGRESSION`, `RANDOM_FOREST_CLASSIFIER` | Classification estimator identifiers. |
| `RIDGE_REGRESSION`, `RANDOM_FOREST_REGRESSOR` | Regression estimator identifiers. |
| `CLASSIFICATION_ESTIMATORS`, `REGRESSION_ESTIMATORS`, `ALL_ESTIMATORS` | Supported estimator collections. |
| `CLASSIFICATION_METRICS` | Stable classification metrics accepted as benchmark objectives. |

If `run_store` is omitted, `train` writes to `mlruns/`. Expected failures after a run begins are
recorded and raised as `TrainingFailedError`; invalid configuration fails before a run starts.

## Benchmarks

Import from `mlforge.benchmarks`.

| Public name | Purpose |
| --- | --- |
| `BenchmarkConfig` | Unique classification estimators, primary metric, shared split, preprocessing, and feature roles. |
| `benchmark(dataset, config, *, run_store=None, benchmark_store=None)` | Train, rank, and record several classification baselines. |
| `BenchmarkResult` | Aggregate manifest/path, fitted successful pipelines, all referenced run manifests, and fitted winner. |
| `BenchmarkManifest`, `BenchmarkEntry`, `BenchmarkStatus` | Versioned aggregate evidence and terminal per-estimator outcomes. |
| `BenchmarkConfiguration` | Serialized effective benchmark configuration. |
| `LocalBenchmarkStore` | Create-only validated local benchmark-manifest storage. |
| `DEFAULT_CLASSIFICATION_BENCHMARK_ESTIMATORS` | Dummy, logistic-regression, and random-forest defaults. |
| `BENCHMARK_MANIFEST_SCHEMA_VERSION` | Independent aggregate manifest schema version. |
| `CrossValidationConfig` | Unique classifiers, primary metric, shared fold plan, preprocessing, and feature roles. |
| `cross_validate_benchmark(dataset, config, *, store=None)` | Fit and evaluate classifiers independently across one shared stratified fold plan. |
| `CrossValidationResult` | Immutable selection manifest and its persisted path; no fitted deployment model. |
| `CrossValidationManifest`, `CrossValidationEntry` | Strict aggregate protocol and terminal per-estimator outcomes. |
| `CrossValidationFoldSnapshot`, `CrossValidationFoldResult` | Exact shared partition identity and one estimator's fold metrics. |
| `CrossValidationMetricSummary` | Ordered fold values, arithmetic mean, population standard deviation, and direction. |
| `CrossValidationConfiguration` | Serialized effective cross-validation configuration. |
| `LocalCrossValidationStore` | Create-only validated cross-validation manifest storage. |
| `CROSS_VALIDATION_MANIFEST_SCHEMA_VERSION` | Independent cross-validation manifest schema version. |

The service calls the ordinary `train` application service once per estimator. Each run therefore
retains its complete lineage and fitted preprocessing boundary. Successful runs must share the
same source fingerprint, target, split configuration, actual stratification policy, and exact row
partition before ranking. Ties are resolved by estimator identifier, and failed estimators remain
in the aggregate manifest without a rank. `BenchmarkResult.winner` returns the fitted rank-one
`TrainingResult` for optional artifact persistence.

The holdout benchmark's winner means best observed for that selected metric and partition; it does
not establish a universally best model. If every model fails, MLForge writes the aggregate and
raises `BenchmarkFailedError`.

Cross-validation uses every row for validation exactly once and gives every estimator the same
ordered partition fingerprints. Each fold receives a fresh estimator clone and a newly fitted
preprocessing pipeline learned only from that fold's training rows. It records every metric's fold
values, arithmetic mean, and population standard deviation. Rank is primary-metric mean first,
then lower standard deviation, then estimator identifier. A failed estimator records the failing
fold and partition while successful peers remain ranked; an all-failed manifest is persisted
before `BenchmarkFailedError` is raised. `CrossValidationResult` deliberately has no fitted winner:
selecting an estimator is separate from training a final model, and non-nested cross-validation is
not an unbiased estimate after tuning or repeated selection.

## Final models

Import from `mlforge.final_models`.

| Public name | Purpose |
| --- | --- |
| `fit_selected_model(dataset, selection, *, final_model_store=None, artifact_store=None)` | Verify persisted CV selection and exact data, refit the rank-one estimator on all rows, and save its artifact. |
| `FinalModelResult` | Fitted pipeline, strict manifest/path, artifact path, input schema, and raw dtypes. |
| `FinalModelManifest` | Terminal all-row fit record separating selection evidence, final-fit scope, and artifact payload lineage. |
| `FinalModelArtifact` | Artifact/model identity, serialization format, pipeline size, and pipeline SHA-256 contract. |
| `FinalModelSelection` | Benchmark UUID/digest, selected estimator/metric aggregates, fold count, and fold-plan digest. |
| `FinalModelConfiguration` | Reconstructed preprocessing, feature roles, seed, estimator, and exact parameters. |
| `LocalFinalModelStore` | Create-only validated final-model manifest storage. |
| `FINAL_MODEL_FIT_SCOPE` | Canonical `all_rows` scope recorded for every final fitting attempt. |
| `FINAL_MODEL_MANIFEST_SCHEMA_VERSION` | Independent final-model record schema version. |

`fit_selected_model` accepts only a `CrossValidationResult` backed by its unchanged regular-file
manifest. The loaded dataset must still match its own ingestion metadata, source CSV, and the
selection's dataset identity. The service fits every selected row, writes the immutable final-model
record, persists through `LocalArtifactStore`, and returns its artifact path. It deliberately
records no training-set metric. Expected fit failures receive an immutable failed manifest;
lineage failures stop before fitting or writing.

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
| `LocalArtifactStore` | Create-only local model-artifact storage keyed by a run or final-model UUID. |
| `LocalArtifactStore.save_final(result)` | Persist a successful final pipeline with verified final-manifest lineage. |
| `inspect_artifact(path)` | Validate archive structure, manifest, size, and checksum without deserializing. |
| `load_artifact(path, *, trusted=False)` | Load a compatible pickle pipeline only with explicit trust. |
| `verify_run_manifest(artifact, run)` | Verify artifact lineage against a canonical run manifest. |
| `verify_final_model_manifest(artifact, final_model)` | Verify artifact lineage against a canonical final-model manifest. |
| `ArtifactManifest`, `ArtifactEnvironment`, `ArtifactFeature`, `FeatureRole`, `ArtifactLineageKind` | Safe metadata, lineage kind, and ordered feature contract. |
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
| `write_predictions_csv(result, path)` | Atomically create a UTF-8 prediction CSV without overwriting. |
| `PredictionResult`, `PredictionRecord`, `PredictionValue` | JSON-safe result, one-based row records, and scalar output type. |

Prediction requires a `LoadedArtifact`, every recorded feature exactly once, and no extra columns.
The input is safely reordered to training order after validation. CSV output contains
`row_number` and `prediction`; it creates missing parents and refuses existing destinations.

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
|- BenchmarkError
|  |- BenchmarkStoreError
|  `- BenchmarkFailedError
|- FinalModelError
|  |- FinalModelLineageError
|  |- FinalModelStoreError
|  `- FinalModelFailedError
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
