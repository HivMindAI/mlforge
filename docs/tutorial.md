# Complete Local Workflow Tutorial

This tutorial exercises MLForge's supported product path: validate data, inspect it, train and
evaluate a baseline, retain the run record, save the exact fitted pipeline, and make
schema-validated predictions.

## 1. Install a development checkout

MLForge needs Python 3.11 or newer. Users of a published release install the distribution with
`python -m pip install hivmind-mlforge`; both `import mlforge` and the `mlforge` command keep the
short project name. From a development clone:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate with `source .venv/bin/activate`. Confirm that the package and console
entrypoint agree:

```powershell
mlforge --version
python -m mlforge --version
```

## 2. Validate and profile the data

The bundled model-ready dataset has a binary `churn` target, numeric columns, a categorical region,
and one missing numeric value:

```powershell
mlforge dataset profile examples/customer_churn.csv --target churn
mlforge dataset profile examples/customer_churn.csv --target churn --json
```

Profiling reports facts and conservative warnings. It does not drop columns, repair missing values,
or automatically select classification. Training uses every non-target column, so prepare the CSV
without identifiers, timestamps, post-outcome fields, or other columns that should not become model
features. Those remain explicit modeling decisions.

## 3. Train and save an artifact

Run the complete leakage-safe training path:

```powershell
mlforge train examples/customer_churn.csv --target churn --task classification --estimator logistic-regression --runs-dir mlruns --artifacts-dir artifacts --json
```

The command prints a successful run manifest containing a UUID. It writes:

- `mlruns/RUN_ID.json`: immutable lineage, configuration, split identity, metrics, and versions;
- `artifacts/RUN_ID.mlforge`: the fitted preprocessing-and-model pipeline plus its strict manifest.

The run is evaluated only on held-out rows. Preprocessing is fitted inside the pipeline using only
training rows. If expected fitting or evaluation fails after the run starts, MLForge writes a
failed terminal run instead of pretending the run did not exist.

## 4. Inspect before loading

Replace `RUN_ID` with the UUID printed above:

```powershell
mlforge runs show RUN_ID --runs-dir mlruns --json
mlforge artifacts inspect artifacts/RUN_ID.mlforge --json
```

Artifact inspection validates bounded structure and the pipeline checksum without deserializing
the executable payload. A valid checksum proves consistency with the artifact manifest, not who
created it.

## 5. Predict only after a trust decision

The bundled prediction CSV contains the four feature columns and no target:

```powershell
mlforge predict artifacts/RUN_ID.mlforge examples/prediction_customers.csv --trust-artifact --json
```

For larger batches, write only the compact save summary to the terminal and place prediction rows
in a new UTF-8 CSV:

```powershell
mlforge predict artifacts/RUN_ID.mlforge examples/prediction_customers.csv --trust-artifact --output predictions.csv
```

The file contains `row_number` and `prediction` columns. Missing parent directories are created;
an existing output is never overwritten.

The trust flag is required because the artifact contains pickle data. Use it only for an artifact
whose source and custody you have verified. MLForge then requires the exact recorded environment,
checks every feature name and role, restores training column order, and emits one-based JSON-safe
prediction records. See [artifact security](security.md) for the complete boundary.

## 6. Use the Python API

The same workflow is available without the CLI:

```python
from pathlib import Path

from mlforge.artifacts import LocalArtifactStore
from mlforge.datasets import load_csv
from mlforge.inference import predict_csv, write_predictions_csv
from mlforge.pipelines import TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import LOGISTIC_REGRESSION, TrainingConfig, train

dataset = load_csv(Path("examples/customer_churn.csv"), target="churn")
configuration = TrainingConfig(
    task=TaskType.CLASSIFICATION,
    estimator=LOGISTIC_REGRESSION,
)
result = train(
    dataset,
    configuration,
    run_store=LocalRunStore(Path("mlruns")),
)

artifact_store = LocalArtifactStore(Path("artifacts"))
saved = artifact_store.save(result)
print(saved.manifest.to_json())

# Pickle loading executes code. This assertion is appropriate only for a verified source.
artifact = artifact_store.load(result.manifest.run_id, trusted=True)
predictions = predict_csv(artifact, Path("examples/prediction_customers.csv"))
write_predictions_csv(predictions, Path("predictions.csv"))
```

`result.pipeline` is already fitted. Artifact saving is a separate explicit operation in the
Python API so applications control whether executable model data is persisted.

## 7. Compare compatible experiments

Train a second classifier with the same dataset and split contract:

```powershell
mlforge train examples/customer_churn.csv --target churn --task classification --estimator random-forest-classifier --runs-dir mlruns --json
mlforge runs list --runs-dir mlruns
mlforge runs compare RUN_ID_1 RUN_ID_2 --metric accuracy --runs-dir mlruns --json
```

Comparison rejects runs with different source bytes, target, task, validation fraction, seed,
actual stratification, or row partition. That prevents a polished ranking of metrics that were not
measured on the same holdout.

## 8. Benchmark classification baselines

Run the default local benchmark on one shared deterministic holdout:

```powershell
mlforge benchmark examples/customer_churn.csv --target churn --metric balanced_accuracy --runs-dir mlruns --benchmarks-dir mlbenchmarks
```

MLForge trains a dummy prior baseline, logistic regression, and random forest. Every estimator uses
the same source bytes, target, preprocessing policy, seed, and validation rows. The leaderboard
ranks successful runs by the explicit primary metric, reports observed wall-clock duration, and
keeps estimator failures visible. The aggregate JSON manifest is stored in `mlbenchmarks/`, while
the complete underlying run manifests remain in `mlruns/`.

This fast benchmark is based on one holdout partition. “Best observed” means first for this
dataset, metric, seed, and partition; it is not a universal claim about an algorithm.

Use the same operation from Python when the fitted winning pipeline is needed:

```python
from pathlib import Path

from mlforge.benchmarks import BenchmarkConfig, LocalBenchmarkStore, benchmark
from mlforge.datasets import load_csv
from mlforge.runs import LocalRunStore

dataset = load_csv(Path("examples/customer_churn.csv"), target="churn")
result = benchmark(
    dataset,
    BenchmarkConfig(primary_metric="balanced_accuracy"),
    run_store=LocalRunStore(Path("mlruns")),
    benchmark_store=LocalBenchmarkStore(Path("mlbenchmarks")),
)
winning_pipeline = result.winner.pipeline
```

## 9. Compare stability with cross-validation

Use a shared deterministic stratified fold plan when one holdout is too fragile:

```powershell
mlforge benchmark examples/customer_churn.csv --target churn --metric balanced_accuracy --cross-validation-folds 3 --benchmarks-dir mlbenchmarks
```

Regression uses deterministic shuffled K-folds and lower-is-better RMSE by default:

```powershell
mlforge benchmark examples/house_prices.csv --target price --task regression --cross-validation-folds 5 --benchmarks-dir mlbenchmarks
```

The bundled data supports three folds because its smallest target class has three rows. For every
estimator, MLForge clones a fresh model and fits a new preprocessing pipeline on each fold's
training rows. The fold's validation rows never contribute imputation values, scaling state,
categories, or estimator parameters. Every source row becomes validation data exactly once.

The leaderboard displays the primary metric as mean ± population standard deviation. Its strict
manifest in `mlbenchmarks/cross-validation/` also records all fold values for every task metric,
exact partition fingerprints, warnings, timing, and any failed fold. Mean determines rank;
lower variability and then estimator name break exact mean ties deterministically.
The aggregate is self-contained and no ordinary `mlruns/` records are produced in this mode;
accordingly, `--runs-dir` is a holdout-only option and is rejected when folds are requested.

The same protocol is importable:

```python
from pathlib import Path

from mlforge.benchmarks import (
    CrossValidationConfig,
    LocalCrossValidationStore,
    cross_validate_benchmark,
)
from mlforge.datasets import load_csv
from mlforge.pipelines import CrossValidationSplitConfig

dataset = load_csv(Path("examples/customer_churn.csv"), target="churn")
result = cross_validate_benchmark(
    dataset,
    CrossValidationConfig(
        primary_metric="balanced_accuracy",
        split=CrossValidationSplitConfig(fold_count=3, random_seed=42),
    ),
    store=LocalCrossValidationStore(Path("mlbenchmarks/cross-validation")),
)
print(result.manifest.to_json())
```

This result selects an estimator under the declared folds. It intentionally does not return a
fitted pipeline. Because the same folds informed selection, do not describe the winning
cross-validation mean as a nested-tuning or untouched final-test estimate.

For regression in Python, set `task=TaskType.REGRESSION`, choose Ridge Regression and Random Forest
Regressor, and set `primary_metric="root_mean_squared_error"`. The same finalization call below
then fits the selected regressor on every verified row.

## 10. Explicitly fit the selected final model

Copy the cross-validation benchmark UUID and explicitly refit its winner on every selected row:

```powershell
mlforge finalize examples/customer_churn.csv --target churn --benchmark-id BENCHMARK_ID --benchmarks-dir mlbenchmarks --final-models-dir mlfinalmodels --artifacts-dir artifacts --json
```

Before fitting, MLForge re-reads the immutable selection manifest, verifies its digest and UUID,
revalidates the source CSV, and requires the exact selected SHA-256, parser settings, target, and
shape. It reconstructs the recorded preprocessing, feature overrides, random seed, estimator, and
estimator parameters. A new pipeline learns from all rows, so no new validation metric is created.

The command creates:

- `mlfinalmodels/FINAL_MODEL_ID.json`: terminal all-row fitting evidence and CV lineage;
- `artifacts/FINAL_MODEL_ID.mlforge`: fitted pipeline, input contract, and final-manifest digest.

The Python API keeps selection, fitting, persistence, and executable loading explicit:

```python
from pathlib import Path

from mlforge.artifacts import LocalArtifactStore
from mlforge.final_models import LocalFinalModelStore, fit_selected_model

final_model = fit_selected_model(
    dataset,
    result,
    final_model_store=LocalFinalModelStore(Path("mlfinalmodels")),
    artifact_store=LocalArtifactStore(Path("artifacts")),
)
print(final_model.artifact_path)
```

Safe inspection and trusted prediction use the same commands as evaluated-run artifacts. The
version-2 artifact manifest identifies `lineage_kind` as `final-model`. The immutable final-model
record separates cross-validation values under `selection_evidence` from the metric-free
`final_fit` block and records the artifact payload SHA-256. See the runnable
`examples/finalize_customer_churn.py` workflow.

## 11. Verify the checkout

Run the executable examples and all contributor checks:

```powershell
python examples/profile_dataset.py
python examples/preprocess_dataset.py
python examples/train_customer_churn.py
python examples/benchmark_customer_churn.py
python examples/cross_validate_customer_churn.py
python examples/finalize_customer_churn.py
python examples/train_and_predict.py
ruff check .
ruff format --check .
mypy src tests
python -m pytest
python -m build
```

Generated `mlruns/`, `mlbenchmarks/`, `mlfinalmodels/`, and `artifacts/` directories are ignored by
Git. Delete them when you no longer need the local outputs. Continue with the
[API reference](api.md) for lower-level pipeline and storage interfaces, and
[architecture](architecture.md) for design boundaries.
