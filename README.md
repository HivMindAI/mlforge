# MLForge

MLForge is a pre-alpha Python toolkit being built for reproducible tabular supervised
machine-learning workflows.

> **Current status:** the local workflow now covers validated CSV ingestion, profiling,
> deterministic splitting, leakage-safe preprocessing, baseline training, held-out evaluation, and
> immutable JSON run records, versioned trusted-local model artifacts, and schema-validated batch
> inference.

## Why MLForge

Tabular ML projects often repeat the same error-prone work: validating input data, preventing
training leakage, recording run configuration, comparing metrics, and keeping a fitted pipeline
together with the schema it expects. MLForge aims to provide one small, understandable local
workflow for those responsibilities before adding any service infrastructure.

The intended developer journey is:

```text
CSV dataset -> validation and profile -> split and preprocessing -> training and evaluation
            -> reproducible run artifact -> schema-validated batch prediction
```

## Scope and capabilities

| Area | Status |
| --- | --- |
| Python package, editable install, wheel and source build | Implemented |
| CLI bootstrap with help and version output | Implemented |
| Typed application configuration, domain errors, and explicit logging | Implemented |
| Ruff, strict mypy, pytest, and Python 3.11/3.12 CI | Implemented |
| Validated local CSV ingestion and deterministic profiling | Implemented |
| Deterministic train/validation splitting | Implemented |
| Unfitted numeric/categorical preprocessing pipelines | Implemented |
| Four local baseline estimators and held-out evaluation | Implemented |
| Versioned, immutable local run manifests and fair comparison | Implemented |
| Trusted-local model artifacts and schema-validated batch inference | Implemented |
| Database, API, workers, web UI, deployment, and monitoring | Deferred and conditional |

MLForge will initially support local classification and regression on tabular data. It is not
intended to replace notebooks for exploration, distributed training systems, or general-purpose
workflow orchestrators.

## Requirements

- Python 3.11 or newer
- Git, only when installing a source checkout

MLForge uses pandas 3.x for tabular-data behavior and scikit-learn `>=1.9,<2` for splitting and
estimator-compatible pipelines. Dependencies needed by later ML capabilities will be introduced
only in the milestone that uses them.

## Installation

Install the published distribution into a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install hivmind-mlforge
```

The distribution is named `hivmind-mlforge`; the Python import package and console command both
remain `mlforge`:

```python
import mlforge
```

For development, clone the repository and install the development extra:

```powershell
git clone https://github.com/HivMindAI/mlforge.git
cd mlforge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Quick start

Profile the bundled example dataset, then train a classification baseline and save its artifact:

```powershell
mlforge --help
mlforge --version
mlforge dataset profile examples/customer_churn.csv --target churn
mlforge dataset profile examples/customer_churn.csv --target churn --json
mlforge train examples/customer_churn.csv --target churn `
  --task classification --estimator logistic-regression `
  --runs-dir mlruns --artifacts-dir artifacts --json
mlforge runs list --runs-dir mlruns
```

The JSON result identifies the run UUID. The artifact is `artifacts/RUN_ID.mlforge`. Verify its
strict manifest and payload checksum without loading executable model bytes, then predict a
target-free CSV only after making an explicit source-trust decision:

```powershell
mlforge artifacts inspect artifacts/RUN_ID.mlforge --json
mlforge predict artifacts/RUN_ID.mlforge examples/prediction_customers.csv `
  --trust-artifact --output predictions.csv
```

The output CSV contains `row_number` and `prediction` columns. MLForge creates missing parent
directories but refuses to overwrite an existing output. Omit `--output` to retain the existing
terminal output; add `--json` for structured terminal output.

Every command supports normal `--help`. Add `--json` to `train`, `runs list`, or `runs show` for
machine-readable output. A run ID from the training output can be inspected with:

```powershell
mlforge runs show RUN_ID --runs-dir mlruns
```

Compare two successful runs only when they use the same dataset fingerprint, target, task, split
fraction, seed, actual stratification policy, and exact row-partition fingerprint:

```powershell
mlforge runs compare RUN_ID_1 RUN_ID_2 --metric accuracy --runs-dir mlruns
```

The complete training workflow is also available as an importable API:

```python
from pathlib import Path

from mlforge.artifacts import LocalArtifactStore
from mlforge.datasets import load_csv
from mlforge.inference import predict_csv, write_predictions_csv
from mlforge.pipelines import TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import LOGISTIC_REGRESSION, TrainingConfig, train

dataset = load_csv(Path("examples/customer_churn.csv"), target="churn")
config = TrainingConfig(
    task=TaskType.CLASSIFICATION,
    estimator=LOGISTIC_REGRESSION,
)
result = train(dataset, config, run_store=LocalRunStore(Path("mlruns")))
print(result.manifest.to_json())

artifact_store = LocalArtifactStore(Path("artifacts"))
saved = artifact_store.save(result)

# Pickle-based model loading can execute code. Opt in only for a verified source.
trusted_artifact = artifact_store.load(result.manifest.run_id, trusted=True)
predictions = predict_csv(
    trusted_artifact,
    Path("examples/prediction_customers.csv"),
)
write_predictions_csv(predictions, Path("predictions.csv"))
```

The returned `result.pipeline` is the fitted preprocessing-and-model pipeline. The lower-level
split and pipeline-builder APIs remain available when a caller needs to own fitting directly:

```python
from sklearn.linear_model import LogisticRegression

from mlforge.pipelines import TaskType, build_model_pipeline, split_dataset

split = split_dataset(dataset, task=TaskType.CLASSIFICATION)
pipeline = build_model_pipeline(split, LogisticRegression(max_iter=1_000))

# The builder returns an unfitted pipeline. Fitting is explicit and training-only.
pipeline.fit(split.train_features, split.train_target)
predictions = pipeline.predict(split.validation_features)
```

`train` fits preprocessing and the estimator together on training rows, predicts only the held-out
validation rows, evaluates them, and atomically creates a terminal manifest. The Python API returns
the fitted pipeline in memory. Artifact persistence remains an explicit second operation; the CLI
performs it only when `--artifacts-dir` is supplied. A JSON run record is lineage metadata, while a
`.mlforge` archive contains the executable fitted pipeline.

Task selection is explicit: profiling hints never silently choose classification or regression.
Classification defaults to a target-stratified split; regression defaults to an unstratified split.
Both use a recorded integer seed (`42` by default), retain source row indices, and keep the target
out of feature frames. Missing targets, non-finite numeric targets, unusable tiny partitions, and
impossible stratification fail with actionable domain errors.

Numeric features are median-imputed and standardized by default. Categorical and boolean features
use a collision-checked missing marker and one-hot encoding that safely handles categories first
seen during validation or prediction. Feature roles come from training pandas dtypes. Use
`FeatureOverrides` for ambiguous columns such as an all-missing categorical field; MLForge does not
guess datetime feature engineering or coerce arbitrary text into numbers.

Run the preprocessing-only and complete training examples with:

```powershell
python examples/preprocess_dataset.py
python examples/train_customer_churn.py
python examples/train_and_predict.py
```

Supported classification estimators are logistic regression and random forest classifier;
supported regression estimators are ridge regression and random forest regressor. Classification
runs record accuracy, balanced accuracy, and weighted F1. Regression runs record mean absolute
error, R-squared, and root mean squared error. Randomized estimators and splits use the configured
seed; random forests use one process to avoid execution-policy-dependent parallel behavior.

Run manifests use a strict versioned JSON schema and contain the effective estimator and
preprocessing parameters, CSV parser choices, source-file SHA-256, dimensions, exact split
fingerprint, dependency versions, UTC timestamps, warnings, metrics, and terminal failure details
when an expected training failure occurs. Manifests are create-only: an existing run ID is never
overwritten. The default directory is `mlruns/`, which is ignored by Git.

Artifacts are create-only ZIP containers using the successful run UUID as identity. Each contains
exactly a UTF-8 JSON manifest and a protocol-5 pickle payload. Safe inspection validates archive
structure, manifest version, payload size, SHA-256, ordered input schema, environment versions, and
the canonical run-manifest hash without deserializing Python objects. Trusted loading additionally
requires `trusted=True` or `--trust-artifact` and exact Python, MLForge, pandas, NumPy, SciPy, and
scikit-learn versions. Checksums detect accidental or unauthorized byte changes; they do not prove
who created an artifact and cannot make hostile pickle content safe.

Inference requires every recorded feature exactly once, rejects extra columns such as a leftover
target, safely restores training column order, enforces numeric/categorical roles, rejects
non-finite numeric values and reserved missing markers, and emits one-based JSON-safe prediction
records. Unseen categorical values are handled by the fitted encoder. See
[docs/security.md](docs/security.md) before loading or sharing model artifacts.

The loader accepts local uncompressed `.csv` files, uses strict row-width validation, defaults to
UTF-8 with optional BOM handling, and enforces a 100 MiB limit. Encoding, one-character delimiter,
and size limit are explicit `CsvLoadOptions` or CLI options. It calculates a SHA-256 source
fingerprint, preserves missing values, requires a named target column, and never modifies the
source file. Training treats every non-target column as a feature; prepare model-ready input that
omits identifiers, post-outcome fields, and other columns that should not influence a model.

Profiles report shape and physical pandas dtypes, missingness, cardinality, duplicate rows,
finite-only numeric summaries, infinite values, constants, high-cardinality text, likely
name-and-uniqueness-based identifiers, a conservative task hint, and classification balance. Task,
identifier, cardinality, and imbalance results are warnings/heuristics—not automatic modeling
decisions.

## Configuration

The CLI log level defaults to `WARNING`. Set it through the process environment or an explicit CLI
option:

```powershell
$env:MLFORGE_LOG_LEVEL = "INFO"
mlforge
mlforge --log-level DEBUG
```

An explicit CLI value takes precedence over the environment. Valid values are `DEBUG`, `INFO`,
`WARNING`, `ERROR`, and `CRITICAL`, case-insensitively. MLForge does not automatically read `.env`
files; `.env.example` documents supported variables for shells or environment managers.

Configuration and logging happen only when an application entrypoint runs. Importing `mlforge` or
its foundation modules does not read the environment, configure handlers, or write files.
Dataset parser options are passed explicitly through Python or CLI flags; they are not hidden in
environment variables.

## Architecture

MLForge is library-first. Domain modules own dataset, pipeline, training, evaluation, run,
artifact, and inference behavior. The CLI remains a thin adapter over those importable APIs.
Remote services will not be introduced until the local workflow proves they are needed.

See [docs/architecture.md](docs/architecture.md) for module responsibilities, data flow, public API
direction, extension points, security boundaries, and testing strategy.

## Documentation

- [Complete local workflow tutorial](docs/tutorial.md)
- [Python API reference](docs/api.md)
- [Architecture and design boundaries](docs/architecture.md)
- [Compatibility and versioning policy](docs/compatibility.md)
- [Artifact trust and secure-use guidance](docs/security.md)
- [Release validation record](docs/release-validation.md)
- [Maintainer release procedure](docs/releasing.md)
- [Changelog](CHANGELOG.md)
- [Vulnerability reporting policy](SECURITY.md)

## Development

Install the development dependencies and run all current checks:

```powershell
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src tests
python -m pytest
python -m build
```

Tests require an installed package because MLForge uses the standard `src/` layout. Do not add the
repository's `src` directory to `PYTHONPATH` as a substitute for an editable install.

## Project structure

```text
mlforge/
|- .github/workflows/    # Automated quality checks and trusted release publishing
|- examples/             # Runnable profiling, training, artifact, and inference examples
|- src/mlforge/          # Importable production package
|- tests/                # Unit and integration tests
|- scripts/              # Installed-wheel verification scripts
|- docs/api.md           # Supported Python API contract
|- docs/architecture.md  # Target architecture and design boundaries
|- docs/compatibility.md # Version, runtime, and schema compatibility policy
|- docs/releasing.md     # Owner-only release and Trusted Publishing procedure
|- docs/release-validation.md # Offline real-data and clean-wheel validation record
|- docs/security.md      # Artifact trust model and secure-use guidance
|- docs/tutorial.md      # Clone-to-prediction guided workflow
|- CONTRIBUTING.md       # Contributor setup and review contract
|- CHANGELOG.md          # Versioned user-visible changes
|- LICENSE               # Apache License 2.0 terms
|- SECURITY.md           # Vulnerability reporting and support policy
|- pyproject.toml        # Package and tool configuration
|- ROADMAP.md            # Ordered milestones and definitions of done
`- AGENTS.md             # Durable contribution and validation rules
```

## Roadmap and contributing

Work proceeds one milestone at a time. The dependency-ordered plan and acceptance criteria are in
[ROADMAP.md](ROADMAP.md). Contributors should read [CONTRIBUTING.md](CONTRIBUTING.md) and
[AGENTS.md](AGENTS.md) before making changes. Security concerns should follow [SECURITY.md](SECURITY.md).
Changes should stay small, include behavioral tests, and keep documentation aligned with
implemented functionality.

## License

MLForge is licensed under the [Apache License 2.0](LICENSE). It permits use, modification, and
distribution, including commercial use, subject to its notice and other terms, and includes an
explicit patent license from contributors.
