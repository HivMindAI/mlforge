# MLForge Completion Roadmap

MLForge will become useful by completing one local tabular-ML workflow before adding platform
infrastructure. Each milestone is a review gate. The next milestone starts only after the project
owner accepts the current one with `DONE`.

## Priorities

- **Critical:** Milestones 0-4 establish a correct, usable local workflow.
- **Important:** Milestones 5-6 make artifacts usable and prepare a responsible public release.
- **Local benchmarking:** Milestones 7-8 are accepted for the v0.2.0 local release. Milestones 9+
  remain conditional and require separate product decisions before implementation.

## Critical milestones

### Milestone 0 - Repository stabilization

**Status:** Accepted by the project owner on 2026-08-12.

**What changes:** Keep one source of package version metadata; provide truthful CLI help and
version behavior; test installed metadata and entrypoints; make `.env.example` trackable; include
the package build frontend in development dependencies; and document the real status, target
architecture, development workflow, and ordered roadmap.

**Why:** Every later milestone depends on a package that installs, builds, tests, and communicates
its capabilities without metadata drift or placeholder claims.

**Affected files:** `pyproject.toml`, `.gitignore`, `src/mlforge/cli.py`,
`src/mlforge/__main__.py`, `tests/`, `README.md`, `ROADMAP.md`, and
`docs/architecture.md`.

**Dependencies:** None.

**Verification:** Ruff lint and format checks; pytest unit/integration tests; editable install;
`mlforge --help`; `mlforge --version`; `python -m mlforge --version`; wheel and source builds; and
inspection of the clean Git diff.

**Done means:** All verification passes, docs make no unimplemented capability claims, and the
owner reviews the milestone. License selection was intentionally deferred to Milestone 6 before a
public release.

### Milestone 1 - Typed application foundation

**Status:** Accepted by the project owner on 2026-08-12.

**What changes:** Add a small domain error hierarchy, immutable application configuration with
default/environment/CLI precedence, logging setup configured only by an entrypoint, strict type
checking, CI for supported Python versions, and contributor setup guidance. Dataset-specific
configuration remains with the dataset implementation in Milestone 2 rather than existing unused.

**Why:** Dataset code needs consistent errors and settings, while contributors need automated
feedback before core ML behavior expands.

**Affected modules:** `src/mlforge/errors.py`, `src/mlforge/config.py`,
`src/mlforge/logging_config.py`, CLI wiring, `pyproject.toml`, `.env.example`,
`.github/workflows/`, tests, `AGENTS.md`, and contributor/user/architecture documentation.

**Dependencies:** Milestone 0.

**Testing:** Configuration validation and precedence tests, logging capture tests, type checks, CI
matrix execution, plus all existing checks.

**Done means:** Invalid settings fail clearly, importing MLForge has no logging or filesystem side
effects, static checks pass, and CI reproduces the documented local commands.

### Milestone 2 - Dataset ingestion and profiling

**Status:** Accepted by the project owner on 2026-08-12.

**What changes:** Add explicit CSV loading options, path/file validation, a typed loaded-dataset
result, stable metadata, target-column validation, and a JSON-serializable profile covering schema,
missingness, cardinality, likely identifiers, target balance, and basic numeric summaries.

**Why:** Every downstream operation needs one validated interpretation of the source data. A
profile makes problems visible before splitting or training.

**Affected modules:** `src/mlforge/datasets/`, configuration and errors, CLI `dataset` commands,
fixtures, tests, and a small example dataset or generated example.

**Dependencies:** Milestone 1. Pandas `>=3.0,<4` is the sole runtime dependency and
`pandas-stubs>=3.0,<4` supports development type checks.

**Testing:** Empty, missing, malformed, oversized, duplicate-header, mixed-type, missing-value,
high-cardinality, identifier-like, classification-target, and regression-target cases; profile JSON
round trips; CLI success and error paths.

**Done means:** A developer can load a supported CSV, receive deterministic metadata/profile JSON,
and get actionable errors for unsupported inputs without changing the source file.

### Milestone 3 - Leakage-safe splitting and preprocessing

**Status:** Accepted by the project owner on 2026-08-12.

**What changes:** Define task and preprocessing configuration; split features and target before
fitting transformations; implement deterministic train/validation splitting; build numeric and
categorical transformers; and return an unfitted estimator-compatible pipeline.

**Why:** Fitting encoders, imputers, or scalers before the validation split produces misleading
metrics. Leakage prevention must be an architectural boundary rather than caller discipline.

**Affected modules:** `src/mlforge/pipelines/`, dataset types, configuration, errors, tests, and
architecture documentation. Add scikit-learn only in this milestone.

**Dependencies:** Milestone 2.

**Testing:** Transformer fit boundaries, unseen categories, all-missing columns, unsupported target
shapes, deterministic splits, stratification behavior, tiny datasets, and fitted/unfitted state.

**Done means:** Tests prove validation rows do not influence fitted preprocessing state, and the
resulting pipeline handles documented numeric/categorical inputs consistently.

### Milestone 4 - Local training, evaluation, and run records

**Status:** Accepted by the project owner on 2026-08-13.

**What changes:** Support a deliberately small set of baseline classification and regression
estimators; fit the complete pipeline; calculate task-appropriate metrics; record configuration,
data fingerprint, library versions, timestamps, warnings, and outcomes in an immutable local run
manifest; and expose thin CLI commands for training and run inspection.

**Why:** This is the first complete value path and the point at which MLForge becomes more than data
utilities.

**Affected modules:** `src/mlforge/training/`, `src/mlforge/runs/`, pipeline APIs, CLI, examples,
tests, and user documentation.

**Dependencies:** Milestone 3.

**Testing:** Classification/regression integrations, metric correctness, fixed-seed reproducibility,
failed-run recording, atomic manifest writes, unsupported estimator/task combinations, and a
documented end-to-end example.

**Done means:** One command and one importable API can train and compare real local runs, with enough
metadata to explain what data, configuration, code dependencies, and metrics produced each result.

## Important milestones

### Milestone 5 - Versioned artifacts and batch inference

**Status:** Accepted by the project owner on 2026-08-13.

**What changes:** Save the fitted pipeline with a versioned manifest and input schema; use atomic
writes and integrity checks; load only explicitly trusted local artifacts; validate prediction
inputs; and emit structured batch predictions and errors.

**Why:** A metric is not useful unless the exact fitted pipeline can be applied consistently. Python
model serialization can execute code, so the trust boundary must be prominent and tested.

**Affected modules:** `src/mlforge/artifacts/`, `src/mlforge/inference.py`, run records, CLI, tests,
security documentation, and examples.

**Dependencies:** Milestone 4.

**Testing:** Save/load parity, manifest-version rejection, schema mismatch, missing/extra columns,
checksum mismatch, partial writes, explicit trust requirements, and end-to-end batch prediction.

**Done means:** A locally trained artifact produces repeatable predictions for valid data and fails
closed with clear errors for incompatible, corrupted, unsupported, or untrusted inputs.

### Milestone 6 - Local product and release readiness

**Status:** Accepted by the project owner on 2026-08-16.

**What changes:** Stabilize the intentionally small public API; complete tutorials and reference
documentation; add package-install smoke tests from built wheels; choose a license with owner
approval; add contribution and security policies; define compatibility/versioning rules; and remove
dead or experimental public code.

**Why:** Passing unit tests is not sufficient for a reliable open-source release. Installation,
documentation, legal permissions, and compatibility are part of the product contract.

**Affected files:** Public package exports, examples, docs, CI/release configuration, project
metadata, `LICENSE`, `CONTRIBUTING.md`, and `SECURITY.md`.

**Dependencies:** Milestone 5 and an owner license decision.

**Testing:** Clean-environment wheel installation on supported Python versions, tutorial execution,
API/interface tests, documentation link/code checks, and the full quality suite.

**Done means:** A new developer can clone or install MLForge, complete the documented workflow, and
understand supported behavior and limitations without reverse-engineering the source.

### Milestone 7 - Local classification benchmarking

**Status:** Accepted by the project owner on 2026-08-18.

**What changes:** Add a local benchmark application service that trains a small explicit set of
classification baselines against one shared deterministic holdout contract; includes a dummy
baseline; ranks successful runs by a user-selected classification metric; records observed
per-model duration and failures; writes a separate immutable benchmark manifest that references
the underlying run manifests; and exposes the same workflow through Python and a thin CLI command.

**Why:** MLForge already records trustworthy individual experiments. A benchmark turns those
experiments into one reproducible answer to a common user question: which supported baseline
performed best under this declared evaluation protocol, and what evidence produced that ranking?

**Affected modules:** `src/mlforge/benchmarks/`, classification estimator and metric definitions,
CLI and public API surfaces, local manifest storage, tests, examples, and user/architecture
documentation.

**Dependencies:** Milestone 6. The first version remains local, single-process, classification-only,
and scikit-learn-only.

**Testing:** Configuration validation, dummy-baseline behavior, identical split fingerprints,
metric-direction ranking and deterministic tie-breaking, partial and complete estimator failures,
manifest schema/atomicity/path safety, CLI JSON and human output, public interfaces, and the full
quality suite.

**Done means:** One command and one importable API run at least two classification baselines on the
same dataset partition, preserve every underlying run, create a validated immutable benchmark
record, identify the best observed model for the selected metric, and state the single-holdout
limitation without claiming a universally best model.

### Milestone 8 - Cross-validated classification benchmarking

**Status:** Accepted by the project owner on 2026-08-18.

**What changes:** Add deterministic stratified K-fold partitioning; fit preprocessing and each
classifier independently inside every training fold; evaluate every estimator on the exact same
fold plan; aggregate all classification metrics as per-fold values, means, and population standard
deviations; rank by primary-metric mean with stability-aware deterministic tie-breaking; preserve
failures and warnings; and store the complete protocol in a separate immutable cross-validation
manifest exposed through Python and the benchmark CLI.

**Why:** A single holdout is useful for a fast baseline but can produce a fragile ranking. Shared
K-fold evaluation uses every row for validation once, shows fold-to-fold variability, and provides
stronger evidence without claiming that model selection has produced an unbiased final deployment
estimate.

**Affected modules:** Fold configuration/splitting and partition fingerprints in
`src/mlforge/pipelines/`; cross-validation services, types, storage, and CLI adapters in
`src/mlforge/benchmarks/`; examples, tests, and user/architecture/API documentation.

**Dependencies:** Milestone 7. The protocol remains local, single-process, classification-only,
scikit-learn-only, and resource-bounded to 2-10 folds.

**Testing:** Fold coverage/disjointness and deterministic fingerprints, insufficient class counts,
same-fold fairness across estimators, preprocessing fit boundaries, metric aggregation and ranking,
partial/complete estimator failures, strict manifest round trips and atomic storage, CLI JSON/human
output, public interfaces, runnable examples, and the full quality suite.

**Done means:** One command and one importable API run a reproducible shared stratified K-fold
classification benchmark, expose mean and variability for every metric, keep all validation rows
out of their fold's fitted preprocessing, record a strict immutable aggregate, and clearly separate
model-selection evidence from final model fitting and nested tuning claims.

## Conditional improvements

### Milestone 9 - Service adapters and shared experiment storage

**What changes:** Only if multi-user or remote execution is a demonstrated requirement, introduce a
versioned HTTP API, transactional persistence, shared artifact storage, and background execution as
separate adapters around the existing application services. Add a model-registry concept only when
promotion/version lifecycle requirements are defined.

**Why:** These capabilities solve coordination and scale problems but would obscure the core design
if added before the local workflow is stable.

**Affected areas:** New `api`, persistence, worker, and storage adapter packages; migrations;
deployment configuration; integration tests; and operational docs.

**Dependencies:** Milestone 6 plus written product requirements for concurrency, tenancy, artifact
volume, and deployment environment.

**Testing:** Contract tests against local services, transaction and idempotency tests, migration
tests, authorization boundaries where applicable, worker retry/timeout behavior, and failure
recovery.

**Done means:** Remote behavior preserves the same domain semantics as the local API, failures are
observable and recoverable, and no infrastructure concern leaks into core ML modules.

### Milestone 10 - Deployment, observability, and user interface

**What changes:** Evaluate a web interface, online inference, model promotion/rollback, prediction
telemetry, drift analysis, authentication/authorization, containerization, and production release
automation individually. Implement only capabilities backed by accepted use cases and measurable
operational requirements.

**Why:** A dashboard or monitoring stack is useful only when there are real users, deployed models,
service-level goals, and feedback loops to support.

**Affected areas:** Separate frontend and deployment projects or adapters, API surfaces,
observability, security, operations docs, and system tests.

**Dependencies:** Milestone 9 and explicit product decisions for deployment, privacy, retention,
security, and monitoring.

**Testing:** Threat modeling, authorization tests, deployment smoke/rollback tests, load and latency
checks, telemetry privacy tests, drift-method validation, and end-to-end user journeys.

**Done means:** Each shipped capability has a named user problem, documented operational contract,
tested failure behavior, and a maintainer rather than existing only for platform appearance.
