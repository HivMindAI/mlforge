# MLForge Completion Roadmap

MLForge will become useful by completing one local tabular-ML workflow before adding platform
infrastructure. Each milestone is a review gate. The next milestone starts only after the project
owner accepts the current one with `DONE`.

## Priorities

- **Critical:** Milestones 0-4 establish a correct, usable local workflow.
- **Important:** Milestones 5-6 make artifacts usable and prepare a responsible public release.
- **Feature-complete local product:** Milestones 7-8.1 complete the Python workflow in v0.3.0;
  Milestone 8.2 packages that workflow for one local web operator in v0.4.0, and Milestone 8.3
  brings regression parity in v0.5.0.
- **Conditional platform work:** Shared-service portions of Milestones 9-10 remain postponed
  proposals, not active development commitments.

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

### Milestone 8.1 - Explicit selection-driven final-model fitting

**Status:** Accepted by the project owner on 2026-08-22; complete for v0.3.0.

**What changes:** Verify one persisted successful cross-validation result and the exact selected
dataset; reconstruct the rank-one estimator, preprocessing, feature overrides, seed, and recorded
parameters; fit one new pipeline on every selected row; persist a separate immutable final-model
manifest; extend artifact lineage without breaking version-1 training-run inspection/loading; and
expose the workflow through Python, CLI, examples, and installed-wheel smoke validation.

**Why:** Cross-validation identifies a supported estimator but intentionally returns no fitted
pipeline. A developer needs one explicit, auditable step that uses all selected data without
pretending training-set scores or reused cross-validation means are new performance evidence.

**Affected modules:** `src/mlforge/final_models/`, final-fit pipeline construction, artifact schema
and persistence, CLI, examples, smoke validation, tests, and user/architecture/security/API docs.

**Dependencies:** Milestone 8. The first version remains local, single-process,
classification-only, scikit-learn-only, and restricted to MLForge's persisted cross-validation
winner and exact dataset.

**Testing:** Full-row fitting boundaries, selection/dataset/parameter drift, success and terminal
failure manifests, create-only storage, artifact version compatibility and lineage, trusted load,
prediction, CLI JSON/human behavior, runnable examples, clean-package smoke, and the full quality
suite.

**Done means:** One explicit command and API convert verified cross-validation selection evidence
into a fitted all-row artifact with immutable lineage, while refusing changed evidence/data and
making no new evaluation or deployment-performance claim.

### Milestone 8.2 - Single-user web workflow and private release profile

**Status:** Accepted by the project owner on 2026-09-02; complete for v0.4.0.

**What changes:** Add a thin FastAPI adapter, SQLite metadata, one bounded worker, a Next.js
interface for the complete supported classification journey, and a private two-container profile.
Version the web database, include all application sources in the source distribution, and automate
the critical browser journey without changing the core ML evidence model.

**Why:** The stable local core is usable from Python and the CLI, but one trusted operator also needs
an approachable interface and a repeatable private deployment path.

**Dependencies:** Milestone 8.1 and the existing immutable local stores.

**Testing:** Core and HTTP integration tests, schema adoption/restore tests, frontend lint and
production build, Playwright golden-path coverage, package/source-archive validation, and Compose
build/start/readiness smoke testing.

**Done means:** A clean checkout or source archive can run the private application, complete upload
through prediction download, preserve a versioned workspace across restart/restore, and present
documentation and release metadata that match the shipped product.

### Milestone 8.3 - Regression workflow parity

**Status:** Accepted by the project owner on 2026-09-02; complete for v0.5.0.

**What changes:** Extend deterministic cross-validation, winner selection, verified all-row final
fitting, CLI orchestration, the single-user web workflow, and prediction to regression. Use ordinary
shuffled K-folds without target binning, rank by lower-is-better RMSE by default, preserve full MAE,
R-squared, and RMSE evidence, and migrate the web workspace without losing existing lineage.

**Why:** Regression already worked for individual held-out training, but users could not compare
regressors, select a stable winner, finalize it, or complete that journey in the web interface.

**Dependencies:** Milestone 8.2 and the existing regression training/artifact contracts.

**Testing:** Task-aware fold policy and metric direction, v1 manifest read compatibility, v1-to-v2
SQLite migration, core/CLI/HTTP finalization and prediction, frontend lint/build, and both browser
golden paths.

**Done means:** Classification remains unchanged, while a regression CSV can move from explicit
target selection through fair comparison, immutable selection evidence, final fitting, and
schema-validated predictions in Python, CLI, and the local web product.

## Project state for v0.5.0

**Status:** Single-user release candidate.

v0.3.0 established the local Python core, v0.4.0 added the repository- and
source-archive-distributed web interface and private deployment profile, and v0.5.0 brings
classification/regression parity to comparison, finalization, and prediction.
The API, SQLite database, worker, uploaded files, model evidence, artifacts, and prediction outputs
still belong to one local workspace and one active MLForge process.

Milestone 9 remains postponed for shared or multi-user infrastructure. MLForge does not claim
distributed execution, multi-user support, request authentication, hosted public services, public
online model serving, or a model-registry server.

## Conditional improvements

### Milestone 9 - Service adapters and shared experiment storage

**Status:** Postponed / conditional for shared and multi-user infrastructure. The v0.5.0 local API
does not activate this milestone.

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

### Milestone 10 - Public deployment, observability, and multi-user interface

**Status:** Partially superseded by Milestone 8.2 for one private operator. Public and multi-user
capabilities remain postponed and are not active development.

**What changes:** Evaluate public online inference, model promotion/rollback, prediction telemetry,
drift analysis, authentication/authorization, shared storage, and production release automation
individually. Implement only capabilities backed by accepted use cases and measurable operational
requirements.

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
