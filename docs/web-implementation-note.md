# Web Application Implementation Note

**Status:** The local single-user application and regression-parity extension are complete. Both
classification and regression journeys pass their final automated gates and browser verification.

## Current architecture

MLForge 0.5.0 is a typed, library-first Python package under `src/mlforge`. The command-line
interface is an adapter: dataset validation, profiling, preprocessing, training, comparison,
cross-validation, final fitting, artifact persistence, and inference live in importable domain and
application modules. Expected failures use the `MLForgeError` hierarchy.

Runs, holdout benchmarks, cross-validation selections, final-model records, and artifacts use
separate create-only filesystem stores. Their JSON manifests are strict and immutable. Model
artifacts are local ZIP archives containing a pickle payload and a safe-to-inspect JSON manifest.
Loading the payload is deliberately blocked until the caller explicitly establishes trust.

The Python application remains synchronous, single-process, and local. Phase 3 adds a thin FastAPI
adapter for dataset uploads, plus SQLite metadata and UUID-named immutable CSV storage inside a
configurable local workspace. Phase 4 profiles the exact stored CSV on request through the existing
core API and returns a path-free DTO. Phase 5 adds immutable web-owned experiment configuration in
the same SQLite database, while all ML evidence remains owned by the existing stores. Phase 6 adds
a one-worker local executor and durable job state. Phase 7 reads completed results directly from
the strict `LocalCrossValidationStore` and exposes only path-free response fields. Phase 8 passes
that persisted selection to `fit_selected_model`, storing immutable final-model and artifact records
while keeping retryable web job state in SQLite. Phase 9 safely joins completed web lineage back to
those strict manifests for model discovery and display. Phase 10 accepts a feature CSV only for a
web-owned finalized model, crosses the executable-artifact trust boundary after lineage
verification, and delegates schema validation and inference to `predict_csv`. There is still no
request authentication or supported public-network deployment. The repository's private Docker
Compose adapter keeps the browser-facing service on loopback. Phase 11 reads the already-created
result CSV without
rerunning inference, validates its stored row contract, exposes a bounded preview, and streams the
complete file for download. Phase 12 adds a read-only history projection over persisted experiment,
dataset, and durable job metadata; selecting an item continues to read metrics and results from the
existing immutable core stores.

## Reusable MLForge modules

| Web capability | Existing source of truth | Integration note |
| --- | --- | --- |
| Pre-target CSV validation | `mlforge.datasets.load_feature_csv` | Validates the file and returns its real columns without inventing a target. |
| Target selection and dataset identity | `mlforge.datasets.load_csv` | Requires an explicit target and records the resolved path, size, SHA-256, parser settings, shape, and dtypes. |
| Data-quality review | `mlforge.datasets.profile_dataset` | Provides JSON-safe column, missingness, cardinality, identifier, target, and warning data. |
| One-model training | `mlforge.training.train` | Supports three classification baselines and two regression baselines with held-out metrics. |
| Classification comparison | `mlforge.benchmarks.benchmark` | Uses one shared holdout partition and returns a fitted in-memory winner. |
| Task-aware cross-validation | `mlforge.benchmarks.cross_validate_benchmark` | Supports deterministic stratified classification folds or shuffled regression folds, with persisted ranking evidence. |
| Compatible run comparison | `mlforge.runs.compare_runs` | Can rank compatible successful runs but is not a regression benchmark orchestration service. |
| Final model fitting | `mlforge.final_models.fit_selected_model` | Supports classification and regression; requires a persisted cross-validation result and the exact unchanged source CSV. |
| History | `LocalRunStore`, `LocalBenchmarkStore`, `LocalCrossValidationStore`, `LocalFinalModelStore` | Lists and validates immutable local records; it must not be rewritten for web labels or UI state. |
| Artifact inspection and loading | `inspect_artifact`, `load_artifact`, `LocalArtifactStore` | Inspection is safe; loading crosses the pickle trust boundary and requires exact dependency versions. |
| Prediction and download | `predict_frame`, `predict_csv`, `write_predictions_csv` | Enforces the stored input schema and creates a non-overwriting UTF-8 result CSV. |

The web adapter should call these public APIs rather than CLI handlers or private helper functions.
The core does not need an ML algorithm redesign.

## Proposed single-user web architecture

Use two deliberately small application surfaces. Phase 1 established the first, Phase 3 introduced
the dataset boundary of the second, Phase 4 connected both through a profiling endpoint, and Phase
5 added configuration persistence. Phase 6 connects configured experiments to the existing
cross-validation service through a bounded worker, and Phase 7 adapts its immutable results for the
browser:

- `frontend/`: a strict TypeScript Next.js application using Tailwind CSS, semantic HTML, and
  accessible components. Add Lucide icons, shadcn/ui primitives, TanStack Query, or Recharts only
  where the implemented workflow benefits from them.
- `src/mlforge/web/`: a FastAPI adapter containing typed request/response schemas, thin routes,
  application services, web metadata storage, and error translation. It imports the existing
  MLForge public APIs and contains no duplicate preprocessing, ranking, or model logic.

Use one configurable local web workspace, ignored by Git, with generated paths similar to:

```text
.mlforge-web/
|- mlforge.sqlite3       # Mutable web-only metadata and job state
|- uploads/              # UUID-named immutable training CSVs
|- prediction-inputs/    # UUID-named validated prediction inputs
|- mlruns/               # Existing MLForge run store
|- mlbenchmarks/         # Existing benchmark and cross-validation stores
|- mlfinalmodels/        # Existing final-model store
|- artifacts/            # Server-created trusted-local artifacts
`- predictions/          # Downloadable result CSVs
```

SQLite is sufficient for dataset display names, parser settings, target choices, experiment/model
links, job states, and prediction download references. Immutable MLForge manifests remain the
canonical ML evidence. The SQLite rows should reference their UUIDs and paths, not copy or mutate
their contents.

Long-running work should run through a bounded single-worker local executor and persisted job
records. The honest initial states are `waiting`, `running`, `complete`, and `failed`; the existing
core does not expose reliable per-model live progress. Run the API as one process for this version,
and mark interrupted `running` jobs as failed during restart recovery. This boundary can later be
replaced by a real queue without changing route or domain contracts.

Suggested HTTP resources are:

```text
POST /api/datasets
GET  /api/datasets
GET  /api/datasets/{dataset_id}
POST /api/datasets/{dataset_id}/analysis

POST /api/experiments
GET  /api/experiments
GET  /api/experiments/{experiment_id}
POST /api/experiments/{experiment_id}/run
GET  /api/experiments/{experiment_id}/results
POST /api/experiments/{experiment_id}/finalize
GET  /api/experiments/{experiment_id}/finalization

GET  /api/jobs/{job_id}

GET  /api/final-models
GET  /api/final-models/{final_model_id}

POST /api/predictions
GET  /api/predictions/{prediction_id}
GET  /api/predictions/{prediction_id}/download
```

All three prediction resources shown above exist through Phase 11. The detail response is bounded
to the first 20 result rows; the download response streams the complete CSV.

The experiment collection resource now exists through Phase 12. It returns newest-first compact
history rows derived from SQLite display metadata and job state. It does not copy, edit, or replace
immutable benchmark, final-model, or artifact manifests.

Upload endpoints should generate server-side filenames, enforce HTTP and MLForge size limits, and
never trust the client filename as a path. API responses should expose purpose-built DTOs rather
than raw manifests where those manifests contain local source paths. `MLForgeError` subclasses
should become structured, actionable client errors while unexpected failures remain server errors.

The initial server should bind to loopback by default. A public network deployment without
authentication would contradict the explicitly single-user trust model and is outside this
version's scope.

## Files and directories expected across web phases

```text
src/mlforge/web/
|- __init__.py
|- __main__.py           # Loopback-only local API entry point
|- app.py                # FastAPI construction and exception handlers
|- api.py                # Thin HTTP routes
|- errors.py             # Expected adapter-specific failures
|- schemas.py            # Pydantic request/response contracts
|- services.py           # Web workflow orchestration over MLForge APIs
|- settings.py           # Local workspace and upload-limit configuration
|- storage.py            # SQLite metadata and safe workspace paths
`- jobs.py               # Bounded local execution and persisted status

tests/web/               # API, storage, job, and end-to-end integration tests

frontend/
|- app/                  # Routes and application shell
|- components/           # Reused workflow components only
|- lib/                  # Typed API client and server-state helpers
|- public/
|- package.json
`- package-lock.json
```

Phase 1 added `frontend/` with the application shell, responsive navigation, project-specific
metadata, and frontend validation scripts. Phase 3 added the optional web dependency group,
non-secret environment documentation, the ignored local workspace, API integration tests, and
user/developer documentation. Exact files continue to be introduced only in the phase that
exercises them.

## Integration risks and required boundaries

1. **Exact dataset retention:** final fitting reopens and verifies the original selected CSV.
   Training uploads must therefore be stored immutably until their experiments and models are
   intentionally removed.
2. **Artifact trust:** the server may trust only artifacts it created and can match to its local
   immutable lineage records. Version 1 must not accept arbitrary artifact uploads.
3. **Task-specific validation:** classification uses stratified folds and percentage-style metrics;
   regression uses ordinary shuffled K-folds and MAE/R-squared/RMSE. The UI must preserve metric
   direction and must not render regression errors as percentages.
4. **Progress granularity:** current application calls return only terminal results. The first UI
   must show honest job-level progress and never fabricate percentages or per-model states.
5. **Mutable versus immutable state:** friendly names, dataset ownership links, and job status are
   web metadata; MLForge manifests are evidence and must remain create-only.
6. **Sensitive metadata:** profiles and manifests can reveal local paths, column names, labels, and
   class values. Public DTOs and logs must avoid leaking server paths and must not include raw rows.
7. **Resource use:** a valid 100 MiB CSV or fitted model can consume substantial memory and CPU.
   Upload limits, a one-job concurrency limit, bounded previews, and streamed downloads are needed.
8. **Process and storage assumptions:** in-process jobs require a single API process, and MLForge's
   atomic publication expects temporary and final files on the same filesystem.
9. **Version compatibility:** prediction artifacts require the exact Python and ML dependency
   versions recorded at training time; the web runtime must train and predict in one controlled
   environment.
10. **Regression compatibility:** version-2 comparison/final-model manifests and web schema rows
    represent regression; version-1 immutable manifests remain classification-only and readable.

## Phase 0 verification

The repository was clean on `main` and synchronized with `origin/main` before this note. The
existing baseline was run on 2026-08-27: Ruff lint passed, Ruff formatting reported all 80 files
formatted, strict mypy reported no issues in 58 source files, and all 225 tests passed with 83.68%
statement coverage. Node.js and npm are available. Phase 1 uses Next.js, strict TypeScript, ESLint,
and Tailwind CSS in the dedicated `frontend/` directory. FastAPI dependencies were not installed
at Phase 0 because the HTTP adapter was outside that milestone; Phase 3 now isolates them in the
optional `web` dependency group.

## Phase 1 implementation

The application shell uses a 56-pixel top header, a 224-pixel desktop sidebar, a spacious bounded
workspace, neutral surfaces, subtle borders, and restrained blue selected states. The navigation
contains the final information architecture, but later-phase destinations remain visibly
non-interactive so the shell does not pretend unsupported workflows exist.

Below 768 pixels the fixed sidebar is replaced by a native modal `dialog`. The menu reports its
expanded state, closes through its button, backdrop, or Escape key, and preserves visible keyboard
focus. A skip link moves keyboard users directly to the workspace. Reduced-motion preferences
disable the few short interface transitions.

Phase 1 introduced no charts, metrics, upload controls, experiment controls, model controls,
prediction controls, gradients, glass effects, decorative imagery, or speculative product data.
Its root page was deliberately limited to a restrained empty workspace state.

## Phase 2 implementation

The root page now introduces MLForge with direct product language and reserves a simple, divided
section for recent experiments. Because no experiment registry or web adapter exists yet, the
section renders an honest empty state rather than sample records, analytics, or fabricated update
times.

During Phase 2 the empty state included the future `Upload dataset` action required by the
dashboard design, but the control remained semantically disabled. That phase added no route,
upload behavior, client state, backend call, or persisted data; Phase 3 now activates the action.

## Phase 3 implementation

The dashboard action and Datasets navigation now open `/datasets/new`. The page supports a native
file picker, drag and drop, client-side extension/empty/100 MiB checks, real browser upload
progress, server error feedback, and responsive keyboard-accessible controls. After a successful
upload it displays only verified filename, file size, row count, and column count, then requires an
explicit target selection. No target is guessed.

`POST /api/datasets` streams the upload to a server-generated temporary path, enforces the same
100 MiB boundary, and delegates CSV structure and parsing to `load_feature_csv`. Only after that
validation succeeds is the file atomically published under a UUID filename and its path-free
metadata stored in SQLite. `PATCH /api/datasets/{dataset_id}/target` delegates target validation to
`load_csv` before persisting the exact user choice. `GET /api/datasets/{dataset_id}` provides the
same safe DTO for later route restoration.

The local web workspace defaults to `.mlforge-web/`, is ignored by Git, and contains only the
SQLite database and immutable uploads at this phase. Client filenames are retained for display but
never used as server paths. Failed uploads clean up temporary and partially published files. Phase
3 does not profile data, show sample rows, infer task type, create experiments, or start jobs.

## Phase 4 implementation

After target selection, the upload flow opens `/datasets/{dataset_id}`. That page requests a fresh
analysis through `POST /api/datasets/{dataset_id}/analysis`. The thin service reloads the immutable
CSV with `load_csv`, then delegates all type, missingness, cardinality, identifier, task-hint,
distribution, and warning behavior to `profile_dataset`; the web layer contains no competing ML
heuristics.

The response uses explicit Pydantic DTOs and deliberately excludes the core profile's local source
path. It includes no raw rows. The TypeScript client validates the complete response before the UI
uses it and supports cancellation, retry, and structured API error messages.

The Data Overview uses a divided summary strip, evidence-based potential-issue rows, a selected
target section, the core warning text, and compact responsive tables. Counts and warnings come only
from the real profile. The page introduces no charts, sample records, fake metrics, experiment
controls, training calls, model actions, or predictions. Experiment configuration begins only in
the following phase.

## Phase 5 implementation

The Data Overview now links to `/datasets/{dataset_id}/experiment/new`. The screen reloads the real
profile before showing options. For a classification target it exposes only stratified
cross-validation, 2-10 folds, balanced accuracy ranking, and the three estimators exported by the
core: Dummy Classifier, Logistic Regression, and Random Forest Classifier. At least two estimators
are required.

`POST /api/experiments` builds `CrossValidationConfig` and `CrossValidationSplitConfig`, then calls
the public `split_classification_folds` function to validate that the selected fold count is feasible
for the exact dataset. Successful configuration is stored as immutable web metadata in SQLite. It
contains no source path and starts no training process, benchmark, run, or job.

For a regression target, the UI states that Ridge Regression and Random Forest Regression exist as
individual holdout runs but that the core has no shared regression comparison or regression
finalization workflow. The API rejects regression comparison as `invalid_experiment` rather than
inventing orchestration. Execution begins only in the following phase.

## Phase 6 implementation

Saved configurations now open at `/experiments/{experiment_id}`. The page restores the experiment
and any existing job after a refresh. `POST /api/experiments/{experiment_id}/run` idempotently
creates one job for that configuration, while `GET /api/jobs/{job_id}` and
`GET /api/experiments/{experiment_id}/job` expose its latest durable state.

The local `JobManager` uses a `ThreadPoolExecutor` with one worker, so CPU-heavy comparisons run
serially without blocking HTTP requests. It reloads the exact immutable CSV, reconstructs the
validated `CrossValidationConfig`, and calls `cross_validate_benchmark`. The resulting immutable
manifest is written by `LocalCrossValidationStore` under the web workspace; SQLite stores only the
job state and benchmark UUID lineage.

Jobs use exactly `waiting`, `running`, `complete`, and `failed`. Duplicate run requests return the
same job. During API startup, jobs left waiting or running by an interrupted process are marked
failed with a readable recovery message rather than appearing active forever. Expected core
failures are persisted for inspection; unexpected failures receive a safe generic browser message
and remain in API logs.

The current core returns a cross-validation result only after all estimators finish and has no
per-model progress callback. The interface therefore shows one truthful job-level status, never a
percentage or invented model state. On completion it transitions to the result evidence introduced
in Phase 7.

## Phase 7 implementation

`GET /api/experiments/{experiment_id}/results` resolves the completed job's benchmark UUID, reads
the create-only manifest through `LocalCrossValidationStore`, validates that its dataset and
configuration still match the saved experiment, and returns a purpose-built path-free DTO. A job
that has not completed returns `result_not_ready`; missing or corrupt evidence is treated as a
storage failure rather than silently recomputed.

The results screen names the rank-one model and displays only metrics recorded by the core:
accuracy, balanced accuracy, macro and weighted F1, macro precision, and macro recall. It exposes
the core-computed mean, population standard deviation, and every observed fold score. The ranking
table uses the configured primary metric and places failed entries after ranked models. Partial
benchmarks and per-model failure messages remain inspectable.

No chart was added because the compact ranking and fold tables communicate the available evidence
more precisely. No stability category, score, metric, or result is inferred by the frontend. Phase
7 adds no finalization action, final model record, artifact, model list, or prediction workflow;
Phase 8 now adds only the explicit finalization action and final-model detail needed by that action.

## Phase 8 implementation

The result page now offers `Finalize model` for the core-selected rank-one classification model.
`POST /api/experiments/{experiment_id}/finalize` first requires a completed comparison with a
successful winner, then submits the work to the same one-worker executor used by comparisons. The
worker reloads the exact immutable CSV and persisted cross-validation manifest, then calls the
public `fit_selected_model` API with `LocalFinalModelStore` and `LocalArtifactStore`.

Finalization attempts use durable `waiting`, `running`, `complete`, and `failed` states. A partial
SQLite unique index permits only one active or completed finalization per experiment while allowing
a failed attempt to be retried. Interrupted attempts are marked failed during API restart. A
completed experiment is idempotent and cannot silently create multiple finalized models.

`GET /api/final-models/{final_model_id}` reads the strict final-model manifest and safely inspects
the artifact archive without deserializing its pickle payload. It verifies the final-model,
artifact, benchmark, and experiment lineage before returning a path-free DTO containing the model
identity, selected metric evidence, all-row fit scope, artifact filename, size, checksum, target,
and ordered feature contract.

The interface shows the selected estimator, explains the all-row refit, reports honest job-level
progress without a percentage, permits retry after failure, and displays relevant artifact details
after completion. A visible trust warning states that the artifact contains executable Python
pickle data and must only be loaded from a trusted source. No artifact download, arbitrary artifact
upload, trusted loading, prediction, Models screen, or model-history list is added in this phase.

## Phase 9 implementation

`GET /api/final-models` lists only completed models connected to web-owned finalization records. A
partial SQLite index supports the newest-first completed-model query. Each entry is resolved through
the strict final-model and artifact stores rather than treating SQLite as ML evidence; corrupt or
missing lineage fails visibly instead of being skipped.

The existing final-model detail response now includes the original dataset display name and UUID,
source experiment UUID, all metric means and population deviations recorded for the selected
cross-validation entry, the ordered input schema, and the exact Python and ML library versions from
the safe artifact manifest. It still exposes no local paths or executable payload bytes and never
deserializes pickle data.

The Models navigation now opens `/models`, where a compact table shows actual finalized models,
dataset names, task type, primary selection score, and creation time. Selecting a model opens
`/models/{model_id}` with source links, recorded metrics, input schema, runtime versions, collapsed
artifact integrity details, and a persistent trust warning. Empty, loading, error, and responsive
states use the existing restrained shell. No prediction action, artifact download, arbitrary model
import, editable model naming, or experiment-history workflow is added in this phase.

## Phase 10 implementation

The Predictions navigation and the action on each finalized-model detail page now open
`/predictions/new`. The workflow lists only finalized models connected to completed web lineage,
shows the ordered feature contract from the safely inspected artifact manifest, and accepts one
UTF-8 CSV up to the configured upload limit. Browser upload progress represents transferred bytes;
after transfer it changes to an honest validation-and-inference state rather than inventing model
progress.

`POST /api/predictions` streams the input to a server-generated temporary filename, verifies the
selected model through `FinalModelService`, then loads only that workspace-owned artifact with the
required explicit trust flag. It calls the public `predict_csv` function, so missing or unexpected
columns, incompatible dtypes, malformed CSV bytes, and value constraints remain owned by the core
schema contract. `write_predictions_csv` creates the immutable output and SQLite records only the
web lineage and terminal metadata.

Expected input, artifact, and inference failures have distinct structured error codes and retain
actionable core messages without exposing server paths. Failed attempts remove temporary input and
output files. The Phase 10 response deliberately contains only the prediction id, selected model
id, input display filename, terminal status, and timestamps. It exposes no row count, prediction
values, preview, output path, or download URL; those presentation and download resources remain
Phase 11 work.

## Phase 11 implementation

After `POST /api/predictions` completes, the browser now opens
`/predictions/{prediction_id}`. `GET /api/predictions/{prediction_id}` reads the terminal SQLite
record, resolves only its server-generated output filename, rejects symlinks or paths outside the
prediction directory, and validates the CSV header, sequential row numbers, and stored row count.
It returns at most the first 20 predictions as strings, plus the real processed-row count and zero
invalid rows. Successful Phase 10 inference is all-or-nothing, so a completed record cannot contain
partially invalid input rows.

`GET /api/predictions/{prediction_id}/download` performs the same saved-output validation before
returning the complete UTF-8 CSV as `predictions.csv`. The response disables caching and content
type sniffing. A missing prediction returns `prediction_not_found`; a missing, malformed, or
metadata-inconsistent saved output returns the specific `prediction_result_unavailable` error and
is never silently regenerated.

The result screen uses a divided summary for rows processed, invalid rows, and completion time; a
compact horizontally scrollable preview table; and direct download actions. It has intentional
loading, retryable error, success, truncated-preview, and responsive states. Phase 11 adds no
prediction history list, experiment history, pagination over stored results, deletion, or rerun
behavior; those remain outside this milestone.

## Phase 12 implementation

`GET /api/experiments` now returns a newest-first, path-free history of every saved web experiment.
Each compact row combines the immutable experiment configuration with its dataset display name and
durable job status. An experiment without a job is honestly reported as `configured`; submitted
jobs retain `waiting`, `running`, `complete`, or `failed`. The updated timestamp comes from the
latest persisted job timestamp and is never estimated. A SQLite index supports the actual
newest-first query, and startup runs `PRAGMA optimize` after ensuring that index exists.

The Experiments navigation now opens `/experiments`. Its compact responsive table shows the saved
experiment id, dataset name, supported task, selected-model count, status, and last persisted
update. Empty, loading, and retryable error states use the existing restrained shell without fake
names, metrics, or sample history.

Selecting a row reuses the existing experiment detail and immutable result readers. The detail now
also loads the exact stored dataset filename, target, row count, and column count. Completed runs
still expose real rankings, fold metrics, warnings, and finalization state; failed jobs expose their
persisted error; finalized models link to the existing safe model detail containing recorded Python
and ML library versions. No history manifest is modified, no evidence is recomputed, and Phase 12
adds no deletion, editing, pagination, prediction history, or Phase 13 polish.

## Phase 13 implementation

The shell now changes from its fixed sidebar to the native modal navigation below 900 pixels, so
tablet layouts retain a useful workspace instead of compressing forms and detail views beside the
desktop rail. Existing summary grids stack to two columns, action groups remain usable, and tables
keep their deliberate minimum widths inside touch-scrollable containers rather than discarding
columns or converting evidence into oversized cards.

Keyboard and assistive-technology behavior is explicit. The mobile menu button identifies its
controlled dialog, the dialog initially focuses its close action, and dismissal restores focus to
the invoking button unless navigation is already moving to another page. External links announce
that they open a new tab, unavailable Settings text is identified, form help text is associated
with its select or fieldset, and every potentially overflowing table has a named, focusable scroll
region. Long row headings wrap instead of being irretrievably truncated.

Muted text tokens were darkened for stronger contrast while preserving the neutral visual
hierarchy. Current navigation retains its subtle blue treatment but also uses weight and an inset
edge, statuses continue to include readable text rather than relying on color, global visible focus
outlines remain intact, and reduced-motion preferences still suppress transitions. Phase 13 adds
no new product state, backend behavior, error presentation redesign, empty-state redesign, or
Phase 14 functionality.

## Phase 14 implementation

Important routes now use the same restrained page-level loading and server-error patterns, with
short static skeleton lines, retry actions, and a useful secondary route. True empty states remain
separate from failures and contain one direct next action. Validation and action failures stay
inline beside the relevant workflow rather than replacing otherwise useful page context.

The dashboard now reads the real persisted experiment history and intentionally distinguishes
loading, retryable server failure, a genuinely empty workspace, and a compact newest-first success
list. It does not invent activity, metrics, timestamps, or examples. The prediction workflow also
separates a failed finalized-model request from a successful request with no models, so an API
failure can no longer masquerade as an empty registry.

Finalization no longer exposes its write action when the current finalization state could not be
read. Experiment results identify partial success with the exact number of failed models while
retaining the valid ranked evidence. Existing upload validation, durable job failures, prediction
schema errors, and successful result states remain contextual and actionable. Phase 14 adds no
dataset, experiment, model, or prediction capability, and does not begin Phase 15 cleanup.

## Phase 15 verification and cleanup

The final browser verification exercised the complete supported workflow with repository example
data: upload, explicit target selection, data review, three-model cross-validation, immutable
result inspection, rank-one finalization, model/schema inspection, prediction upload, inference,
preview, and CSV download. It also confirmed the disabled missing-target action, a malformed CSV
response, and the exact unexpected-column prediction error. Desktop navigation, the compact mobile
dialog, initial dialog focus, close behavior, and focus restoration were checked at explicit desktop
and mobile viewports. No browser console warnings or errors were recorded.

The automated web integration tests continue to cover the remaining destructive or impractical
manual cases, including the streaming upload limit, unsupported regression comparison, failed
training, missing models and artifacts, artifact integrity failures, and invalid prediction input.
The complete Python suite passes with the configured coverage threshold, and the Python package,
strict typing, lint, formatting, frontend lint, TypeScript compilation, and production frontend
build all complete successfully.

Final cleanup restores the dashboard's direct `Upload dataset` action when history exists, handles
the native mobile-dialog cancel event explicitly, and disables Next.js development-time generation
of redundant nested agent instruction files. The application remains deliberately local: its
FastAPI process, SQLite metadata, uploaded datasets, immutable evidence, artifacts, and predictions
belong to one trusted workspace. Hosted deployment, authentication, multi-user storage, and remote
job execution remain outside this version.
