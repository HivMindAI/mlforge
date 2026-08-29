# MLForge Web

The single-user web interface for the existing MLForge Python application. Phase 1 establishes the
application shell and responsive navigation, and Phase 2 adds the dashboard and its truthful empty
state. Phase 3 adds real local CSV upload and target selection. Phase 4 adds a core-backed data
overview with quality signals, target information, and a compact columns table. Phase 5 adds
core-validated classification comparison configuration and an honest regression capability state.
Phase 6 adds real one-worker comparison execution with durable job-level progress and inspectable
errors. Phase 7 reads the immutable benchmark evidence and shows real model rankings, metric means,
standard deviations, and fold values. Phase 8 explicitly refits the rank-one model on all rows and
safely exposes its immutable final-model and artifact metadata. Phase 9 adds a real Models list and
detail view with source lineage, metrics, input schema, and recorded library versions. Prediction is
introduced in Phase 10: a finalized local model can validate and run a matching CSV while showing
exact schema and artifact errors. Phase 11 validates the saved output, previews at most 20 rows,
and downloads the complete prediction CSV without rerunning the model. Phase 12 adds a real
Experiments history screen and connects each saved row to its existing configuration, dataset,
job state, result evidence, failure details, and finalized-model runtime metadata. Phase 13
strengthens tablet/mobile navigation, keyboard focus, table scrolling, form descriptions, contrast,
and screen-reader labeling without changing product behavior. Phase 14 gives important routes
intentional loading, empty, server-error, validation-error, success, and partial-result states with
restrained retry and next-step actions; the dashboard now reflects real persisted history. Phase 15
completes the automated quality gates, end-to-end browser verification, responsive navigation
checks, and final boilerplate cleanup for the first local single-user web version.

## Local development

```bash
pip install -e ".[web]"
python -m mlforge.web
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies `/api` to `http://127.0.0.1:8000` by default. Override that local origin with
`MLFORGE_API_ORIGIN` when necessary.

Before review, run:

```bash
npm run lint
npm run build
```
