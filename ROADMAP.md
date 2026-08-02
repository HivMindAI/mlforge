# MLForge Roadmap

MLForge will grow gradually from a stable Python package into a local and production-ready MLOps platform for tabular supervised machine learning.

## Milestones

### Milestone 0 - Repository stabilization

Validate the repository, package metadata, editable install, CLI entrypoint, linting, tests, and core project documentation.

### Milestone 1 - Professional Python foundation

Finalize package metadata, development dependencies, source and test layout, logging, configuration, CLI foundations, CI, and contributor instructions.

### Milestone 2 - Dataset ingestion

Load CSV datasets safely, validate file properties and structure, produce metadata, add CLI commands, and test ingestion behavior.

### Milestone 3 - Dataset profiling

Detect column types, summarize dataset quality, identify high cardinality, identifiers, missingness, imbalance, and produce serializable profile reports.

### Milestone 4 - Preprocessing pipelines

Build reusable preprocessing configuration with numeric and categorical handling, `ColumnTransformer`, leakage prevention, and fitted/unfitted tests.

### Milestone 5 - Local training engine

Train initial classification and regression models, evaluate metrics, compare runs, record metadata, save fitted pipelines, and provide a CLI workflow.

### Milestone 6 - Experiment domain layer

Introduce projects, datasets, experiments, runs, models, artifacts, stable IDs, statuses, state transitions, failure handling, and audit information.

### Milestone 7 - Database persistence

Add PostgreSQL-oriented persistence with SQLAlchemy, Alembic, repository abstractions, constraints, transactions, and targeted tests.

### Milestone 8 - FastAPI backend

Expose versioned REST APIs for datasets, experiments, runs, and models with validation, structured errors, pagination, examples, and thin routes.

### Milestone 9 - Background training workers

Move training into Redis-backed Celery workers with queued states, retries, timeouts, idempotency, progress, logs, and worker tests.

### Milestone 10 - MLflow and artifact storage

Track experiments in MLflow, store parameters, metrics, models, and artifacts in object storage, and preserve model lineage.

### Milestone 11 - Web dashboard

Create a React and TypeScript dashboard for projects, datasets, profiling, experiments, runs, model comparison, registry views, and prediction testing.

### Milestone 12 - Model deployment and inference

Deploy selected model versions behind prediction APIs, validate schemas, cache models safely, switch versions, roll back, and record prediction metadata.

### Milestone 13 - Monitoring and drift

Measure latency, throughput, errors, input and prediction distributions, drift signals, dashboards, alerts, and retraining recommendations.

### Milestone 14 - Authentication and authorization

Add registration, login, secure password handling, tokens, organizations, memberships, permissions, API keys, audit logs, and security tests.

### Milestone 15 - Docker and production infrastructure

Containerize API, worker, frontend, and supporting services with Docker Compose, health checks, volumes, Nginx, and production-oriented settings.

### Milestone 16 - CI/CD and releases

Run linting, formatting, type checks, tests, frontend builds, package builds, Docker image builds, dependency checks, release tags, and changelog updates.

### Milestone 17 - Documentation and open-source readiness

Complete setup docs, tutorials, architecture docs, API examples, contribution guides, security policy, templates, model and dataset cards, and demo workflows.
