# Compatibility and Versioning

This document defines what MLForge users may rely on and how incompatible changes will be
communicated. It applies to released distributions; a source checkout may contain unreleased work.

## Version policy

MLForge uses three-part versions (`MAJOR.MINOR.PATCH`) with semantic-versioning intent.

During `0.y.z` development, the public API is not stable. Minor releases may contain incompatible
changes; patch releases should remain backward compatible. When practical, an incompatible change
will be deprecated for at least one minor release and documented with a migration path before
removal.

Starting with `1.0.0`:

- `PATCH` releases contain backward-compatible fixes;
- `MINOR` releases add backward-compatible behavior and may introduce deprecations; and
- `MAJOR` releases may remove or incompatibly change public behavior.

Published release contents are immutable. A fix always receives a new version.

## Maintenance mode after v0.3.0

v0.3.0 is the feature-complete release of the intended local MLForge product. Subsequent releases
should normally contain only real bug fixes, security fixes, compatibility fixes, documentation
corrections, and other justified maintenance changes. Conditional service, multi-user, hosted, and
deployment capabilities are not part of the supported product and are not an active roadmap.

## Public API

The supported Python API consists of the names listed in `__all__` by these modules and documented
in [API reference](api.md):

- `mlforge`
- `mlforge.artifacts`
- `mlforge.benchmarks`
- `mlforge.config`
- `mlforge.datasets`
- `mlforge.errors`
- `mlforge.final_models`
- `mlforge.inference`
- `mlforge.logging_config`
- `mlforge.pipelines`
- `mlforge.runs`
- `mlforge.training`

The top-level `mlforge` package intentionally exports only `__version__`; domain imports make
dependencies and responsibilities visible. Modules, functions, methods, constants, and attributes
whose names start with `_` are internal. `mlforge.cli` is an entrypoint adapter rather than a
Python-library compatibility surface; the documented `mlforge` commands and options are the CLI
contract.

Public dataclass fields, enum values, function parameter names, return types, documented errors,
CLI commands, and serialized schema meaning are part of the compatibility review. Export tests
make accidental additions and removals visible.

## Python and dependency support

- Python 3.11 is the minimum supported interpreter.
- CI verifies CPython 3.11 and 3.12 on Ubuntu and CPython 3.12 on Windows. Newer Python versions may
  work but are unverified until added to the CI matrix.
- The package metadata permits pandas `>=3.0,<4` and scikit-learn `>=1.9,<2`.
- Dependency updates within those ranges may affect numerical results. Reproducible run manifests
  record exact environment versions so results can be interpreted and environments reconstructed.

Dropping a verified Python version or narrowing a dependency range requires an announced
compatibility change. Support claims are updated only after the corresponding CI job passes.

## Serialized runs, benchmarks, final models, and artifacts

Run, holdout-benchmark, cross-validation-benchmark, final-model, and artifact manifests have
independent integer schema versions. Readers validate the complete schema and reject unsupported
versions rather than guessing how to migrate them. Adding or changing a required field therefore
requires a schema-version decision and fixture tests. Cross-validation records are intentionally
separate from ordinary run manifests because one estimator outcome contains several fitted folds.
Final-model records are separate because all selected rows are fitted and no held-out metric exists.
Their selection values live under `selection_evidence`; `final_fit` records only `all_rows` scope and
dimensions, while the artifact contract records identity plus executable-payload size and SHA-256.

Artifact manifest version 1 remains readable and represents evaluated training-run lineage.
Version 2 uses a generic model identity plus an explicit lineage kind and manifest digest; the
initial version-2 writer is reserved for final-model lineage.

Model artifacts have a deliberately narrower boundary than the Python API. Trusted loading
requires exact Python, MLForge, pandas, NumPy, SciPy, and scikit-learn versions recorded at training
time. MLForge does not promise cross-version pickle compatibility. See [artifact security](security.md).

## Deprecation and change review

A public change should include:

1. an explicit compatibility classification;
2. updated API and tutorial documentation;
3. interface and behavior tests;
4. a schema-version decision when serialized data changes; and
5. a migration note for users when existing code or artifacts are affected.

Internal refactoring may happen without notice when public behavior and serialized contracts remain
unchanged.
