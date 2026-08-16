# MLForge Artifact Security

## Trust model

An MLForge `.mlforge` file contains a fitted scikit-learn pipeline serialized with Python pickle
protocol 5. Deserializing pickle is equivalent to executing code supplied by the artifact creator.
Do not load an artifact merely because it has an MLForge filename, a valid manifest, or a matching
checksum.

MLForge separates two operations:

- `inspect_artifact(path)` and `mlforge artifacts inspect` parse bounded JSON and stream the
  pipeline bytes only to verify structure, size, and SHA-256. They do not deserialize the pipeline.
- `load_artifact(path, trusted=True)` and `mlforge predict ... --trust-artifact` cross the executable
  boundary. The caller is asserting that the artifact source and custody are trusted.

The default is refusal. There is no environment variable, global setting, filename convention, or
manifest field that silently enables loading.

## What integrity checks establish

An artifact is one create-only ZIP archive containing exactly `manifest.json` and `pipeline.pkl` as
uncompressed regular members. The loader rejects unexpected members, encryption, compression,
oversized content, malformed or unsupported manifests, a filename/run-ID mismatch, and pipeline
bytes whose size or SHA-256 differs from the manifest. It never extracts archive paths to disk.

The artifact manifest also records a SHA-256 of the canonical immutable run manifest. Saving fails
unless the in-memory successful run still matches its persisted run file. Use
`verify_run_manifest(artifact_manifest, run_manifest)` when independently checking lineage.

These checks detect corruption and changes relative to the manifest. They do not authenticate the
manifest itself: an attacker who can replace the archive can replace both the payload and its
checksum. MLForge does not yet provide signatures, certificates, provenance attestations, or a
remote trust service.

## Compatibility boundary

Before deserialization, MLForge requires exact matches for Python, MLForge, pandas, NumPy, SciPy,
and scikit-learn versions recorded during training. It also converts scikit-learn's inconsistent
version warning into a hard failure. This is intentionally stricter than attempting a best-effort
load because cross-version model persistence is unsupported and can produce incorrect behavior.

An exact version match improves compatibility; it does not make an untrusted pickle safe.

## Prediction-data boundary

Prediction CSVs receive the same encoding, delimiter, row-width, null-byte, regular-file, and
100 MiB default limits as training CSVs. Before prediction, MLForge requires the exact recorded
feature names, safely restores their training order, rejects duplicates and extra columns, checks
numeric and categorical roles, rejects infinite numeric values, and prevents a real category from
colliding with the configured missing-value marker. Model output is normalized to finite JSON
scalar records.

These are schema and resource checks, not adversarial compute isolation. A trusted model may still
consume substantial CPU or memory. Run artifacts inside a constrained process or container when
resource isolation matters.

Dataset profiles and run records are metadata, not anonymized output. Profiles expose resolved
source paths, column names, target class values/counts, and quality statistics. Run records expose
the resolved source path, configuration, warnings, metrics, and bounded failure messages. They do
not copy raw rows, but paths and labels can still reveal sensitive context. Review and sanitize
these JSON values before publishing or attaching them to an issue.

## Safe local workflow

1. Train the model yourself or establish the artifact's source and chain of custody.
2. Inspect the artifact without trust-enabled loading.
3. Compare its run UUID and canonical run-manifest hash with the expected local run record.
4. Reproduce the exact dependency environment recorded in the artifact.
5. Only then pass `trusted=True` or `--trust-artifact`.
6. Keep generated `artifacts/`, `mlruns/`, datasets, and prediction outputs out of Git.

Do not load artifacts received through untrusted email, public file sharing, issue attachments, or
unknown URLs. A checksum supplied beside an artifact by the same untrusted source is not an
independent trust signal.

## Development versus production

The current product is a single-user trusted-local workflow. A production artifact system would need
authenticated writers, signed provenance, access control, encryption where required, retention and
revocation policies, isolated execution, audit logs, and deployment-specific resource limits.
Formats such as ONNX or carefully reviewed `skops.io` may offer a different execution boundary, but
they have compatibility and estimator-support tradeoffs and are not silently substituted by
MLForge.
