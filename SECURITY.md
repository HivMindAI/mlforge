# Security Policy

MLForge handles local datasets and executable Python model artifacts. Please report security
problems privately when possible so maintainers can assess them before exploit details become
public.

## Supported versions

MLForge remains in the `0.y.z` development series. Security fixes are made on the latest `0.2.x`
line; older minor lines, snapshots, and locally modified copies are not supported.

| Version | Supported |
| --- | --- |
| `0.2.x` | Yes |
| `0.1.x` | No |
| Unreleased snapshots | Best effort |

## Reporting a vulnerability

Use the repository's **Security** tab and **Report a vulnerability** when that private-reporting
option is available. Include:

- the affected MLForge version or commit;
- the operating system and Python version;
- the affected command or Python API;
- minimal reproduction steps and expected impact;
- whether artifact loading or untrusted input is involved; and
- any mitigation you have already verified.

If private reporting is unavailable, open a minimal GitHub issue asking the maintainers to provide
a private contact channel. Do not include exploit code, malicious artifacts, credentials, private
data, or details that would enable abuse in that public issue.

Maintainers will confirm receipt, assess scope and severity, coordinate a fix, and agree on
disclosure timing with the reporter when contact is possible. Response and remediation times are
not guaranteed while the project is in its `0.y.z` development series.

## Security boundaries

- A `.mlforge` artifact contains pickle data and can execute code when loaded. Inspection and
  checksums do not make an untrusted artifact safe.
- `trusted=True` and `--trust-artifact` are explicit source-trust assertions, not sandboxing or
  authentication.
- MLForge is a single-user local tool. It does not provide authorization, tenant isolation,
  network serving, artifact signing, or hostile-code isolation.
- Dataset and prediction loaders validate local CSV structure and size, but callers remain
  responsible for filesystem permissions and sensitive-data handling.
- Profiles and run manifests contain paths, schema/target metadata, metrics, warnings, and bounded
  error text. Review them before public sharing; they are not anonymized reports.

Read the complete [artifact security model](docs/security.md) before loading, sharing, or deploying
a model artifact.
