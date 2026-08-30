# Private single-user deployment

MLForge's first web version is a single-process application. This deployment profile preserves
that boundary: one FastAPI worker, one Next.js server, one durable workspace, and one trusted
operator. It is not a public SaaS architecture and does not add authentication, multi-tenancy,
distributed jobs, or online model serving.

## Architecture

- `web` is the only container with a host port. Compose binds it to `127.0.0.1` by default.
- `api` is reachable only from the internal Compose network. It is never published to the host.
- Next.js proxies `/api` requests to `api:8000`, so the browser uses one origin.
- `api` always runs with one worker because SQLite and the bounded in-process job manager are
  intentionally single-process.
- `mlforge-data` mounts at `/var/lib/mlforge` and contains SQLite metadata, uploaded CSVs,
  benchmark evidence, fitted model artifacts, and prediction files.
- Both containers run as unprivileged users with Linux capabilities dropped.

The public interface must remain behind a private access layer. The safest provider-neutral option
is an SSH tunnel. A private overlay such as Tailscale or an identity-aware gateway may replace the
tunnel later, but direct internet exposure is not supported by this profile.

## Prerequisites

- A Linux host or VM with Docker Engine and the Docker Compose plugin.
- Enough persistent disk for uploaded datasets, cross-validation evidence, model archives, and
  prediction output.
- A private access path to the host. Do not open port `3000` in a public firewall.

Docker is deliberately the only host runtime dependency. Python, Node.js, scikit-learn, and Next.js
are installed inside their respective images.

## Configure and start

From a clean checkout:

```bash
cp deployment/.env.example .env.private
docker compose --env-file .env.private --file compose.private.yaml config
docker compose --env-file .env.private --file compose.private.yaml build --pull
docker compose --env-file .env.private --file compose.private.yaml up --detach
docker compose --env-file .env.private --file compose.private.yaml ps
```

The browser-facing readiness probe should return HTTP 200:

```bash
curl --fail http://127.0.0.1:3000/api/health/ready
```

From the operator's computer, create a private tunnel and open `http://127.0.0.1:3000`:

```bash
ssh -N -L 3000:127.0.0.1:3000 operator@example-host
```

The loopback bind in `compose.private.yaml` is a security boundary. Do not replace it with
`0.0.0.0` unless a reviewed authentication and TLS gateway is already in front of the application.

## Health and operations

- `/api/health/live` verifies that the API process can serve a request.
- `/api/health/ready` also verifies that SQLite is readable and the upload volume is writable.
- The frontend container checks readiness through its own `/api` proxy, covering both containers
  and the internal network.
- `docker compose --file compose.private.yaml logs --follow` shows process logs without embedding
  uploaded CSV data or model payloads.

Use these commands for ordinary lifecycle operations:

```bash
docker compose --env-file .env.private --file compose.private.yaml restart
docker compose --env-file .env.private --file compose.private.yaml stop
docker compose --env-file .env.private --file compose.private.yaml start
docker compose --env-file .env.private --file compose.private.yaml down
```

Do not use `down --volumes` in normal operation: that deletes MLForge's durable workspace.

## Backup and restore

The `mlforge-data` volume is the complete application state. Stop both containers before taking a
filesystem or provider volume snapshot so SQLite, CSVs, manifests, and model archives remain from
the same point in time. Encrypt backups because they contain user datasets and fitted models.

To restore, provision an empty volume, restore the entire directory tree, preserve ownership for
container UID/GID `10001`, and then start the stack. Never restore only `mlforge.sqlite3`; its rows
refer to immutable files elsewhere in the same workspace.

## Upgrade and rollback

1. Take a consistent volume snapshot.
2. Check out the reviewed commit to deploy.
3. Run `docker compose --file compose.private.yaml build --pull`.
4. Run `docker compose --file compose.private.yaml up --detach`.
5. Verify `/api/health/ready` and one non-destructive application journey.

For rollback, check out the previously deployed commit, rebuild both images, restore the matching
volume snapshot if a storage format changed, and start the stack again. The current web metadata
schema has no migration framework, so deployment changes must not silently mutate it.

## Production limits

This profile is suitable only for one trusted operator and one active MLForge process. It does not
provide request-level authentication, concurrent workers, zero-downtime migrations, shared object
storage, hostile model isolation, rate limits, monitoring, or public uptime guarantees. A public or
multi-user deployment requires the conditional service-infrastructure milestone in `ROADMAP.md`.
