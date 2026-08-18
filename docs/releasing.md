# Releasing MLForge

Only a repository owner should perform a release. Publishing a GitHub Release triggers
`.github/workflows/release.yml` and, after any `pypi` environment approval, publishes the verified
wheel and source archive to the real Python Package Index.

## One-time owner configuration

1. Create a GitHub environment named `pypi`. Restrict it to protected release tags and add a
   required maintainer reviewer where the repository plan supports that control.
2. On PyPI, create a pending Trusted Publisher for the new `hivmind-mlforge` project with:

   | Setting | Value |
   | --- | --- |
   | PyPI project name | `hivmind-mlforge` |
   | GitHub owner | `HivMindAI` |
   | Repository | `mlforge` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` |

3. Protect `main` and the `v*` release-tag pattern so unreviewed changes cannot alter the release
   workflow or create a publishing identity.

Trusted Publishing uses GitHub OIDC and short-lived credentials. Do not create a PyPI API-token
secret for this workflow.

## Release procedure

1. Confirm the release commit is merged to `main`, the working tree is clean, and required CI is
   green on Python 3.11 and 3.12.
2. Confirm `mlforge.__version__`, `CHANGELOG.md`, and the intended tag agree.
3. Run the full contributor suite and clean-wheel workflow described in
   [release validation](release-validation.md).
4. Create and push the annotated tag:

   ```powershell
   $releaseVersion = "X.Y.Z"
   git switch main
   git pull --ff-only
   git tag -a "v$releaseVersion" -m "MLForge v$releaseVersion"
   git push origin "v$releaseVersion"
   ```

5. Create a **draft** GitHub Release for the existing tag, copy the matching versioned changelog
   section into its notes, and review it. Never reuse a previously published version or tag.
6. Publish the GitHub Release only when the PyPI publisher and GitHub `pypi` environment exactly
   match the table above.
7. Approve the protected `pypi` deployment, if configured. Verify the workflow builds once,
   transfers immutable artifacts to the isolated publish job, and completes without a password or
   repository secret.
8. Install from PyPI in a fresh environment and repeat `mlforge --help`, `pip check`, and the quick
   start.

Published package files and versions are immutable. Correct a release problem with a new version;
never replace an existing wheel or source archive.
