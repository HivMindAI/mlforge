"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import {
  CLASSIFICATION_ESTIMATOR_LABELS,
  type ClassificationEstimator,
  type FinalizationRecord,
  type FinalModelRecord,
  getExperimentFinalization,
  getFinalModel,
  startFinalization,
} from "@/lib/datasets";

const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function metricLabel(name: string): string {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

type FinalizeModelProps = Readonly<{
  experimentId: string;
  estimator: ClassificationEstimator;
}>;

export function FinalizeModel({ experimentId, estimator }: FinalizeModelProps) {
  const [finalization, setFinalization] = useState<FinalizationRecord | null>(null);
  const [model, setModel] = useState<FinalModelRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  const retryLoad = useCallback(() => {
    setLoading(true);
    setError(null);
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getExperimentFinalization(experimentId, controller.signal)
      .then(setFinalization)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Finalization status failed to load.",
        );
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [experimentId, requestVersion]);

  useEffect(() => {
    if (
      !finalization ||
      (finalization.status !== "waiting" && finalization.status !== "running")
    ) {
      return;
    }

    const controller = new AbortController();
    let stopped = false;
    let timer: number | undefined;

    async function poll() {
      try {
        const updated = await getExperimentFinalization(experimentId, controller.signal);
        if (stopped || updated === null) return;
        setFinalization(updated);
        if (updated.status === "waiting" || updated.status === "running") {
          timer = window.setTimeout(poll, 1000);
        }
      } catch (requestError) {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        if (!stopped) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Finalization status could not be refreshed.",
          );
        }
      }
    }

    timer = window.setTimeout(poll, 1000);
    return () => {
      stopped = true;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [experimentId, finalization]);

  useEffect(() => {
    if (finalization?.status !== "complete" || !finalization.final_model_id || model) return;
    const controller = new AbortController();
    void getFinalModel(finalization.final_model_id, controller.signal)
      .then(setModel)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(
          requestError instanceof Error ? requestError.message : "Final model failed to load.",
        );
      });
    return () => controller.abort();
  }, [finalization, model]);

  async function handleFinalize() {
    setStarting(true);
    setError(null);
    setModel(null);
    try {
      setFinalization(await startFinalization(experimentId));
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "Finalization could not be started.",
      );
    } finally {
      setStarting(false);
    }
  }

  const active = finalization?.status === "waiting" || finalization?.status === "running";

  return (
    <section className="finalize-section" aria-labelledby="finalize-title">
      <div className="configuration-section-heading">
        <div>
          <h3 id="finalize-title">Finalize model</h3>
          <p>Fit the rank-one model and its preprocessing pipeline on every selected row.</p>
        </div>
        {finalization ? (
          <span className={`job-status job-status-${finalization.status}`} aria-live="polite">
            {finalization.status === "complete"
              ? "Complete"
              : finalization.status.charAt(0).toUpperCase() + finalization.status.slice(1)}
          </span>
        ) : null}
      </div>

      {error ? (
        <div className="form-message form-message-error finalization-message" role="alert">
          <span>{error}</span>
          <button className="secondary-button" type="button" onClick={retryLoad}>
            Reload
          </button>
        </div>
      ) : null}

      {loading ? (
        <div className="finalization-loading" role="status" aria-live="polite" aria-busy="true">
          <span>Loading finalization state.</span>
          <div className="loading-lines loading-lines-compact" aria-hidden="true">
            <span />
            <span />
          </div>
        </div>
      ) : null}

      {!loading && !finalization && !error ? (
        <div className="finalize-action">
          <div>
            <strong>{CLASSIFICATION_ESTIMATOR_LABELS[estimator]}</strong>
            <p>
              This creates a new immutable final-model record and a prediction-ready local artifact.
            </p>
          </div>
          <button
            className="primary-button"
            type="button"
            onClick={handleFinalize}
            disabled={starting}
          >
            {starting ? "Starting..." : "Finalize model"}
          </button>
        </div>
      ) : null}

      {active ? (
        <div className="finalization-active" role="status" aria-live="polite">
          <strong>Fitting final model</strong>
          <p>
            MLForge is revalidating the selected dataset and fitting the saved winner on all rows.
            A percentage is not estimated.
          </p>
        </div>
      ) : null}

      {finalization?.status === "failed" ? (
        <div className="finalization-failed">
          <strong>Finalization failed</strong>
          <p>{finalization.error_message ?? "No error detail was recorded."}</p>
          <button
            className="secondary-button"
            type="button"
            onClick={handleFinalize}
            disabled={starting}
          >
            {starting ? "Starting..." : "Try finalization again"}
          </button>
        </div>
      ) : null}

      {finalization?.status === "complete" && !model && !error ? (
        <div className="finalization-loading" role="status" aria-live="polite" aria-busy="true">
          <span>Inspecting the saved model artifact.</span>
          <div className="loading-lines loading-lines-compact" aria-hidden="true">
            <span />
            <span />
          </div>
        </div>
      ) : null}

      {model ? (
        <div className="final-model-details">
          <header>
            <span className="section-kicker">Model finalized</span>
            <h4>{CLASSIFICATION_ESTIMATOR_LABELS[model.estimator]}</h4>
            <p>Created {formatDate(model.created_at)}</p>
          </header>

          <dl className="final-model-facts">
            <div>
              <dt>Training rows</dt>
              <dd>{model.training_rows.toLocaleString("en-US")}</dd>
            </div>
            <div>
              <dt>Input features</dt>
              <dd>{model.feature_count}</dd>
            </div>
            <div>
              <dt>{metricLabel(model.primary_metric)}</dt>
              <dd>{percentFormatter.format(model.primary_metric_mean)}</dd>
            </div>
            <div>
              <dt>CV std. dev.</dt>
              <dd>{percentFormatter.format(model.primary_metric_standard_deviation)}</dd>
            </div>
          </dl>

          <Link className="text-link final-model-runtime-link" href={`/models/${model.final_model_id}`}>
            View model details and recorded runtime versions
          </Link>

          <section className="artifact-details" aria-labelledby="artifact-title">
            <div className="configuration-section-heading">
              <div>
                <h4 id="artifact-title">Artifact details</h4>
                <p>Safe metadata inspected without loading the executable pipeline.</p>
              </div>
            </div>
            <dl>
              <div>
                <dt>Filename</dt>
                <dd><code>{model.artifact.filename}</code></dd>
              </div>
              <div>
                <dt>Format</dt>
                <dd>{model.artifact.serialization_format}</dd>
              </div>
              <div>
                <dt>Pipeline size</dt>
                <dd>{formatBytes(model.artifact.pipeline_size_bytes)}</dd>
              </div>
              <div>
                <dt>Target</dt>
                <dd>{model.artifact.target}</dd>
              </div>
              <div>
                <dt>SHA-256</dt>
                <dd><code>{model.artifact.pipeline_sha256}</code></dd>
              </div>
            </dl>
          </section>

          <details className="artifact-features">
            <summary>Expected input features ({model.artifact.features.length})</summary>
            <div
              className="data-table-wrap"
              role="region"
              aria-label="Finalized model expected input features table"
              tabIndex={0}
            >
              <table className="data-table final-model-feature-table">
                <thead>
                  <tr>
                    <th scope="col">Name</th>
                    <th scope="col">Role</th>
                    <th scope="col">Recorded dtype</th>
                  </tr>
                </thead>
                <tbody>
                  {model.artifact.features.map((feature) => (
                    <tr key={feature.name}>
                      <th scope="row">{feature.name}</th>
                      <td>{feature.role}</td>
                      <td>{feature.pandas_dtype}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>

          <p className="artifact-trust-note">
            MLForge artifacts contain an executable Python pickle payload. Load only artifacts
            created and kept in a trusted local workspace; never trust a model file from an unknown
            or unverified source.
          </p>

          {model.warnings.length > 0 ? (
            <details className="result-warnings">
              <summary>Final-fit warnings ({model.warnings.length})</summary>
              <ul>
                {model.warnings.map((warning) => <li key={warning}>{warning}</li>)}
              </ul>
            </details>
          ) : null}

          <div className="configuration-id final-model-id">
            <span>Final model ID</span>
            <code>{model.final_model_id}</code>
          </div>
        </div>
      ) : null}

      {!model ? (
        <p className="artifact-trust-note artifact-trust-note-pending">
          Final artifacts contain executable Python pickle data. Only load artifacts from a source
          whose origin and custody you trust.
        </p>
      ) : null}
    </section>
  );
}
