"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { PageErrorState, PageLoadingState } from "@/components/async-state";
import {
  type FinalModelRecord,
  getFinalModel,
} from "@/lib/datasets";
import { estimatorLabel, formatMetricValue, metricLabel, taskLabel } from "@/lib/model-display";

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

type ModelDetailProps = Readonly<{ modelId: string }>;

export function ModelDetail({ modelId }: ModelDetailProps) {
  const [model, setModel] = useState<FinalModelRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  const retry = useCallback(() => {
    setModel(null);
    setError(null);
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getFinalModel(modelId, controller.signal)
      .then(setModel)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(requestError instanceof Error ? requestError.message : "Model failed to load.");
      });
    return () => controller.abort();
  }, [modelId, requestVersion]);

  if (error) {
    return (
      <PageErrorState
        kicker="Model registry"
        title="Model unavailable"
        description={error}
        onRetry={retry}
        secondaryHref="/models"
        secondaryLabel="Back to models"
      />
    );
  }

  if (!model) {
    return (
      <PageLoadingState
        kicker="Model registry"
        title="Loading model"
        description="MLForge is verifying the final-model record and artifact metadata."
      />
    );
  }

  const environment = model.artifact.environment;

  return (
    <article className="page model-detail-page">
      <Link className="text-link model-back-link" href="/models">
        Back to models
      </Link>

      <header className="model-detail-header">
        <div>
          <span className="section-kicker">Finalized model</span>
          <h1>{estimatorLabel(model.estimator)}</h1>
          <p>{model.dataset_name}</p>
        </div>
        <span className="model-status">Finalized</span>
      </header>

      <div className="model-primary-action">
        <div>
          <strong>Use this model</strong>
          <p>Upload a CSV that matches the recorded input schema.</p>
        </div>
        <Link className="primary-button" href={`/predictions/new?model=${model.final_model_id}`}>
          Run prediction
        </Link>
      </div>

      <dl className="model-detail-facts">
        <div>
          <dt>Task</dt>
          <dd>{taskLabel(model.task)}</dd>
        </div>
        <div>
          <dt>Created</dt>
          <dd>{formatDate(model.created_at)}</dd>
        </div>
        <div>
          <dt>Training rows</dt>
          <dd>{model.training_rows.toLocaleString("en-US")}</dd>
        </div>
        <div>
          <dt>Input features</dt>
          <dd>{model.feature_count}</dd>
        </div>
      </dl>

      <section className="model-detail-section" aria-labelledby="model-source-title">
        <div className="configuration-section-heading">
          <div>
            <h2 id="model-source-title">Source</h2>
            <p>The dataset and comparison evidence used to create this model.</p>
          </div>
        </div>
        <dl className="model-source-list">
          <div>
            <dt>Dataset</dt>
            <dd>
              <Link className="text-link" href={`/datasets/${model.dataset_id}`}>
                {model.dataset_name}
              </Link>
            </dd>
          </div>
          <div>
            <dt>Experiment</dt>
            <dd>
              <Link className="text-link" href={`/experiments/${model.experiment_id}`}>
                {model.experiment_id}
              </Link>
            </dd>
          </div>
          <div>
            <dt>Final model ID</dt>
            <dd><code>{model.final_model_id}</code></dd>
          </div>
        </dl>
      </section>

      <section className="model-detail-section" aria-labelledby="model-metrics-title">
        <div className="configuration-section-heading">
          <div>
            <h2 id="model-metrics-title">Selection metrics</h2>
            <p>
              Recorded by the source cross-validation. The all-row final fit creates no new
              evaluation score.
            </p>
          </div>
        </div>
        <div
          className="data-table-wrap"
          role="region"
          aria-label="Model selection metrics table"
          tabIndex={0}
        >
          <table className="data-table model-metrics-table">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">Mean</th>
                <th scope="col">Std. dev.</th>
              </tr>
            </thead>
            <tbody>
              {model.metrics.map((metric) => (
                <tr key={metric.name}>
                  <th scope="row">{metricLabel(metric.name)}</th>
                  <td>{formatMetricValue(metric.name, metric.mean)}</td>
                  <td>{formatMetricValue(metric.name, metric.standard_deviation)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="model-detail-section" aria-labelledby="model-schema-title">
        <div className="configuration-section-heading">
          <div>
            <h2 id="model-schema-title">Input schema</h2>
            <p>Ordered feature contract stored with the preprocessing pipeline.</p>
          </div>
          <span>{model.artifact.features.length}</span>
        </div>
        <div
          className="data-table-wrap"
          role="region"
          aria-label="Model input schema table"
          tabIndex={0}
        >
          <table className="data-table model-schema-table">
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
      </section>

      <section className="model-detail-section" aria-labelledby="model-runtime-title">
        <div className="configuration-section-heading">
          <div>
            <h2 id="model-runtime-title">Recorded runtime</h2>
            <p>Exact versions required before the executable artifact can be loaded.</p>
          </div>
        </div>
        <dl className="model-runtime-list">
          <div><dt>Python</dt><dd>{environment.python}</dd></div>
          <div><dt>MLForge</dt><dd>{environment.mlforge}</dd></div>
          <div><dt>scikit-learn</dt><dd>{environment.scikit_learn}</dd></div>
          <div><dt>pandas</dt><dd>{environment.pandas}</dd></div>
          <div><dt>NumPy</dt><dd>{environment.numpy}</dd></div>
          <div><dt>SciPy</dt><dd>{environment.scipy}</dd></div>
        </dl>
      </section>

      <details className="model-artifact-integrity">
        <summary>Artifact integrity</summary>
        <dl>
          <div><dt>Filename</dt><dd><code>{model.artifact.filename}</code></dd></div>
          <div><dt>Format</dt><dd>{model.artifact.serialization_format}</dd></div>
          <div><dt>Pipeline size</dt><dd>{formatBytes(model.artifact.pipeline_size_bytes)}</dd></div>
          <div><dt>Target</dt><dd>{model.artifact.target}</dd></div>
          <div><dt>SHA-256</dt><dd><code>{model.artifact.pipeline_sha256}</code></dd></div>
        </dl>
      </details>

      <p className="artifact-trust-note model-trust-note">
        This MLForge artifact contains an executable Python pickle payload. Load it only when its
        origin and custody are trusted; never load a model file from an unknown source.
      </p>

      {model.warnings.length > 0 ? (
        <details className="result-warnings model-detail-warnings">
          <summary>Final-fit warnings ({model.warnings.length})</summary>
          <ul>
            {model.warnings.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        </details>
      ) : null}
    </article>
  );
}
