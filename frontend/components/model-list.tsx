"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { EmptyState, PageErrorState, PageLoadingState } from "@/components/async-state";
import {
  type FinalModelList,
  getFinalModels,
} from "@/lib/datasets";
import { estimatorLabel, formatMetricValue, metricLabel, taskLabel } from "@/lib/model-display";

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

export function ModelList() {
  const [models, setModels] = useState<FinalModelList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  const retry = useCallback(() => {
    setModels(null);
    setError(null);
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getFinalModels(controller.signal)
      .then(setModels)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(requestError instanceof Error ? requestError.message : "Models failed to load.");
      });
    return () => controller.abort();
  }, [requestVersion]);

  if (error) {
    return (
      <PageErrorState
        kicker="Model registry"
        title="Models unavailable"
        description={error}
        onRetry={retry}
        secondaryHref="/experiments"
        secondaryLabel="Experiments"
      />
    );
  }

  if (!models) {
    return (
      <PageLoadingState
        kicker="Model registry"
        title="Loading models"
        description="MLForge is inspecting completed local final-model records."
      />
    );
  }

  return (
    <div className="page models-page">
      <header className="page-header">
        <span className="section-kicker">Model registry</span>
        <h1>Models</h1>
        <p>Finalized models created in this local MLForge workspace.</p>
      </header>

      <section className="models-section" aria-labelledby="model-list-title">
        <div className="configuration-section-heading">
          <div>
            <h2 id="model-list-title">Finalized models</h2>
            <p>Open a model to inspect its source evidence and input contract.</p>
          </div>
          <span>{models.count}</span>
        </div>

        {models.count === 0 ? (
          <EmptyState
            className="models-empty"
            title="No finalized models"
            description="Complete an experiment, then finalize its rank-one model."
            actionHref="/experiments"
            actionLabel="Review experiments"
          />
        ) : (
          <div
            className="data-table-wrap models-table-wrap"
            role="region"
            aria-label="Finalized models table"
            tabIndex={0}
          >
            <table className="data-table models-table">
              <thead>
                <tr>
                  <th scope="col">Name</th>
                  <th scope="col">Dataset</th>
                  <th scope="col">Type</th>
                  <th scope="col">Selection metric</th>
                  <th scope="col">Created</th>
                </tr>
              </thead>
              <tbody>
                {models.models.map((model) => (
                  <tr key={model.final_model_id}>
                    <th scope="row">
                      <Link className="model-name-link" href={`/models/${model.final_model_id}`}>
                        {estimatorLabel(model.estimator)}
                      </Link>
                      <code className="model-short-id">{model.final_model_id.slice(0, 8)}</code>
                    </th>
                    <td>{model.dataset_name}</td>
                    <td>{taskLabel(model.task)}</td>
                    <td>
                      <span className="model-metric-value">
                        {formatMetricValue(model.primary_metric, model.primary_metric_mean)}
                      </span>
                      <span className="model-metric-label">
                        {metricLabel(model.primary_metric)}
                      </span>
                    </td>
                    <td>{formatDate(model.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
