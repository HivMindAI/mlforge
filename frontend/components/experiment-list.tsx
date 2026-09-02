"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { EmptyState, PageErrorState, PageLoadingState } from "@/components/async-state";
import {
  type ExperimentList as ExperimentListRecord,
  type ExperimentStatus,
  getExperiments,
} from "@/lib/datasets";

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function statusLabel(status: ExperimentStatus): string {
  if (status === "complete") return "Completed";
  return status.charAt(0).toUpperCase() + status.slice(1);
}

export function ExperimentList() {
  const [history, setHistory] = useState<ExperimentListRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  const retry = useCallback(() => {
    setHistory(null);
    setError(null);
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getExperiments(controller.signal)
      .then(setHistory)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Experiment history failed to load.",
        );
      });
    return () => controller.abort();
  }, [requestVersion]);

  if (error) {
    return (
      <PageErrorState
        kicker="Experiment history"
        title="Experiments unavailable"
        description={error}
        onRetry={retry}
        secondaryHref="/"
        secondaryLabel="Dashboard"
      />
    );
  }

  if (!history) {
    return (
      <PageLoadingState
        kicker="Experiment history"
        title="Loading experiments"
        description="MLForge is reading saved configurations and durable job states."
      />
    );
  }

  return (
    <div className="page experiments-page">
      <header className="page-header">
        <span className="section-kicker">Experiment history</span>
        <h1>Experiments</h1>
        <p>Saved model comparisons in this local MLForge workspace.</p>
      </header>

      <section className="experiments-section" aria-labelledby="experiment-list-title">
        <div className="configuration-section-heading">
          <div>
            <h2 id="experiment-list-title">Saved experiments</h2>
            <p>Open an experiment to inspect its configuration, evidence, and failure details.</p>
          </div>
          <span>{history.count}</span>
        </div>

        {history.count === 0 ? (
          <EmptyState
            className="experiments-empty"
            title="No experiments yet"
            description="Upload a dataset and save a supported model comparison to begin."
            actionHref="/datasets/new"
            actionLabel="Upload dataset"
          />
        ) : (
          <div
            className="data-table-wrap experiments-table-wrap"
            role="region"
            aria-label="Saved experiments table"
            tabIndex={0}
          >
            <table className="data-table experiments-table">
              <thead>
                <tr>
                  <th scope="col">Experiment</th>
                  <th scope="col">Dataset</th>
                  <th scope="col">Task</th>
                  <th scope="col">Models</th>
                  <th scope="col">Status</th>
                  <th scope="col">Updated</th>
                </tr>
              </thead>
              <tbody>
                {history.experiments.map((experiment) => (
                  <tr key={experiment.experiment_id}>
                    <th scope="row">
                      <Link
                        className="experiment-list-link"
                        href={`/experiments/${experiment.experiment_id}`}
                      >
                        #{experiment.experiment_id.slice(0, 8)}
                      </Link>
                    </th>
                    <td>{experiment.dataset_name}</td>
                    <td>
                      {experiment.task === "classification" ? "Classification" : "Regression"}
                    </td>
                    <td>{experiment.model_count}</td>
                    <td>
                      <span className={`job-status job-status-${experiment.status}`}>
                        {statusLabel(experiment.status)}
                      </span>
                    </td>
                    <td>
                      <time dateTime={experiment.updated_at}>{formatDate(experiment.updated_at)}</time>
                    </td>
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
