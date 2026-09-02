"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { EmptyState } from "@/components/async-state";
import {
  type ExperimentList,
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

export function Dashboard() {
  const [history, setHistory] = useState<ExperimentList | null>(null);
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
            : "Recent experiments failed to load.",
        );
      });
    return () => controller.abort();
  }, [requestVersion]);

  const recentExperiments = history?.experiments.slice(0, 5) ?? [];

  return (
    <div className="page">
      <header className="page-header">
        <h1>MLForge</h1>
        <p>Train, compare, and use machine-learning models from your datasets.</p>
        <Link className="primary-button dashboard-primary-action" href="/datasets/new">
          Upload dataset
        </Link>
      </header>

      <section className="dashboard-section" aria-labelledby="recent-experiments-title">
        <div className="section-heading dashboard-section-heading">
          <h2 id="recent-experiments-title">Recent experiments</h2>
          {history && history.count > 0 ? (
            <Link className="text-link" href="/experiments">
              View all
            </Link>
          ) : null}
        </div>

        {error ? (
          <div className="dashboard-error" role="alert">
            <strong>Recent experiments unavailable</strong>
            <p>{error}</p>
            <button className="secondary-button" type="button" onClick={retry}>
              Try again
            </button>
          </div>
        ) : !history ? (
          <div className="dashboard-loading" role="status" aria-live="polite" aria-busy="true">
            <span>Loading saved experiments.</span>
            <div className="loading-lines" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
          </div>
        ) : history.count === 0 ? (
          <EmptyState
            className="dashboard-empty"
            title="No experiments yet"
            description="Upload a CSV dataset to create your first experiment."
            actionHref="/datasets/new"
            actionLabel="Upload dataset"
          />
        ) : (
          <ul className="recent-experiment-list">
            {recentExperiments.map((experiment) => (
              <li key={experiment.experiment_id}>
                <div>
                  <Link href={`/experiments/${experiment.experiment_id}`}>
                    {experiment.dataset_name}
                  </Link>
                  <span>
                    {experiment.task === "classification" ? "Classification" : "Regression"},{" "}
                    {experiment.model_count} models
                  </span>
                </div>
                <div>
                  <span className={`job-status job-status-${experiment.status}`}>
                    {statusLabel(experiment.status)}
                  </span>
                  <time dateTime={experiment.updated_at}>{formatDate(experiment.updated_at)}</time>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
