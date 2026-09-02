"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { PageErrorState, PageLoadingState } from "@/components/async-state";
import { ExperimentResults } from "@/components/experiment-results";
import {
  type DatasetRecord,
  type ExperimentRecord,
  getDataset,
  getExperiment,
  getExperimentJob,
  getJob,
  type JobRecord,
  startExperiment,
} from "@/lib/datasets";
import { estimatorLabel, metricLabel, taskLabel } from "@/lib/model-display";

function statusLabel(job: JobRecord | null): string {
  if (!job) return "Configured";
  return job.status.charAt(0).toUpperCase() + job.status.slice(1);
}

function pageTitle(job: JobRecord | null): string {
  if (!job) return "Experiment configured";
  if (job.status === "complete") return "Comparison complete";
  if (job.status === "failed") return "Comparison failed";
  return "Running experiment";
}

type ExperimentRunProps = Readonly<{ experimentId: string }>;

export function ExperimentRun({ experimentId }: ExperimentRunProps) {
  const [experiment, setExperiment] = useState<ExperimentRecord | null>(null);
  const [dataset, setDataset] = useState<DatasetRecord | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [requestVersion, setRequestVersion] = useState(0);

  const retry = useCallback(() => {
    setExperiment(null);
    setDataset(null);
    setJob(null);
    setLoadError(null);
    setActionError(null);
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getExperiment(experimentId, controller.signal)
      .then(async (loadedExperiment) => {
        const [loadedDataset, loadedJob] = await Promise.all([
          getDataset(loadedExperiment.dataset_id, controller.signal),
          getExperimentJob(experimentId, controller.signal),
        ]);
        return { loadedDataset, loadedExperiment, loadedJob };
      })
      .then(({ loadedDataset, loadedExperiment, loadedJob }) => {
        setDataset(loadedDataset);
        setExperiment(loadedExperiment);
        setJob(loadedJob);
      })
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setLoadError(requestError instanceof Error ? requestError.message : "Experiment failed to load.");
      });
    return () => controller.abort();
  }, [experimentId, requestVersion]);

  useEffect(() => {
    if (!job || (job.status !== "waiting" && job.status !== "running")) return;

    const jobId = job.job_id;
    const controller = new AbortController();
    let stopped = false;
    let timer: number | undefined;

    async function poll() {
      try {
        const updated = await getJob(jobId, controller.signal);
        if (stopped) return;
        setJob(updated);
        if (updated.status === "waiting" || updated.status === "running") {
          timer = window.setTimeout(poll, 1000);
        }
      } catch (requestError) {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        if (!stopped) {
          setActionError(
            requestError instanceof Error
              ? requestError.message
              : "Comparison status could not be refreshed.",
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
  }, [job]);

  async function handleStart() {
    setStarting(true);
    setActionError(null);
    try {
      setJob(await startExperiment(experimentId));
    } catch (requestError) {
      setActionError(
        requestError instanceof Error ? requestError.message : "Comparison could not be started.",
      );
    } finally {
      setStarting(false);
    }
  }

  if (loadError) {
    return (
      <PageErrorState
        kicker="Experiment"
        title="Experiment unavailable"
        description={loadError}
        onRetry={retry}
        secondaryHref="/experiments"
        secondaryLabel="Experiments"
      />
    );
  }

  if (!experiment || !dataset) {
    return (
      <PageLoadingState
        kicker="Experiment"
        title="Loading experiment"
        description="MLForge is restoring the saved configuration and execution state."
      />
    );
  }

  const active = job?.status === "waiting" || job?.status === "running";

  return (
    <div className="page experiment-page">
      <Link className="text-link experiment-back-link" href="/experiments">
        Back to experiments
      </Link>
      <header className="experiment-header">
        <div>
          <span className="section-kicker">Experiment</span>
          <h1>{pageTitle(job)}</h1>
          <p>
            {active
              ? "MLForge is running the saved cross-validation comparison."
              : job?.status === "complete"
                ? "The complete benchmark evidence has been saved."
                : job?.status === "failed"
                  ? "The comparison stopped without a successful result."
                  : "The configuration is saved. No training job has been started."}
          </p>
        </div>
      </header>

      <dl className="configuration-summary" aria-label="Experiment configuration">
        <div>
          <dt>Problem type</dt>
          <dd>{taskLabel(experiment.task)}</dd>
        </div>
        <div>
          <dt>Validation</dt>
          <dd>Cross-validation</dd>
        </div>
        <div>
          <dt>Folds</dt>
          <dd>{experiment.fold_count}</dd>
        </div>
        <div>
          <dt>Ranking metric</dt>
          <dd>{metricLabel(experiment.primary_metric)}</dd>
        </div>
      </dl>

      <section className="experiment-dataset" aria-labelledby="experiment-dataset-title">
        <div className="configuration-section-heading">
          <div>
            <h2 id="experiment-dataset-title">Dataset</h2>
            <p>Stored metadata for the exact dataset used by this configuration.</p>
          </div>
          <Link className="text-link" href={`/datasets/${dataset.dataset_id}`}>
            Open data overview
          </Link>
        </div>
        <dl className="experiment-dataset-facts">
          <div>
            <dt>Filename</dt>
            <dd>{dataset.filename}</dd>
          </div>
          <div>
            <dt>Target</dt>
            <dd>{dataset.target ?? "Not recorded"}</dd>
          </div>
          <div>
            <dt>Rows</dt>
            <dd>{dataset.row_count.toLocaleString("en-US")}</dd>
          </div>
          <div>
            <dt>Columns</dt>
            <dd>{dataset.column_count.toLocaleString("en-US")}</dd>
          </div>
        </dl>
      </section>

      {actionError ? (
        <div className="form-message form-message-error experiment-message" role="alert">
          {actionError}
        </div>
      ) : null}

      <section className="execution-section" aria-labelledby="execution-title">
        <div className="configuration-section-heading">
          <h2 id="execution-title">Execution</h2>
          <span>Job level</span>
        </div>
        <div className="execution-row" aria-live="polite" aria-atomic="true">
          <div>
            <strong>Model comparison</strong>
            <p>All selected models across {experiment.fold_count} shared folds.</p>
          </div>
          <span className={`job-status job-status-${job?.status ?? "configured"}`}>
            {statusLabel(job)}
          </span>
        </div>

        {active ? (
          <p className="execution-note">
            The MLForge core returns only terminal comparison results. Per-model live status and
            progress percentages are not available, so they are not estimated here.
          </p>
        ) : null}

        {!job ? (
          <div className="run-action">
            <div>
              <strong>Ready to run</strong>
              <p>This executes real cross-validation and writes an immutable benchmark record.</p>
            </div>
            <button
              className="primary-button"
              type="button"
              onClick={handleStart}
              disabled={starting}
            >
              {starting ? "Starting..." : "Run comparison"}
            </button>
          </div>
        ) : null}

        {job?.status === "failed" ? (
          <div className="terminal-job-details terminal-job-failed">
            <strong>Comparison failed</strong>
            <p>No successful comparison result was published.</p>
            <details open>
              <summary>Error details</summary>
              <p>{job.error_message ?? "No error detail was recorded."}</p>
            </details>
          </div>
        ) : null}
      </section>

      {job?.status === "complete" ? <ExperimentResults experimentId={experimentId} /> : null}

      <section className="configured-models" aria-labelledby="configured-models-title">
        <div className="configuration-section-heading">
          <h2 id="configured-models-title">Selected models</h2>
          <span>{experiment.estimators.length}</span>
        </div>
        <ul>
          {experiment.estimators.map((estimator) => (
            <li key={estimator}>{estimatorLabel(estimator)}</li>
          ))}
        </ul>
      </section>

      <div className="configuration-id">
        <span>Experiment ID</span>
        <code>{experiment.experiment_id}</code>
        {job ? (
          <>
            <span>Job ID</span>
            <code>{job.job_id}</code>
            {job.benchmark_id ? (
              <>
                <span>Benchmark ID</span>
                <code>{job.benchmark_id}</code>
              </>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}
