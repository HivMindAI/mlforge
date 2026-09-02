"use client";

import { useCallback, useEffect, useState } from "react";

import { FinalizeModel } from "@/components/finalize-model";
import {
  type BenchmarkEntry,
  type BenchmarkMetricSummary,
  type ExperimentResult,
  getExperimentResults,
} from "@/lib/datasets";
import { estimatorLabel, formatMetricValue, metricLabel } from "@/lib/model-display";

function formatDuration(seconds: number): string {
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
}

function metricFor(entry: BenchmarkEntry, name: string): BenchmarkMetricSummary | null {
  return entry.metrics.find((metric) => metric.name === name) ?? null;
}

function foldWarnings(entry: BenchmarkEntry): readonly string[] {
  return entry.folds.flatMap((fold) =>
    fold.warnings.map((warning) => `Fold ${fold.fold_number}: ${warning}`),
  );
}

function rankedEntries(result: ExperimentResult): readonly BenchmarkEntry[] {
  return [...result.entries].sort((left, right) => {
    if (left.rank === null) return right.rank === null ? 0 : 1;
    if (right.rank === null) return -1;
    return left.rank - right.rank;
  });
}

type ExperimentResultsProps = Readonly<{ experimentId: string }>;

export function ExperimentResults({ experimentId }: ExperimentResultsProps) {
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  const retry = useCallback(() => {
    setResult(null);
    setError(null);
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getExperimentResults(experimentId, controller.signal)
      .then(setResult)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(
          requestError instanceof Error ? requestError.message : "Experiment results failed to load.",
        );
      });
    return () => controller.abort();
  }, [experimentId, requestVersion]);

  if (error) {
    return (
      <section className="results-section" aria-labelledby="results-title">
        <div className="configuration-section-heading">
          <h2 id="results-title">Results unavailable</h2>
        </div>
        <div className="results-error" role="alert">
          <p>{error}</p>
          <button className="secondary-button" type="button" onClick={retry}>
            Try again
          </button>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="results-section" aria-labelledby="results-title" aria-live="polite">
        <div className="configuration-section-heading">
          <h2 id="results-title">Loading results</h2>
          <span>Saved evidence</span>
        </div>
        <div className="results-loading" role="status" aria-busy="true">
          <span>Reading the completed cross-validation benchmark.</span>
          <div className="loading-lines loading-lines-compact" aria-hidden="true">
            <span />
            <span />
          </div>
        </div>
      </section>
    );
  }

  const entries = rankedEntries(result);
  const winner = entries.find((entry) => entry.rank === 1) ?? null;
  const secondaryMetricNames =
    result.task === "classification"
      ? (["accuracy", "f1_macro"] as const)
      : (["mean_absolute_error", "r2"] as const);
  const winnerSecondaryMetrics = secondaryMetricNames.map((name) =>
    winner ? metricFor(winner, name) : null,
  );
  const failedEntryCount = entries.filter((entry) => entry.status === "failed").length;

  return (
    <section className="results-section" aria-labelledby="results-title">
      <header className="results-header">
        <div>
          <span className="section-kicker">Results</span>
          <h2 id="results-title">Best model</h2>
          <p>{winner ? estimatorLabel(winner.estimator) : "No ranked model"}</p>
        </div>
        <dl className="results-dataset-context" aria-label="Result dataset context">
          <div>
            <dt>Target</dt>
            <dd>{result.target}</dd>
          </div>
          <div>
            <dt>Rows</dt>
            <dd>{result.row_count.toLocaleString("en-US")}</dd>
          </div>
        </dl>
      </header>

      {winner ? (
        <dl className="result-highlights" aria-label="Best model metrics">
          <div>
            <dt>{metricLabel(result.primary_metric)}</dt>
            <dd>{formatMetricValue(result.primary_metric, winner.primary_metric_mean)}</dd>
          </div>
          <div>
            <dt>Standard deviation</dt>
            <dd>
              {formatMetricValue(result.primary_metric, winner.primary_metric_standard_deviation)}
            </dd>
          </div>
          <div>
            <dt>{metricLabel(secondaryMetricNames[0])}</dt>
            <dd>
              {formatMetricValue(
                secondaryMetricNames[0],
                winnerSecondaryMetrics[0]?.mean ?? null,
              )}
            </dd>
          </div>
          <div>
            <dt>{metricLabel(secondaryMetricNames[1])}</dt>
            <dd>
              {formatMetricValue(
                secondaryMetricNames[1],
                winnerSecondaryMetrics[1]?.mean ?? null,
              )}
            </dd>
          </div>
        </dl>
      ) : null}

      {result.status === "partial" ? (
        <p className="partial-result-note" role="status">
          <strong>Partial result</strong>
          <span>
            {failedEntryCount} of {entries.length} models failed. Successful models remain ranked
            using the recorded comparison metric.
          </span>
        </p>
      ) : null}

      <section className="result-subsection" aria-labelledby="ranking-title">
        <div className="configuration-section-heading">
          <div>
            <h3 id="ranking-title">Model ranking</h3>
            <p>Ranked by mean {metricLabel(result.primary_metric).toLowerCase()}.</p>
          </div>
          <span>{entries.length}</span>
        </div>
        <div
          className="data-table-wrap"
          role="region"
          aria-label="Model ranking table"
          tabIndex={0}
        >
          <table className="data-table results-ranking-table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Model</th>
                <th scope="col">{metricLabel(result.primary_metric)}</th>
                <th scope="col">Std. dev.</th>
                <th scope="col">{metricLabel(secondaryMetricNames[0])}</th>
                <th scope="col">{metricLabel(secondaryMetricNames[1])}</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.estimator}>
                  <td>{entry.rank ?? "—"}</td>
                  <th scope="row">{estimatorLabel(entry.estimator)}</th>
                  <td>{formatMetricValue(result.primary_metric, entry.primary_metric_mean)}</td>
                  <td>
                    {formatMetricValue(
                      result.primary_metric,
                      entry.primary_metric_standard_deviation,
                    )}
                  </td>
                  {secondaryMetricNames.map((name) => (
                    <td key={name}>{formatMetricValue(name, metricFor(entry, name)?.mean ?? null)}</td>
                  ))}
                  <td>
                    <span className={`result-entry-status result-entry-${entry.status}`}>
                      {entry.status === "succeeded" ? "Complete" : "Failed"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="result-subsection" aria-labelledby="evidence-title">
        <div className="configuration-section-heading">
          <div>
            <h3 id="evidence-title">Metric evidence</h3>
            <p>Mean, variation, and the observed score from every shared fold.</p>
          </div>
          <span>{result.fold_count} folds</span>
        </div>
        <div className="model-evidence-list">
          {entries.map((entry) => (
            <details key={entry.estimator} open={entry.rank === 1}>
              <summary>
                <span>{estimatorLabel(entry.estimator)}</span>
                <span>{entry.rank === null ? "Failed" : `Rank ${entry.rank}`}</span>
              </summary>
              {entry.status === "succeeded" ? (
                <div
                  className="data-table-wrap"
                  role="region"
                  aria-label={`${estimatorLabel(entry.estimator)} metric evidence table`}
                  tabIndex={0}
                >
                  <table className="data-table metric-evidence-table">
                    <thead>
                      <tr>
                        <th scope="col">Metric</th>
                        <th scope="col">Mean</th>
                        <th scope="col">Std. dev.</th>
                        {entry.folds.map((fold) => (
                          <th scope="col" key={fold.fold_number}>Fold {fold.fold_number}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {entry.metrics.map((metric) => (
                        <tr key={metric.name}>
                          <th scope="row">{metricLabel(metric.name)}</th>
                          <td>{formatMetricValue(metric.name, metric.mean)}</td>
                          <td>{formatMetricValue(metric.name, metric.standard_deviation)}</td>
                          {metric.fold_values.map((value, index) => (
                            <td key={`${metric.name}-${index}`}>
                              {formatMetricValue(metric.name, value)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="model-failure" role="alert">
                  <strong>{entry.failure?.error_type ?? "Model failed"}</strong>
                  <p>{entry.failure?.message ?? "No failure detail was recorded."}</p>
                  {entry.failure_fold ? <span>Stopped at fold {entry.failure_fold}.</span> : null}
                </div>
              )}
              {foldWarnings(entry).length > 0 ? (
                <div className="model-fold-warnings">
                  <strong>Fold warnings</strong>
                  <ul>
                    {foldWarnings(entry).map((warning) => <li key={warning}>{warning}</li>)}
                  </ul>
                </div>
              ) : null}
              <p className="model-duration">Observed duration: {formatDuration(entry.duration_seconds)}</p>
            </details>
          ))}
        </div>
      </section>

      <section className="result-subsection" aria-labelledby="fold-plan-title">
        <div className="configuration-section-heading">
          <div>
            <h3 id="fold-plan-title">Shared fold plan</h3>
            <p>Every model used the same deterministic train and validation partitions.</p>
          </div>
        </div>
        <dl className="fold-plan-list">
          {result.folds.map((fold) => (
            <div key={fold.fold_number}>
              <dt>Fold {fold.fold_number}</dt>
              <dd>{fold.train_rows} train · {fold.validation_rows} validation</dd>
            </div>
          ))}
        </dl>
      </section>

      {result.warnings.length > 0 ? (
        <details className="result-warnings">
          <summary>MLForge warnings ({result.warnings.length})</summary>
          <ul>
            {result.warnings.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        </details>
      ) : null}

      {winner ? (
        <FinalizeModel experimentId={experimentId} estimator={winner.estimator} />
      ) : null}
    </section>
  );
}
