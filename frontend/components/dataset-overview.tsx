"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { PageErrorState, PageLoadingState } from "@/components/async-state";
import {
  analyzeDataset,
  type ColumnProfile,
  type DatasetAnalysis,
} from "@/lib/datasets";

const integerFormatter = new Intl.NumberFormat("en-US");
const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 1,
});

function formatKind(kind: ColumnProfile["kind"]): string {
  return kind.charAt(0).toUpperCase() + kind.slice(1);
}

type Issue = Readonly<{ key: string; title: string; detail: string }>;

function potentialIssues(analysis: DatasetAnalysis): readonly Issue[] {
  const issues: Issue[] = [];
  if (analysis.duplicate_row_count > 0) {
    issues.push({
      key: "duplicate-rows",
      title: "Duplicate rows",
      detail: `${integerFormatter.format(analysis.duplicate_row_count)} duplicate rows detected.`,
    });
  }

  for (const column of analysis.columns) {
    if (column.missing_count > 0) {
      issues.push({
        key: `missing-${column.name}`,
        title: column.name,
        detail: `${integerFormatter.format(column.missing_count)} missing values (${percentFormatter.format(column.missing_ratio)}).`,
      });
    }
    if (column.is_likely_identifier) {
      issues.push({
        key: `identifier-${column.name}`,
        title: column.name,
        detail: "Possible identifier column.",
      });
    }
    if (column.is_constant) {
      issues.push({
        key: `constant-${column.name}`,
        title: column.name,
        detail: "Constant column.",
      });
    }
    if (column.is_high_cardinality) {
      issues.push({
        key: `cardinality-${column.name}`,
        title: column.name,
        detail: "High-cardinality column.",
      });
    }
    if (column.infinite_count > 0) {
      issues.push({
        key: `infinite-${column.name}`,
        title: column.name,
        detail: `${integerFormatter.format(column.infinite_count)} infinite values.`,
      });
    }
  }
  return issues;
}

type DatasetOverviewProps = Readonly<{ datasetId: string }>;

export function DatasetOverview({ datasetId }: DatasetOverviewProps) {
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  const retry = useCallback(() => {
    setAnalysis(null);
    setError(null);
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    void analyzeDataset(datasetId, controller.signal)
      .then(setAnalysis)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(requestError instanceof Error ? requestError.message : "Analysis failed.");
      });

    return () => controller.abort();
  }, [datasetId, requestVersion]);

  if (error) {
    return (
      <PageErrorState
        kicker="Dataset review"
        title="Data overview unavailable"
        description={error}
        onRetry={retry}
        secondaryHref="/datasets/new"
        secondaryLabel="Upload another dataset"
      />
    );
  }

  if (!analysis) {
    return (
      <PageLoadingState
        kicker="Dataset review"
        title="Analyzing dataset"
        description="MLForge is profiling column types, missing values, and the selected target."
      />
    );
  }

  const numericalCount = analysis.columns.filter(
    (column) => column.kind === "integer" || column.kind === "float",
  ).length;
  const categoricalCount = analysis.columns.filter((column) =>
    ["boolean", "categorical", "string"].includes(column.kind),
  ).length;
  const columnsWithMissing = analysis.columns.filter((column) => column.missing_count > 0).length;
  const issues = potentialIssues(analysis);

  return (
    <div className="page dataset-overview">
      <header className="overview-header">
        <div>
          <span className="section-kicker">Data overview</span>
          <h1>{analysis.dataset.filename}</h1>
          <p>Review MLForge&apos;s detected structure before configuring an experiment.</p>
        </div>
        <Link className="secondary-button secondary-link" href="/datasets/new">
          New dataset
        </Link>
      </header>

      <dl className="overview-facts" aria-label="Dataset summary">
        <div>
          <dt>Rows</dt>
          <dd>{integerFormatter.format(analysis.dataset.row_count)}</dd>
        </div>
        <div>
          <dt>Columns</dt>
          <dd>{integerFormatter.format(analysis.dataset.column_count)}</dd>
        </div>
        <div>
          <dt>Numerical</dt>
          <dd>{integerFormatter.format(numericalCount)}</dd>
        </div>
        <div>
          <dt>Categorical</dt>
          <dd>{integerFormatter.format(categoricalCount)}</dd>
        </div>
        <div>
          <dt>With missing</dt>
          <dd>{integerFormatter.format(columnsWithMissing)}</dd>
        </div>
        <div>
          <dt>Detected task</dt>
          <dd className="task-value">{analysis.target.task_hint}</dd>
        </div>
      </dl>

      <section className="overview-section" aria-labelledby="issues-title">
        <div className="overview-section-heading">
          <div>
            <h2 id="issues-title">Potential issues</h2>
            <p>Signals reported by MLForge&apos;s dataset profiler.</p>
          </div>
          <span>{integerFormatter.format(issues.length)}</span>
        </div>
        {issues.length > 0 ? (
          <ul className="issue-list">
            {issues.map((issue) => (
              <li key={issue.key}>
                <strong>{issue.title}</strong>
                <span>{issue.detail}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="quiet-state">No potential issues were detected by the current profiler.</p>
        )}

        {analysis.warnings.length > 0 ? (
          <details className="core-warnings">
            <summary>MLForge warnings ({analysis.warnings.length})</summary>
            <ul>
              {analysis.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </details>
        ) : null}
      </section>

      <section className="overview-section" aria-labelledby="target-title">
        <div className="overview-section-heading">
          <div>
            <h2 id="target-title">Target</h2>
            <p>The column selected for prediction.</p>
          </div>
        </div>
        <dl className="target-overview">
          <div>
            <dt>Column</dt>
            <dd>{analysis.target.name}</dd>
          </div>
          <div>
            <dt>Task hint</dt>
            <dd>{analysis.target.task_hint}</dd>
          </div>
          <div>
            <dt>Missing</dt>
            <dd>{integerFormatter.format(analysis.target.missing_count)}</dd>
          </div>
          <div>
            <dt>Unique</dt>
            <dd>{integerFormatter.format(analysis.target.unique_count)}</dd>
          </div>
        </dl>

        {analysis.target.class_distribution.length > 0 ? (
          <div
            className="compact-table-wrap target-distribution"
            role="region"
            aria-label="Target distribution table"
            tabIndex={0}
          >
            <table className="data-table compact-table">
              <caption>Observed target distribution</caption>
              <thead>
                <tr>
                  <th scope="col">Value</th>
                  <th scope="col">Count</th>
                  <th scope="col">Share</th>
                </tr>
              </thead>
              <tbody>
                {analysis.target.class_distribution.map((item) => (
                  <tr key={item.value}>
                    <th scope="row">{item.value}</th>
                    <td>{integerFormatter.format(item.count)}</td>
                    <td>{percentFormatter.format(item.ratio)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {analysis.target.distribution_truncated ? (
              <p className="table-note">Only the most frequent target values are shown.</p>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="overview-section" aria-labelledby="columns-title">
        <div className="overview-section-heading">
          <div>
            <h2 id="columns-title">Columns</h2>
            <p>Detected types and completeness across the uploaded CSV.</p>
          </div>
          <span>{integerFormatter.format(analysis.columns.length)}</span>
        </div>
        <div
          className="data-table-wrap"
          role="region"
          aria-label="Dataset columns table"
          tabIndex={0}
        >
          <table className="data-table columns-table">
            <thead>
              <tr>
                <th scope="col">Name</th>
                <th scope="col">Type</th>
                <th scope="col">Missing</th>
                <th scope="col">Unique</th>
              </tr>
            </thead>
            <tbody>
              {analysis.columns.map((column) => (
                <tr key={column.name}>
                  <th scope="row">{column.name}</th>
                  <td>
                    <span className="type-label">{formatKind(column.kind)}</span>
                  </td>
                  <td>
                    {integerFormatter.format(column.missing_count)}{" "}
                    <span className="table-muted">
                      ({percentFormatter.format(column.missing_ratio)})
                    </span>
                  </td>
                  <td>{integerFormatter.format(column.unique_count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="overview-actions">
        <div>
          <strong>Training configuration</strong>
          <p>
            {analysis.target.task_hint !== "undetermined"
              ? "Configure a supported cross-validation model comparison."
              : "Review the training capabilities available for this target."}
          </p>
        </div>
        <Link
          className="primary-button"
          href={`/datasets/${analysis.dataset.dataset_id}/experiment/new`}
        >
          {analysis.target.task_hint !== "undetermined"
            ? "Configure comparison"
            : "Review training options"}
        </Link>
      </footer>
    </div>
  );
}
