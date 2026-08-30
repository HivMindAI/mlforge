"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { PageErrorState, PageLoadingState } from "@/components/async-state";
import { type PredictionResultRecord, getPrediction } from "@/lib/datasets";

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

type PredictionResultProps = Readonly<{ predictionId: string }>;

export function PredictionResult({ predictionId }: PredictionResultProps) {
  const [result, setResult] = useState<PredictionResultRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  const retry = useCallback(() => {
    setResult(null);
    setError(null);
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getPrediction(predictionId, controller.signal)
      .then(setResult)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(
          requestError instanceof Error ? requestError.message : "Prediction results failed to load.",
        );
      });
    return () => controller.abort();
  }, [predictionId, requestVersion]);

  if (error) {
    return (
      <PageErrorState
        kicker="Prediction result"
        title="Result unavailable"
        description={error}
        onRetry={retry}
        secondaryHref="/predictions/new"
        secondaryLabel="Back to predictions"
      />
    );
  }

  if (!result) {
    return (
      <PageLoadingState
        kicker="Prediction result"
        title="Loading results"
        description="MLForge is validating the saved output before displaying its preview."
      />
    );
  }

  return (
    <article className="page prediction-result-page">
      <Link className="text-link model-back-link" href="/predictions/new">
        Run another prediction
      </Link>

      <header className="prediction-result-header">
        <div>
          <span className="section-kicker">Prediction complete</span>
          <h1>Prediction complete</h1>
          <p>{result.input_filename}</p>
        </div>
        <a
          className="primary-button"
          href={`/api/predictions/${result.prediction_id}/download`}
          download="predictions.csv"
        >
          Download predictions.csv
        </a>
      </header>

      <dl className="prediction-result-facts">
        <div>
          <dt>Rows processed</dt>
          <dd>{result.row_count.toLocaleString("en-US")}</dd>
        </div>
        <div>
          <dt>Invalid rows</dt>
          <dd>{result.invalid_row_count}</dd>
        </div>
        <div>
          <dt>Completed</dt>
          <dd>{formatDate(result.completed_at)}</dd>
        </div>
      </dl>

      <section className="prediction-preview" aria-labelledby="prediction-preview-title">
        <div className="configuration-section-heading">
          <div>
            <h2 id="prediction-preview-title">Result preview</h2>
            <p>
              First {result.preview_rows.length.toLocaleString("en-US")} of{" "}
              {result.row_count.toLocaleString("en-US")} rows.
            </p>
          </div>
          {result.preview_truncated ? <span>Preview limited to {result.preview_limit}</span> : null}
        </div>
        <div
          className="data-table-wrap"
          role="region"
          aria-label="Prediction result preview table"
          tabIndex={0}
        >
          <table className="data-table prediction-preview-table">
            <thead>
              <tr>
                <th scope="col">Row</th>
                <th scope="col">Prediction</th>
              </tr>
            </thead>
            <tbody>
              {result.preview_rows.map((row) => (
                <tr key={row.row_number}>
                  <th scope="row">{row.row_number.toLocaleString("en-US")}</th>
                  <td>{row.prediction}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="prediction-result-footer">
        <div>
          <strong>Complete CSV</strong>
          <p>The download contains every processed row and its prediction.</p>
        </div>
        <a
          className="secondary-button"
          href={`/api/predictions/${result.prediction_id}/download`}
          download="predictions.csv"
        >
          Download CSV
        </a>
      </footer>

      <p className="configuration-id prediction-result-id">
        <span>Prediction ID</span>
        <code>{result.prediction_id}</code>
      </p>
    </article>
  );
}
