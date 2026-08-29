"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import type { FormEvent } from "react";
import { useCallback, useEffect, useState } from "react";

import { PageErrorState, PageLoadingState } from "@/components/async-state";
import {
  analyzeDataset,
  type ClassificationEstimator,
  createExperiment,
  type DatasetAnalysis,
} from "@/lib/datasets";

const estimatorOptions: readonly Readonly<{
  id: ClassificationEstimator;
  label: string;
  description: string;
}>[] = [
  {
    id: "logistic-regression",
    label: "Logistic Regression",
    description: "Linear classification baseline with scaled numerical features.",
  },
  {
    id: "random-forest-classifier",
    label: "Random Forest Classifier",
    description: "Tree ensemble with deterministic, resource-bounded defaults.",
  },
  {
    id: "dummy-classifier",
    label: "Dummy Classifier",
    description: "Prior-based reference baseline for judging useful performance.",
  },
];

type ExperimentConfigurationProps = Readonly<{ datasetId: string }>;

export function ExperimentConfiguration({ datasetId }: ExperimentConfigurationProps) {
  const router = useRouter();
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);
  const [foldCount, setFoldCount] = useState(5);
  const [selectedEstimators, setSelectedEstimators] = useState<
    readonly ClassificationEstimator[]
  >(estimatorOptions.map((option) => option.id));
  const [formError, setFormError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const retry = useCallback(() => {
    setLoadError(null);
    setAnalysis(null);
    setRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void analyzeDataset(datasetId, controller.signal)
      .then(setAnalysis)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setLoadError(requestError instanceof Error ? requestError.message : "Dataset review failed.");
      });
    return () => controller.abort();
  }, [datasetId, requestVersion]);

  function toggleEstimator(estimator: ClassificationEstimator, checked: boolean) {
    setFormError(null);
    setSelectedEstimators((current) =>
      checked ? [...current, estimator] : current.filter((item) => item !== estimator),
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (selectedEstimators.length < 2) {
      setFormError("Select at least two models for a comparison.");
      return;
    }

    setSaving(true);
    setFormError(null);
    try {
      const configured = await createExperiment(datasetId, selectedEstimators, foldCount);
      router.push(`/experiments/${configured.experiment_id}`);
    } catch (requestError) {
      setFormError(
        requestError instanceof Error ? requestError.message : "Configuration could not be saved.",
      );
    } finally {
      setSaving(false);
    }
  }

  if (loadError) {
    return (
      <PageErrorState
        kicker="Experiment"
        title="Training options unavailable"
        description={loadError}
        onRetry={retry}
        secondaryHref={`/datasets/${datasetId}`}
        secondaryLabel="Back to data overview"
      />
    );
  }

  if (!analysis) {
    return (
      <PageLoadingState
        kicker="Experiment"
        title="Loading training options"
        description="MLForge is checking the selected target and supported comparison workflow."
      />
    );
  }

  return (
    <div className="page experiment-page">
      <header className="experiment-header">
        <div>
          <span className="section-kicker">Experiment configuration</span>
          <h1>Compare models</h1>
          <p>Choose a supported validation setup for {analysis.dataset.filename}.</p>
        </div>
        <Link className="secondary-button secondary-link" href={`/datasets/${datasetId}`}>
          Data overview
        </Link>
      </header>

      <dl className="experiment-context" aria-label="Experiment dataset context">
        <div>
          <dt>Problem type</dt>
          <dd>{analysis.target.task_hint}</dd>
        </div>
        <div>
          <dt>Target</dt>
          <dd>{analysis.target.name}</dd>
        </div>
        <div>
          <dt>Rows</dt>
          <dd>{new Intl.NumberFormat("en-US").format(analysis.dataset.row_count)}</dd>
        </div>
      </dl>

      {analysis.target.task_hint === "classification" ? (
        <form className="experiment-form" onSubmit={handleSubmit} aria-busy={saving}>
          {formError ? (
            <div className="form-message form-message-error" role="alert">
              {formError}
            </div>
          ) : null}

          <fieldset className="configuration-section" disabled={saving}>
            <legend>Validation</legend>
            <div className="configuration-row">
              <div>
                <strong>Cross-validation</strong>
                <p>Shared deterministic stratified folds for every selected model.</p>
              </div>
              <span className="fixed-value">Balanced accuracy</span>
            </div>
            <label className="fold-field" htmlFor="fold-count">
              <span>Folds</span>
              <select
                id="fold-count"
                value={foldCount}
                onChange={(event) => {
                  setFoldCount(Number(event.currentTarget.value));
                  setFormError(null);
                }}
              >
                {Array.from({ length: 9 }, (_, index) => index + 2).map((count) => (
                  <option key={count} value={count}>
                    {count}
                  </option>
                ))}
              </select>
            </label>
          </fieldset>

          <fieldset
            className="configuration-section model-selection"
            disabled={saving}
            aria-describedby="model-selection-description"
          >
            <legend>Models</legend>
            <p className="fieldset-description" id="model-selection-description">
              Select at least two supported classifiers.
            </p>
            <div className="model-options">
              {estimatorOptions.map((option) => (
                <label key={option.id} className="model-option">
                  <input
                    type="checkbox"
                    checked={selectedEstimators.includes(option.id)}
                    onChange={(event) => toggleEstimator(option.id, event.currentTarget.checked)}
                  />
                  <span>
                    <strong>{option.label}</strong>
                    <small>{option.description}</small>
                  </span>
                </label>
              ))}
            </div>
          </fieldset>

          <aside className="capability-note">
            Final model fitting is available only after a completed classification
            cross-validation comparison. Training is not started when this configuration is saved.
          </aside>

          <div className="experiment-actions">
            <span>{selectedEstimators.length} models selected</span>
            <button className="primary-button" type="submit" disabled={saving}>
              {saving ? "Saving..." : "Save configuration"}
            </button>
          </div>
        </form>
      ) : (
        <section className="unsupported-comparison" aria-labelledby="unsupported-title">
          <h2 id="unsupported-title">
            {analysis.target.task_hint === "regression"
              ? "Regression comparison is not available"
              : "Problem type could not be determined"}
          </h2>
          {analysis.target.task_hint === "regression" ? (
            <>
              <p>
                MLForge currently supports Ridge Regression and Random Forest Regression as
                individual holdout runs. It does not provide a shared regression comparison or
                regression finalization workflow.
              </p>
              <dl>
                <div>
                  <dt>Available individual estimators</dt>
                  <dd>Ridge Regression, Random Forest Regression</dd>
                </div>
                <div>
                  <dt>Comparison</dt>
                  <dd>Classification only</dd>
                </div>
              </dl>
            </>
          ) : (
            <p>
              Choose a target with enough non-missing variation for MLForge to infer a supported
              task before configuring a comparison.
            </p>
          )}
        </section>
      )}
    </div>
  );
}
