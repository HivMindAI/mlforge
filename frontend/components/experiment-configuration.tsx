"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import type { FormEvent } from "react";
import { useCallback, useEffect, useState } from "react";

import { PageErrorState, PageLoadingState } from "@/components/async-state";
import {
  analyzeDataset,
  createExperiment,
  type DatasetAnalysis,
  type Estimator,
  type SupervisedTask,
} from "@/lib/datasets";

type EstimatorOption = Readonly<{
  id: Estimator;
  label: string;
  description: string;
}>;

const estimatorOptions: Readonly<Record<SupervisedTask, readonly EstimatorOption[]>> = {
  classification: [
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
  ],
  regression: [
    {
      id: "ridge-regression",
      label: "Ridge Regression",
      description: "Regularized linear baseline with scaled numerical features.",
    },
    {
      id: "random-forest-regressor",
      label: "Random Forest Regressor",
      description: "Nonlinear tree ensemble with deterministic, resource-bounded defaults.",
    },
  ],
};

type ExperimentConfigurationProps = Readonly<{ datasetId: string }>;

export function ExperimentConfiguration({ datasetId }: ExperimentConfigurationProps) {
  const router = useRouter();
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);
  const [foldCount, setFoldCount] = useState(5);
  const [selectedEstimators, setSelectedEstimators] = useState<readonly Estimator[]>(
    estimatorOptions.classification.map((option) => option.id),
  );
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
      .then((loadedAnalysis) => {
        setAnalysis(loadedAnalysis);
        if (loadedAnalysis.target.task_hint !== "undetermined") {
          setSelectedEstimators(
            estimatorOptions[loadedAnalysis.target.task_hint].map((option) => option.id),
          );
        }
      })
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setLoadError(requestError instanceof Error ? requestError.message : "Dataset review failed.");
      });
    return () => controller.abort();
  }, [datasetId, requestVersion]);

  function toggleEstimator(estimator: Estimator, checked: boolean) {
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

  const task = analysis.target.task_hint;
  const supportedTask = task === "classification" || task === "regression" ? task : null;
  const options = supportedTask ? estimatorOptions[supportedTask] : [];

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

      {supportedTask ? (
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
                <p>
                  Shared deterministic {supportedTask === "classification" ? "stratified" : "shuffled"}
                  {" "}folds for every selected model.
                </p>
              </div>
              <span className="fixed-value">
                {supportedTask === "classification" ? "Balanced accuracy" : "Root mean squared error"}
              </span>
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
              Select at least two supported {supportedTask === "classification" ? "classifiers" : "regressors"}.
            </p>
            <div className="model-options">
              {options.map((option) => (
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
            Final model fitting is available after a completed cross-validation comparison.
            Training is not started when this configuration is saved.
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
          <h2 id="unsupported-title">Problem type could not be determined</h2>
          <p>
            Choose a target with enough non-missing variation for MLForge to infer a supported
            task before configuring a comparison.
          </p>
        </section>
      )}
    </div>
  );
}
