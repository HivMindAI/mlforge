"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import type { ChangeEvent, DragEvent, FormEvent } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import { EmptyState, PageErrorState } from "@/components/async-state";
import {
  MAX_UPLOAD_BYTES,
  type FinalModelRecord,
  type FinalModelSummary,
  getFinalModel,
  getFinalModels,
  runPrediction,
} from "@/lib/datasets";
import { estimatorLabel } from "@/lib/model-display";

type RunState = "idle" | "running" | "opening";

function validateFile(file: File): string | null {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    return "Choose a file with the .csv extension.";
  }
  if (file.size === 0) {
    return "The selected prediction CSV file is empty.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "Prediction CSV files must be 100 MB or smaller.";
  }
  return null;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

type PredictionWorkflowProps = Readonly<{ initialModelId?: string }>;

export function PredictionWorkflow({ initialModelId }: PredictionWorkflowProps) {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const [models, setModels] = useState<readonly FinalModelSummary[]>([]);
  const [modelId, setModelId] = useState("");
  const [model, setModel] = useState<FinalModelRecord | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<RunState>("idle");
  const [progress, setProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelsRequestVersion, setModelsRequestVersion] = useState(0);

  const retryModels = useCallback(() => {
    setModels([]);
    setModelId("");
    setModel(null);
    setModelsError(null);
    setError(null);
    setModelsLoading(true);
    setModelsRequestVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getFinalModels(controller.signal)
      .then((response) => {
        setModels(response.models);
        if (
          initialModelId &&
          response.models.some((candidate) => candidate.final_model_id === initialModelId)
        ) {
          setModelLoading(true);
          setModelId(initialModelId);
        }
      })
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setModelsError(
          requestError instanceof Error ? requestError.message : "Models failed to load.",
        );
      })
      .finally(() => setModelsLoading(false));
    return () => controller.abort();
  }, [initialModelId, modelsRequestVersion]);

  useEffect(() => {
    if (!modelId) return;
    const controller = new AbortController();
    void getFinalModel(modelId, controller.signal)
      .then(setModel)
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(requestError instanceof Error ? requestError.message : "Model failed to load.");
      })
      .finally(() => setModelLoading(false));
    return () => controller.abort();
  }, [modelId]);

  function chooseFile(selected: File) {
    const validationError = validateFile(selected);
    if (validationError) {
      setFile(null);
      setError(validationError);
      if (fileInput.current) fileInput.current.value = "";
      return;
    }
    setFile(selected);
    setState("idle");
    setProgress(0);
    setError(null);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.currentTarget.files?.[0];
    if (selected) chooseFile(selected);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    if (state !== "idle") return;
    const selected = event.dataTransfer.files[0];
    if (selected) chooseFile(selected);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!modelId || !model || !file || state !== "idle") return;
    setError(null);
    setProgress(0);
    setState("running");
    try {
      const created = await runPrediction(modelId, file, setProgress);
      setProgress(100);
      setState("opening");
      router.push(`/predictions/${created.prediction_id}`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Prediction failed.");
      setState("idle");
    }
  }

  if (modelsError) {
    return (
      <PageErrorState
        kicker="Predictions"
        title="Finalized models unavailable"
        description={modelsError}
        onRetry={retryModels}
        secondaryHref="/models"
        secondaryLabel="Model registry"
      />
    );
  }

  const busy = state !== "idle";

  return (
    <div className="page prediction-page">
      <header className="page-header">
        <span className="section-kicker">Predictions</span>
        <h1>Run prediction</h1>
        <p>Choose a finalized local model, then provide a CSV matching its input schema.</p>
      </header>

      {error ? (
        <div className="form-message form-message-error prediction-message" role="alert">
          {error}
        </div>
      ) : null}

      {modelsLoading ? (
        <div className="prediction-loading" role="status" aria-live="polite" aria-busy="true">
          <span>Loading finalized models.</span>
          <div className="loading-lines loading-lines-compact" aria-hidden="true">
            <span />
            <span />
          </div>
        </div>
      ) : models.length === 0 ? (
        <EmptyState
          className="prediction-empty"
          headingLevel={2}
          title="No finalized model is available"
          description="Complete a comparison and finalize its selected model before running predictions."
          actionHref="/models"
          actionLabel="Review models"
        />
      ) : (
        <form className="prediction-form" onSubmit={handleSubmit} aria-busy={busy}>
          <section className="prediction-section" aria-labelledby="prediction-model-title">
            <div className="configuration-section-heading">
              <div>
                <h2 id="prediction-model-title">Finalized model</h2>
                <p id="prediction-model-description">
                  Only models created in this local MLForge workspace are eligible.
                </p>
              </div>
            </div>
            <div className="form-field prediction-model-field">
              <label htmlFor="prediction-model">Model</label>
              <select
                id="prediction-model"
                aria-describedby="prediction-model-description"
                value={modelId}
                onChange={(event) => {
                  const nextModelId = event.currentTarget.value;
                  setModel(null);
                  setModelLoading(Boolean(nextModelId));
                  setModelId(nextModelId);
                  setError(null);
                }}
                disabled={busy}
                required
              >
                <option value="">Choose a finalized model</option>
                {models.map((candidate) => (
                  <option key={candidate.final_model_id} value={candidate.final_model_id}>
                    {estimatorLabel(candidate.estimator)} — {candidate.dataset_name}
                  </option>
                ))}
              </select>
            </div>

            {modelLoading ? (
              <p className="prediction-loading" role="status">Verifying model metadata...</p>
            ) : model ? (
              <div className="prediction-schema">
                <div className="prediction-schema-heading">
                  <div>
                    <h3>Expected input columns</h3>
                    <p>Include every column exactly once. The target column is not included.</p>
                  </div>
                  <Link className="text-link" href={`/models/${model.final_model_id}`}>
                    Model details
                  </Link>
                </div>
                <div
                  className="data-table-wrap"
                  role="region"
                  aria-label="Expected prediction input columns table"
                  tabIndex={0}
                >
                  <table className="data-table prediction-schema-table">
                    <thead><tr><th scope="col">Column</th><th scope="col">Role</th><th scope="col">Recorded dtype</th></tr></thead>
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
              </div>
            ) : null}
          </section>

          <section className="prediction-section" aria-labelledby="prediction-file-title">
            <div className="configuration-section-heading">
              <div>
                <h2 id="prediction-file-title">Prediction CSV</h2>
                <p>UTF-8 CSV, up to 100 MB. MLForge validates the file before inference.</p>
              </div>
            </div>
            <input
              ref={fileInput}
              className="visually-hidden"
              type="file"
              accept=".csv,text/csv"
              onChange={handleFileChange}
              disabled={busy}
              tabIndex={-1}
              aria-hidden="true"
            />
            <div
              className={`prediction-dropzone${dragActive ? " is-dragging" : ""}`}
              onDragEnter={(event) => { event.preventDefault(); if (!busy) setDragActive(true); }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                  setDragActive(false);
                }
              }}
              onDrop={handleDrop}
            >
              <div>
                <strong>{file ? file.name : "Drop prediction CSV here"}</strong>
                <span>{file ? formatFileSize(file.size) : "or choose a file from this computer"}</span>
              </div>
              <button
                className="secondary-button"
                type="button"
                onClick={() => fileInput.current?.click()}
                disabled={busy}
              >
                {file ? "Choose another file" : "Choose file"}
              </button>
            </div>
          </section>

          {busy ? (
            <div className="prediction-progress" role="status" aria-live="polite">
              <div><strong>{state === "opening" ? "Opening results" : progress < 100 ? "Uploading CSV" : "Validating schema and running model"}</strong><span>{progress}%</span></div>
              <progress value={progress} max="100" aria-label={`Upload ${progress}% complete`} />
            </div>
          ) : null}

          <div className="prediction-actions">
            <p>Errors from CSV parsing, schema validation, and artifact checks are shown here exactly.</p>
            <button className="primary-button" type="submit" disabled={!model || !file || busy}>
              {state === "opening"
                ? "Opening results..."
                : state === "running"
                  ? "Running prediction..."
                  : "Run prediction"}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
