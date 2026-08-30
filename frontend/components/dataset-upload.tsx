"use client";

import { useRouter } from "next/navigation";
import type { ChangeEvent, DragEvent, FormEvent } from "react";
import { useRef, useState } from "react";

import {
  MAX_UPLOAD_BYTES,
  type DatasetRecord,
  selectDatasetTarget,
  uploadDataset,
} from "@/lib/datasets";

type UploadState = "idle" | "uploading" | "uploaded" | "saving" | "saved";

function validateFile(file: File): string | null {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    return "Choose a file with the .csv extension.";
  }
  if (file.size === 0) {
    return "The selected CSV file is empty.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "CSV files must be 100 MB or smaller.";
  }
  return null;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const kilobytes = bytes / 1024;
  if (kilobytes < 1024) {
    return `${kilobytes.toFixed(kilobytes >= 10 ? 0 : 1)} KB`;
  }
  const megabytes = kilobytes / 1024;
  return `${megabytes.toFixed(megabytes >= 10 ? 0 : 1)} MB`;
}

export function DatasetUpload() {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [dataset, setDataset] = useState<DatasetRecord | null>(null);
  const [target, setTarget] = useState("");
  const [savedTarget, setSavedTarget] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const busy = state === "uploading" || state === "saving";

  async function processFile(file: File) {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      if (fileInput.current) {
        fileInput.current.value = "";
      }
      return;
    }

    setError(null);
    setDataset(null);
    setTarget("");
    setSavedTarget(null);
    setProgress(0);
    setState("uploading");

    try {
      const uploadedDataset = await uploadDataset(file, setProgress);
      setDataset(uploadedDataset);
      setProgress(100);
      setState("uploaded");
      if (fileInput.current) {
        fileInput.current.value = "";
      }
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Upload failed.");
      setState("idle");
      if (fileInput.current) {
        fileInput.current.value = "";
      }
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    if (file) {
      void processFile(file);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    if (busy) {
      return;
    }
    const file = event.dataTransfer.files[0];
    if (file) {
      void processFile(file);
    }
  }

  async function handleTargetSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!dataset || !target || target === savedTarget) {
      return;
    }

    setError(null);
    setState("saving");
    try {
      const updatedDataset = await selectDatasetTarget(dataset.dataset_id, target);
      setDataset(updatedDataset);
      setSavedTarget(updatedDataset.target);
      setState("saved");
      router.push(`/datasets/${updatedDataset.dataset_id}`);
    } catch (targetError) {
      setError(targetError instanceof Error ? targetError.message : "Target selection failed.");
      setState("uploaded");
    }
  }

  return (
    <div className="dataset-upload">
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

      {error ? (
        <div className="form-message form-message-error" role="alert">
          {error}
        </div>
      ) : null}

      {dataset === null ? (
        <div
          className={`file-dropzone${dragActive ? " is-dragging" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            if (!busy) setDragActive(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
              setDragActive(false);
            }
          }}
          onDrop={handleDrop}
          aria-busy={state === "uploading"}
        >
          {state === "uploading" ? (
            <div className="upload-progress" aria-live="polite">
              <strong>Uploading CSV</strong>
              <progress value={progress} max="100" aria-label={`Upload ${progress}% complete`} />
              <span>{progress}%</span>
            </div>
          ) : (
            <>
              <strong>Drop CSV here</strong>
              <span>or</span>
              <button
                className="file-picker-button"
                type="button"
                onClick={() => fileInput.current?.click()}
              >
                Choose a file
              </button>
              <small>Maximum 100 MB</small>
            </>
          )}
        </div>
      ) : (
        <section className="dataset-summary" aria-labelledby="uploaded-dataset-title">
          <div className="dataset-summary-heading">
            <div>
              <span className="section-kicker">Uploaded dataset</span>
              <h2 id="uploaded-dataset-title">{dataset.filename}</h2>
            </div>
            <button
              className="secondary-button"
              type="button"
              onClick={() => fileInput.current?.click()}
              disabled={busy}
            >
              Choose another file
            </button>
          </div>

          <dl className="dataset-facts">
            <div>
              <dt>Rows</dt>
              <dd>{new Intl.NumberFormat("en-US").format(dataset.row_count)}</dd>
            </div>
            <div>
              <dt>Columns</dt>
              <dd>{new Intl.NumberFormat("en-US").format(dataset.column_count)}</dd>
            </div>
            <div>
              <dt>File size</dt>
              <dd>{formatFileSize(dataset.file_size_bytes)}</dd>
            </div>
          </dl>

          <form className="target-form" onSubmit={handleTargetSubmit} aria-busy={state === "saving"}>
            <div className="form-field">
              <label htmlFor="target-column">Target column</label>
              <select
                id="target-column"
                aria-describedby="target-column-description"
                value={target}
                onChange={(event) => {
                  setTarget(event.currentTarget.value);
                  if (state === "saved") setState("uploaded");
                }}
                disabled={state === "saving"}
                required
              >
                <option value="">Choose a column</option>
                {dataset.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
              <p id="target-column-description">
                Select the column MLForge should learn to predict.
              </p>
            </div>

            <div className="target-actions">
              {state === "saved" && savedTarget === target ? (
                <span className="save-confirmation" role="status">
                  Target column saved.
                </span>
              ) : null}
              <button
                className="primary-button"
                type="submit"
                disabled={!target || state === "saving" || target === savedTarget}
              >
                {state === "saving" ? "Saving..." : "Continue"}
              </button>
            </div>
          </form>
        </section>
      )}
    </div>
  );
}
