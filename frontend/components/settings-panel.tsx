"use client";

import type { FormEvent } from "react";
import { useCallback, useEffect, useState, useSyncExternalStore } from "react";

import { PageErrorState, PageLoadingState } from "@/components/async-state";
import {
  CLASSIFICATION_ESTIMATOR_IDS,
  ESTIMATOR_LABELS,
  REGRESSION_ESTIMATOR_IDS,
  type ClassificationEstimator,
  type RegressionEstimator,
} from "@/lib/datasets";
import {
  CLASSIFICATION_METRIC_IDS,
  METRIC_LABELS,
  REGRESSION_METRIC_IDS,
  downloadWorkspaceBackup,
  getApplicationSettings,
  saveApplicationSettings,
  type ApplicationSettings,
  type ClassificationMetric,
  type RegressionMetric,
  type ThemePreference,
} from "@/lib/settings";

const THEME_STORAGE_KEY = "mlforge-theme";

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = value;
  let unit = -1;
  do {
    amount /= 1024;
    unit += 1;
  } while (amount >= 1024 && unit < units.length - 1);
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[unit]}`;
}

function initialTheme(): ThemePreference {
  if (typeof window === "undefined") return "system";
  const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
  return saved === "light" || saved === "dark" ? saved : "system";
}

function subscribeToTheme(onChange: () => void): () => void {
  window.addEventListener("storage", onChange);
  window.addEventListener("mlforge-theme-change", onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener("mlforge-theme-change", onChange);
  };
}

export function SettingsPanel() {
  const [settings, setSettings] = useState<ApplicationSettings | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);
  const theme = useSyncExternalStore(subscribeToTheme, initialTheme, () => "system");
  const [foldCount, setFoldCount] = useState(5);
  const [classificationMetric, setClassificationMetric] =
    useState<ClassificationMetric>("balanced_accuracy");
  const [regressionMetric, setRegressionMetric] =
    useState<RegressionMetric>("root_mean_squared_error");
  const [classificationEstimators, setClassificationEstimators] = useState<
    readonly ClassificationEstimator[]
  >([...CLASSIFICATION_ESTIMATOR_IDS]);
  const [regressionEstimators, setRegressionEstimators] = useState<
    readonly RegressionEstimator[]
  >([...REGRESSION_ESTIMATOR_IDS]);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [backingUp, setBackingUp] = useState(false);
  const [backupMessage, setBackupMessage] = useState<string | null>(null);
  const [backupError, setBackupError] = useState<string | null>(null);

  const retry = useCallback(() => {
    setLoadError(null);
    setSettings(null);
    setRequestVersion((current) => current + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void getApplicationSettings(controller.signal)
      .then((loaded) => {
        setSettings(loaded);
        setFoldCount(loaded.preferences.default_fold_count);
        setClassificationMetric(loaded.preferences.classification_metric);
        setRegressionMetric(loaded.preferences.regression_metric);
        setClassificationEstimators(loaded.preferences.classification_estimators);
        setRegressionEstimators(loaded.preferences.regression_estimators);
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLoadError(error instanceof Error ? error.message : "Settings could not be loaded.");
      });
    return () => controller.abort();
  }, [requestVersion]);

  function updateTheme(value: ThemePreference) {
    document.documentElement.dataset.theme = value;
    if (value === "system") {
      window.localStorage.removeItem(THEME_STORAGE_KEY);
    } else {
      window.localStorage.setItem(THEME_STORAGE_KEY, value);
    }
    window.dispatchEvent(new Event("mlforge-theme-change"));
  }

  function toggleClassificationEstimator(estimator: ClassificationEstimator, checked: boolean) {
    setSaveMessage(null);
    setFormError(null);
    setClassificationEstimators((current) =>
      checked ? [...current, estimator] : current.filter((item) => item !== estimator),
    );
  }

  function toggleRegressionEstimator(estimator: RegressionEstimator, checked: boolean) {
    setSaveMessage(null);
    setFormError(null);
    setRegressionEstimators((current) =>
      checked ? [...current, estimator] : current.filter((item) => item !== estimator),
    );
  }

  async function handleSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaveMessage(null);
    setFormError(null);
    if (classificationEstimators.length < 2 || regressionEstimators.length < 2) {
      setFormError("Choose at least two default models for both supported problem types.");
      return;
    }
    setSaving(true);
    try {
      const updated = await saveApplicationSettings({
        default_fold_count: foldCount,
        classification_metric: classificationMetric,
        regression_metric: regressionMetric,
        classification_estimators: classificationEstimators,
        regression_estimators: regressionEstimators,
      });
      setSettings(updated);
      setSaveMessage("Defaults saved. Existing experiments were not changed.");
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Settings could not be saved.");
    } finally {
      setSaving(false);
    }
  }

  async function handleBackup() {
    setBackingUp(true);
    setBackupMessage(null);
    setBackupError(null);
    try {
      await downloadWorkspaceBackup();
      setBackupMessage("Backup created and downloaded.");
      const refreshed = await getApplicationSettings();
      setSettings(refreshed);
    } catch (error) {
      setBackupError(error instanceof Error ? error.message : "Backup could not be created.");
    } finally {
      setBackingUp(false);
    }
  }

  if (loadError) {
    return (
      <PageErrorState
        kicker="Settings"
        title="Settings unavailable"
        description={loadError}
        onRetry={retry}
        secondaryHref="/"
        secondaryLabel="Back to dashboard"
      />
    );
  }

  if (!settings) {
    return (
      <PageLoadingState
        kicker="Settings"
        title="Loading local settings"
        description="MLForge is checking workspace configuration and runtime health."
      />
    );
  }

  const { workspace, system, diagnostics } = settings;

  return (
    <div className="page settings-page">
      <header className="page-header settings-header">
        <div>
          <span className="section-kicker">Local application</span>
          <h1>Settings</h1>
          <p>Manage future experiment defaults and inspect this trusted local workspace.</p>
        </div>
      </header>

      <section className="settings-section" aria-labelledby="appearance-title">
        <div className="settings-section-heading">
          <div>
            <h2 id="appearance-title">General</h2>
            <p>Appearance is stored only in this browser.</p>
          </div>
        </div>
        <label className="settings-select-row" htmlFor="theme-preference">
          <span>
            <strong>Appearance</strong>
            <small>Follow the operating system or choose a fixed theme.</small>
          </span>
          <select
            id="theme-preference"
            value={theme}
            onChange={(event) => updateTheme(event.currentTarget.value as ThemePreference)}
          >
            <option value="system">System</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>
      </section>

      <form className="settings-section" onSubmit={handleSave} aria-busy={saving}>
        <div className="settings-section-heading">
          <div>
            <h2>Experiment defaults</h2>
            <p>Applied when a new comparison form opens; saved experiments remain immutable.</p>
          </div>
          <button className="primary-button" type="submit" disabled={saving}>
            {saving ? "Saving..." : "Save defaults"}
          </button>
        </div>

        {formError ? (
          <div className="form-message form-message-error" role="alert">
            {formError}
          </div>
        ) : null}
        {saveMessage ? (
          <div className="settings-success" role="status">
            {saveMessage}
          </div>
        ) : null}

        <div className="settings-default-grid">
          <label className="settings-field" htmlFor="default-fold-count">
            <span>Cross-validation folds</span>
            <select
              id="default-fold-count"
              value={foldCount}
              disabled={saving}
              onChange={(event) => {
                setFoldCount(Number(event.currentTarget.value));
                setSaveMessage(null);
              }}
            >
              {Array.from({ length: 9 }, (_, index) => index + 2).map((count) => (
                <option key={count} value={count}>
                  {count}
                </option>
              ))}
            </select>
          </label>
          <label className="settings-field" htmlFor="classification-metric">
            <span>Classification ranking metric</span>
            <select
              id="classification-metric"
              value={classificationMetric}
              disabled={saving}
              onChange={(event) => {
                setClassificationMetric(event.currentTarget.value as ClassificationMetric);
                setSaveMessage(null);
              }}
            >
              {CLASSIFICATION_METRIC_IDS.map((metric) => (
                <option key={metric} value={metric}>
                  {METRIC_LABELS[metric]}
                </option>
              ))}
            </select>
          </label>
          <label className="settings-field" htmlFor="regression-metric">
            <span>Regression ranking metric</span>
            <select
              id="regression-metric"
              value={regressionMetric}
              disabled={saving}
              onChange={(event) => {
                setRegressionMetric(event.currentTarget.value as RegressionMetric);
                setSaveMessage(null);
              }}
            >
              {REGRESSION_METRIC_IDS.map((metric) => (
                <option key={metric} value={metric}>
                  {METRIC_LABELS[metric]}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="settings-model-grid">
          <fieldset disabled={saving}>
            <legend>Classification models</legend>
            <p>Select at least two defaults.</p>
            {CLASSIFICATION_ESTIMATOR_IDS.map((estimator) => (
              <label key={estimator} className="settings-checkbox">
                <input
                  type="checkbox"
                  checked={classificationEstimators.includes(estimator)}
                  onChange={(event) =>
                    toggleClassificationEstimator(estimator, event.currentTarget.checked)
                  }
                />
                <span>{ESTIMATOR_LABELS[estimator]}</span>
              </label>
            ))}
          </fieldset>
          <fieldset disabled={saving}>
            <legend>Regression models</legend>
            <p>Both supported models are required for comparison.</p>
            {REGRESSION_ESTIMATOR_IDS.map((estimator) => (
              <label key={estimator} className="settings-checkbox">
                <input
                  type="checkbox"
                  checked={regressionEstimators.includes(estimator)}
                  onChange={(event) =>
                    toggleRegressionEstimator(estimator, event.currentTarget.checked)
                  }
                />
                <span>{ESTIMATOR_LABELS[estimator]}</span>
              </label>
            ))}
          </fieldset>
        </div>
      </form>

      <section className="settings-section" aria-labelledby="workspace-title">
        <div className="settings-section-heading">
          <div>
            <h2 id="workspace-title">Storage and backup</h2>
            <p>Workspace controls are read-only while the API is running.</p>
          </div>
          <button className="secondary-button" type="button" onClick={handleBackup} disabled={backingUp}>
            {backingUp ? "Creating backup..." : "Download backup"}
          </button>
        </div>
        {backupError ? (
          <div className="form-message form-message-error" role="alert">
            {backupError}
          </div>
        ) : null}
        {backupMessage ? (
          <div className="settings-success" role="status">
            {backupMessage}
          </div>
        ) : null}
        <dl className="settings-count-grid">
          <div><dt>Datasets</dt><dd>{workspace.counts.datasets}</dd></div>
          <div><dt>Experiments</dt><dd>{workspace.counts.experiments}</dd></div>
          <div><dt>Final models</dt><dd>{workspace.counts.final_models}</dd></div>
          <div><dt>Predictions</dt><dd>{workspace.counts.predictions}</dd></div>
        </dl>
        <dl className="settings-detail-list">
          <div><dt>Workspace</dt><dd>{workspace.name}</dd></div>
          <div><dt>Disk usage</dt><dd>{formatBytes(workspace.usage_bytes)}</dd></div>
          <div><dt>Maximum CSV upload</dt><dd>{formatBytes(workspace.max_upload_bytes)}</dd></div>
          <div>
            <dt>Process configuration</dt>
            <dd>
              <code>{workspace.workspace_environment_variable}</code> and{" "}
              <code>{workspace.upload_limit_environment_variable}</code>
            </dd>
          </div>
        </dl>
        <p className="settings-restart-note">
          Changing either environment variable requires an API restart. Backups never include older
          backup archives. Protect each download because it contains datasets, fitted models, and
          prediction data.
        </p>
      </section>

      <section className="settings-section" aria-labelledby="system-title">
        <div className="settings-section-heading">
          <div>
            <h2 id="system-title">System and diagnostics</h2>
            <p>Version evidence and live local service readiness.</p>
          </div>
        </div>
        <dl className="settings-detail-list settings-system-list">
          <div><dt>MLForge</dt><dd>{system.mlforge_version}</dd></div>
          <div><dt>Python</dt><dd>{system.python_version}</dd></div>
          <div><dt>pandas</dt><dd>{system.pandas_version}</dd></div>
          <div><dt>scikit-learn</dt><dd>{system.scikit_learn_version}</dd></div>
          <div><dt>Web database schema</dt><dd>Version {system.web_schema_version}</dd></div>
        </dl>
        <div className="settings-health" aria-label="Local service health">
          <span><i aria-hidden="true" />API {diagnostics.api}</span>
          <span><i aria-hidden="true" />Database {diagnostics.database}</span>
          <span><i aria-hidden="true" />Worker {diagnostics.worker}</span>
        </div>
      </section>
    </div>
  );
}
