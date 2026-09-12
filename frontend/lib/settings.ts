import {
  CLASSIFICATION_ESTIMATOR_IDS,
  REGRESSION_ESTIMATOR_IDS,
  type ClassificationEstimator,
  type RegressionEstimator,
} from "@/lib/datasets";

export type ThemePreference = "system" | "light" | "dark";

export const CLASSIFICATION_METRIC_IDS = [
  "accuracy",
  "balanced_accuracy",
  "f1_macro",
  "f1_weighted",
  "precision_macro",
  "recall_macro",
] as const;

export const REGRESSION_METRIC_IDS = [
  "mean_absolute_error",
  "r2",
  "root_mean_squared_error",
] as const;

export type ClassificationMetric = (typeof CLASSIFICATION_METRIC_IDS)[number];
export type RegressionMetric = (typeof REGRESSION_METRIC_IDS)[number];

export const METRIC_LABELS: Readonly<
  Record<ClassificationMetric | RegressionMetric, string>
> = {
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced accuracy",
  f1_macro: "Macro F1",
  f1_weighted: "Weighted F1",
  precision_macro: "Macro precision",
  recall_macro: "Macro recall",
  mean_absolute_error: "Mean absolute error",
  r2: "R-squared",
  root_mean_squared_error: "Root mean squared error",
};

export type ApplicationPreferences = Readonly<{
  default_fold_count: number;
  classification_metric: ClassificationMetric;
  regression_metric: RegressionMetric;
  classification_estimators: readonly ClassificationEstimator[];
  regression_estimators: readonly RegressionEstimator[];
  updated_at: string | null;
}>;

export type ApplicationSettings = Readonly<{
  preferences: ApplicationPreferences;
  system: Readonly<{
    mlforge_version: string;
    python_version: string;
    pandas_version: string;
    scikit_learn_version: string;
    web_schema_version: number;
  }>;
  workspace: Readonly<{
    name: string;
    usage_bytes: number;
    max_upload_bytes: number;
    workspace_environment_variable: "MLFORGE_WEB_WORKSPACE";
    upload_limit_environment_variable: "MLFORGE_WEB_MAX_UPLOAD_BYTES";
    restart_required: boolean;
    counts: Readonly<{
      datasets: number;
      experiments: number;
      final_models: number;
      predictions: number;
    }>;
  }>;
  diagnostics: Readonly<{
    api: "ready";
    database: "ready";
    worker: "available";
  }>;
}>;

type JsonObject = Record<string, unknown>;

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function parseError(value: unknown, fallback: string): string {
  if (!isObject(value) || !isObject(value.error) || typeof value.error.message !== "string") {
    return fallback;
  }
  return value.error.message;
}

function isClassificationMetric(value: unknown): value is ClassificationMetric {
  return CLASSIFICATION_METRIC_IDS.includes(value as ClassificationMetric);
}

function isRegressionMetric(value: unknown): value is RegressionMetric {
  return REGRESSION_METRIC_IDS.includes(value as RegressionMetric);
}

function parseSettings(value: unknown): ApplicationSettings {
  if (
    !isObject(value) ||
    !isObject(value.preferences) ||
    !isObject(value.system) ||
    !isObject(value.workspace) ||
    !isObject(value.diagnostics)
  ) {
    throw new Error("The MLForge API returned an invalid settings response.");
  }

  const preferences = value.preferences;
  const system = value.system;
  const workspace = value.workspace;
  const diagnostics = value.diagnostics;
  if (!isObject(workspace.counts)) {
    throw new Error("The MLForge API returned an invalid settings response.");
  }
  const counts = workspace.counts;
  if (
    !Number.isInteger(preferences.default_fold_count) ||
    !isClassificationMetric(preferences.classification_metric) ||
    !isRegressionMetric(preferences.regression_metric) ||
    !Array.isArray(preferences.classification_estimators) ||
    !preferences.classification_estimators.every((item) =>
      CLASSIFICATION_ESTIMATOR_IDS.includes(item as ClassificationEstimator),
    ) ||
    !Array.isArray(preferences.regression_estimators) ||
    !preferences.regression_estimators.every((item) =>
      REGRESSION_ESTIMATOR_IDS.includes(item as RegressionEstimator),
    ) ||
    (preferences.updated_at !== null && typeof preferences.updated_at !== "string") ||
    typeof system.mlforge_version !== "string" ||
    typeof system.python_version !== "string" ||
    typeof system.pandas_version !== "string" ||
    typeof system.scikit_learn_version !== "string" ||
    !Number.isInteger(system.web_schema_version) ||
    typeof workspace.name !== "string" ||
    !isFiniteNumber(workspace.usage_bytes) ||
    !isFiniteNumber(workspace.max_upload_bytes) ||
    workspace.workspace_environment_variable !== "MLFORGE_WEB_WORKSPACE" ||
    workspace.upload_limit_environment_variable !== "MLFORGE_WEB_MAX_UPLOAD_BYTES" ||
    typeof workspace.restart_required !== "boolean" ||
    !Number.isInteger(counts.datasets) ||
    !Number.isInteger(counts.experiments) ||
    !Number.isInteger(counts.final_models) ||
    !Number.isInteger(counts.predictions) ||
    diagnostics.api !== "ready" ||
    diagnostics.database !== "ready" ||
    diagnostics.worker !== "available"
  ) {
    throw new Error("The MLForge API returned an invalid settings response.");
  }

  return value as ApplicationSettings;
}

export async function getApplicationSettings(signal?: AbortSignal): Promise<ApplicationSettings> {
  let response: Response;
  try {
    response = await fetch("/api/settings", { signal, cache: "no-store" });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(parseError(body, "Application settings could not be loaded."));
  }
  return parseSettings(body);
}

export async function saveApplicationSettings(
  preferences: Omit<ApplicationPreferences, "updated_at">,
): Promise<ApplicationSettings> {
  let response: Response;
  try {
    response = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(preferences),
    });
  } catch {
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(parseError(body, "Application settings could not be saved."));
  }
  return parseSettings(body);
}

export async function downloadWorkspaceBackup(): Promise<void> {
  let response: Response;
  try {
    response = await fetch("/api/settings/backup", { method: "POST" });
  } catch {
    throw new Error("Could not reach the local MLForge API.");
  }
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as unknown;
    throw new Error(parseError(body, "The workspace backup could not be created."));
  }
  const disposition = response.headers.get("content-disposition") ?? "";
  const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);
  const filename = filenameMatch?.[1] ?? "mlforge-backup.zip";
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
