export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

export type DatasetRecord = Readonly<{
  dataset_id: string;
  filename: string;
  file_size_bytes: number;
  row_count: number;
  column_count: number;
  columns: readonly string[];
  target: string | null;
  created_at: string;
}>;

export type ColumnKind =
  | "boolean"
  | "integer"
  | "float"
  | "datetime"
  | "categorical"
  | "string"
  | "other";

export type TaskHint = "classification" | "regression" | "undetermined";

export type NumericSummary = Readonly<{
  minimum: number;
  maximum: number;
  mean: number;
  median: number;
  standard_deviation: number | null;
}>;

export type ColumnProfile = Readonly<{
  name: string;
  kind: ColumnKind;
  pandas_dtype: string;
  non_missing_count: number;
  missing_count: number;
  missing_ratio: number;
  unique_count: number;
  unique_ratio: number;
  infinite_count: number;
  is_constant: boolean;
  is_high_cardinality: boolean;
  is_likely_identifier: boolean;
  numeric_summary: NumericSummary | null;
}>;

export type ValueFrequency = Readonly<{
  value: string;
  count: number;
  ratio: number;
}>;

export type TargetProfile = Readonly<{
  name: string;
  task_hint: TaskHint;
  non_missing_count: number;
  missing_count: number;
  unique_count: number;
  class_distribution: readonly ValueFrequency[];
  distribution_truncated: boolean;
  imbalance_warning: boolean;
}>;

export type DatasetAnalysis = Readonly<{
  dataset: DatasetRecord;
  missing_cell_count: number;
  missing_cell_ratio: number;
  duplicate_row_count: number;
  columns: readonly ColumnProfile[];
  target: TargetProfile;
  warnings: readonly string[];
}>;

export const CLASSIFICATION_ESTIMATOR_IDS = [
  "dummy-classifier",
  "logistic-regression",
  "random-forest-classifier",
] as const;

export type ClassificationEstimator = (typeof CLASSIFICATION_ESTIMATOR_IDS)[number];

export const REGRESSION_ESTIMATOR_IDS = [
  "ridge-regression",
  "random-forest-regressor",
] as const;

export type RegressionEstimator = (typeof REGRESSION_ESTIMATOR_IDS)[number];
export type Estimator = ClassificationEstimator | RegressionEstimator;
export type SupervisedTask = "classification" | "regression";

export const CLASSIFICATION_ESTIMATOR_LABELS: Readonly<
  Record<ClassificationEstimator, string>
> = {
  "dummy-classifier": "Dummy Classifier",
  "logistic-regression": "Logistic Regression",
  "random-forest-classifier": "Random Forest Classifier",
};

export const ESTIMATOR_LABELS: Readonly<Record<Estimator, string>> = {
  ...CLASSIFICATION_ESTIMATOR_LABELS,
  "ridge-regression": "Ridge Regression",
  "random-forest-regressor": "Random Forest Regressor",
};

export type ExperimentRecord = Readonly<{
  experiment_id: string;
  dataset_id: string;
  task: SupervisedTask;
  validation_strategy: "cross-validation";
  fold_count: number;
  estimators: readonly Estimator[];
  primary_metric: string;
  created_at: string;
}>;

export type JobStatus = "waiting" | "running" | "complete" | "failed";

export type ExperimentStatus = "configured" | JobStatus;

export type ExperimentHistoryItem = Readonly<{
  experiment_id: string;
  dataset_id: string;
  dataset_name: string;
  task: SupervisedTask;
  status: ExperimentStatus;
  model_count: number;
  created_at: string;
  updated_at: string;
}>;

export type ExperimentList = Readonly<{
  experiments: readonly ExperimentHistoryItem[];
  count: number;
}>;

export type JobRecord = Readonly<{
  job_id: string;
  experiment_id: string;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  benchmark_id: string | null;
  error_message: string | null;
}>;

export type FinalizationRecord = Readonly<{
  finalization_id: string;
  experiment_id: string;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  final_model_id: string | null;
  error_message: string | null;
}>;

export type FinalModelFeature = Readonly<{
  name: string;
  pandas_dtype: string;
  role: "numeric" | "categorical";
}>;

export type FinalModelEnvironment = Readonly<{
  python: string;
  mlforge: string;
  pandas: string;
  numpy: string;
  scipy: string;
  scikit_learn: string;
}>;

export type FinalModelArtifact = Readonly<{
  filename: string;
  created_at: string;
  serialization_format: string;
  pipeline_size_bytes: number;
  pipeline_sha256: string;
  target: string;
  features: readonly FinalModelFeature[];
  environment: FinalModelEnvironment;
}>;

export type FinalModelMetric = Readonly<{
  name: string;
  mean: number;
  standard_deviation: number;
}>;

export type FinalModelSummary = Readonly<{
  final_model_id: string;
  dataset_id: string;
  dataset_name: string;
  experiment_id: string;
  estimator: Estimator;
  task: SupervisedTask;
  created_at: string;
  primary_metric: string;
  primary_metric_mean: number;
  primary_metric_standard_deviation: number;
}>;

export type FinalModelList = Readonly<{
  models: readonly FinalModelSummary[];
  count: number;
}>;

export type FinalModelRecord = Readonly<{
  final_model_id: string;
  dataset_id: string;
  dataset_name: string;
  experiment_id: string;
  benchmark_id: string;
  status: "succeeded";
  estimator: Estimator;
  task: SupervisedTask;
  created_at: string;
  fit_scope: "all_rows";
  training_rows: number;
  feature_count: number;
  primary_metric: string;
  primary_metric_mean: number;
  primary_metric_standard_deviation: number;
  metrics: readonly FinalModelMetric[];
  warnings: readonly string[];
  artifact: FinalModelArtifact;
}>;

export type PredictionRecord = Readonly<{
  prediction_id: string;
  final_model_id: string;
  input_filename: string;
  status: "complete";
  created_at: string;
  completed_at: string;
}>;

export type PredictionPreviewRow = Readonly<{
  row_number: number;
  prediction: string;
}>;

export type PredictionResultRecord = PredictionRecord &
  Readonly<{
    row_count: number;
    invalid_row_count: 0;
    preview_rows: readonly PredictionPreviewRow[];
    preview_limit: number;
    preview_truncated: boolean;
  }>;

export type BenchmarkStatus = "succeeded" | "partial" | "failed";
export type BenchmarkEntryStatus = "succeeded" | "failed";

export type BenchmarkFoldPlan = Readonly<{
  fold_number: number;
  train_rows: number;
  validation_rows: number;
}>;

export type BenchmarkMetricValue = Readonly<{
  name: string;
  value: number;
}>;

export type BenchmarkEstimatorFold = Readonly<{
  fold_number: number;
  metrics: readonly BenchmarkMetricValue[];
  duration_seconds: number;
  warnings: readonly string[];
}>;

export type BenchmarkMetricSummary = Readonly<{
  name: string;
  fold_values: readonly number[];
  mean: number;
  standard_deviation: number;
  higher_is_better: boolean;
}>;

export type BenchmarkFailure = Readonly<{
  error_type: string;
  message: string;
}>;

export type BenchmarkEntry = Readonly<{
  estimator: Estimator;
  status: BenchmarkEntryStatus;
  rank: number | null;
  primary_metric_mean: number | null;
  primary_metric_standard_deviation: number | null;
  metrics: readonly BenchmarkMetricSummary[];
  folds: readonly BenchmarkEstimatorFold[];
  duration_seconds: number;
  failure_fold: number | null;
  failure: BenchmarkFailure | null;
}>;

export type ExperimentResult = Readonly<{
  experiment_id: string;
  benchmark_id: string;
  status: BenchmarkStatus;
  started_at: string;
  completed_at: string;
  task: SupervisedTask;
  target: string;
  row_count: number;
  column_count: number;
  primary_metric: string;
  fold_count: number;
  warnings: readonly string[];
  folds: readonly BenchmarkFoldPlan[];
  entries: readonly BenchmarkEntry[];
}>;

type JsonObject = Record<string, unknown>;

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseDataset(value: unknown): DatasetRecord {
  if (
    !isJsonObject(value) ||
    typeof value.dataset_id !== "string" ||
    typeof value.filename !== "string" ||
    typeof value.file_size_bytes !== "number" ||
    typeof value.row_count !== "number" ||
    typeof value.column_count !== "number" ||
    !Array.isArray(value.columns) ||
    !value.columns.every((column) => typeof column === "string") ||
    (value.target !== null && typeof value.target !== "string") ||
    typeof value.created_at !== "string"
  ) {
    throw new Error("The MLForge API returned an invalid dataset response.");
  }

  return {
    dataset_id: value.dataset_id,
    filename: value.filename,
    file_size_bytes: value.file_size_bytes,
    row_count: value.row_count,
    column_count: value.column_count,
    columns: value.columns,
    target: value.target,
    created_at: value.created_at,
  };
}

const columnKinds: readonly ColumnKind[] = [
  "boolean",
  "integer",
  "float",
  "datetime",
  "categorical",
  "string",
  "other",
];

const taskHints: readonly TaskHint[] = ["classification", "regression", "undetermined"];

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function parseNumericSummary(value: unknown): NumericSummary | null {
  if (value === null) return null;
  if (
    !isJsonObject(value) ||
    !isNumber(value.minimum) ||
    !isNumber(value.maximum) ||
    !isNumber(value.mean) ||
    !isNumber(value.median) ||
    (value.standard_deviation !== null && !isNumber(value.standard_deviation))
  ) {
    throw new Error("The MLForge API returned an invalid numeric summary.");
  }
  return {
    minimum: value.minimum,
    maximum: value.maximum,
    mean: value.mean,
    median: value.median,
    standard_deviation: value.standard_deviation,
  };
}

function parseColumn(value: unknown): ColumnProfile {
  if (
    !isJsonObject(value) ||
    typeof value.name !== "string" ||
    !columnKinds.includes(value.kind as ColumnKind) ||
    typeof value.pandas_dtype !== "string" ||
    !isNumber(value.non_missing_count) ||
    !isNumber(value.missing_count) ||
    !isNumber(value.missing_ratio) ||
    !isNumber(value.unique_count) ||
    !isNumber(value.unique_ratio) ||
    !isNumber(value.infinite_count) ||
    typeof value.is_constant !== "boolean" ||
    typeof value.is_high_cardinality !== "boolean" ||
    typeof value.is_likely_identifier !== "boolean"
  ) {
    throw new Error("The MLForge API returned an invalid column profile.");
  }
  return {
    name: value.name,
    kind: value.kind as ColumnKind,
    pandas_dtype: value.pandas_dtype,
    non_missing_count: value.non_missing_count,
    missing_count: value.missing_count,
    missing_ratio: value.missing_ratio,
    unique_count: value.unique_count,
    unique_ratio: value.unique_ratio,
    infinite_count: value.infinite_count,
    is_constant: value.is_constant,
    is_high_cardinality: value.is_high_cardinality,
    is_likely_identifier: value.is_likely_identifier,
    numeric_summary: parseNumericSummary(value.numeric_summary),
  };
}

function parseTarget(value: unknown): TargetProfile {
  if (
    !isJsonObject(value) ||
    typeof value.name !== "string" ||
    !taskHints.includes(value.task_hint as TaskHint) ||
    !isNumber(value.non_missing_count) ||
    !isNumber(value.missing_count) ||
    !isNumber(value.unique_count) ||
    !Array.isArray(value.class_distribution) ||
    typeof value.distribution_truncated !== "boolean" ||
    typeof value.imbalance_warning !== "boolean"
  ) {
    throw new Error("The MLForge API returned an invalid target profile.");
  }
  const classDistribution = value.class_distribution.map((item) => {
    if (
      !isJsonObject(item) ||
      typeof item.value !== "string" ||
      !isNumber(item.count) ||
      !isNumber(item.ratio)
    ) {
      throw new Error("The MLForge API returned an invalid class distribution.");
    }
    return { value: item.value, count: item.count, ratio: item.ratio };
  });
  return {
    name: value.name,
    task_hint: value.task_hint as TaskHint,
    non_missing_count: value.non_missing_count,
    missing_count: value.missing_count,
    unique_count: value.unique_count,
    class_distribution: classDistribution,
    distribution_truncated: value.distribution_truncated,
    imbalance_warning: value.imbalance_warning,
  };
}

function parseDatasetAnalysis(value: unknown): DatasetAnalysis {
  if (
    !isJsonObject(value) ||
    !isNumber(value.missing_cell_count) ||
    !isNumber(value.missing_cell_ratio) ||
    !isNumber(value.duplicate_row_count) ||
    !Array.isArray(value.columns) ||
    !Array.isArray(value.warnings) ||
    !value.warnings.every((warning) => typeof warning === "string")
  ) {
    throw new Error("The MLForge API returned an invalid dataset analysis.");
  }
  return {
    dataset: parseDataset(value.dataset),
    missing_cell_count: value.missing_cell_count,
    missing_cell_ratio: value.missing_cell_ratio,
    duplicate_row_count: value.duplicate_row_count,
    columns: value.columns.map(parseColumn),
    target: parseTarget(value.target),
    warnings: value.warnings,
  };
}

function isClassificationEstimator(value: unknown): value is ClassificationEstimator {
  return CLASSIFICATION_ESTIMATOR_IDS.includes(value as ClassificationEstimator);
}

function isRegressionEstimator(value: unknown): value is RegressionEstimator {
  return REGRESSION_ESTIMATOR_IDS.includes(value as RegressionEstimator);
}

function isEstimator(value: unknown): value is Estimator {
  return isClassificationEstimator(value) || isRegressionEstimator(value);
}

function isSupervisedTask(value: unknown): value is SupervisedTask {
  return value === "classification" || value === "regression";
}

function parseExperiment(value: unknown): ExperimentRecord {
  if (
    !isJsonObject(value) ||
    typeof value.experiment_id !== "string" ||
    typeof value.dataset_id !== "string" ||
    !isSupervisedTask(value.task) ||
    value.validation_strategy !== "cross-validation" ||
    !isNumber(value.fold_count) ||
    !Array.isArray(value.estimators) ||
    !value.estimators.every(isEstimator) ||
    typeof value.primary_metric !== "string" ||
    typeof value.created_at !== "string"
  ) {
    throw new Error("The MLForge API returned an invalid experiment configuration.");
  }
  return {
    experiment_id: value.experiment_id,
    dataset_id: value.dataset_id,
    task: value.task,
    validation_strategy: value.validation_strategy,
    fold_count: value.fold_count,
    estimators: value.estimators,
    primary_metric: value.primary_metric,
    created_at: value.created_at,
  };
}

const jobStatuses: readonly JobStatus[] = ["waiting", "running", "complete", "failed"];
const experimentStatuses: readonly ExperimentStatus[] = ["configured", ...jobStatuses];

function parseExperimentHistoryItem(value: unknown): ExperimentHistoryItem {
  if (
    !isJsonObject(value) ||
    typeof value.experiment_id !== "string" ||
    typeof value.dataset_id !== "string" ||
    typeof value.dataset_name !== "string" ||
    !isSupervisedTask(value.task) ||
    !experimentStatuses.includes(value.status as ExperimentStatus) ||
    !isNumber(value.model_count) ||
    typeof value.created_at !== "string" ||
    typeof value.updated_at !== "string"
  ) {
    throw new Error("The MLForge API returned an invalid experiment history item.");
  }
  return {
    experiment_id: value.experiment_id,
    dataset_id: value.dataset_id,
    dataset_name: value.dataset_name,
    task: value.task,
    status: value.status as ExperimentStatus,
    model_count: value.model_count,
    created_at: value.created_at,
    updated_at: value.updated_at,
  };
}

function parseExperimentList(value: unknown): ExperimentList {
  if (
    !isJsonObject(value) ||
    !Array.isArray(value.experiments) ||
    !isNumber(value.count) ||
    value.count !== value.experiments.length
  ) {
    throw new Error("The MLForge API returned an invalid experiment history.");
  }
  return {
    experiments: value.experiments.map(parseExperimentHistoryItem),
    count: value.count,
  };
}

function parseJob(value: unknown): JobRecord {
  if (
    !isJsonObject(value) ||
    typeof value.job_id !== "string" ||
    typeof value.experiment_id !== "string" ||
    !jobStatuses.includes(value.status as JobStatus) ||
    typeof value.created_at !== "string" ||
    (value.started_at !== null && typeof value.started_at !== "string") ||
    (value.completed_at !== null && typeof value.completed_at !== "string") ||
    (value.benchmark_id !== null && typeof value.benchmark_id !== "string") ||
    (value.error_message !== null && typeof value.error_message !== "string")
  ) {
    throw new Error("The MLForge API returned an invalid comparison job.");
  }
  return {
    job_id: value.job_id,
    experiment_id: value.experiment_id,
    status: value.status as JobStatus,
    created_at: value.created_at,
    started_at: value.started_at,
    completed_at: value.completed_at,
    benchmark_id: value.benchmark_id,
    error_message: value.error_message,
  };
}

function parseFinalization(value: unknown): FinalizationRecord {
  if (
    !isJsonObject(value) ||
    typeof value.finalization_id !== "string" ||
    typeof value.experiment_id !== "string" ||
    !jobStatuses.includes(value.status as JobStatus) ||
    typeof value.created_at !== "string" ||
    (value.started_at !== null && typeof value.started_at !== "string") ||
    (value.completed_at !== null && typeof value.completed_at !== "string") ||
    (value.final_model_id !== null && typeof value.final_model_id !== "string") ||
    (value.error_message !== null && typeof value.error_message !== "string")
  ) {
    throw new Error("The MLForge API returned an invalid finalization job.");
  }
  return {
    finalization_id: value.finalization_id,
    experiment_id: value.experiment_id,
    status: value.status as JobStatus,
    created_at: value.created_at,
    started_at: value.started_at,
    completed_at: value.completed_at,
    final_model_id: value.final_model_id,
    error_message: value.error_message,
  };
}

function parseFinalModel(value: unknown): FinalModelRecord {
  if (
    !isJsonObject(value) ||
    typeof value.final_model_id !== "string" ||
    typeof value.dataset_id !== "string" ||
    typeof value.dataset_name !== "string" ||
    typeof value.experiment_id !== "string" ||
    typeof value.benchmark_id !== "string" ||
    value.status !== "succeeded" ||
    !isEstimator(value.estimator) ||
    !isSupervisedTask(value.task) ||
    typeof value.created_at !== "string" ||
    value.fit_scope !== "all_rows" ||
    !isNumber(value.training_rows) ||
    !isNumber(value.feature_count) ||
    typeof value.primary_metric !== "string" ||
    !isNumber(value.primary_metric_mean) ||
    !isNumber(value.primary_metric_standard_deviation) ||
    !Array.isArray(value.metrics) ||
    !isStringArray(value.warnings) ||
    !isJsonObject(value.artifact) ||
    typeof value.artifact.filename !== "string" ||
    typeof value.artifact.created_at !== "string" ||
    typeof value.artifact.serialization_format !== "string" ||
    !isNumber(value.artifact.pipeline_size_bytes) ||
    typeof value.artifact.pipeline_sha256 !== "string" ||
    typeof value.artifact.target !== "string" ||
    !Array.isArray(value.artifact.features) ||
    !isJsonObject(value.artifact.environment) ||
    typeof value.artifact.environment.python !== "string" ||
    typeof value.artifact.environment.mlforge !== "string" ||
    typeof value.artifact.environment.pandas !== "string" ||
    typeof value.artifact.environment.numpy !== "string" ||
    typeof value.artifact.environment.scipy !== "string" ||
    typeof value.artifact.environment.scikit_learn !== "string"
  ) {
    throw new Error("The MLForge API returned invalid final-model details.");
  }
  const features = value.artifact.features.map((feature): FinalModelFeature => {
    if (
      !isJsonObject(feature) ||
      typeof feature.name !== "string" ||
      typeof feature.pandas_dtype !== "string" ||
      (feature.role !== "numeric" && feature.role !== "categorical")
    ) {
      throw new Error("The MLForge API returned an invalid final-model feature.");
    }
    return { name: feature.name, pandas_dtype: feature.pandas_dtype, role: feature.role };
  });
  const metrics = value.metrics.map((metric): FinalModelMetric => {
    if (
      !isJsonObject(metric) ||
      typeof metric.name !== "string" ||
      !isNumber(metric.mean) ||
      !isNumber(metric.standard_deviation)
    ) {
      throw new Error("The MLForge API returned an invalid final-model metric.");
    }
    return {
      name: metric.name,
      mean: metric.mean,
      standard_deviation: metric.standard_deviation,
    };
  });
  return {
    final_model_id: value.final_model_id,
    dataset_id: value.dataset_id,
    dataset_name: value.dataset_name,
    experiment_id: value.experiment_id,
    benchmark_id: value.benchmark_id,
    status: value.status,
    estimator: value.estimator,
    task: value.task,
    created_at: value.created_at,
    fit_scope: value.fit_scope,
    training_rows: value.training_rows,
    feature_count: value.feature_count,
    primary_metric: value.primary_metric,
    primary_metric_mean: value.primary_metric_mean,
    primary_metric_standard_deviation: value.primary_metric_standard_deviation,
    metrics,
    warnings: value.warnings,
    artifact: {
      filename: value.artifact.filename,
      created_at: value.artifact.created_at,
      serialization_format: value.artifact.serialization_format,
      pipeline_size_bytes: value.artifact.pipeline_size_bytes,
      pipeline_sha256: value.artifact.pipeline_sha256,
      target: value.artifact.target,
      features,
      environment: {
        python: value.artifact.environment.python,
        mlforge: value.artifact.environment.mlforge,
        pandas: value.artifact.environment.pandas,
        numpy: value.artifact.environment.numpy,
        scipy: value.artifact.environment.scipy,
        scikit_learn: value.artifact.environment.scikit_learn,
      },
    },
  };
}

function parseFinalModelSummary(value: unknown): FinalModelSummary {
  if (
    !isJsonObject(value) ||
    typeof value.final_model_id !== "string" ||
    typeof value.dataset_id !== "string" ||
    typeof value.dataset_name !== "string" ||
    typeof value.experiment_id !== "string" ||
    !isEstimator(value.estimator) ||
    !isSupervisedTask(value.task) ||
    typeof value.created_at !== "string" ||
    typeof value.primary_metric !== "string" ||
    !isNumber(value.primary_metric_mean) ||
    !isNumber(value.primary_metric_standard_deviation)
  ) {
    throw new Error("The MLForge API returned an invalid model-list entry.");
  }
  return {
    final_model_id: value.final_model_id,
    dataset_id: value.dataset_id,
    dataset_name: value.dataset_name,
    experiment_id: value.experiment_id,
    estimator: value.estimator,
    task: value.task,
    created_at: value.created_at,
    primary_metric: value.primary_metric,
    primary_metric_mean: value.primary_metric_mean,
    primary_metric_standard_deviation: value.primary_metric_standard_deviation,
  };
}

function parseFinalModelList(value: unknown): FinalModelList {
  if (!isJsonObject(value) || !Array.isArray(value.models) || !isNumber(value.count)) {
    throw new Error("The MLForge API returned an invalid model list.");
  }
  const models = value.models.map(parseFinalModelSummary);
  if (value.count !== models.length) {
    throw new Error("The MLForge API returned an inconsistent model count.");
  }
  return { models, count: value.count };
}

function parsePrediction(value: unknown): PredictionRecord {
  if (
    !isJsonObject(value) ||
    typeof value.prediction_id !== "string" ||
    typeof value.final_model_id !== "string" ||
    typeof value.input_filename !== "string" ||
    value.status !== "complete" ||
    typeof value.created_at !== "string" ||
    typeof value.completed_at !== "string"
  ) {
    throw new Error("The MLForge API returned an invalid prediction record.");
  }
  return {
    prediction_id: value.prediction_id,
    final_model_id: value.final_model_id,
    input_filename: value.input_filename,
    status: value.status,
    created_at: value.created_at,
    completed_at: value.completed_at,
  };
}

function parsePredictionResult(value: unknown): PredictionResultRecord {
  const record = parsePrediction(value);
  if (
    !isJsonObject(value) ||
    !Number.isInteger(value.row_count) ||
    (value.row_count as number) <= 0 ||
    value.invalid_row_count !== 0 ||
    !Array.isArray(value.preview_rows) ||
    !Number.isInteger(value.preview_limit) ||
    (value.preview_limit as number) <= 0 ||
    typeof value.preview_truncated !== "boolean"
  ) {
    throw new Error("The MLForge API returned invalid prediction results.");
  }
  const previewRows = value.preview_rows.map((row): PredictionPreviewRow => {
    if (
      !isJsonObject(row) ||
      !Number.isInteger(row.row_number) ||
      (row.row_number as number) <= 0 ||
      typeof row.prediction !== "string"
    ) {
      throw new Error("The MLForge API returned an invalid prediction preview row.");
    }
    return { row_number: row.row_number as number, prediction: row.prediction };
  });
  if (
    previewRows.length > (value.preview_limit as number) ||
    previewRows.length > (value.row_count as number)
  ) {
    throw new Error("The MLForge API returned an inconsistent prediction preview.");
  }
  return {
    ...record,
    row_count: value.row_count as number,
    invalid_row_count: 0,
    preview_rows: previewRows,
    preview_limit: value.preview_limit as number,
    preview_truncated: value.preview_truncated,
  };
}

const benchmarkStatuses: readonly BenchmarkStatus[] = ["succeeded", "partial", "failed"];
const benchmarkEntryStatuses: readonly BenchmarkEntryStatus[] = ["succeeded", "failed"];

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function parseBenchmarkMetricSummary(value: unknown): BenchmarkMetricSummary {
  if (
    !isJsonObject(value) ||
    typeof value.name !== "string" ||
    !Array.isArray(value.fold_values) ||
    !value.fold_values.every(isNumber) ||
    !isNumber(value.mean) ||
    !isNumber(value.standard_deviation) ||
    typeof value.higher_is_better !== "boolean"
  ) {
    throw new Error("The MLForge API returned an invalid metric summary.");
  }
  return {
    name: value.name,
    fold_values: value.fold_values,
    mean: value.mean,
    standard_deviation: value.standard_deviation,
    higher_is_better: value.higher_is_better,
  };
}

function parseBenchmarkEstimatorFold(value: unknown): BenchmarkEstimatorFold {
  if (
    !isJsonObject(value) ||
    !isNumber(value.fold_number) ||
    !Array.isArray(value.metrics) ||
    !isNumber(value.duration_seconds) ||
    !isStringArray(value.warnings)
  ) {
    throw new Error("The MLForge API returned invalid fold results.");
  }
  const metrics = value.metrics.map((metric) => {
    if (!isJsonObject(metric) || typeof metric.name !== "string" || !isNumber(metric.value)) {
      throw new Error("The MLForge API returned an invalid fold metric.");
    }
    return { name: metric.name, value: metric.value };
  });
  return {
    fold_number: value.fold_number,
    metrics,
    duration_seconds: value.duration_seconds,
    warnings: value.warnings,
  };
}

function parseBenchmarkEntry(value: unknown): BenchmarkEntry {
  if (
    !isJsonObject(value) ||
    !isEstimator(value.estimator) ||
    !benchmarkEntryStatuses.includes(value.status as BenchmarkEntryStatus) ||
    (value.rank !== null && !isNumber(value.rank)) ||
    (value.primary_metric_mean !== null && !isNumber(value.primary_metric_mean)) ||
    (value.primary_metric_standard_deviation !== null &&
      !isNumber(value.primary_metric_standard_deviation)) ||
    !Array.isArray(value.metrics) ||
    !Array.isArray(value.folds) ||
    !isNumber(value.duration_seconds) ||
    (value.failure_fold !== null && !isNumber(value.failure_fold))
  ) {
    throw new Error("The MLForge API returned an invalid model result.");
  }
  let failure: BenchmarkFailure | null = null;
  if (value.failure !== null) {
    if (
      !isJsonObject(value.failure) ||
      typeof value.failure.error_type !== "string" ||
      typeof value.failure.message !== "string"
    ) {
      throw new Error("The MLForge API returned invalid model failure details.");
    }
    failure = { error_type: value.failure.error_type, message: value.failure.message };
  }
  return {
    estimator: value.estimator,
    status: value.status as BenchmarkEntryStatus,
    rank: value.rank,
    primary_metric_mean: value.primary_metric_mean,
    primary_metric_standard_deviation: value.primary_metric_standard_deviation,
    metrics: value.metrics.map(parseBenchmarkMetricSummary),
    folds: value.folds.map(parseBenchmarkEstimatorFold),
    duration_seconds: value.duration_seconds,
    failure_fold: value.failure_fold,
    failure,
  };
}

function parseExperimentResult(value: unknown): ExperimentResult {
  if (
    !isJsonObject(value) ||
    typeof value.experiment_id !== "string" ||
    typeof value.benchmark_id !== "string" ||
    !benchmarkStatuses.includes(value.status as BenchmarkStatus) ||
    typeof value.started_at !== "string" ||
    typeof value.completed_at !== "string" ||
    !isSupervisedTask(value.task) ||
    typeof value.target !== "string" ||
    !isNumber(value.row_count) ||
    !isNumber(value.column_count) ||
    typeof value.primary_metric !== "string" ||
    !isNumber(value.fold_count) ||
    !isStringArray(value.warnings) ||
    !Array.isArray(value.folds) ||
    !Array.isArray(value.entries)
  ) {
    throw new Error("The MLForge API returned invalid experiment results.");
  }
  const folds = value.folds.map((fold): BenchmarkFoldPlan => {
    if (
      !isJsonObject(fold) ||
      !isNumber(fold.fold_number) ||
      !isNumber(fold.train_rows) ||
      !isNumber(fold.validation_rows)
    ) {
      throw new Error("The MLForge API returned an invalid fold plan.");
    }
    return {
      fold_number: fold.fold_number,
      train_rows: fold.train_rows,
      validation_rows: fold.validation_rows,
    };
  });
  return {
    experiment_id: value.experiment_id,
    benchmark_id: value.benchmark_id,
    status: value.status as BenchmarkStatus,
    started_at: value.started_at,
    completed_at: value.completed_at,
    task: value.task,
    target: value.target,
    row_count: value.row_count,
    column_count: value.column_count,
    primary_metric: value.primary_metric,
    fold_count: value.fold_count,
    warnings: value.warnings,
    folds,
    entries: value.entries.map(parseBenchmarkEntry),
  };
}

function errorMessage(value: unknown, fallback: string): string {
  if (!isJsonObject(value)) {
    return fallback;
  }
  const error = value.error;
  if (isJsonObject(error) && typeof error.message === "string") {
    return error.message;
  }
  if (typeof value.detail === "string") {
    return value.detail;
  }
  return fallback;
}

function parseJson(text: string): unknown {
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export function uploadDataset(
  file: File,
  onProgress: (progress: number) => void,
): Promise<DatasetRecord> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const body = new FormData();
    body.append("file", file);

    request.open("POST", "/api/datasets");
    request.responseType = "text";

    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    });

    request.addEventListener("load", () => {
      const response = parseJson(request.responseText);
      if (request.status >= 200 && request.status < 300) {
        try {
          resolve(parseDataset(response));
        } catch (error) {
          reject(error);
        }
        return;
      }
      reject(new Error(errorMessage(response, "The CSV file could not be uploaded.")));
    });

    request.addEventListener("error", () => {
      reject(new Error("Could not reach the local MLForge API."));
    });

    request.send(body);
  });
}

export function runPrediction(
  finalModelId: string,
  file: File,
  onProgress: (progress: number) => void,
): Promise<PredictionRecord> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const body = new FormData();
    body.append("model_id", finalModelId);
    body.append("file", file);

    request.open("POST", "/api/predictions");
    request.responseType = "text";
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    });
    request.addEventListener("load", () => {
      const response = parseJson(request.responseText);
      if (request.status >= 200 && request.status < 300) {
        try {
          resolve(parsePrediction(response));
        } catch (error) {
          reject(error);
        }
        return;
      }
      reject(new Error(errorMessage(response, "The prediction could not be completed.")));
    });
    request.addEventListener("error", () => {
      reject(new Error("Could not reach the local MLForge API."));
    });
    request.send(body);
  });
}

export async function getPrediction(
  predictionId: string,
  signal?: AbortSignal,
): Promise<PredictionResultRecord> {
  let response: Response;
  try {
    response = await fetch(`/api/predictions/${predictionId}`, {
      signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The prediction results could not be loaded."));
  }
  return parsePredictionResult(body);
}

export async function selectDatasetTarget(
  datasetId: string,
  target: string,
): Promise<DatasetRecord> {
  let response: Response;
  try {
    response = await fetch(`/api/datasets/${datasetId}/target`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    });
  } catch {
    throw new Error("Could not reach the local MLForge API.");
  }

  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The target column could not be saved."));
  }
  return parseDataset(body);
}

export async function getDataset(
  datasetId: string,
  signal?: AbortSignal,
): Promise<DatasetRecord> {
  let response: Response;
  try {
    response = await fetch(`/api/datasets/${datasetId}`, { signal, cache: "no-store" });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The dataset metadata could not be loaded."));
  }
  return parseDataset(body);
}

export async function analyzeDataset(
  datasetId: string,
  signal?: AbortSignal,
): Promise<DatasetAnalysis> {
  let response: Response;
  try {
    response = await fetch(`/api/datasets/${datasetId}/analysis`, {
      method: "POST",
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }

  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The dataset could not be analyzed."));
  }
  return parseDatasetAnalysis(body);
}

export async function createExperiment(
  datasetId: string,
  estimators: readonly Estimator[],
  foldCount: number,
): Promise<ExperimentRecord> {
  let response: Response;
  try {
    response = await fetch("/api/experiments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: datasetId,
        estimators,
        fold_count: foldCount,
      }),
    });
  } catch {
    throw new Error("Could not reach the local MLForge API.");
  }

  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The experiment configuration could not be saved."));
  }
  return parseExperiment(body);
}

export async function getExperiment(
  experimentId: string,
  signal?: AbortSignal,
): Promise<ExperimentRecord> {
  let response: Response;
  try {
    response = await fetch(`/api/experiments/${experimentId}`, { signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The experiment configuration could not be loaded."));
  }
  return parseExperiment(body);
}

export async function getExperiments(signal?: AbortSignal): Promise<ExperimentList> {
  let response: Response;
  try {
    response = await fetch("/api/experiments", { signal, cache: "no-store" });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The experiment history could not be loaded."));
  }
  return parseExperimentList(body);
}

export async function getExperimentJob(
  experimentId: string,
  signal?: AbortSignal,
): Promise<JobRecord | null> {
  let response: Response;
  try {
    response = await fetch(`/api/experiments/${experimentId}/job`, { signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The comparison status could not be loaded."));
  }
  return body === null ? null : parseJob(body);
}

export async function startExperiment(experimentId: string): Promise<JobRecord> {
  let response: Response;
  try {
    response = await fetch(`/api/experiments/${experimentId}/run`, { method: "POST" });
  } catch {
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The comparison could not be started."));
  }
  return parseJob(body);
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobRecord> {
  let response: Response;
  try {
    response = await fetch(`/api/jobs/${jobId}`, { signal, cache: "no-store" });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The comparison status could not be loaded."));
  }
  return parseJob(body);
}

export async function getExperimentResults(
  experimentId: string,
  signal?: AbortSignal,
): Promise<ExperimentResult> {
  let response: Response;
  try {
    response = await fetch(`/api/experiments/${experimentId}/results`, {
      signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The experiment results could not be loaded."));
  }
  return parseExperimentResult(body);
}

export async function getExperimentFinalization(
  experimentId: string,
  signal?: AbortSignal,
): Promise<FinalizationRecord | null> {
  let response: Response;
  try {
    response = await fetch(`/api/experiments/${experimentId}/finalization`, {
      signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The finalization status could not be loaded."));
  }
  return body === null ? null : parseFinalization(body);
}

export async function startFinalization(experimentId: string): Promise<FinalizationRecord> {
  let response: Response;
  try {
    response = await fetch(`/api/experiments/${experimentId}/finalize`, { method: "POST" });
  } catch {
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The final model could not be started."));
  }
  return parseFinalization(body);
}

export async function getFinalModel(
  finalModelId: string,
  signal?: AbortSignal,
): Promise<FinalModelRecord> {
  let response: Response;
  try {
    response = await fetch(`/api/final-models/${finalModelId}`, {
      signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The final model could not be loaded."));
  }
  return parseFinalModel(body);
}

export async function getFinalModels(signal?: AbortSignal): Promise<FinalModelList> {
  let response: Response;
  try {
    response = await fetch("/api/final-models", {
      signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the local MLForge API.");
  }
  const body = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    throw new Error(errorMessage(body, "The model list could not be loaded."));
  }
  return parseFinalModelList(body);
}
