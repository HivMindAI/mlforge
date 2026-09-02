import { ESTIMATOR_LABELS, type Estimator, type SupervisedTask } from "@/lib/datasets";

const percentageMetrics = new Set([
  "accuracy",
  "balanced_accuracy",
  "f1_macro",
  "f1_weighted",
  "precision_macro",
  "recall_macro",
]);

const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
});

export function estimatorLabel(estimator: Estimator): string {
  return ESTIMATOR_LABELS[estimator];
}

export function taskLabel(task: SupervisedTask): string {
  return task === "classification" ? "Classification" : "Regression";
}

export function metricLabel(name: string): string {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatMetricValue(name: string, value: number | null): string {
  if (value === null) return "—";
  return percentageMetrics.has(name)
    ? percentFormatter.format(value)
    : numberFormatter.format(value);
}
