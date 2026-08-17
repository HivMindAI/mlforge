"""Command-line interface for MLForge."""

import json
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from mlforge import __version__
from mlforge.artifacts import (
    ArtifactManifest,
    LocalArtifactStore,
    inspect_artifact,
    load_artifact,
)
from mlforge.config import ApplicationConfig, LogLevel
from mlforge.datasets import CsvLoadOptions, DatasetProfile, load_csv, profile_dataset
from mlforge.errors import ConfigurationError, MLForgeError
from mlforge.inference import PredictionResult, predict_csv
from mlforge.logging_config import configure_logging
from mlforge.pipelines import (
    FeatureOverrides,
    NumericImputationStrategy,
    PreprocessingConfig,
    SplitConfig,
    TaskType,
)
from mlforge.runs import LocalRunStore, RunComparison, RunManifest, compare_runs
from mlforge.training import ALL_ESTIMATORS, TrainingConfig, train

_BYTES_PER_MEBIBYTE = 1024 * 1024


def _parse_log_level(value: str) -> LogLevel:
    """Translate argparse input into a validated log level."""
    try:
        return LogLevel.parse(value)
    except ConfigurationError as error:
        raise ArgumentTypeError(str(error)) from error


def _positive_integer(value: str) -> int:
    """Parse one strictly positive CLI integer."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise ArgumentTypeError(f"expected a positive integer, received {value!r}") from error
    if parsed <= 0:
        raise ArgumentTypeError(f"expected a positive integer, received {value!r}")
    return parsed


def _fraction(value: str) -> float:
    """Parse one strict fractional CLI value."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise ArgumentTypeError(f"expected a number between 0 and 1, received {value!r}") from error
    if not 0 < parsed < 1:
        raise ArgumentTypeError(f"expected a number between 0 and 1, received {value!r}")
    return parsed


def _random_seed(value: str) -> int:
    """Parse an integer accepted by NumPy/scikit-learn random state."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise ArgumentTypeError(f"expected an integer random seed, received {value!r}") from error
    if not 0 <= parsed <= 2**32 - 1:
        raise ArgumentTypeError("random seed must be between 0 and 4294967295")
    return parsed


def _add_csv_options(parser: ArgumentParser) -> None:
    """Add shared explicit local CSV parser and resource options."""
    parser.add_argument("path", type=Path, help="path to a local .csv file")
    parser.add_argument("--target", required=True, help="target column name")
    _add_csv_reader_options(parser)


def _add_csv_reader_options(parser: ArgumentParser) -> None:
    """Add target-independent CSV decoding and resource options."""
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV text encoding (default: utf-8-sig)",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="single-character CSV delimiter (default: comma)",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=_positive_integer,
        default=100,
        metavar="MEBIBYTES",
        help="maximum accepted file size in MiB (default: 100)",
    )


def _add_commands(parser: ArgumentParser) -> None:
    """Register the complete implemented command tree."""
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    dataset_parser = commands.add_parser("dataset", help="inspect local tabular datasets")
    dataset_commands = dataset_parser.add_subparsers(
        dest="dataset_command",
        metavar="DATASET_COMMAND",
    )
    profile_parser = dataset_commands.add_parser(
        "profile",
        help="validate and profile a CSV dataset",
        description="Validate a local CSV dataset and report deterministic quality metadata.",
    )
    _add_csv_options(profile_parser)
    profile_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the complete profile as JSON",
    )

    train_parser = commands.add_parser(
        "train",
        help="fit, evaluate, and record one local baseline run",
        description="Fit one leakage-safe baseline and atomically record its held-out metrics.",
    )
    _add_csv_options(train_parser)
    train_parser.add_argument(
        "--task",
        required=True,
        type=TaskType,
        choices=tuple(TaskType),
        help="supervised task type",
    )
    train_parser.add_argument(
        "--estimator",
        required=True,
        choices=ALL_ESTIMATORS,
        help="supported baseline estimator",
    )
    train_parser.add_argument(
        "--validation-fraction",
        type=_fraction,
        default=0.2,
        help="held-out row fraction between 0 and 1 (default: 0.2)",
    )
    train_parser.add_argument(
        "--random-seed",
        type=_random_seed,
        default=42,
        help="reproducible split/estimator seed (default: 42)",
    )
    train_parser.add_argument(
        "--no-stratify",
        action="store_false",
        default=None,
        dest="stratify",
        help="disable the default classification target stratification",
    )
    train_parser.add_argument(
        "--numeric-imputation",
        type=NumericImputationStrategy,
        choices=tuple(NumericImputationStrategy),
        default=NumericImputationStrategy.MEDIAN,
        help="numeric missing-value statistic (default: median)",
    )
    train_parser.add_argument(
        "--no-scale-numeric",
        action="store_false",
        default=True,
        dest="scale_numeric",
        help="disable numeric standardization",
    )
    train_parser.add_argument(
        "--numeric-feature",
        action="append",
        default=[],
        metavar="COLUMN",
        help="force a feature to the numeric transformer; repeat as needed",
    )
    train_parser.add_argument(
        "--categorical-feature",
        action="append",
        default=[],
        metavar="COLUMN",
        help="force a feature to the categorical transformer; repeat as needed",
    )
    train_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("mlruns"),
        help="local immutable run-manifest directory (default: mlruns)",
    )
    train_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="optionally save the fitted pipeline as a trusted-local .mlforge artifact",
    )
    train_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the completed run manifest as JSON",
    )

    runs_parser = commands.add_parser("runs", help="inspect and compare local run records")
    run_commands = runs_parser.add_subparsers(dest="runs_command", metavar="RUNS_COMMAND")
    list_parser = run_commands.add_parser("list", help="list validated local runs")
    list_parser.add_argument("--runs-dir", type=Path, default=Path("mlruns"))
    list_parser.add_argument("--json", action="store_true", dest="as_json")
    show_parser = run_commands.add_parser("show", help="show one validated local run")
    show_parser.add_argument("run_id", help="canonical run UUID")
    show_parser.add_argument("--runs-dir", type=Path, default=Path("mlruns"))
    show_parser.add_argument("--json", action="store_true", dest="as_json")
    compare_parser = run_commands.add_parser(
        "compare",
        help="rank compatible successful runs by one metric",
    )
    compare_parser.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    compare_parser.add_argument("--metric", required=True, help="metric name recorded in every run")
    compare_parser.add_argument("--runs-dir", type=Path, default=Path("mlruns"))
    compare_parser.add_argument("--json", action="store_true", dest="as_json")

    artifacts_parser = commands.add_parser(
        "artifacts",
        help="safely inspect local model artifacts without loading executable bytes",
    )
    artifact_commands = artifacts_parser.add_subparsers(
        dest="artifacts_command",
        metavar="ARTIFACTS_COMMAND",
    )
    inspect_parser = artifact_commands.add_parser(
        "inspect",
        help="verify artifact structure and checksums without deserializing its pipeline",
    )
    inspect_parser.add_argument("artifact", type=Path, help="path to a .mlforge artifact")
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

    predict_parser = commands.add_parser(
        "predict",
        help="run schema-validated batch inference with an explicitly trusted artifact",
    )
    predict_parser.add_argument("artifact", type=Path, help="path to a .mlforge artifact")
    predict_parser.add_argument("path", type=Path, help="path to a target-free prediction CSV")
    _add_csv_reader_options(predict_parser)
    predict_parser.add_argument(
        "--trust-artifact",
        action="store_true",
        help="confirm the artifact source is trusted; loading can execute Python code",
    )
    predict_parser.add_argument("--json", action="store_true", dest="as_json")


def build_parser() -> ArgumentParser:
    """Build the top-level command-line parser."""
    parser = ArgumentParser(
        prog="mlforge",
        description="Build reproducible tabular machine-learning workflows.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--log-level",
        type=_parse_log_level,
        metavar="LEVEL",
        help="set logging to DEBUG, INFO, WARNING, ERROR, or CRITICAL",
    )
    _add_commands(parser)
    return parser


def _render_profile(profile: DatasetProfile) -> str:
    """Render a concise human-readable dataset profile."""
    metadata = profile.metadata
    lines = [
        f"Dataset: {metadata.source_path}",
        f"Rows: {metadata.row_count}",
        f"Columns: {metadata.column_count}",
        f"Target: {metadata.target} ({profile.target.task_hint.value})",
        f"SHA-256: {metadata.sha256}",
        f"Missing cells: {profile.missing_cell_count} ({profile.missing_cell_ratio:.2%})",
        f"Duplicate rows: {profile.duplicate_row_count}",
        "Column profiles:",
    ]
    lines.extend(
        f"  - {column.name}: {column.kind.value}, dtype={column.pandas_dtype}, "
        f"missing={column.missing_count}, unique={column.unique_count}"
        for column in profile.columns
    )
    if profile.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in profile.warnings)
    else:
        lines.append("Warnings: none")
    return "\n".join(lines)


def _run_dataset_profile(arguments: Namespace) -> int:
    """Run CSV ingestion and profiling for parsed CLI arguments."""
    path = cast(Path, arguments.path)
    target = cast(str, arguments.target)
    encoding = cast(str, arguments.encoding)
    delimiter = cast(str, arguments.delimiter)
    maximum_mebibytes = cast(int, arguments.max_file_size_mb)
    as_json = cast(bool, arguments.as_json)
    options = CsvLoadOptions(
        encoding=encoding,
        delimiter=delimiter,
        max_file_size_bytes=maximum_mebibytes * _BYTES_PER_MEBIBYTE,
    )
    profile = profile_dataset(load_csv(path, target=target, options=options))
    print(profile.to_json() if as_json else _render_profile(profile))
    return 0


def _csv_options(arguments: Namespace) -> CsvLoadOptions:
    encoding = cast(str, arguments.encoding)
    delimiter = cast(str, arguments.delimiter)
    maximum_mebibytes = cast(int, arguments.max_file_size_mb)
    return CsvLoadOptions(
        encoding=encoding,
        delimiter=delimiter,
        max_file_size_bytes=maximum_mebibytes * _BYTES_PER_MEBIBYTE,
    )


def _render_run(
    manifest: RunManifest,
    *,
    manifest_path: Path | None = None,
    artifact_path: Path | None = None,
) -> str:
    lines = [
        f"Run: {manifest.run_id}",
        f"Status: {manifest.status.value}",
        f"Task: {manifest.configuration.task}",
        f"Estimator: {manifest.configuration.estimator}",
        f"Dataset SHA-256: {manifest.dataset.sha256}",
    ]
    if manifest_path is not None:
        lines.append(f"Manifest: {manifest_path}")
    if artifact_path is not None:
        lines.append(f"Artifact: {artifact_path}")
    if manifest.metrics:
        lines.append("Metrics:")
        lines.extend(f"  - {metric.name}: {metric.value:.6g}" for metric in manifest.metrics)
    if manifest.failure is not None:
        lines.append(f"Failure: {manifest.failure.error_type}: {manifest.failure.message}")
    if manifest.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in manifest.warnings)
    return "\n".join(lines)


def _run_train(arguments: Namespace) -> int:
    path = cast(Path, arguments.path)
    target = cast(str, arguments.target)
    task = cast(TaskType, arguments.task)
    estimator = cast(str, arguments.estimator)
    validation_fraction = cast(float, arguments.validation_fraction)
    random_seed = cast(int, arguments.random_seed)
    stratify = cast(bool | None, arguments.stratify)
    numeric_imputation = cast(NumericImputationStrategy, arguments.numeric_imputation)
    scale_numeric = cast(bool, arguments.scale_numeric)
    numeric_features = tuple(cast(list[str], arguments.numeric_feature))
    categorical_features = tuple(cast(list[str], arguments.categorical_feature))
    runs_directory = cast(Path, arguments.runs_dir)
    artifacts_directory = cast(Path | None, arguments.artifacts_dir)
    as_json = cast(bool, arguments.as_json)

    dataset = load_csv(path, target=target, options=_csv_options(arguments))
    result = train(
        dataset,
        TrainingConfig(
            task=task,
            estimator=estimator,
            split=SplitConfig(
                validation_fraction=validation_fraction,
                random_seed=random_seed,
                stratify=stratify,
            ),
            preprocessing=PreprocessingConfig(
                numeric_imputation=numeric_imputation,
                scale_numeric=scale_numeric,
            ),
            feature_overrides=FeatureOverrides(
                numeric=numeric_features,
                categorical=categorical_features,
            ),
        ),
        run_store=LocalRunStore(runs_directory),
    )
    saved_artifact = (
        LocalArtifactStore(artifacts_directory).save(result)
        if artifacts_directory is not None
        else None
    )
    if as_json and saved_artifact is not None:
        print(
            json.dumps(
                {
                    "run": result.manifest.to_dict(),
                    "artifact": {
                        "path": str(saved_artifact.path),
                        "manifest": saved_artifact.manifest.to_dict(),
                    },
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif as_json:
        print(result.manifest.to_json())
    else:
        print(
            _render_run(
                result.manifest,
                manifest_path=result.manifest_path,
                artifact_path=saved_artifact.path if saved_artifact is not None else None,
            )
        )
    return 0


def _render_run_list(manifests: tuple[RunManifest, ...]) -> str:
    if not manifests:
        return "No runs found."
    return "\n".join(
        f"{manifest.run_id}  {manifest.status.value:<9}  "
        f"{manifest.configuration.task:<14}  {manifest.configuration.estimator}"
        for manifest in manifests
    )


def _run_runs(arguments: Namespace) -> int:
    command = cast(str | None, arguments.runs_command)
    runs_directory = cast(Path, arguments.runs_dir)
    store = LocalRunStore(runs_directory)
    as_json = cast(bool, arguments.as_json)
    if command == "list":
        manifests = store.list_manifests()
        print(
            json.dumps(
                [manifest.to_dict() for manifest in manifests],
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            if as_json
            else _render_run_list(manifests)
        )
        return 0
    if command == "show":
        manifest = store.read(cast(str, arguments.run_id))
        print(manifest.to_json() if as_json else _render_run(manifest))
        return 0
    if command == "compare":
        run_ids = cast(list[str], arguments.run_ids)
        comparison = compare_runs(
            tuple(store.read(run_id) for run_id in run_ids),
            metric=cast(str, arguments.metric),
        )
        print(
            json.dumps(comparison.to_dict(), allow_nan=False, indent=2, sort_keys=True)
            if as_json
            else _render_comparison(comparison)
        )
        return 0
    return 0


def _render_comparison(comparison: RunComparison) -> str:
    direction = "higher is better" if comparison.higher_is_better else "lower is better"
    lines = [f"Metric: {comparison.metric} ({direction})"]
    lines.extend(
        f"{entry.rank}. {entry.run_id}  {entry.value:.6g}  {entry.estimator}"
        for entry in comparison.entries
    )
    return "\n".join(lines)


def _render_artifact(manifest: ArtifactManifest, *, path: Path) -> str:
    lines = [
        f"Artifact: {path}",
        f"Run: {manifest.run_id}",
        f"Task: {manifest.task}",
        f"Target: {manifest.target}",
        f"Serialization: {manifest.serialization_format}",
        f"Pipeline SHA-256: {manifest.pipeline_sha256}",
        "Input features:",
    ]
    lines.extend(
        f"  - {feature.name}: {feature.role.value}, dtype={feature.pandas_dtype}"
        for feature in manifest.features
    )
    return "\n".join(lines)


def _run_artifacts(arguments: Namespace) -> int:
    command = cast(str | None, arguments.artifacts_command)
    if command == "inspect":
        path = cast(Path, arguments.artifact)
        manifest = inspect_artifact(path)
        print(
            manifest.to_json()
            if cast(bool, arguments.as_json)
            else _render_artifact(manifest, path=path)
        )
    return 0


def _render_predictions(result: PredictionResult) -> str:
    lines = [
        f"Run: {result.run_id}",
        f"Task: {result.task}",
        f"Target: {result.target}",
        f"Rows: {result.row_count}",
        "Predictions:",
    ]
    lines.extend(
        f"  - row {record.row_number}: {record.prediction}" for record in result.predictions
    )
    return "\n".join(lines)


def _run_predict(arguments: Namespace) -> int:
    artifact = load_artifact(
        cast(Path, arguments.artifact),
        trusted=cast(bool, arguments.trust_artifact),
    )
    result = predict_csv(
        artifact,
        cast(Path, arguments.path),
        options=_csv_options(arguments),
    )
    print(result.to_json() if cast(bool, arguments.as_json) else _render_predictions(result))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MLForge command-line interface."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    cli_log_level = cast(LogLevel | None, arguments.log_level)

    try:
        config = ApplicationConfig.from_environment().with_overrides(log_level=cli_log_level)
    except ConfigurationError as error:
        parser.error(str(error))

    configure_logging(config.log_level)
    command = cast(str | None, arguments.command)
    try:
        if command == "dataset":
            dataset_command = cast(str | None, arguments.dataset_command)
            if dataset_command == "profile":
                return _run_dataset_profile(arguments)
        if command == "train":
            return _run_train(arguments)
        if command == "runs":
            runs_command = cast(str | None, arguments.runs_command)
            if runs_command is not None:
                return _run_runs(arguments)
        if command == "artifacts":
            artifacts_command = cast(str | None, arguments.artifacts_command)
            if artifacts_command is not None:
                return _run_artifacts(arguments)
        if command == "predict":
            return _run_predict(arguments)
    except MLForgeError as error:
        print(f"mlforge: error: {error}", file=sys.stderr)
        return 1

    parser.print_help()
    return 0
