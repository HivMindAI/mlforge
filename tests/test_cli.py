"""Tests for the MLForge command-line interface."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from mlforge import __version__
from mlforge.cli import main
from mlforge.config import LOG_LEVEL_ENVIRONMENT_VARIABLE, LogLevel


def _write_training_csv(path: Path) -> Path:
    rows = ["row_id,value,region,target"]
    for index in range(40):
        target = "yes" if index % 2 else "no"
        rows.append(f"{index},{index / 2},{'north' if index % 3 else 'south'},{target}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_no_arguments_display_help(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The top-level command should explain its interface truthfully."""
    configured_levels: list[LogLevel] = []
    monkeypatch.setattr("mlforge.cli.configure_logging", configured_levels.append)

    assert main([]) == 0

    output = capsys.readouterr().out
    assert "usage: mlforge" in output
    assert "--version" in output
    assert "--log-level LEVEL" in output
    assert configured_levels == [LogLevel.WARNING]


def test_cli_log_level_overrides_environment(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit CLI configuration should win over environment configuration."""
    configured_levels: list[LogLevel] = []
    monkeypatch.setenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, "INFO")
    monkeypatch.setattr("mlforge.cli.configure_logging", configured_levels.append)

    assert main(["--log-level", "debug"]) == 0

    assert "usage: mlforge" in capsys.readouterr().out
    assert configured_levels == [LogLevel.DEBUG]


def test_dataset_profile_json_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should expose the complete dataset profile as machine-readable JSON."""
    path = tmp_path / "data.csv"
    path.write_text("feature,label\n1,yes\n2,no\n", encoding="utf-8")
    monkeypatch.delenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr("mlforge.cli.configure_logging", lambda level: None)

    assert main(["dataset", "profile", str(path), "--target", "label", "--json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["metadata"]["row_count"] == 2
    assert result["target"]["task_hint"] == "classification"


def test_dataset_profile_human_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default CLI result should be concise and readable."""
    path = tmp_path / "data.csv"
    path.write_text("feature,label\n1,yes\n2,no\n", encoding="utf-8")
    monkeypatch.delenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr("mlforge.cli.configure_logging", lambda level: None)

    assert main(["dataset", "profile", str(path), "--target", "label"]) == 0

    output = capsys.readouterr().out
    assert f"Dataset: {path.resolve()}" in output
    assert "Rows: 2" in output
    assert "Target: label (classification)" in output
    assert "Column profiles:" in output


def test_dataset_profile_domain_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset failures should be concise and return a stable runtime exit code."""
    monkeypatch.delenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr("mlforge.cli.configure_logging", lambda level: None)

    assert (
        main(
            [
                "dataset",
                "profile",
                str(tmp_path / "missing.csv"),
                "--target",
                "label",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "mlforge: error: Dataset path does not exist" in captured.err


def test_invalid_environment_configuration_is_a_cli_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Domain configuration failures should become stable CLI usage errors."""
    monkeypatch.setenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, "verbose")

    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    assert "Invalid log level 'verbose'" in capsys.readouterr().err


def test_version_option(capsys: pytest.CaptureFixture[str]) -> None:
    """The version option should report the distribution version."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"mlforge {__version__}\n"


def test_module_entrypoint() -> None:
    """The python -m entrypoint should use the same CLI implementation."""
    completed = subprocess.run(
        [sys.executable, "-m", "mlforge", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == f"mlforge {__version__}\n"
    assert completed.stderr == ""


def test_train_json_command_creates_a_successful_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One CLI command should train, evaluate, and persist a machine-readable run."""
    path = _write_training_csv(tmp_path / "training.csv")
    runs_directory = tmp_path / "runs"
    monkeypatch.delenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr("mlforge.cli.configure_logging", lambda level: None)

    exit_code = main(
        [
            "train",
            str(path),
            "--target",
            "target",
            "--task",
            "classification",
            "--estimator",
            "logistic-regression",
            "--runs-dir",
            str(runs_directory),
            "--json",
        ]
    )

    assert exit_code == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["status"] == "succeeded"
    assert manifest["configuration"]["estimator"] == "logistic-regression"
    assert {metric["name"] for metric in manifest["metrics"]} == {
        "accuracy",
        "balanced_accuracy",
        "f1_weighted",
    }
    assert (runs_directory / f"{manifest['run_id']}.json").is_file()


def test_runs_list_show_and_compare_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted runs should be inspectable and fairly comparable from the CLI."""
    path = _write_training_csv(tmp_path / "training.csv")
    runs_directory = tmp_path / "runs"
    monkeypatch.delenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr("mlforge.cli.configure_logging", lambda level: None)
    run_ids: list[str] = []
    for estimator in ("logistic-regression", "random-forest-classifier"):
        assert (
            main(
                [
                    "train",
                    str(path),
                    "--target",
                    "target",
                    "--task",
                    "classification",
                    "--estimator",
                    estimator,
                    "--runs-dir",
                    str(runs_directory),
                    "--json",
                ]
            )
            == 0
        )
        run_ids.append(json.loads(capsys.readouterr().out)["run_id"])

    assert main(["runs", "list", "--runs-dir", str(runs_directory), "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {item["run_id"] for item in listed} == set(run_ids)

    assert (
        main(
            [
                "runs",
                "show",
                run_ids[0],
                "--runs-dir",
                str(runs_directory),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["run_id"] == run_ids[0]

    assert (
        main(
            [
                "runs",
                "compare",
                *run_ids,
                "--metric",
                "accuracy",
                "--runs-dir",
                str(runs_directory),
                "--json",
            ]
        )
        == 0
    )
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["metric"] == "accuracy"
    assert [entry["rank"] for entry in comparison["entries"]] == [1, 2]


def test_failed_train_command_returns_one_and_records_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected runtime failures should be visible in stderr and the run store."""
    path = tmp_path / "invalid.csv"
    path.write_text(
        "value,target\n1,no\n2,yes\ninf,no\n4,yes\n5,no\n6,yes\n7,no\n8,yes\n",
        encoding="utf-8",
    )
    runs_directory = tmp_path / "runs"
    monkeypatch.delenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr("mlforge.cli.configure_logging", lambda level: None)

    exit_code = main(
        [
            "train",
            str(path),
            "--target",
            "target",
            "--task",
            "classification",
            "--estimator",
            "logistic-regression",
            "--runs-dir",
            str(runs_directory),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Training run" in captured.err
    manifests = list(runs_directory.glob("*.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text(encoding="utf-8"))["status"] == "failed"


def test_invalid_training_fraction_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Malformed numeric CLI configuration should retain argparse's exit code two."""
    with pytest.raises(SystemExit) as captured:
        main(
            [
                "train",
                "data.csv",
                "--target",
                "label",
                "--task",
                "classification",
                "--estimator",
                "logistic-regression",
                "--validation-fraction",
                "1",
            ]
        )

    assert captured.value.code == 2
    assert "expected a number between 0 and 1" in capsys.readouterr().err


def test_train_artifact_inspection_and_trusted_prediction_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should complete train-to-artifact-to-batch-prediction without hidden trust."""
    training_path = _write_training_csv(tmp_path / "training.csv")
    prediction_path = tmp_path / "prediction.csv"
    prediction_path.write_text(
        "row_id,value,region\n101,2.5,north\n102,7.0,west\n",
        encoding="utf-8",
    )
    runs_directory = tmp_path / "runs"
    artifacts_directory = tmp_path / "artifacts"
    monkeypatch.delenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr("mlforge.cli.configure_logging", lambda level: None)

    assert (
        main(
            [
                "train",
                str(training_path),
                "--target",
                "target",
                "--task",
                "classification",
                "--estimator",
                "logistic-regression",
                "--runs-dir",
                str(runs_directory),
                "--artifacts-dir",
                str(artifacts_directory),
                "--json",
            ]
        )
        == 0
    )
    training_output = json.loads(capsys.readouterr().out)
    artifact_path = training_output["artifact"]["path"]
    assert training_output["run"]["run_id"] == training_output["artifact"]["manifest"]["run_id"]
    assert Path(artifact_path).is_file()

    assert main(["artifacts", "inspect", artifact_path, "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert (
        inspected["pipeline_sha256"] == training_output["artifact"]["manifest"]["pipeline_sha256"]
    )

    assert main(["predict", artifact_path, str(prediction_path), "--json"]) == 1
    assert "trusted=True" in capsys.readouterr().err

    assert (
        main(
            [
                "predict",
                artifact_path,
                str(prediction_path),
                "--trust-artifact",
                "--json",
            ]
        )
        == 0
    )
    predictions = json.loads(capsys.readouterr().out)
    assert predictions["row_count"] == 2
    assert predictions["run_id"] == training_output["run"]["run_id"]

    output_path = tmp_path / "output" / "predictions.csv"
    assert (
        main(
            [
                "predict",
                artifact_path,
                str(prediction_path),
                "--trust-artifact",
                "--output",
                str(output_path),
                "--json",
            ]
        )
        == 0
    )
    saved = json.loads(capsys.readouterr().out)
    assert saved["output_path"] == str(output_path.resolve())
    assert saved["row_count"] == 2
    assert "predictions" not in saved
    assert pd.read_csv(output_path).to_dict(orient="records") == [
        {"row_number": 1, "prediction": predictions["predictions"][0]["prediction"]},
        {"row_number": 2, "prediction": predictions["predictions"][1]["prediction"]},
    ]

    assert (
        main(
            [
                "predict",
                artifact_path,
                str(prediction_path),
                "--trust-artifact",
                "--output",
                str(output_path),
            ]
        )
        == 1
    )
    assert "will not be overwritten" in capsys.readouterr().err
