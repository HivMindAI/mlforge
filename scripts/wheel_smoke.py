"""Exercise the installed MLForge wheel without relying on repository package sources."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from mlforge import __version__
from mlforge.artifacts import LocalArtifactStore, inspect_artifact, load_artifact
from mlforge.datasets import load_csv
from mlforge.inference import predict_frame
from mlforge.pipelines import TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import LOGISTIC_REGRESSION, TrainingConfig, train

TRAINING_CSV = """age,monthly_spend,region,churn
24,42.50,north,no
31,68.20,south,no
45,95.10,east,yes
28,51.00,west,no
52,110.75,north,yes
39,,south,no
34,73.40,east,no
61,125.00,west,yes
"""


def main() -> None:
    """Run one complete installed-package workflow in an isolated temporary directory."""
    assert __version__ == version("mlforge")
    version_command = subprocess.run(
        [sys.executable, "-m", "mlforge", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert version_command.stdout.strip() == f"mlforge {__version__}"

    with TemporaryDirectory(prefix="mlforge-wheel-smoke-") as directory:
        workspace = Path(directory)
        source = workspace / "training.csv"
        source.write_text(TRAINING_CSV, encoding="utf-8", newline="\n")

        dataset = load_csv(source, target="churn")
        result = train(
            dataset,
            TrainingConfig(
                task=TaskType.CLASSIFICATION,
                estimator=LOGISTIC_REGRESSION,
            ),
            run_store=LocalRunStore(workspace / "runs"),
        )
        saved = LocalArtifactStore(workspace / "artifacts").save(result)
        inspected = inspect_artifact(saved.path)
        loaded = load_artifact(saved.path, trusted=True)
        predictions = predict_frame(
            loaded,
            pd.DataFrame(
                {
                    "age": [29, 57],
                    "monthly_spend": [62.0, 118.5],
                    "region": ["south", "north"],
                }
            ),
        )

        assert inspected.run_id == result.manifest.run_id
        assert predictions.run_id == result.manifest.run_id
        assert predictions.row_count == 2
        print(
            json.dumps(
                {
                    "artifact": str(saved.path),
                    "predictions": predictions.to_dict(),
                    "version": __version__,
                },
                allow_nan=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
