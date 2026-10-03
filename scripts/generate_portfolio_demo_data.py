"""Generate the deterministic synthetic CSVs used by MLForge portfolio screenshots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

SEED = 20_260_817
TRAINING_ROWS = 25_000
PREDICTION_ROWS = 2_500

FEATURES = (
    "tenure_months",
    "age",
    "monthly_spend",
    "support_tickets",
    "contract_type",
    "region",
    "autopay",
    "paperless_billing",
)
TARGET = "churn"


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _make_row(rng: random.Random, *, include_target: bool) -> dict[str, str | int]:
    tenure_months = rng.randint(0, 120)
    age = rng.randint(18, 82)
    monthly_spend = max(15.0, min(180.0, rng.gauss(72.0, 27.0)))
    support_tickets = rng.choices(
        population=[0, 1, 2, 3, 4, 5, 6, 7],
        weights=[30, 26, 18, 11, 7, 4, 2, 1],
        k=1,
    )[0]
    contract_type = rng.choices(
        population=["month-to-month", "one-year", "two-year"],
        weights=[58, 27, 15],
        k=1,
    )[0]
    region = rng.choice(["north", "south", "east", "west", "central"])
    autopay = rng.choices(["yes", "no"], weights=[62, 38], k=1)[0]
    paperless_billing = rng.choices(["yes", "no"], weights=[68, 32], k=1)[0]

    churn_score = (
        -0.55
        - (0.018 * tenure_months)
        + (0.016 * (monthly_spend - 70.0))
        + (0.30 * support_tickets)
        + (0.95 if contract_type == "month-to-month" else 0.0)
        - (0.35 if contract_type == "two-year" else 0.0)
        - (0.45 if autopay == "yes" else 0.0)
        + (0.18 if paperless_billing == "yes" else 0.0)
        + (0.006 * (age - 40))
        + rng.gauss(0.0, 0.35)
    )

    row: dict[str, str | int] = {
        "tenure_months": tenure_months,
        "age": age,
        "monthly_spend": "" if rng.random() < 0.02 else f"{monthly_spend:.2f}",
        "support_tickets": support_tickets,
        "contract_type": contract_type,
        "region": "" if rng.random() < 0.01 else region,
        "autopay": autopay,
        "paperless_billing": paperless_billing,
    }
    if include_target:
        row[TARGET] = "yes" if rng.random() < _sigmoid(churn_score) else "no"
    return row


def _write_dataset(
    path: Path,
    *,
    row_count: int,
    rng: random.Random,
    include_target: bool,
) -> dict[str, int | str]:
    fieldnames = [*FEATURES, TARGET] if include_target else list(FEATURES)
    positive_targets = 0
    missing_monthly_spend = 0
    missing_region = 0

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(row_count):
            row = _make_row(rng, include_target=include_target)
            positive_targets += int(row.get(TARGET) == "yes")
            missing_monthly_spend += int(row["monthly_spend"] == "")
            missing_region += int(row["region"] == "")
            writer.writerow(row)

    return {
        "path": str(path),
        "rows": row_count,
        "churn_yes": positive_targets,
        "missing_monthly_spend": missing_monthly_spend,
        "missing_region": missing_region,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory for generated CSVs (default: repository data directory).",
    )
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    summaries = [
        _write_dataset(
            output_directory / "large_customer_churn_train.csv",
            row_count=TRAINING_ROWS,
            rng=rng,
            include_target=True,
        ),
        _write_dataset(
            output_directory / "large_customer_churn_predict.csv",
            row_count=PREDICTION_ROWS,
            rng=rng,
            include_target=False,
        ),
    ]
    print(json.dumps({"seed": SEED, "datasets": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
