#!/usr/bin/env python
# Copyright (c) 2025 The VLA4AD Authors. All rights reserved.
"""Compute ADE / FDE / heading error for predicted trajectories.

Prediction JSONL records need a predicted trajectory and a ground-truth
trajectory, e.g.::

    {"id": "...", "prediction": [[x, y, h], ...], "trajectory": [[x, y, h], ...]}

Example:
    python tools/evaluate_trajectory.py \
        --input results/vla_predictions.jsonl \
        --output results/vla_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trajectory_utils import trajectory_metrics  # noqa: E402


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory ADE/FDE evaluation")
    parser.add_argument("--input", required=True, help="Prediction JSONL.")
    parser.add_argument("--pred-key", default="prediction")
    parser.add_argument("--gt-key", default="trajectory")
    parser.add_argument("--output", default="results/vla_metrics.json")
    parser.add_argument("--csv", default=None, help="Optional per-sample CSV.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = load_records(Path(args.input))
    if args.limit:
        records = records[: args.limit]

    rows = []
    skipped = 0
    for record in records:
        prediction = record.get(args.pred_key)
        ground_truth = record.get(args.gt_key)
        if prediction is None or ground_truth is None:
            skipped += 1
            continue
        try:
            metrics = trajectory_metrics(prediction, ground_truth)
        except ValueError:
            skipped += 1
            continue
        rows.append({"id": record.get("id", ""), **metrics})

    if not rows:
        raise SystemExit("No valid prediction/ground-truth pairs found.")

    steps = len(rows[0]["per_step_l2"])
    summary = {
        "num_samples": len(rows),
        "skipped": skipped,
        "ADE": mean(row["ade"] for row in rows),
        "FDE": mean(row["fde"] for row in rows),
        "heading_error": mean(row["heading_error"] for row in rows),
        "per_step_l2": [
            mean(row["per_step_l2"][index] for row in rows) for index in range(steps)
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "ade", "fde", "heading_error"])
            for row in rows:
                writer.writerow(
                    [row["id"], row["ade"], row["fde"], row["heading_error"]]
                )


if __name__ == "__main__":
    main()
