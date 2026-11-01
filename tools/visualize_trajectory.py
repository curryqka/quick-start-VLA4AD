#!/usr/bin/env python
# Copyright (c) 2025 The VLA4AD Authors. All rights reserved.
"""Plot predicted vs ground-truth trajectories in the ego frame.

Input record:
    {"id": "...", "prediction": [[x,y,h], ...], "trajectory": [[x,y,h], ...]}

Example:
    python tools/visualize_trajectory.py \
        --input results/vla_predictions.jsonl \
        --id sample_001 --output results/trajectory.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a VLA trajectory")
    parser.add_argument("--input", required=True)
    parser.add_argument("--id", default=None, help="Record id; default: first record.")
    parser.add_argument("--pred-key", default="prediction")
    parser.add_argument("--gt-key", default="trajectory")
    parser.add_argument("--output", default="results/trajectory.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_records(Path(args.input))
    record = next(
        (item for item in records if args.id is None or str(item.get("id")) == args.id),
        None,
    )
    if record is None:
        raise SystemExit(f"Record {args.id!r} not found")

    def xy(key):
        value: Any = record.get(key)
        if value is None:
            return None
        points = value[0] if len(value) == 1 and isinstance(value[0], list) else value
        return [float(point[0]) for point in points], [float(point[1]) for point in points]

    prediction = xy(args.pred_key)
    ground_truth = xy(args.gt_key)

    plt.figure(figsize=(5, 5))
    if ground_truth:
        plt.plot(*ground_truth, "o-", label="ground truth")
    if prediction:
        plt.plot(*prediction, "x--", label="prediction")
    plt.scatter([0], [0], c="k", marker="s", label="ego")
    plt.xlabel("x (m, forward)")
    plt.ylabel("y (m, left)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(str(record.get("id", "")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
