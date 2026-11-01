#!/usr/bin/env python
# Copyright (c) 2025 The VLA4AD Authors. All rights reserved.
"""Validate an OmniDrive/VLA JSONL before training.

Checks messages, images and the optional ``(6, 3)`` trajectory label, and
prints a short dataset summary. This catches the most common MS-Swift errors
(wrong roles, missing images, string trajectories, dropped columns).

Example:
    python tools/validate_dataset.py \
        --input preprocessed_data/nuscenes_navigation/nuscenes_navigation_train.jsonl \
        --check-images --require-trajectory --strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trajectory_utils import normalize_trajectory  # noqa: E402


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list of records")
    return data


def validate_record(
    record: Dict[str, Any], *, check_images: bool, require_trajectory: bool
) -> List[str]:
    errors: List[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
    else:
        roles = {message.get("role") for message in messages if isinstance(message, dict)}
        if "user" not in roles:
            errors.append("messages has no user role")
        if "assistant" not in roles:
            errors.append("messages has no assistant role")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"messages[{index}] is not an object")
                continue
            if "content" not in message:
                errors.append(f"messages[{index}] has no content")

    images = record.get("images")
    if images is not None:
        if isinstance(images, str):
            images = [images]
        if not isinstance(images, list):
            errors.append("images must be a list or string")
        elif check_images:
            for image in images:
                if not isinstance(image, str) or not os.path.exists(image):
                    errors.append(f"image not found: {image!r}")

    trajectory = record.get("trajectory")
    if trajectory is None:
        if require_trajectory:
            errors.append("missing top-level trajectory")
    else:
        try:
            normalize_trajectory(trajectory)
        except ValueError as exc:
            errors.append(f"invalid trajectory: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a VLA JSONL dataset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--require-trajectory", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any error.")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    if args.limit:
        records = records[: args.limit]

    bad: List[Tuple[int, List[str]]] = []
    with_trajectory = 0
    with_images = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            bad.append((index, ["record is not an object"]))
            continue
        if record.get("trajectory") is not None:
            with_trajectory += 1
        if record.get("images"):
            with_images += 1
        errors = validate_record(
            record,
            check_images=args.check_images,
            require_trajectory=args.require_trajectory,
        )
        if errors:
            bad.append((index, errors))

    print(f"records:              {len(records)}")
    print(f"with trajectory:      {with_trajectory}")
    print(f"with images:          {with_images}")
    print(f"invalid records:      {len(bad)}")
    for index, errors in bad[:10]:
        print(f"  [{index}] " + "; ".join(errors))
    if len(bad) > 10:
        print(f"  ... {len(bad) - 10} more")

    if bad and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
