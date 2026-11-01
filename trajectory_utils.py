# Copyright (c) 2025 The VLA4AD Authors. All rights reserved.
"""Dependency-free trajectory helpers shared by data preparation and tests."""

from __future__ import annotations

import math
from typing import Any, Dict, List

TRAJECTORY_STEPS = 6
TRAJECTORY_FEATURES = 3


def normalize_trajectory(value: Any) -> List[List[float]]:
    """Convert ``(6, 3)`` or ``(1, 6, 3)`` data to nested Python lists."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
        and len(value[0]) == TRAJECTORY_STEPS
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) != TRAJECTORY_STEPS:
        raise ValueError(
            f"trajectory must contain {TRAJECTORY_STEPS} points; got {value!r}"
        )
    output = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != TRAJECTORY_FEATURES:
            raise ValueError(
                f"each trajectory point must be (x, y, heading); got {point!r}"
            )
        output.append([float(component) for component in point])
    return output


def format_trajectory_text(
    trajectory: List[List[float]],
    *,
    format_type: str = "compact",
    round_decimals: int = 6,
) -> str:
    """Render a trajectory for text supervision or inspection."""
    points = normalize_trajectory(trajectory)
    if format_type == "list":
        return str(
            [[round(component, round_decimals) for component in point] for point in points]
        )
    if format_type == "detailed":
        return "; ".join(
            f"t={index * 0.5:.1f}s: x={x:.{round_decimals}f}, "
            f"y={y:.{round_decimals}f}, heading={heading:.{round_decimals}f}"
            for index, (x, y, heading) in enumerate(points)
        )
    return "; ".join(
        f"[{x:.{round_decimals}f},{y:.{round_decimals}f},{heading:.{round_decimals}f}]"
        for x, y, heading in points
    )


def trajectory_statistics(trajectory: List[List[float]]) -> Dict[str, Any]:
    points = normalize_trajectory(trajectory)
    distances = [
        math.dist(points[index][:2], points[index - 1][:2])
        for index in range(1, len(points))
    ]
    return {
        "displacement": math.dist(points[-1][:2], points[0][:2]),
        "total_distance": sum(distances),
        "avg_speed": (sum(distances) / len(distances) / 0.5) if distances else 0.0,
        "heading_change": points[-1][2] - points[0][2],
        "num_points": len(points),
    }
