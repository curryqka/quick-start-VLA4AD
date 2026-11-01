# Copyright (c) Alibaba, Inc. and its affiliates.
"""ms-swift dataset helpers for the VLA trajectory task.

The VLA model consumes the usual ``messages`` / ``images`` fields plus a
top-level numeric ``trajectory`` field of shape ``(6, 3)`` (x, y, heading).
ms-swift places unknown columns in ``StdTemplateInputs.extra_kwargs``, where
``Qwen2_5VLATemplate`` reads them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, MessagesPreprocessor, register_dataset

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trajectory_utils import (  # noqa: E402
    format_trajectory_text,
    normalize_trajectory,
    trajectory_statistics,
)


class OmniDriveTrajectoryPreprocessor(MessagesPreprocessor):
    """Attach the numeric trajectory used by the VLA head.

    Input row (produced by the preparation scripts)::

        {
          "messages": [system, user, assistant],
          "images": [...],
          "metadata": {"trajectory": [[x, y, h], ...]}
        }

    Output adds top-level ``trajectory`` (numeric, for the model) and
    ``trajectory_text`` (for inspection). The assistant message stays the
    navigation/text supervision target.
    """

    def __init__(
        self,
        trajectory_format: str = "compact",
        round_decimals: int = 6,
        **kwargs,
    ) -> None:
        self.trajectory_format = trajectory_format
        self.round_decimals = round_decimals
        super().__init__(**kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row.get("messages") or []
        if not messages:
            raise ValueError("No messages found in row")
        if not any(message.get("role") == "user" for message in messages):
            raise ValueError("No user message found in messages")
        if not any(message.get("role") == "assistant" for message in messages):
            raise ValueError("No assistant message found in messages")

        metadata = row.get("metadata") or {}
        trajectory = row.get("trajectory", metadata.get("trajectory"))
        if trajectory is None:
            raise ValueError(
                "No trajectory found in row['trajectory'] or metadata['trajectory']"
            )

        trajectory = normalize_trajectory(trajectory)
        row["trajectory"] = trajectory
        row["trajectory_text"] = format_trajectory_text(
            trajectory,
            format_type=self.trajectory_format,
            round_decimals=self.round_decimals,
        )
        row["question_type"] = "omnidrive_trajectory"

        metadata["formatted_trajectory"] = row["trajectory_text"]
        metadata["trajectory_stats"] = trajectory_statistics(trajectory)
        row["metadata"] = metadata
        return row


def maybe_register_dataset() -> Optional[str]:
    """Register a trajectory dataset only when ``VLA4AD_TRAJECTORY_DATASET`` is set.

    Passing a JSONL path directly to ``--dataset`` does not require registration;
    this hook is for users who prefer a named ms-swift dataset.
    """
    dataset_path = os.getenv("VLA4AD_TRAJECTORY_DATASET")
    if not dataset_path:
        return None
    register_dataset(
        DatasetMeta(
            dataset_path=dataset_path,
            preprocess_func=OmniDriveTrajectoryPreprocessor(),
            tags=["trajectory", "planning", "autonomous_driving", "vla"],
        )
    )
    return dataset_path


if os.getenv("VLA4AD_TRAJECTORY_DATASET"):
    maybe_register_dataset()


if __name__ == "__main__":
    print("Trajectory dataset helper. Set VLA4AD_TRAJECTORY_DATASET to register a JSONL.")
