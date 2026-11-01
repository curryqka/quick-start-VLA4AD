# Copyright (c) Alibaba, Inc. and its affiliates.
"""Register the Qwen2.5-VLA model and template with ms-swift.

The model is created by ``save_custom_model.py`` (a Qwen2.5-VL checkpoint plus a
randomly initialised trajectory head). Set ``VLA4AD_MODEL_PATH`` to that
directory before launching training/inference.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from swift.llm import (
    Model,
    ModelArch,
    ModelGroup,
    ModelMeta,
    register_model,
    register_template,
)
from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_5_vl
from swift.llm.template.template.qwen import Qwen2_5TemplateMeta, Qwen2_5VLTemplate

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAJECTORY_STEPS = 6
TRAJECTORY_FEATURES = 3


class Qwen2_5VLATemplate(Qwen2_5VLTemplate):
    """Qwen2.5-VL template that also carries the numeric trajectory label."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.traj_timesteps = TRAJECTORY_STEPS
        self.traj_feat_dim = TRAJECTORY_FEATURES

    def _dummy_trajectory(self) -> torch.Tensor:
        return torch.zeros(
            1, self.traj_timesteps, self.traj_feat_dim, dtype=torch.float32
        )

    def _encode(self, inputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        trajectory = inputs.extra_kwargs.get("trajectory")
        if trajectory is None:
            encoded["trajectory"] = self._dummy_trajectory()
            encoded["trajectory_mask"] = torch.zeros(1, dtype=torch.float32)
            return encoded

        if not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, dtype=torch.float32)
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
        expected = (self.traj_timesteps, self.traj_feat_dim)
        if tuple(trajectory.shape[-2:]) != expected:
            raise ValueError(
                f"trajectory must end with {expected}, got {tuple(trajectory.shape)}"
            )
        encoded["trajectory"] = trajectory
        encoded["trajectory_mask"] = torch.ones(trajectory.shape[0], dtype=torch.float32)
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None):
        result = super()._data_collator(batch, padding_to=padding_to)
        trajectories = []
        masks = []
        for row in batch:
            trajectory = row.get("trajectory")
            if trajectory is None:
                trajectory = self._dummy_trajectory()
                mask = 0.0
            else:
                if not isinstance(trajectory, torch.Tensor):
                    trajectory = torch.tensor(trajectory, dtype=torch.float32)
                mask = float(row.get("trajectory_mask", 1.0))
            trajectories.append(trajectory)
            masks.append(mask)
        result["trajectory"] = torch.cat(trajectories, dim=0)
        result["trajectory_mask"] = torch.tensor(masks, dtype=torch.float32)
        return result

    def prepare_model_inputs(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_inputs = super().prepare_model_inputs(model, inputs)
        for key in ("trajectory", "trajectory_mask"):
            if key in inputs:
                model_inputs[key] = inputs[key]
        return model_inputs


def get_model_tokenizer_qwen2_5_vla(model_dir, *args, **kwargs):
    """Load the custom class while reusing ms-swift's Qwen2.5-VL pipeline."""
    from modeling_qwen2_5_vla import Qwen2_5_VLAForConditionalGeneration

    kwargs["automodel_class"] = (
        kwargs.get("automodel_class") or Qwen2_5_VLAForConditionalGeneration
    )
    kwargs["_check_qwen_vl_utils"] = False
    return get_model_tokenizer_qwen2_5_vl(model_dir, *args, **kwargs)


register_template(
    Qwen2_5TemplateMeta(
        "qwen2_5_vla",
        template_cls=Qwen2_5VLATemplate,
        default_system=None,
    )
)

MODEL_PATH = os.getenv("VLA4AD_MODEL_PATH", str(REPO_ROOT / "diy_model"))
MODEL_ID = os.getenv("VLA4AD_MODEL_ID") or None

register_model(
    ModelMeta(
        model_type="qwen2_5_vla",
        model_groups=[
            ModelGroup([Model(model_path=MODEL_PATH, hf_model_id=MODEL_ID)])
        ],
        template="qwen2_5_vla",
        get_function=get_model_tokenizer_qwen2_5_vla,
        model_arch=ModelArch.qwen2_vl,
        architectures=["Qwen2_5_VLAForConditionalGeneration"],
        requires=["transformers>=4.50"],
        tags=["vision", "language", "vla", "autonomous-driving"],
        is_multimodal=True,
    )
)
