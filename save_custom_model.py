# Copyright (c) Alibaba, Inc. and its affiliates.
"""Create a Qwen2.5-VLA checkpoint (base VLM + random trajectory head).

The language/vision weights are copied from the base Qwen2.5-VL checkpoint; only
``trajectory_head`` is newly initialised. The output directory can be passed to
ms-swift with ``VLA4AD_MODEL_PATH`` and ``custom_model.py``.

Example:
    python save_custom_model.py \
        --base-model Qwen/Qwen2.5-VL-3B-Instruct \
        --save-path ./diy_model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_utils import no_init_weights

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeling_qwen2_5_vla import Qwen2_5_VLAForConditionalGeneration

LOGGER = logging.getLogger("vla4ad.save")


def resolve_base_model(base_model: str) -> str:
    if os.path.isdir(base_model):
        return base_model
    from huggingface_hub import snapshot_download

    LOGGER.info("Downloading %s from Hugging Face", base_model)
    return snapshot_download(base_model)


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value == "auto":
        return (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    if value not in mapping:
        raise ValueError(f"unknown dtype: {value}")
    return mapping[value]


def init_and_save_qwen_vla_model(
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    save_path: str = "diy_model",
    trajectory_steps: int = 6,
    trajectory_feature_dim: int = 3,
    trajectory_head_hidden_size: int = 512,
    trajectory_loss_weight: float = 0.5,
    dtype: str = "auto",
) -> str:
    """Create and save the VLA checkpoint. Returns ``save_path``."""
    torch_dtype = parse_dtype(dtype)
    base_model_name = base_model
    base_model_path = resolve_base_model(base_model)
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(base_model_path)
    output_dim = trajectory_steps * trajectory_feature_dim
    config.trajectory_steps = trajectory_steps
    config.trajectory_timesteps = trajectory_steps
    config.trajectory_feature_dim = trajectory_feature_dim
    config.trajectory_output_dim = output_dim
    config.trajectory_head_hidden_size = trajectory_head_hidden_size
    config.trajectory_loss_weight = trajectory_loss_weight
    config.architectures = ["Qwen2_5_VLAForConditionalGeneration"]
    config.model_type = "qwen2_5_vl"  # keep the Qwen config class

    LOGGER.info("Building the VLA model on CPU")
    with no_init_weights():
        model = Qwen2_5_VLAForConditionalGeneration(config)

    LOGGER.info("Loading base weights (%s)", torch_dtype)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    base_state = base_model.state_dict()
    model_state = model.state_dict()
    copied = 0
    for key, value in model_state.items():
        if key in base_state:
            model_state[key] = base_state[key]
            copied += 1
    model.load_state_dict(model_state, strict=False)
    del base_model, base_state
    LOGGER.info("Copied %d/%d parameters; trajectory head keeps its init", copied, len(model_state))

    model.eval()
    model.save_pretrained(save_dir, max_shard_size="4GB", safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)
    for file_name in os.listdir(base_model_path):
        if (
            file_name.endswith((".bin", ".safetensors", ".pt"))
            or file_name == "config.json"
            or ".index.json" in file_name
        ):
            continue
        source = Path(base_model_path) / file_name
        target = save_dir / file_name
        if source.is_file():
            shutil.copy2(source, target)
        elif source.is_dir() and not target.exists():
            shutil.copytree(source, target)

    card = {
        "model_type": "qwen2_5_vla",
        "base_model": base_model_name,
        "trajectory_steps": trajectory_steps,
        "trajectory_feature_dim": trajectory_feature_dim,
        "trajectory_output_dim": output_dim,
        "trajectory_head_hidden_size": trajectory_head_hidden_size,
        "trajectory_loss_weight": trajectory_loss_weight,
        "description": "Qwen2.5-VL with a (6, 3) trajectory regression head",
    }
    (save_dir / "model_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved VLA model to %s", save_dir)
    return str(save_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Qwen2.5-VLA checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--save-path", default="diy_model")
    parser.add_argument("--trajectory-steps", type=int, default=6)
    parser.add_argument("--trajectory-feature-dim", type=int, default=3)
    parser.add_argument("--trajectory-head-hidden-size", type=int, default=512)
    parser.add_argument("--trajectory-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    init_and_save_qwen_vla_model(
        base_model=args.base_model,
        save_path=args.save_path,
        trajectory_steps=args.trajectory_steps,
        trajectory_feature_dim=args.trajectory_feature_dim,
        trajectory_head_hidden_size=args.trajectory_head_hidden_size,
        trajectory_loss_weight=args.trajectory_loss_weight,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
