#!/usr/bin/env python
# Copyright (c) 2025 The VLA4AD Authors. All rights reserved.
"""Print the VLA4AD runtime environment and check the model path.

Example:
    python tools/check_env.py
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


def version(module: str) -> str:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({exc})"
    try:
        from importlib.metadata import version as dist_version

        return dist_version(module.replace("_", "-"))
    except Exception:  # noqa: BLE001
        return getattr(imported, "__version__", "unknown")


def main() -> None:
    print(f"Python:       {sys.version.split()[0]} ({platform.platform()})")
    for module in ("torch", "transformers", "accelerate", "swift", "qwen_vl_utils"):
        print(f"{module:13s} {version(module)}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices:   {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                memory = torch.cuda.get_device_properties(index).total_memory / 2**30
                print(f"  [{index}] {name} ({memory:.1f} GiB)")
    except Exception as exc:  # noqa: BLE001
        print(f"torch check failed: {exc}")

    model_path = os.getenv("VLA4AD_MODEL_PATH", "")
    print(f"VLA4AD_MODEL_PATH: {model_path or '(not set)'}")
    if model_path:
        path = Path(model_path)
        for required in ("config.json", "preprocessor_config.json"):
            print(f"  {required}: {(path / required).exists()}")
        weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
        print(f"  weights: {len(weights)} file(s)")


if __name__ == "__main__":
    main()
