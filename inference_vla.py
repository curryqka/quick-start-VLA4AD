#!/usr/bin/env python
# Copyright (c) 2025 The VLA4AD Authors. All rights reserved.
"""Run a merged Qwen2.5-VLA checkpoint and print text + trajectory.

This bypasses ms-swift's CLI because the trajectory is a custom model output.
For batched benchmark inference, use ``swift infer`` with the text-only output.

Example:
    python inference_vla.py \
        --model ./output/vla4ad_merged \
        --image data/nuscenes/samples/CAM_FRONT/xxx.jpg \
        --question "Plan the next 3 seconds."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeling_qwen2_5_vla import Qwen2_5_VLAForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VLA single-sample inference")
    parser.add_argument("--model", required=True, help="Merged VLA checkpoint.")
    parser.add_argument("--image", action="append", required=True, help="Image path; repeat for multiple cameras.")
    parser.add_argument("--question", default="Plan a 3-second trajectory with 6 waypoints.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--id", default="sample", help="Record id for the output JSONL.")
    parser.add_argument("--output", default=None, help="Optional prediction JSONL.")
    parser.add_argument(
        "--gt-trajectory",
        default=None,
        help="Optional JSON file with the ground-truth 6x3 trajectory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = Qwen2_5_VLAForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    content = [
        {"type": "image", "image": Image.open(path).convert("RGB")}
        for path in args.image
    ]
    content.append({"type": "text", "text": args.question})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated = output.sequences[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("=== Text answer ===")
    print(answer)
    prediction = None
    if hasattr(output, "trajectory"):
        prediction = output.trajectory.float().cpu().tolist()[0]
        print("\n=== Trajectory (6 x [x, y, heading]) ===")
        print(prediction)

    if args.output:
        import json

        record = {"id": args.id, "text": answer, "prediction": prediction}
        if args.gt_trajectory:
            record["trajectory"] = json.loads(
                Path(args.gt_trajectory).read_text(encoding="utf-8")
            )
        with Path(args.output).open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nSaved prediction to {args.output}")


if __name__ == "__main__":
    main()
