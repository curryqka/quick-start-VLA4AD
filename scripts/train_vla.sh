#!/usr/bin/env bash
# Train Qwen2.5-VLA with the custom model/template registered by this repo.
#
# Usage:
#   VLA4AD_MODEL_PATH=./diy_model bash scripts/train_vla.sh DATA.jsonl [OUTPUT_DIR]
#
# Common overrides: NPROC_PER_NODE, MAX_PIXELS, EPOCHS, BATCH_SIZE,
# GRAD_ACCUM, MAX_LENGTH, LEARNING_RATE, LORA_RANK, LORA_ALPHA, DEEPSPEED.
set -euo pipefail

DATA="${1:?usage: train_vla.sh DATA.jsonl [OUTPUT_DIR]}"
OUTPUT="${2:-./output/vla4ad_model}"
MODEL="${VLA4AD_MODEL_PATH:?set VLA4AD_MODEL_PATH to the diy_model directory}"
NPROC="${NPROC_PER_NODE:-1}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
DEEPSPEED="${DEEPSPEED-zero2}"   # set DEEPSPEED="" to disable
MAX_PIXELS="${MAX_PIXELS:-1003520}"
MAX_STEPS="${MAX_STEPS:-}"
VAL_DATASET="${VAL_DATASET:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${DATA}" ]]; then
  echo "Dataset not found: ${DATA}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NPROC_PER_NODE="${NPROC}"
export MAX_PIXELS

EXTRA_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
  EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${VAL_DATASET}" ]]; then
  EXTRA_ARGS+=(--val_dataset "${VAL_DATASET}" --eval_steps 500)
fi
if [[ -n "${DEEPSPEED}" ]]; then
  EXTRA_ARGS+=(--deepspeed "${DEEPSPEED}")
fi

swift sft \
  --model "${MODEL}" \
  --model_type qwen2_5_vla \
  --dataset "${DATA}" \
  --custom_register_path custom_dataset.py custom_model.py \
  --remove_unused_columns false \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size 1 \
  --learning_rate "${LEARNING_RATE}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --modules_to_save trajectory_head \
  --freeze_vit true \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --save_steps 500 \
  --save_total_limit 5 \
  --logging_steps 5 \
  --max_length "${MAX_LENGTH}" \
  --output_dir "${OUTPUT}" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
