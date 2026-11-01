# VLA4AD 教程：从数据到轨迹推理

本教程按“最短可跑通”的顺序组织。每一步都有明确输入/输出，建议先用 `--convert_ratio 0.01` 或 1 条样本验证流程，再跑全量。

## 0. 环境自检

```bash
pip install -r requirements.txt
python tools/check_env.py
```

检查 `torch / transformers / swift / qwen_vl_utils` 是否齐全，以及 GPU 显存。ms-swift 固定为 **3.11.3**。

## 1. 数据准备

需要的目录：

```text
data/
├── nuscenes/samples/                # 六相机图像
└── omnidrive/
    ├── omnidrive_train.jsonl
    ├── omnidrive_val.jsonl
    ├── nuscenes_ego_infos_train.pkl
    └── nuscenes_ego_infos_val.pkl
```

下载说明见 README。

## 2. 转换 VQA 数据

```bash
python convert_omnidrive.py \
    --input_file data/omnidrive/omnidrive_train.jsonl \
    --output_dir preprocessed_data/OmniDrive \
    --data_split train \
    --img_root data/nuscenes/samples \
    --check_images
```

输出为 `messages + images` 的 MS-Swift JSONL，用于语言/场景理解。

## 3. 转换轨迹数据

文本轨迹（语言监督）：

```bash
python prepare_omnidrive_planning_data.py \
    --input_file data/omnidrive/nuscenes_ego_infos_train.pkl \
    --output_dir preprocessed_data/nuscenes_trajectory \
    --data_split train --img_root data/nuscenes/samples \
    --trajectory_format string --check_images
```

高层导航 + 数值轨迹（VLA 训练推荐）：

```bash
python prepare_omnidrive_numerical_planning_data.py \
    --input_file data/omnidrive/nuscenes_ego_infos_train.pkl \
    --output_dir preprocessed_data/nuscenes_navigation \
    --data_split train --img_root data/nuscenes/samples \
    --navigation_format structured --user_prompt_type structured --check_images
```

输出中每条样本都有顶层 `trajectory`（6×3）和 assistant 文本答案。

## 4. 校验数据

```bash
python tools/validate_dataset.py \
    --input preprocessed_data/nuscenes_navigation/nuscenes_navigation_train.jsonl \
    --check-images --require-trajectory --strict
```

输出会统计记录数、轨迹数、图片数，并列出前 10 条错误。

## 5. 基础 VLM 推理 / LoRA

```bash
# 零样本推理
swift infer --model /path/to/Qwen2.5-VL-3B-Instruct --infer_backend pt \
    --val_dataset preprocessed_data/OmniDrive/omnidrive_val.jsonl \
    --result_path infer_results/omnidrive

# 纯文本 LoRA
swift sft --model /path/to/Qwen2.5-VL-3B-Instruct \
    --dataset preprocessed_data/OmniDrive/omnidrive_train.jsonl \
    --train_type lora --lora_rank 8 --lora_alpha 32 ...
```

这一步只训练文本能力，不涉及轨迹头。

## 6. 创建 Qwen2.5-VLA

```bash
python save_custom_model.py \
    --base-model Qwen/Qwen2.5-VL-3B-Instruct \
    --save-path ./diy_model
```

生成一个包含基础 VLM 权重 + 随机初始化 `trajectory_head` 的 checkpoint。

## 7. 训练 VLA

```bash
export VLA4AD_MODEL_PATH=$PWD/diy_model

VLA4AD_MODEL_PATH=$VLA4AD_MODEL_PATH \
MAX_PIXELS=1003520 \
NPROC_PER_NODE=1 \
bash scripts/train_vla.sh \
    preprocessed_data/nuscenes_navigation/nuscenes_navigation_train.jsonl \
    output/vla4ad_model
```

脚本默认 LoRA + `modules_to_save trajectory_head`，并保留 `trajectory` 列。小数据冒烟：

```bash
MAX_STEPS=1 BATCH_SIZE=1 GRAD_ACCUM=1 DEEPSPEED="" \
bash scripts/train_vla.sh <data.jsonl> output/smoke
```

## 8. 合并与推理

```bash
swift export --model "$VLA4AD_MODEL_PATH" --model_type qwen2_5_vla \
    --custom_register_path custom_model.py \
    --adapters output/vla4ad_model/vx-xxx/checkpoint-xxx \
    --merge_lora true --output_dir output/vla4ad_merged

python inference_vla.py \
    --model output/vla4ad_merged \
    --image data/nuscenes/samples/CAM_FRONT/xxx.jpg \
    --question "Plan the next 3 seconds." \
    --id sample_001 \
    --output results/vla_predictions.jsonl
```

## 9. 轨迹评测与可视化

如果有 GT 轨迹（JSON 6×3）：

```bash
python inference_vla.py ... --gt-trajectory gt.json --output pred.jsonl

python tools/evaluate_trajectory.py \
    --input pred.jsonl --output results/metrics.json --csv results/metrics.csv

python tools/visualize_trajectory.py \
    --input pred.jsonl --output results/trajectory.png
```

指标：ADE（6 点平均 L2）、FDE（终点 L2）、heading error。

## 10. 产物一览

| 阶段 | 产物 |
|---|---|
| VQA 转换 | `preprocessed_data/OmniDrive/omnidrive_{split}.jsonl` |
| 轨迹转换 | `preprocessed_data/nuscenes_*/..._{split}.jsonl`（含 `trajectory`） |
| VLA checkpoint | `diy_model/` |
| LoRA 训练 | `output/vla4ad_model/.../checkpoint-*` |
| 合并模型 | `output/vla4ad_merged/` |
| 推理 | `results/vla_predictions.jsonl` |
| 评测/可视化 | `results/metrics.json`、`results/trajectory.png` |

## 常见问题

- **`--model_type qwen2_5_vla` 未注册**：确保 `--custom_register_path custom_model.py` 且从仓库根目录运行。
- **轨迹列被丢弃**：`--remove_unused_columns false`。
- **head 不训练**：使用 `--modules_to_save trajectory_head`。
- **OOM**：降低 `BATCH_SIZE`、`MAX_LENGTH`、`MAX_PIXELS`，或增加 GPU/ZeRO。
- **轨迹形状错误**：必须是顶层 `(6, 3)`，`(1, 6, 3)` 会在预处理时展平。

## 当前限制

- 轨迹头只从最后一个 token 的 hidden state 回归，没有时序 decoder。
- 未内置碰撞率、舒适度等规划指标。
- 未做轨迹归一化/多模态 token 化。
- 参考实现不等同于论文级复现。
