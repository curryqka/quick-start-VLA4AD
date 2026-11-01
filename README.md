# VLA4AD Quick Start

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: MS-Swift](https://img.shields.io/badge/Framework-MS--Swift-blue)](https://github.com/modelscope/ms-swift)
[![Topic: VLA](https://img.shields.io/badge/Topic-Vision--Language--Action-orange)](https://arxiv.org/abs/2506.24044)

> **中文** | [English summary](#english-summary)

本仓库是 [Awesome-VLA4AD](https://github.com/SicongJiang/Awesome-VLA4AD) 配套的实战代码，基于 **MS-Swift** 提供自动驾驶 VLA（Vision-Language-Action）模型的最小可运行流程：

- OmniDrive / nuScenes 数据转换
- Qwen2.5-VL 基础推理与 LoRA 微调
- 在 Qwen2.5-VL 上增加轨迹回归头，构建 **Qwen2.5-VLA**
- 文本答案 + `(6, 3)` 轨迹联合训练
- 合并 LoRA 并输出文本与轨迹

> **状态说明**：数据转换和 LoRA 流程已经可用；`Qwen2.5-VLA` 轨迹头是参考实现，尚未在完整 OmniDrive 基准上做论文级评测。轨迹格式为 6 个未来路点，每点 `(x, y, heading)`。

## 目录

- [仓库内容](#-仓库内容)
- [教程与工具](#-教程与工具)
- [环境](#-环境)
- [模型与数据](#-模型与数据)
- [数据转换](#-数据转换)
- [基础模型推理](#-基础模型推理)
- [LoRA 微调](#-lora-微调)
- [构建 Qwen2.5-VLA](#-构建-qwen25-vla)
- [训练 VLA](#-训练-vla)
- [合并与推理](#-合并与推理)
- [VLA 数据格式](#-vla-数据格式)
- [限制与排错](#-限制与排错)
- [引用与许可](#-引用与许可)

## 📦 仓库内容

| 文件 | 作用 |
|---|---|
| `convert_omnidrive.py` | OmniDrive VQA JSONL → MS-Swift JSONL |
| `prepare_omnidrive_planning_data.py` | 轨迹 pkl → 文本轨迹数据（附带数值 `trajectory`） |
| `prepare_omnidrive_numerical_planning_data.py` | 轨迹 pkl → 高层导航文本 + 数值 `trajectory` |
| `custom_dataset.py` | 轨迹数据预处理器（把 metadata 中的轨迹提升为模型字段） |
| `custom_model.py` | 注册 `qwen2_5_vla` 模型与模板 |
| `save_custom_model.py` | 基于 Qwen2.5-VL 创建带轨迹头的 VLA checkpoint |
| `modeling_qwen2_5_vla.py` | Qwen2.5-VL + 轨迹回归头，文本 loss + 轨迹 MSE |
| `inference_vla.py` | 单样本推理，打印文本答案与轨迹，可导出 prediction JSONL |
| `trajectory_utils.py` | 轨迹校验、格式化、统计、ADE/FDE（无第三方依赖） |
| `tools/check_env.py` | 检查 Python / torch / swift / CUDA / 模型路径 |
| `tools/validate_dataset.py` | 训练前校验 messages、images、trajectory |
| `tools/evaluate_trajectory.py` | 计算 ADE / FDE / heading error |
| `tools/visualize_trajectory.py` | 俯视图对比预测与 GT 轨迹 |
| `scripts/train_vla.sh` | 封装 VLA LoRA 训练命令 |
| `docs/TUTORIAL.md` | 从数据到推理的完整教程 |

## 🎓 教程与工具

完整、按步骤的版本见 **[docs/TUTORIAL.md](docs/TUTORIAL.md)**。最短路径：

```bash
# 0. 环境
python tools/check_env.py

# 1. 数据转换后先校验
python tools/validate_dataset.py \
    --input preprocessed_data/nuscenes_navigation/nuscenes_navigation_train.jsonl \
    --check-images --require-trajectory --strict

# 2. 创建 VLA 并训练
python save_custom_model.py --base-model Qwen/Qwen2.5-VL-3B-Instruct --save-path ./diy_model
VLA4AD_MODEL_PATH=$PWD/diy_model bash scripts/train_vla.sh <train.jsonl> output/vla4ad_model

# 3. 推理、评测、可视化
python inference_vla.py --model output/vla4ad_merged --image <img.jpg> \
    --gt-trajectory gt.json --output results/pred.jsonl
python tools/evaluate_trajectory.py --input results/pred.jsonl --output results/metrics.json
python tools/visualize_trajectory.py --input results/pred.jsonl --output results/trajectory.png
```

## ⚙️ 环境

```bash
git clone https://github.com/curryqka/quick-start-VLA4AD.git
cd quick-start-VLA4AD
conda create -n vla4ad python=3.10 -y
conda activate vla4ad
pip install -r requirements.txt
```

`requirements.txt` 固定了论文环境附近可用的组合：

- Python 3.10
- PyTorch ≥ 2.0（推荐 2.6）
- transformers 4.50.1
- **ms-swift 3.11.3**（4.x 重构了包结构，本教程不兼容）
- qwen-vl-utils 0.0.14

如果希望使用 ms-swift 源码，可另外克隆；`pip` 安装版本已足够运行本仓库：

```bash
git clone https://github.com/modelscope/ms-swift.git
```

## 📥 模型与数据

### 基础模型

| 资源 | 链接 |
|---|---|
| Qwen2.5-VL-3B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen2.5-VL 文档 | [Transformers Docs](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl) |

### OmniDrive

- 数据发布：[OmniDrive releases](https://github.com/NVlabs/OmniDrive/releases/tag/v1.0)
- `data_nusc.zip`：VQA / 转换后的 info 文件
- `nuscenes_ego_infos_train.pkl`、`nuscenes_ego_infos_val.pkl`：轨迹数据

nuScenes 图像需自行从 [nuScenes](https://www.nuscenes.org/download) 下载。推荐目录：

```text
data/
├── nuscenes/
│   ├── samples/                 # 六个相机图像
│   ├── v1.0-trainval/
│   └── ...
└── omnidrive/
    ├── omnidrive_train.jsonl    # VQA
    ├── omnidrive_val.jsonl
    ├── nuscenes_ego_infos_train.pkl
    └── nuscenes_ego_infos_val.pkl
```

## 🔄 数据转换

### 1. OmniDrive VQA

```bash
python convert_omnidrive.py \
    --input_file data/omnidrive/omnidrive_train.jsonl \
    --output_dir ./preprocessed_data/OmniDrive \
    --data_split train \
    --img_root data/nuscenes/samples \
    --check_images
```

输出 `preprocessed_data/OmniDrive/omnidrive_train.jsonl`，格式为 MS-Swift 的 `messages` + `images`。

### 2. 轨迹转文本（数值轨迹单独保留）

```bash
python prepare_omnidrive_planning_data.py \
    --input_file data/omnidrive/nuscenes_ego_infos_val.pkl \
    --output_dir ./preprocessed_data/nuscenes_trajectory \
    --data_split val \
    --img_root data/nuscenes/samples \
    --trajectory_format string \
    --use_scene_description \
    --check_images
```

`--trajectory_format` 支持 `list` / `string` / `detailed` / `compact`。

### 3. 高层语义导航 + 数值轨迹

```bash
python prepare_omnidrive_numerical_planning_data.py \
    --input_file data/omnidrive/nuscenes_ego_infos_val.pkl \
    --output_dir ./preprocessed_data/nuscenes_navigation \
    --data_split val \
    --img_root data/nuscenes/samples \
    --navigation_format structured \
    --user_prompt_type structured \
    --check_images
```

这两个轨迹脚本都会在每条 JSONL 顶层写入：

```json
"trajectory": [[x1, y1, h1], [x2, y2, h2], "... 6 points ..."]
```

这是 VLA 轨迹头的监督标签；文本 `messages[-1]` 仍然是语言监督目标。

## 🧠 基础模型推理

```bash
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct

swift infer \
    --model "$MODEL_PATH" \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048 \
    --model_type qwen2_5_vl \
    --val_dataset ./preprocessed_data/OmniDrive/omnidrive_val.jsonl \
    --max_batch_size 1 \
    --result_path ./infer_results/omnidrive
```

## 🎛 LoRA 微调

```bash
export CUDA_VISIBLE_DEVICES=0

swift sft \
    --model /path/to/Qwen2.5-VL-3B-Instruct \
    --train_type lora \
    --dataset ./preprocessed_data/OmniDrive/omnidrive_train.jsonl \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_length 2048 \
    --output_dir ./output/lora_omnidrive \
    --dataset_num_proc 4
```

## 🏗 构建 Qwen2.5-VLA

VLA 模型 = Qwen2.5-VL + 轨迹回归头：

```text
图像 + 文本 → Qwen2.5-VL hidden states
                         ├─ LM head → 文本答案（交叉熵）
                         └─ trajectory_head(MLP) → (6, 3) 轨迹（MSE）
```

### 1. 创建带轨迹头的 checkpoint

```bash
python save_custom_model.py \
    --base-model Qwen/Qwen2.5-VL-3B-Instruct \
    --save-path ./diy_model
```

该脚本复制基础模型权重，仅随机初始化 `trajectory_head`，并写入：

- `trajectory_timesteps = 6`
- `trajectory_feature_dim = 3`
- `trajectory_output_dim = 18`
- `trajectory_head_hidden_size = 512`
- `trajectory_loss_weight = 0.5`

> 需要同时保存 tokenizer / preprocessor 文件；脚本会自动从基础模型目录复制。

### 2. 训练

推荐直接用封装脚本（默认值与下面的完整命令一致）：

```bash
VLA4AD_MODEL_PATH=$PWD/diy_model \
  bash scripts/train_vla.sh \
    preprocessed_data/nuscenes_navigation/nuscenes_navigation_train.jsonl \
    output/vla4ad_model
```

也可以直接调用 ms-swift：

```bash
export VLA4AD_MODEL_PATH=$PWD/diy_model
export CUDA_VISIBLE_DEVICES=0
export MAX_PIXELS=1003520   # 控制图像分辨率，降低显存

swift sft \
    --model "$VLA4AD_MODEL_PATH" \
    --model_type qwen2_5_vla \
    --dataset ./preprocessed_data/nuscenes_navigation/nuscenes_navigation_train.jsonl \
    --custom_register_path custom_dataset.py custom_model.py \
    --remove_unused_columns false \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --modules_to_save trajectory_head \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir ./output/vla4ad_model \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
```

要点：

- `--model_type qwen2_5_vla` 必须与 `custom_model.py` 中注册的类型一致；
- `--remove_unused_columns false` 保留顶层 `trajectory` 列；
- 没有 `trajectory` 的 VQA 样本会使用全零占位并设置 `trajectory_mask=0`，不参与轨迹 loss；
- 只有 `trajectory` 样本会计算轨迹 MSE。

### 3. 合并 LoRA

```bash
swift export \
    --model "$VLA4AD_MODEL_PATH" \
    --model_type qwen2_5_vla \
    --custom_register_path custom_model.py \
    --adapters ./output/vla4ad_model/vx-xxx/checkpoint-xxx \
    --merge_lora true \
    --output_dir ./output/vla4ad_merged
```

## 🚀 合并与推理

### 基础 VLM 推理

见上文“基础模型推理”。

### VLA 推理（文本 + 轨迹）

```bash
python inference_vla.py \
    --model ./output/vla4ad_merged \
    --image data/nuscenes/samples/CAM_FRONT/xxx.jpg \
    --image data/nuscenes/samples/CAM_FRONT_LEFT/xxx.jpg \
    --question "Plan the next 3 seconds."
```

输出包括：

```text
=== Text answer ===
...
=== Trajectory (6 x [x, y, heading]) ===
[[...], ...]
```

该脚本绕过 `swift infer`，因为轨迹是自定义模型输出；批量基准推理仍可使用 `swift infer` 的文本输出。

### 轨迹评测与可视化

`inference_vla.py --output` 会写出包含 `prediction` 的 JSONL；如果同时提供 GT：

```bash
cat > gt.json <<'EOF'
[[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0],[4.0,0.0,0.0],[5.0,0.0,0.0]]
EOF

python inference_vla.py --model output/vla4ad_merged --image demo.jpg \
    --gt-trajectory gt.json --id sample_001 --output results/pred.jsonl

python tools/evaluate_trajectory.py \
    --input results/pred.jsonl --output results/metrics.json

python tools/visualize_trajectory.py \
    --input results/pred.jsonl --output results/trajectory.png
```

输出 ADE、FDE、平均 heading error 和逐点 L2；可视化脚本生成俯视图。

## 📈 VLA 数据格式

轨迹训练 JSONL 每条记录：

```json
{
  "messages": [
    {"role": "system", "content": "You are an autonomous driving planner."},
    {"role": "user", "content": "<image> ... Plan the next 3 seconds."},
    {"role": "assistant", "content": "Continue straight and keep a safe distance."}
  ],
  "images": [
    "data/nuscenes/samples/CAM_FRONT/xxx.jpg",
    "data/nuscenes/samples/CAM_FRONT_LEFT/xxx.jpg"
  ],
  "trajectory": [
    [0.02, 0.00, 0.00],
    [0.05, 0.01, 0.00],
    [0.09, 0.01, 0.00],
    [0.14, 0.02, 0.01],
    [0.20, 0.02, 0.01],
    [0.27, 0.03, 0.01]
  ]
}
```

约束：

- `trajectory` 必须是 **顶层字段**，形状 `(6, 3)`；`metadata.trajectory` 只作为数据准备阶段的中间字段。
- 轨迹坐标来自 OmniDrive ego 坐标系：x 纵向、y 横向、heading 弧度。
- `images` 的顺序应与 `<image>` 标签顺序一致。

## ⚠️ 限制与排错

| 问题 | 处理 |
|---|---|
| `ModuleNotFoundError: swift.llm` | 安装 `ms-swift==3.11.3`，不要使用 4.x |
| `trajectory must have shape (6, 3)` | 检查数据转换脚本输出；数值轨迹必须为 6 个 `(x, y, h)` |
| 自定义模型加载失败 | 确认 `VLA4AD_MODEL_PATH` 指向 `save_custom_model.py` 的输出目录，且包含 `config.json`、权重、`preprocessor_config.json` |
| `trajectory` 列被丢弃 | 训练命令加 `--remove_unused_columns false` |
| CUDA OOM | 降低 batch / `MAX_PIXELS` / `max_length`，或使用多卡 |
| loss 不下降 | 降低学习率到 `5e-5`，检查轨迹归一化与文本标签 |

已知限制：

- 轨迹头从最后一个有效 token 的 hidden state 回归，未做时序 decoder；
- 未做坐标归一化/多模态轨迹 token 化；
- 轨迹评估指标（L2、碰撞率）尚未内置；
- 当前只提供参考实现，不保证复现论文数值。

## 📚 引用

- [A Survey on Vision-Language-Action Models for Autonomous Driving](https://arxiv.org/abs/2506.24044)
- [OmniDrive](https://arxiv.org/abs/2405.01533)
- [MS-Swift](https://github.com/modelscope/ms-swift)
- [Awesome-VLA4AD](https://github.com/SicongJiang/Awesome-VLA4AD)

## 📄 许可

MIT License，见 [LICENSE](LICENSE)。

---

## English summary

A minimal, hands-on VLA4AD stack built on MS-Swift: OmniDrive conversion, Qwen2.5-VL inference/LoRA, and an experimental Qwen2.5-VLA model that adds a `(6, 3)` trajectory regression head to Qwen2.5-VL. Text and trajectory losses are trained jointly; VQA-only rows are masked out of the trajectory loss. See the Chinese sections above for exact commands and data format.
