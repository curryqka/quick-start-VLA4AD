
# VLA4AD Quick Start

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: MS-Swift](https://img.shields.io/badge/Framework-MS--Swift-blue)](https://github.com/modelscope/ms-swift)
[![Topic: VLA](https://img.shields.io/badge/Topic-Vision--Language--Action-orange)](https://arxiv.org/abs/2506.24044)

> **English** | [中文](#vla4ad-快速入门)

This project was built based on the our previous survey of vision-language-action for autonomous driving[Awesome-VLA4AD](https://github.com/SicongJiang/Awesome-VLA4AD). Welcome to have a look~

A hands-on, beginner-friendly repository for quickly getting started with **Vision-Language-Action (VLA)** models for autonomous driving using the **MS-Swift** framework. This project provides practical code examples, configuration templates, and step-by-step tutorials to help you build, fine-tune, and evaluate VLA-based autonomous driving models with minimal setup.

---

## 🚗 VLA4AD 快速入门

基于 MS-Swift 框架的自动驾驶 VLA 模型实战指南。本仓库提供从数据准备、模型下载、训练到推理的完整流程，帮助您快速上手构建自动驾驶 Vision-Language-Action 模型。

---

## 📋 目录

- [环境搭建](#-环境搭建)
- [模型下载](#-模型下载)
- [数据下载](#-数据下载)
- [数据准备](#-数据准备)
  - [OmniDrive VQA 数据格式转换](#omnidrive-vqa-数据格式转换)
  - [轨迹数据转换（数值型）](#轨迹数据转换数值型)
  - [轨迹数据转换（语义导航型）](#轨迹数据转换语义导航型)
- [基础模型推理](#-基础模型推理)
- [LoRA 微调](#-lora-微调)
- [权重合并](#-权重合并)
- [构建 VLA4AD 模型](#-构建-vla4ad-模型)
- [轨迹数据文件说明](#-轨迹数据文件说明)
- [注意事项](#-注意事项)
- [故障排除](#-故障排除)
- [参考资料](#-参考资料)

---

## 🛠 环境搭建

### 安装 MS-Swift

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

**环境要求：**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA ≥ 11.7

---

## 📥 模型下载

### Qwen2.5-VL-3B-Instruct

本教程使用 Qwen2.5-VL-3B-Instruct 作为基础视觉语言模型。

| 资源类型 | 链接 |
|---------|------|
| 🤗 Hugging Face | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| 📖 官方文档 | [Transformers Docs](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl) |
| 💻 示例代码 | [qwen2.5vl GitHub](https://github.com/dongyaolin/qwen2.5vl) |

> **提示**：首次使用前建议运行官方推理脚本验证模型功能是否正常。

---

## 📊 数据下载

### OmniDrive 数据集

OmniDrive 是一个面向自动驾驶的综合视觉语言数据集，基于 nuScenes 构建。
完整的数据下载链接:
[Download](https://github.com/NVlabs/OmniDrive/releases/tag/v1.0)

| 文件 | 说明 | 下载链接 |
|------|------|----------|
| `data_nusc.zip` | VQA 问答数据 | [Download](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/data_nusc.zip) |
| `nuscenes_ego_infos_train.pkl` | 训练集轨迹数据 | [Download](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/nuscenes_ego_infos_train.pkl) |
| `nuscenes_ego_infos_val.pkl` | 验证集轨迹数据 | 同上，替换文件名 |

**数据集特点：**
- 基于 nuScenes 700+ 驾驶场景
- 包含 3D 感知、推理和规划的 VQA 标注
- 支持反事实推理（Counterfactual Reasoning）

---

## 🔄 数据准备

### OmniDrive VQA 数据格式转换

将原始 OmniDrive 数据转换为 MS-Swift 兼容格式：

```bash
python convert_omnidrive.py \
    --input_file /path/to/omnidrive_train.json \
    --output_dir ./preprocessed_data/OmniDrive \
    --data_split train \
    --img_root /path/to/nuscenes/samples \
    --check_images \
    --verbose
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_file` | str | **必需** | 原始 OmniDrive JSON 文件路径 |
| `--output_dir` | str | `./preprocessed_data/OmniDrive` | 输出目录 |
| `--data_split` | str | `train` | 数据分割（train/val/test） |
| `--convert_ratio` | float | `1.0` | 数据转换比例（0.0-1.0） |
| `--img_root` | str | `/lustre/.../omnidrive` | 图像根目录 |
| `--system_prompt_type` | str | `default` | 系统提示类型（default/detailed/simple） |
| `--camera_order` | str | `None` | 自定义相机顺序，如 `CAM_FRONT,CAM_LEFT` |
| `--check_images` | bool | `False` | 检查图像文件是否存在 |
| `--allow_no_images` | bool | `False` | 允许没有图像的样本 |
| `--verbose` | bool | `False` | 显示详细处理信息 |

**输出文件：**
- `omnidrive_{split}.jsonl`：转换后的 MS-Swift 格式数据
- `conversion_stats_{split}.txt`：转换统计信息

---

### 轨迹数据转换（数值转到文本内容型）

将 nuScenes 轨迹数据转换为文本内容格式：

```bash
python prepare_omnidrive_planning_data.py \
    --input_file /path/to/nuscenes_ego_infos_val.pkl \
    --output_dir ./preprocessed_data/nuscenes_trajectory \
    --data_split val \
    --img_root /path/to/nuscenes/samples \
    --trajectory_format string \
    --use_scene_description \
    --check_images
```

**轨迹格式选项（`--trajectory_format`）：**

| 格式 | 描述 | 示例 |
|------|------|------|
| `list` | Python 列表 | `[[x, y, heading], ...]` |
| `string` | 字符串格式 | `"[-0.023, 0.004, -0.003], ..."` |
| `detailed` | 带时间戳的自然语言 | "0.5秒后，车辆位于..." |
| `compact` | 精简坐标序列 | `x1,y1,h1,x2,y2,h2...` |

**新增参数：**
- `--use_scene_description`：在提示中包含场景描述（如 "停车场, 障碍物, 出口"）

---

### 轨迹数据转换（语义导航型，数值单独放一个key）

生成高层级语义导航指令（如"向左转，进入主路"）：

```bash
python prepare_omnidrive_numerical_planning_data.py \
    --input_file /path/to/nuscenes_ego_infos_val.pkl \
    --output_dir ./preprocessed_data/nuscenes_navigation \
    --data_split val \
    --img_root /path/to/nuscenes/samples \
    --navigation_format structured \
    --user_prompt_type structured \
    --check_images
```

**导航格式选项（`--navigation_format`）：**

| 格式 | 描述 | 示例 |
|------|------|------|
| `structured` | 结构化 JSON | `{"action": "turn_left", "target": "main_road"}` |
| `detailed` | 自然语言长描述 | "在前方路口左转进入主干道..." |
| `simple` | 简短指令 | "左转" |

**用户提示类型（`--user_prompt_type`）：**
- `structured`：结构化提问
- `default`："请规划下一步动作"

---

## 🧠 基础模型推理

使用转换后的验证集进行零样本（Zero-shot）推理：

```bash
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"

swift infer \
    --model $MODEL_PATH \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048 \
    --model_type qwen2_5_vl \
    --val_dataset /path/to/omnidrive_val.jsonl \
    --max_batch_size 1 \
    --result_path ./infer_results/omnidrive-bench
```

**输出：**
- 推理结果保存为 JSONL 格式，包含模型回答和元数据
- 可用于后续评估或可视化分析

---

## 🎛 LoRA 微调

使用 OmniDrive 数据对模型进行高效参数微调：

```bash
export CUDA_VISIBLE_DEVICES=0

swift sft \
    --model /path/to/Qwen2.5-VL-3B-Instruct \
    --train_type lora \
    --dataset /path/to/omnidrive_train.jsonl \
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

**关键参数说明：**

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `--train_type` | `lora` | 训练类型（lora/full/...） |
| `--lora_rank` | 8 | LoRA 低秩维度 |
| `--lora_alpha` | 32 | LoRA 缩放系数 |
| `--target_modules` | `all-linear` | 目标模块（Qwen-VL 通常用 all-linear） |
| `--gradient_accumulation_steps` | 16 | 梯度累积步数 |

> ⚠️ **注意**：请将 `--dataset` 替换为您实际预处理好的 `.jsonl` 文件路径。

---

## 🔗 权重合并

微调完成后，将 LoRA 权重合并回基础模型：

```bash
swift export \
    --model /path/to/Qwen2.5-VL-3B-Instruct \
    --adapters /path/to/lora_output \
    --merge_lora true \
    --save_directory /path/to/merged_model
```

**合并后模型特点：**
- 单模型文件，无需额外加载适配器
- 可直接用于推理或进一步微调
- 支持导出为 Hugging Face 格式

---

## 🏗 构建 VLA4AD 模型
我想实现一个vla模型，要求在qwen2.5-vl的vlm上加入head，head解码得到轨迹，然后数据是需要图片，输入的文本；输出的文本答案用以监督。除此之外，然后还有一个轨迹的输出，轨迹训练和语言文本训练两个任务一起；

TODO:
1. 查看ms-swift示例，基础模型如何加载模型，加载数据，加载权重
https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-model.html
https://swift.readthedocs.io/zh-cn/latest/BestPractices/MLLM-Registration.html
特别是看注册model的部分和数据的格式
2. 我理解需要编写一个modeling_qwen2_5_vla.py, 这个是新的模型，以及损失函数也在里面定义了，加入一个轨迹的mseloss；新的模型的mlp是随机初始化的参数;
3. 其次，我需要写一个函数save_custom_model.py, 修改config文件，继承原先qwen的config然后加入几个diy_model中新增的东西（针对vla模型新增的就是动作解码器的维度参数），用一个save函数把这个模型保存下来到本地的model_dir(权重文件，qwen2.5-VL-3B的权重完全继承，只有mlphead部分是需要初始化的)，然后手动把原先除权重文件复制过来到这个文件夹，参考这个https://swift.readthedocs.io/zh-cn/latest/BestPractices/Rapidly-Training-VL-model.html;
4. 写template, custom_dataset.py, 增加轨迹数据加载进去，这部分是用于监督的轨迹数据，继承qwen模型的载入方式;
5. 注册模型文件custom_model.py，ms-swift启动时候把自己的模型正确载入，注册进去，继承qwen模型的载入方式;
6. 写训练sh脚本，把外接自定义的数据和模型脚本注册进去，并且调用正确的数据集;
7. 推理时候模型还应该输出轨迹信息.
### 项目结构

```
vla4ad/
├── custom_model.py          # 模型模板注册与架构定义
├── custom_dataset.py        # 自定义数据集加载器
├── save_custom_model.py     # 模型权重与配置保存
└── modeling_qwen2_5_vla.py  # 扩展 Qwen 模型，添加轨迹解码头
```

**核心组件：**
- `modeling_qwen2_5_vla.py`：在语言模型顶部添加 MLP Head，将文本 token 的隐藏状态映射为 `(x, y, heading)` 轨迹点

### 训练 VLA 模型

```bash
export CUDA_VISIBLE_DEVICES=0
export MAX_PIXELS=1003520  # 控制图像分辨率上限

swift sft \
    --model '/path/to/qwenvla/' \
    --dataset 'omnidrive.jsonl' \
    --remove_unused_columns false \
    --custom_register_path custom_dataset.py custom_model.py \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
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

**优化技巧：**
- `--freeze_vit true`：冻结视觉编码器，加速训练并节省显存
- `MAX_PIXELS`：限制输入图像分辨率，平衡精度与速度

---

## 📈 轨迹数据文件说明

### 获取方式

| 数据集 | 文件 | 下载 |
|--------|------|------|
| 训练集 | `nuscenes_ego_infos_train.pkl` | [Download](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/nuscenes_ego_infos_train.pkl) |
| 验证集 | `nuscenes_ego_infos_val.pkl` | 同上，替换文件名 |

### 数据结构

```python
import pickle

# 加载数据
with open('nuscenes_ego_infos_val.pkl', 'rb') as f:
    infos = pickle.load(f)

sample = infos['data_list'][0]
print(sample.keys())
# dict_keys([
#   'sample_idx', 'token', 'timestamp', 'ego2global', 
#   'images', 'lidar_points', 'instances', 'cam_instances',
#   'can_bus', 'lane_info', 'gt_fut_traj', 'gt_fut_yaw',
#   'gt_fut_traj_mask', 'gt_fut_idx', 'gt_planning',
#   'gt_planning_mask', 'gt_planning_command', 
#   'description', 'location', 'scene_name', 'map_geoms'
# ])
```

### 轨迹数据格式

```python
# 轨迹数据（未来6个时间步，每步包含 x, y, heading）
print(sample['gt_planning'].shape)  # (1, 6, 3)

# 坐标说明：
# - x: 纵向位移（米），前方为正
# - y: 横向位移（米），左侧为正  
# - heading: 航向角变化（弧度）

print(sample['gt_planning'][0])
# array([[-0.023,  0.004, -0.003],
#        [-0.024,  0.006, -0.004],
#        ...
#        [ 0.141,  0.011, -0.009]])
```

### 场景描述示例

```python
print(sample['description'])
# 'Parking lot, barrier, exit parking lot'

print(sample['location'])
# 'boston-seaport' 或 'singapore-onenorth'
```

---

## ⚠️ 注意事项

1. **GPU 显存要求**
   - 推理：≥ 24GB（推荐 RTX 3090/4090 或 A100）
   - 训练：≥ 48GB（推荐 A100 80GB 或多卡训练）

2. **路径配置**
   - 确保 `--img_root` 指向 nuScenes `samples/` 目录
   - 图像路径应包含所有相机子目录（如 `CAM_FRONT`, `CAM_BACK` 等）

3. **小规模测试**
   - 首次运行建议设置 `--convert_ratio 0.01` 验证流程
   - 确认无误后再处理完整数据集

4. **效果评估**
   - 微调前后务必在相同验证集上对比
   - 建议使用 CIDEr、METEOR 等指标量化评估

5. **时序对齐**
   - 确保 `gt_planning` 与图像时间戳严格对齐
   - 检查时序错位可能导致训练不稳定

---

## 🛠 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `CUDA out of memory` | 显存不足 | 降低 `per_device_train_batch_size`，启用 `gradient_checkpointing` |
| 图片加载失败 | 路径错误或文件缺失 | 检查 `--img_root` 是否包含所有相机子目录 |
| 模型加载失败 | 文件不完整 | 确认路径下包含 `config.json`, `pytorch_model.bin`, `tokenizer.json` |
| 数据转换报错 | 格式不兼容 | 添加 `--verbose` 查看具体样本，检查 `--allow_no_images` 是否必要 |
| LoRA 不生效 | 目标模块不匹配 | 确保 `--target_modules` 为 `all-linear`（Qwen-VL 架构） |
| 训练 loss 不下降 | 学习率过高或数据问题 | 尝试降低 `learning_rate` 至 `5e-5`，检查数据格式 |

---

## 📚 参考资料

### 核心论文

- [A Survey on Vision-Language-Action Models for Autonomous Driving](https://arxiv.org/abs/2506.24044) (VLA4AD 综述)
- [OmniDrive: Holistic LLM-Agent for Autonomous Driving](https://arxiv.org/abs/2405.01533)

### 官方文档

- [MS-Swift 官方文档](https://github.com/modelscope/ms-swift)
- [Qwen2.5-VL Hugging Face 文档](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
- [nuScenes 数据集官网](https://www.nuscenes.org/)

### 相关项目
- [Awesome-VLA4AD](https://github.com/SicongJiang/Awesome-VLA4AD)
- [OmniDrive GitHub](https://github.com/NVlabs/OmniDrive)

---

## 📄 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！如有问题，请在 GitHub Issues 中讨论。

---

**Star History**

如果本项目对您有帮助，请给我们一个 ⭐️ Star！
