# quick-start-VLA4AD
A hands-on, beginner-friendly repository to quickly get started with Vision-Language-Action (VLA)​ models for autonomous driving using the MS-Swift​ framework. This project provides practical code examples, configuration templates, and step-by-step tutorials to help you build, fine-tune, and evaluate VLA-based AD models with minimal setup.
VLA4AD: 基于MS-Swift的自动驾驶VLA模型实战指南

这是一个快速上手用ms-swift完成自动驾驶Vision-Language-Action (VLA)模型实战的仓库。本教程将指导你从数据准备、模型下载、训练到推理的完整流程。

环境搭建

1. 安装MS-Swift

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .


模型下载

2. 下载Qwen2.5-VL-3B-Instruct模型

• 从Hugging Face下载模型：https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

• 使用官方推理脚本测试模型功能

• 参考资源：

  • 模型地址: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

  • 官方文档: https://hugging-face.cn/docs/transformers/model_doc/qwen2_5_vl

  • 示例代码: https://github.com/dongyaolin/qwen2.5vl

数据下载

3. 下载OmniDrive数据集

• 发布页面: https://github.com/NVlabs/OmniDrive/releases/tag/v1.0

• VQA数据文件: https://github.com/NVlabs/OmniDrive/releases/download/v1.0/data_nusc.zip

数据准备

4.1 准备OmniDrive VQA数据

将原始OmniDrive数据转换为MS-Swift格式：
python convert_omnidrive_to_msswift.py \
    --input_file /path/to/omnidrive_train.jsonl \
    --output_dir ./preprocessed_data/OmniDrive \
    --data_split train \
    --img_root /lustre/MLM_evaluator/data/omnidrive \
    --check_images \
    --verbose


脚本参数说明：
• --input_file: 原始jsonl文件路径（必需）

• --output_dir: 输出目录，默认./preprocessed_data/OmniDrive

• --data_split: 数据分割，可选train/val/test，默认train

• --convert_ratio: 数据转换比例，0.0-1.0，默认1.0

• --img_root: 图片根目录，默认/lustre/MLM_evaluator/data/omnidrive

• --output_name: 输出文件名（自动生成时忽略）

• --system_prompt_type: 系统提示类型，可选default/detailed/simple

• --camera_order: 自定义相机顺序，逗号分隔

• --custom_descriptions: 自定义相机描述，格式CAM1:desc1,CAM2:desc2

• --image_first_question: 在问题开头添加图片标记

• --check_images: 检查图片文件是否存在

• --allow_no_images: 允许没有图片的样本

• --verbose: 详细输出

• --seed: 随机种子，默认42

输出文件：
• 转换后的jsonl文件，格式为MS-Swift兼容格式

• 转换统计文件conversion_stats_{split}.txt

4.2 准备轨迹数据（数值型）

从nuScenes轨迹数据转换为MS-Swift格式：
python convert_nuscenes_to_msswift.py \
    --input_file /path/to/nuscenes_ego_infos_val.pkl \
    --output_dir ./preprocessed_data/nuscenes_trajectory \
    --data_split val \
    --img_root /path/to/nuscenes/images \
    --trajectory_format string \
    --use_scene_description \
    --check_images


脚本参数说明：
• --input_file: 输入pkl文件路径（必需）

• --output_dir: 输出目录，默认./preprocessed_data/nuscenes_trajectory

• --data_split: 数据分割，可选train/val/test，默认train

• --convert_ratio: 数据转换比例，0.0-1.0，默认1.0

• --img_root: 图片根目录

• --output_name: 输出文件名（自动生成时忽略）

• --system_prompt_type: 系统提示类型，可选default/detailed/planning_focused

• --camera_order: 自定义相机顺序

• --custom_descriptions: 自定义相机描述

• --image_first_question: 在问题开头添加图片标记

• --check_images: 检查图片文件是否存在

• --allow_no_images: 允许没有图片的样本

• --use_scene_description: 在提示中包含场景描述

• --trajectory_format: 轨迹输出格式，可选list/string/detailed/compact

• --verbose: 详细输出

• --seed: 随机种子，默认42

4.3 准备轨迹数据（语义导航型）

生成高层级语义导航指令：
python convert_nuscenes_to_msswift_navigation.py \
    --input_file /path/to/nuscenes_ego_infos_val.pkl \
    --output_dir ./preprocessed_data/nuscenes_navigation \
    --data_split val \
    --img_root /path/to/nuscenes/images \
    --navigation_format structured \
    --user_prompt_type structured \
    --check_images


脚本参数说明：
• --input_file: 输入pkl文件路径（必需）

• --output_dir: 输出目录，默认./preprocessed_data/nuscenes_navigation

• --data_split: 数据分割，可选train/val/test，默认train

• --convert_ratio: 数据转换比例，0.0-1.0，默认1.0

• --img_root: 图片根目录

• --output_name: 输出文件名（自动生成时忽略）

• --system_prompt_type: 系统提示类型，可选default/detailed/instruction_focused

• --user_prompt_type: 用户提示类型，可选default/structured/simple

• --camera_order: 自定义相机顺序

• --custom_descriptions: 自定义相机描述

• --image_first_question: 在问题开头添加图片标记

• --check_images: 检查图片文件是否存在

• --allow_no_images: 允许没有图片的样本

• --navigation_format: 导航指令格式，可选structured/detailed/simple

• --include_scene_in_user: 当没有图片时在用户提示中包含场景描述

• --verbose: 详细输出

• --seed: 随机种子，默认42

基础模型推理

使用转换后的验证集数据进行推理：
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


LoRA微调

使用OmniDrive数据进行LoRA微调：
export CUDA_VISIBLE_DEVICES=0

swift sft \
    --model /path/to/Qwen2.5-VL-3B-Instruct \
    --train_type lora \
    --dataset swift/stsb \
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
    --output_dir /path/to/output_folder \
    --dataset_num_proc 4


权重合并

微调完成后，合并LoRA权重：
./ms-swift-main/examples/export/merge_lora.sh


构建VLA4AD模型

文件结构


vla4ad/
├── custom_model.py          # 定义template和模型注册载入
├── custom_dataset.py        # 定义数据加载的内容
├── save_custom_model.py     # 保存新的模型权重和config文件
└── modeling_qwen2_5_vla.py  # 修改qwen模型加入mlp head解码轨迹


训练VLA模型

export CUDA_VISIBLE_DEVICES=0
export MAX_PIXELS=1003520

swift sft \
    --model '/path/to/qwenvla/' \
    --dataset 'omnidrive.jsonl' \
    --remove_unused_columns false \
    --custom_register_path custom_dataset.py \
                            custom_model.py \
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
    --output_dir /path/to/output_folder \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4


轨迹数据文件

轨迹数据可从以下位置获取：
• 轨迹数据文件: https://github.com/NVlabs/OmniDrive/releases/download/v1.0/nuscenes_ego_infos_train.pkl

数据结构示例：
# nuscenes_ego_infos_val.pkl
infos['data_list'][0].keys()
# dict_keys(['sample_idx', 'token', 'timestamp', 'ego2global', 'images', 'lidar_points', 
#           'instances', 'cam_instances', 'can_bus', 'lane_info', 'gt_fut_traj', 
#           'gt_fut_yaw', 'gt_fut_traj_mask', 'gt_fut_idx', 'gt_planning', 
#           'gt_planning_mask', 'gt_planning_command', 'description', 'location', 
#           'scene_name', 'map_geoms'])

# 轨迹数据
infos['data_list'][0]['gt_planning']
# array([[[-0.02338394,  0.00395733, -0.00292365],
#         [-0.02435618,  0.00648969, -0.00425898],
#         [-0.02409667,  0.00829917, -0.00493517],
#         [-0.02323897,  0.00956105, -0.00510122],
#         [-0.01144369,  0.010625  , -0.00544585],
#         [ 0.14093415,  0.01071738, -0.00940987]]])

# 轨迹形状: (1, 6, 3) - 最后三个维度是(x, y, heading)

# 场景描述
infos['data_list'][0]['description']
# 'Parking lot, barrier, exit parking lot'


注意事项

1. 确保有足够的GPU内存进行训练
2. 图片路径需要正确配置
3. 建议先在小规模数据上进行测试
4. 微调前后对比推理结果以评估效果
5. 轨迹数据转换时注意检查图片是否存在

故障排除

• 如果遇到"CUDA out of memory"错误，尝试减少per_device_train_batch_size

• 如果图片加载失败，检查--img_root参数设置

• 如果模型加载失败，检查模型路径和格式

• 如果数据转换出错，使用--verbose参数查看详细错误信息

参考资料

• https://swift.readthedocs.io/

• https://hugging-face.cn/docs/transformers/model_doc/qwen2_5_vl

• https://github.com/NVlabs/OmniDrive

• https://www.nuscenes.org/
