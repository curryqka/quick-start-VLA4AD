import os
import json
import shutil
import torch
import logging
from typing import Any, Dict, Optional, Union
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_utils import no_init_weights
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling_qwen2_5_vl_vla import Qwen2_5_VLAForConditionalGeneration
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from modelscope import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_and_save_qwen_vla_model(
    base_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    save_path="diy/custom_model",
    trajectory_output_dim=7,
    trajectory_head_hidden_size=512,
    trajectory_loss_weight=0.5,
    torch_dtype=None
):
    """
    初始化并保存Qwen-VLA模型 - 修复meta tensor问题
    
    参数:
        base_model_name: 基础Qwen2.5-VL模型名称或路径
        save_path: 保存自定义模型的路径
        trajectory_output_dim: 轨迹输出维度
        trajectory_head_hidden_size: 轨迹头隐藏层大小
        trajectory_loss_weight: 轨迹损失权重
        torch_dtype: 模型数据类型
    """
    # 1. 设置正确的torch_dtype
    if torch_dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    
    logger.info(f"使用数据类型: {torch_dtype}")
    
    # 2. 下载或获取基础模型
    logger.info(f"正在下载基础模型: {base_model_name}")
    if os.path.exists(base_model_name) and os.path.isdir(base_model_name):
        base_model_path = base_model_name
    else:
        base_model_path = snapshot_download(base_model_name)
    
    # 3. 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 4. 加载基础模型配置
    logger.info("加载基础模型配置...")
    base_config = AutoConfig.from_pretrained(base_model_path)
    
    # 5. 创建自定义配置
    logger.info("创建自定义配置...")
    vla_config = base_config
    
    # 添加轨迹相关的配置参数
    vla_config.trajectory_output_dim = trajectory_output_dim
    vla_config.trajectory_head_hidden_size = trajectory_head_hidden_size
    vla_config.trajectory_loss_weight = trajectory_loss_weight
    vla_config.architectures = ["Qwen2_5_VLAForConditionalGeneration"]
    vla_config.model_type = "qwen2_5_vl"  # 保持原始model_type
    
    # 6. 保存配置到临时位置 (供后续使用)
    temp_config_path = os.path.join(save_path, "temp_config")
    os.makedirs(temp_config_path, exist_ok=True)
    vla_config.save_pretrained(temp_config_path)
    
    # 7. 核心修复：直接在CPU上创建模型，避免meta tensor
    logger.info("创建Qwen-VLA模型架构... (在CPU上初始化)")
    
    # 关键修复1: 不再使用init_empty_weights，而是直接在CPU上初始化
    # 使用no_init_weights上下文管理器避免初始化开销
    with no_init_weights():
        vla_model = Qwen2_5_VLAForConditionalGeneration(vla_config)
    
    # 将模型移动到CPU
    vla_model = vla_model.cpu()
    
    # 8. 加载基础模型权重 (在CPU上)
    logger.info("加载基础模型权重... (在CPU上)")
    
    # 关键修复2: 在CPU上加载基础模型，避免设备映射问题
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,  # 使用float32确保兼容性
        device_map="cpu",  # 强制在CPU上加载
        trust_remote_code=True
    )
    
    # 9. 手动复制权重 - 安全且可控的方法
    logger.info("手动复制基础模型权重到新模型...")
    
    base_state_dict = base_model.state_dict()
    vla_state_dict = vla_model.state_dict()
    
    # 创建新的状态字典
    new_state_dict = {}
    
    # 复制所有匹配的参数
    matched_keys = 0
    for key in vla_state_dict.keys():
        if key in base_state_dict:
            new_state_dict[key] = base_state_dict[key]
            matched_keys += 1
            logger.debug(f"复制权重: {key}")
        else:
            # 保留VLA特定参数 (如trajectory_head) 的随机初始化
            new_state_dict[key] = vla_state_dict[key]
            logger.debug(f"保留初始化: {key} (VLA特定参数)")
    
    logger.info(f"匹配的参数数量: {matched_keys}/{len(vla_state_dict)}")
    
    # 加载状态字典 (不严格，允许未匹配的键)
    vla_model.load_state_dict(new_state_dict, strict=False)
    
    # 10. 清理基础模型以释放内存
    del base_model, base_state_dict, vla_state_dict, new_state_dict
    torch.cuda.empty_cache()
    
    # 11. 验证没有meta tensor
    logger.info("验证模型中没有meta tensor...")
    meta_tensors = []
    for name, param in vla_model.named_parameters():
        if param.device.type == 'meta':
            meta_tensors.append(name)
    
    if meta_tensors:
        logger.error(f"发现 {len(meta_tensors)} 个meta tensor! 需要修复")
        logger.error(f"Meta tensors: {meta_tensors[:5]}...")
        raise RuntimeError("模型中存在meta tensor，无法继续")
    else:
        logger.info("✓ 模型中没有meta tensor")
    
    # 12. 验证轨迹头是否正确初始化
    if hasattr(vla_model, 'trajectory_head'):
        logger.info("✓ 轨迹头存在")
        # 检查轨迹头参数
        trajectory_params = {name: param.shape for name, param in vla_model.trajectory_head.named_parameters()}
        logger.info(f"轨迹头参数: {trajectory_params}")
    else:
        logger.warning("✗ 轨迹头不存在!")
    
    # 13. 保存模型 - 先在CPU上保存
    logger.info("保存模型权重...")
    
    # 关键修复3: 保持模型在CPU上保存
    vla_model.eval()
    
    # 保存模型 (在CPU上)
    vla_model.save_pretrained(
        save_path,
        max_shard_size="4GB",
        safe_serialization=True
    )
    
    # 14. 保存tokenizer和配置文件
    logger.info("保存tokenizer和配置文件...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    
    # 15. 复制其他必要文件
    logger.info("复制其他必要文件...")
    for file_name in os.listdir(base_model_path):
        # 跳过模型权重文件
        if file_name.endswith(('.bin', '.safetensors', '.pt')) or 'config.json' in file_name or 'safetensors' in file_name:
            continue
        
        src_path = os.path.join(base_model_path, file_name)
        dst_path = os.path.join(save_path, file_name)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)
    
    # # 16. 检查配置文件是否正确保存
    # config_check_path = os.path.join(save_path, "config.json")
    # if os.path.exists(config_check_path):
    #     logger.info("✓ 配置文件已正确保存")
    # else:
    #     logger.warning("✗ 配置文件未找到，重新保存")
    #     vla_config.save_pretrained(save_path)
    
    # 17. 验证保存的模型可以加载
    logger.info("验证保存的模型可以正确加载...")
    try:
        # 清理当前模型以释放内存
        del vla_model
        torch.cuda.empty_cache()
        
        # 尝试加载保存的模型
        test_model = Qwen2_5_VLAForConditionalGeneration.from_pretrained(
            save_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"✓ 模型成功加载到设备: {next(test_model.parameters()).device}")
        
        # 简单验证前向传播
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # 创建一个简单的测试输入
        test_input_ids = torch.randint(0, 1000, (1, 10)).to(device)
        test_attention_mask = torch.ones_like(test_input_ids).to(device)
        
        with torch.no_grad():
            outputs = test_model(
                input_ids=test_input_ids,
                attention_mask=test_attention_mask,
                use_cache=False
            )
        
        logger.info(f"✓ 前向传播成功! 输出logits形状: {outputs.logits.shape}")
        if hasattr(outputs, 'trajectory_logits'):
            logger.info(f"✓ 轨迹预测成功! 形状: {outputs.trajectory_logits.shape}")
        
        # 清理
        del test_model, outputs
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"✗ 模型验证失败: {str(e)}", exc_info=True)
        # 不抛出异常，继续执行
    
    # 18. 创建模型卡片
    model_card = {
        "model_type": "qwen2_5_vla",
        "base_model": base_model_name,
        "trajectory_output_dim": trajectory_output_dim,
        "trajectory_head_hidden_size": trajectory_head_hidden_size,
        "trajectory_loss_weight": trajectory_loss_weight,
        "torch_dtype": str(torch_dtype),
        "description": "Qwen-VLA: Vision-Language-Action model based on Qwen2.5-VL with trajectory prediction capability"
    }
    
    with open(os.path.join(save_path, "model_card.json"), "w") as f:
        json.dump(model_card, f, indent=2)
    
    # 19. 清理临时文件
    if os.path.exists(temp_config_path):
        shutil.rmtree(temp_config_path)
    
    logger.info(f"✓ Qwen-VLA模型已成功保存到 {save_path}")
    logger.info(f"轨迹维度: {trajectory_output_dim}, 轨迹头隐藏层大小: {trajectory_head_hidden_size}")
    logger.info(f"轨迹损失权重: {trajectory_loss_weight}")
    logger.info(f"模型数据类型: {torch_dtype}")
    
    return save_path

if __name__ == "__main__":
    # 示例用法
    init_and_save_qwen_vla_model(
        base_model_name="/high_perf_store2/tfl-vepfs/qiankangan-vla/BaseModel/model/qwen_vla/Qwen2.5-VL-3B-Instruct",
        save_path="/high_perf_store2/tfl-vepfs/qiankangan-vla/BaseModel/qwenvla/diy_model",
        trajectory_output_dim=12,  # 7D pose (3 translation + 4 rotation + 其他)
        trajectory_head_hidden_size=512,
        trajectory_loss_weight=0.5,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    )
