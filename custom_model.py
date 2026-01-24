from swift.llm import (
    Model, ModelGroup, ModelMeta, register_model, get_model_tokenizer,
    TemplateType, ModelArch  # 添加必要的导入
)
# from template_meta import TemplateMeta
# from diy_template import QwenVLATemplateMeta, Qwen2_5VLATemplate
# from .diy_template
from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_vl
# from swift.utils import require_version

"""
Qwen2.5-VLA Template 完整实现
继承自 Qwen2_5VLTemplate，仅新增 trajectory 字段处理
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from swift.llm import to_float_dtype
import sys
# sys.path.append("./ms-swift-main/swift/llm/template")
from swift.llm import register_template, TemplateMeta
# from template_inputs import StdTemplateInputs
from swift.llm.template.template.qwen import Qwen2_5VLTemplate, Qwen2_5TemplateMeta  # 关键：继承自官方的 Qwen2_5VLTemplate

@dataclass
class QwenVLATemplateMeta(Qwen2_5TemplateMeta):
    """Template 元信息"""
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
    auto_add_bos: bool = False
    stop_words: List[str] = None

    def __post_init__(self):
        if self.stop_words is None:
            self.stop_words = ['<|im_end|>', '<|endoftext|>']

class Qwen2_5VLATemplate(Qwen2_5VLTemplate):
    """
    VLA 专用 Template，在 Qwen2.5-VL 基础上增加 trajectory 字段处理
    """
    version = 'v2_5'
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    
    # 关键：定义需要从数据中提取的额外字段
    additional_input_fields = ['trajectory']  # ← 新增
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 确保 trajectory 维度与 config 一致
        self.traj_dim = getattr(self.config, 'traj_dim', 7)  # 默认 7-DoF
        # 轨迹时间步和维度配置
        self.traj_timesteps = 6  # 6个时间步
        self.traj_feat_dim = 2   # xy坐标
    
    def _generate_dummy_trajectory(self, batch_size=1):
        """
        生成虚拟轨迹数据 [batch_size, 6, 2]
        """
        # 使用零填充生成虚拟轨迹
        return torch.zeros(batch_size, self.traj_timesteps, self.traj_feat_dim, dtype=torch.float32)
    
    def _encode(self, inputs) -> Dict[str, Any]:
        """
        核心编码函数：处理多模态输入 + trajectory
        """
        # 1. 先调用父类方法处理图像/视频/文本，得到基础编码
        encoded = super()._encode(inputs)
        
        # 2. 提取 trajectory 数据（如果存在）
        trajectory = getattr(inputs, 'trajectory', None)
        
        if trajectory is not None:
            # 确保是 tensor 并正确形状
            if not isinstance(trajectory, torch.Tensor):
                trajectory = torch.tensor(trajectory, dtype=torch.float32)
            
            # 验证维度 [batch_size, 6, 2]
            if trajectory.dim() == 2:
                # 单样本：添加 batch 维度 [6,2] -> [1,6,2]
                trajectory = trajectory.unsqueeze(0)
            elif trajectory.dim() == 3:
                # 已经是批处理格式 [batch_size, 6, 2]
                pass
            else:
                raise ValueError(f"Trajectory dimension error: expected 2D or 3D, got {trajectory.dim()}D")
            
            # 验证具体形状
            if trajectory.shape[1:] != (self.traj_timesteps, self.traj_feat_dim):
                raise ValueError(
                    f"Trajectory shape mismatch: expected (..., {self.traj_timesteps}, {self.traj_feat_dim}), "
                    f"got {trajectory.shape}"
                )
            
            encoded['trajectory'] = trajectory
        else:
            #
            batch_size = 1
            encoded['trajectory'] = self._generate_dummy_trajectory(batch_size)
            encoded['is_dummy_trajectory'] = True
        
        return encoded
    
    def _data_collator(self, batch: List[Dict[str, Any]], *, 
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        批处理函数：处理变长序列 + trajectory 批处理
        """
        # 1. 先调用父类方法处理文本和视觉输入的 padding
        res = super()._data_collator(batch, padding_to=padding_to)
        
        # 2. 处理 trajectory 批处理
        trajectories = []
        has_real_trajectory = False
        
        for b in batch:
            traj = b.get('trajectory')
            if traj is not None:
                if not isinstance(traj, torch.Tensor):
                    traj = torch.tensor(traj, dtype=torch.float32)
                trajectories.append(traj)
                if not b.get('is_dummy_trajectory', False):
                    has_real_trajectory = True
            else:
                # 关键修复：每个样本生成batch_size=1的虚拟数据
                dummy_traj = self._generate_dummy_trajectory(batch_size=1)
                trajectories.append(dummy_traj)
        
        # 堆叠成 batch tensor [batch_size, 6, 2]
        if trajectories:
            # 直接堆叠，确保batch_size一致
            res['trajectory'] = torch.cat(trajectories, dim=0)
            res['has_real_trajectory'] = has_real_trajectory
        
        return res
    
    def prepare_model_inputs(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备模型前向输入：确保 trajectory 传递给 model.forward()
        """
        model_inputs = super().prepare_model_inputs(model, inputs)
        
        if 'trajectory' in inputs:
            model_inputs['trajectory'] = inputs['trajectory']
            if 'is_dummy_trajectory' in inputs:
                model_inputs['is_dummy_trajectory'] = inputs['is_dummy_trajectory']
            if 'has_real_trajectory' in inputs:
                model_inputs['has_real_trajectory'] = inputs['has_real_trajectory']
        
        return model_inputs
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        
        print('inputs:', inputs.keys())
        return inputs

    def get_trajectory_info(self) -> Dict[str, Any]:
        """
        获取轨迹配置信息（用于调试和验证）
        """
        return {
            'timesteps': self.traj_timesteps,
            'feature_dim': self.traj_feat_dim,
            'total_dim': self.traj_timesteps * self.traj_feat_dim
        }

# # 注册 template
# register_template(QwenVLATemplateMeta(
#     "qwen2.5_vla",  # 使用 qwen2.5_vl 类型
#     template_cls=Qwen2_5VLATemplate,
#     prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
#     prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
#     chat_sep=['<|im_end|>\n'],
#     suffix=['<|im_end|>'],
#     system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
#     stop_words=['<|im_end|>', '<|endoftext|>'],
#     replace_system=True,
# ))

def get_model_tokenizer_qwen2_5_vla(model_dir, *args, **kwargs):
    """加载Qwen2.5-VLA模型和分词器"""
    # 导入自定义模型类
    import sys
    sys.path.insert(0, "/high_perf_store2/tfl-vepfs/qiankangan-vla/BaseModel/qwenvla/")
    from modeling_qwen2_5_vl_vla import Qwen2_5_VLAForConditionalGeneration
    
    # 检查依赖版本（根据您的模型需要调整版本）
    # require_version('transformers>=4.50')
    # require_version('qwen_vl_utils>=0.0.14')  # 如果使用多模态工具包
    
    # 设置自动模型类
    kwargs['automodel_class'] = kwargs.get('automodel_class') or Qwen2_5_VLAForConditionalGeneration
    kwargs['_check_qwen_vl_utils'] = False
    
    # 使用Qwen2-VL的基础加载函数
    model, processor = get_model_tokenizer_qwen2_vl(model_dir, *args, **kwargs)
    
    # 应用自定义补丁（如果有）
    # patch_Qwen2_5_VLA_dtype()  # 如果需要数据类型补丁
    
    # 兼容性处理（如果需要）
    if model is not None:
        # _compat_qwen2_5_vla_mixed_data(model.model, processor, True)  # 如果需要混合数据兼容性处理
        pass
        
    return model, processor

# 注册 template
# register_template(QwenVLATemplateMeta(
#     "qwen2.5_vla",  # 使用 qwen2.5_vl 类型
#     template_cls=Qwen2_5VLATemplate,
#     prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
#     prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
#     chat_sep=['<|im_end|>\n'],
#     suffix=['<|im_end|>'],
#     system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
#     stop_words=['<|im_end|>', '<|endoftext|>'],
#     # replace_system=True,
# ))

register_template(Qwen2_5TemplateMeta("qwen2_5_vla", template_cls=Qwen2_5VLATemplate, default_system=None))

# 注册模型
register_model(
    ModelMeta(
        model_type='qwen2_5_vla',  # 添加参数名
        model_groups=[
            ModelGroup([
                # 修正模型路径，使用您实际的模型路径
                Model(
                    model_path='/high_perf_store2/tfl-vepfs/qiankangan-vla/BaseModel/qwenvla/diy_model',
                    hf_model_id='curryqka/Qwen2.5-VLA-Custom'  # 可选：HuggingFace模型ID
                ),
            ]),
        ],
        template="qwen2_5_vla",  # 使用正确的模板类型
        get_function=get_model_tokenizer_qwen2_5_vla,
        model_arch=ModelArch.qwen2_vl,  # 使用正确的模型架构
        architectures=['Qwen2_5_VLAForConditionalGeneration'],  # 修正模型类名
        requires=[
            'transformers>=4.50', 
            'qwen_vl_utils>=0.0.14',  # 根据实际需要调整
            'torch>=2.0.0'  # 添加torch依赖
        ],
        tags=['vision', 'language', 'vla', 'pointcloud'],  # 修正标签
        is_multimodal=True,  # 添加多模态标记
        ignore_patterns=[],  # 明确指定忽略模式
    ))

# 测试代码
if __name__ == '__main__':
    pass
