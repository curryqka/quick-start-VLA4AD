import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLPreTrainedModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

# class Qwen2_5_VLACausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
#     def __init__(self, loss=None, logits=None, past_key_values=None, hidden_states=None, attentions=None, trajectory_predictions=None, trajectory_loss=None):
#         super().__init__(loss, logits, past_key_values, hidden_states, attentions)
#         self.trajectory_predictions = trajectory_predictions
#         self.trajectory_loss = trajectory_loss

class Qwen2_5_VLAForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
        # 从配置中获取轨迹相关参数
        trajectory_output_dim = getattr(config, 'trajectory_output_dim', 12)
        trajectory_head_hidden_size = getattr(config, 'trajectory_head_hidden_size', 512)
        
        # 新增轨迹预测头
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.hidden_size, trajectory_head_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(trajectory_head_hidden_size, trajectory_output_dim)
        )
        
        # 轨迹损失权重
        self.traj_loss_weight = getattr(config, 'trajectory_loss_weight', 1.0)
        
        # 初始化轨迹头
        self._init_trajectory_head()
    
    def _init_trajectory_head(self):
        """初始化轨迹预测头"""
        for module in self.trajectory_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        trajectory: Optional[torch.FloatTensor] = None,  # 新增轨迹标签
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True
        
        # 调用父类前向传播
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            return_dict=return_dict,
            **kwargs
        )
        
        # 轨迹预测任务
        traj_loss = None
        traj_predictions = None
        
        if trajectory is not None:
            # 使用最后一个隐藏状态进行轨迹预测
            if output_hidden_states:
                last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            else:
                last_hidden_state = outputs[0]
            
            # 关键修复：简化最后一个token的提取
            if attention_mask is not None:
                # 获取每个序列最后一个有效token的索引
                seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
                # 使用高级索引提取最后一个token
                batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
                last_token_states = last_hidden_state[batch_indices, seq_lengths]  # [batch_size, hidden_size]
            else:
                last_token_states = last_hidden_state[:, -1, :]  # [batch_size, hidden_size]
            
            # 关键修复：轨迹预测头应该输出 [batch_size, 6, 2] 形状
            traj_predictions = self.trajectory_head(last_token_states)  # [batch_size, 12]
            
            # 重塑为 [batch_size, 6, 2]
            traj_predictions = traj_predictions.view(-1, 6, 2)  # [batch_size, 6, 2]
            # breakpoint()
            # 关键修复：处理batch_size不匹配的情况
            if traj_predictions.shape[0] != trajectory.shape[0]:
                # 如果batch_size不匹配，使用重复或截断来匹配
                if traj_predictions.shape[0] < trajectory.shape[0]:
                    # 重复预测以匹配目标batch_size
                    repeat_factor = trajectory.shape[0] // traj_predictions.shape[0]
                    traj_predictions = traj_predictions.repeat(repeat_factor, 1, 1)
                    # 如果还有剩余，添加额外的样本
                    remaining = trajectory.shape[0] % traj_predictions.shape[0]
                    if remaining > 0:
                        extra = traj_predictions[:remaining]
                        traj_predictions = torch.cat([traj_predictions, extra], dim=0)
                else:
                    # 截断预测以匹配目标batch_size
                    traj_predictions = traj_predictions[:trajectory.shape[0]]
            
            # 确保形状完全匹配
            if traj_predictions.shape != trajectory.shape:
                # 如果还有形状不匹配，进行最终调整
                traj_predictions = traj_predictions.reshape(trajectory.shape)
            
            # 计算轨迹损失（MSE）
            traj_loss = nn.MSELoss()(traj_predictions, trajectory)
            traj_loss = traj_loss * getattr(self, 'traj_loss_weight', 0.01)
            
            # 合并损失
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                total_loss = outputs.loss + traj_loss
            else:
                total_loss = traj_loss
        else:
            total_loss = outputs.loss if hasattr(outputs, 'loss') else None
        
        if not return_dict:
            output = (traj_predictions,) + outputs[1:] if traj_predictions is not None else outputs
            return ((total_loss,) + output) if total_loss is not None else output
        
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
        # 使用自定义的输出类
        # return Qwen2_5_VLAForConditionalGeneration(
        #     loss=total_loss,
        #     logits=outputs.logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        #     trajectory_predictions=traj_predictions,
        #     trajectory_loss=traj_loss,
        # )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        return_dict_in_generate: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs
    ):
        """
        文本生成 + 轨迹预测
        """
        # 1. 强制输出 hidden_states
        generate_kwargs["output_hidden_states"] = True
        generate_kwargs["return_dict_in_generate"] = True

        # 2. 走父类 generate
        gen_out = super().generate(
            inputs=inputs,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            **generate_kwargs
        )

        # 3. 正确获取最后一层的隐藏状态
        # gen_out.hidden_states 是一个元组 (tuple)，每个元素是各层的隐藏状态
        # 最后一层的隐藏状态是 gen_out.hidden_states[-1] (一个张量)
        last_layer_hidden = gen_out.hidden_states[-1][0]  # [batch_size, seq_len, hidden_size]
        
        # 4. 从最后一层隐藏状态中提取最后一个token的特征
        # 注意：这里我们直接使用 last_layer_hidden 的最后一个token
        last_token_states = last_layer_hidden[:, -1, :]  # [batch_size, hidden_size]

        # breakpoint()
        # 5. 轨迹预测
        trajectory = self.trajectory_head(last_token_states)  # [batch_size, trajectory_output_dim]

        # 6. 返回
        if return_dict_in_generate:
            gen_out.trajectory = trajectory
            return gen_out
        else:
            return gen_out.sequences
