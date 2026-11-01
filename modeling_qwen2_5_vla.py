# Copyright (c) Alibaba, Inc. and its affiliates.
"""Qwen2.5-VL with a trajectory regression head.

The language model is trained with the usual next-token loss. The trajectory
head maps the final hidden state to ``(timesteps, features)`` (default 6 x 3:
x, y, heading) and adds an MSE loss. Trajectories are masked per sample so a
mixed batch of VQA-only and trajectory samples is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)


@dataclass
class Qwen2_5_VLACausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    trajectory_predictions: Optional[torch.FloatTensor] = None
    trajectory_loss: Optional[torch.FloatTensor] = None


class Qwen2_5_VLAForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.trajectory_timesteps = int(getattr(config, "trajectory_timesteps", 6))
        self.trajectory_feature_dim = int(
            getattr(config, "trajectory_feature_dim", 3)
        )
        self.trajectory_output_dim = (
            self.trajectory_timesteps * self.trajectory_feature_dim
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.hidden_size, getattr(config, "trajectory_head_hidden_size", 512)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                getattr(config, "trajectory_head_hidden_size", 512),
                self.trajectory_output_dim,
            ),
        )
        self.traj_loss_weight = float(getattr(config, "trajectory_loss_weight", 1.0))
        self._init_trajectory_head()

    def _init_trajectory_head(self) -> None:
        for module in self.trajectory_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _last_token_hidden(self, hidden_states, attention_mask):
        last_hidden = hidden_states[-1]
        if attention_mask is None:
            return last_hidden[:, -1, :]
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch = torch.arange(last_hidden.size(0), device=last_hidden.device)
        return last_hidden[batch, lengths]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        trajectory: Optional[torch.FloatTensor] = None,
        trajectory_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            return_dict=True,
            **kwargs,
        )

        last_token_states = self._last_token_hidden(
            outputs.hidden_states, attention_mask
        )
        predictions = self.trajectory_head(last_token_states).view(
            -1, self.trajectory_timesteps, self.trajectory_feature_dim
        )

        trajectory_loss = None
        total_loss = outputs.loss
        if trajectory is not None:
            target = trajectory.to(
                device=predictions.device, dtype=predictions.dtype
            ).view(-1, self.trajectory_timesteps, self.trajectory_feature_dim)
            if target.shape[0] != predictions.shape[0]:
                raise ValueError(
                    "trajectory batch size does not match the model output: "
                    f"{target.shape[0]} vs {predictions.shape[0]}"
                )
            squared = ((predictions - target) ** 2).mean(dim=(1, 2))
            if trajectory_mask is not None:
                mask = trajectory_mask.to(
                    device=squared.device, dtype=squared.dtype
                ).view(-1)
                trajectory_loss = (squared * mask).sum() / mask.sum().clamp(min=1)
            else:
                trajectory_loss = squared.mean()
            weighted = trajectory_loss * self.traj_loss_weight
            total_loss = (
                weighted if outputs.loss is None else outputs.loss + weighted
            )

        if not return_dict:
            output = (
                total_loss,
                outputs.logits,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                outputs.rope_deltas,
            )
            return output
        return Qwen2_5_VLACausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            trajectory_predictions=predictions,
            trajectory_loss=trajectory_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """Generate text and attach ``outputs.trajectory`` (6 x 3)."""
        generate_kwargs.pop("trajectory", None)
        generate_kwargs["output_hidden_states"] = True
        generate_kwargs["return_dict_in_generate"] = True

        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds if inputs_embeds is not None else inputs,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
        model_inputs = {
            key: value for key, value in model_inputs.items() if value is not None
        }
        output = super().generate(**model_inputs, **generate_kwargs)

        if not output.hidden_states:
            raise RuntimeError("generate() returned no hidden states")
        last_layer_hidden = output.hidden_states[-1][-1]
        last_token_states = last_layer_hidden[:, -1, :]
        output.trajectory = self.trajectory_head(last_token_states).view(
            -1, self.trajectory_timesteps, self.trajectory_feature_dim
        )
        return output
