"""Shared-backbone Pokemon Transformer Transformer model."""

from __future__ import annotations

import torch
from torch import nn


class PokemonTransformerTransformer(nn.Module):
    def __init__(
        self,
        design_dimension: int,
        observation_dimension: int,
        action_count: int,
        task_count: int,
        width: int = 256,
        layers: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.configuration = {
            "design_dimension": design_dimension,
            "observation_dimension": observation_dimension,
            "action_count": action_count,
            "task_count": task_count,
            "width": width,
            "layers": layers,
            "heads": heads,
        }
        self.design_input = nn.Linear(design_dimension, width)
        self.state_input = nn.Linear(observation_dimension, width)
        self.task_embedding = nn.Embedding(task_count, width)
        self.generation_tokens = nn.Parameter(torch.zeros(2, width))
        encoder_layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, batch_first=True, activation="gelu"
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, layers)
        self.control_head = nn.Linear(width, action_count)
        self.design_head = nn.Linear(width, design_dimension)

    def forward(self, design, battle_state, task_ids, action_mask=None):
        task = self.task_embedding(task_ids)
        tokens = torch.stack(
            (self.design_input(design) + task, self.state_input(battle_state) + task), dim=1
        )
        encoded = self.backbone(tokens)
        action_logits = self.control_head(encoded[:, 1])
        if action_mask is not None:
            legal = action_mask.bool()
            no_legal = ~legal.any(dim=-1)
            if no_legal.any():
                legal = legal.clone()
                legal[no_legal, 0] = True
            action_logits = action_logits.masked_fill(~legal, float("-inf"))
        generation_input = self.generation_tokens.unsqueeze(0) + task.unsqueeze(1)
        generation_encoding = self.backbone(generation_input)
        generated_design = torch.sigmoid(self.design_head(generation_encoding[:, 0]))
        return action_logits, generated_design

    def generate_design(self, task_ids):
        task = self.task_embedding(task_ids)
        encoded = self.backbone(self.generation_tokens.unsqueeze(0) + task.unsqueeze(1))
        return torch.sigmoid(self.design_head(encoded[:, 0]))
