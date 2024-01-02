# coding=utf-8
# Adapted from
# Copyright 2024 Hust-heptagon team
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings."""
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm._C import ops


#! RotaryEmbedding for a single head
#! in llama, is_neox_style=True
class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: int,
    ) -> None:
        super().__init__()
        self.head_size = head_dim
        self.rotary_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        cache = self._compute_cos_sin_cache()
        cache = cache.to(torch.get_default_dtype())
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = 1.0 / (
            self.base**(torch.arange(0, self.rotary_dim, 2, 
                                     dtype=torch.float, 
                                     device="cuda") 
                        / self.rotary_dim))

        t = torch.arange(self.max_position_embeddings,
                         dtype=torch.float,
                         device="cuda")
        #! tensor product of t & inv_freq
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

# vllm use ops.rotary_embedding to calculate in-place
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 1, 2).unsqueeze(-2)

        query = (query * cos + self.rotate(query) * sin).flatten(-2)
        key = (key * cos + self.rotate(key) * sin).flatten(-2)

        return query, key

# keep for later optimization
    def _forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ops.rotary_embedding() is an in-place operation that
        # updates the query and key tensors.
        ops.rotary_embedding(positions, query, key, self.head_size,
                             self.cos_sin_cache, True)
        return query, key
