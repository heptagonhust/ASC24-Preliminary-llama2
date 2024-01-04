from typing import List, Optional
import torch
import torch.nn as nn
import math
from model.model_metadata import InputMetadata

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class GroupAttention(nn.Module):
    """MHA/MQA/GQA layer with PagedAttention.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Reshape and store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention using either
        xformers or the PagedAttention custom op.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        max_position_embeddings: int,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.seq_mask = subsequent_mask(max_position_embeddings)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        device = query.device  # 获取 query 张量所在的设备

        batch_size, seq_len, hidden_size = query.shape
        
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        key = key[:, :, :, None, :].expand(batch_size, seq_len, self.num_kv_heads, self.num_queries_per_kv, key.shape[-1])
        key = key.reshape(batch_size, seq_len, -1, key.shape[-1]).transpose(1, 2)

        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        value = value[:, :, :, None, :].expand(batch_size, seq_len, self.num_kv_heads, self.num_queries_per_kv, value.shape[-1])
        value = value.reshape(batch_size, seq_len, -1, value.shape[-1]).transpose(1, 2)
        
        seq_len = query.size(2)
        self.seq_mask = subsequent_mask(seq_len).to(device)
        output, _ = attention(query=query, key=key, value=value, mask=self.seq_mask, dropout=None)

        output = output.transpose(1, 2).contiguous()
        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


def attention(query, key, value, mask=None, dropout=None):
    """attention arguments dimension

    Args:
        query: shape = [batch_size, num_kv_heads * num_queries_per_kv, seq_len, head_size]
        key: [batch_size, num_kv_heads * num_queries_per_kv, seq_len, head_size]
        value: [batch_size, num_kv_heads * num_queries_per_kv, seq_len, head_size]
        mask (_type_, optional): _description_. Defaults to None.
        dropout (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: output of attention
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -5e4)  # 使用一个较小的值替代 -1e9
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0