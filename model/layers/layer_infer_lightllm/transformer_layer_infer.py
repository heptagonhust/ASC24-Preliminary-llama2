import torch
import torch.distributed as dist
from typing import Tuple

from model.infer_state_info import InferStateInfo, LlamaInferStateInfo
from model.layer_weight import BaseLayerWeight, LlamaTransformerLayerWeight
from model.layers.triton_kernel.destindex_copy_kv import destindex_copy_kv
from model.layers.triton_kernel.context_flashattention_nopad import context_attention_fwd
from model.layers.triton_kernel.rotary_emb import rotary_emb_fwd
from model.layers.embedding import RotaryEmbedding
from functools import partial
import torch.nn as nn

class BaseLayerInfer(nn.Module):

    def __init__(self) -> None:
        pass

    def context_forward(self, q, k, v, infer_state: InferStateInfo):
        raise Exception("need to impl")

    def token_forward(self, q, k, v, infer_state: InferStateInfo):
        raise Exception("need to impl")
    
    def forward(self, q, k, v, infer_state: InferStateInfo):
        raise Exception("need to impl")

class TransformerLayerInfer(BaseLayerInfer):
    """
    """
    def __init__(self, layer_num, num_attention_heads, num_key_value_heads, head_dim):
        self.layer_num_ = layer_num
        self.num_attention_heads_ = num_attention_heads
        self.head_dim_ = head_dim
        self.num_key_value_heads_ = num_key_value_heads
        return

class TransformerLayerInferTpl(TransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, num_attention_heads, num_key_value_heads, head_dim):
        super().__init__(layer_num, num_attention_heads, num_key_value_heads, head_dim)
        # need to set by subclass
        self.eps_ = 1e-5 
        self.tp_q_head_num_ = -1
        self.tp_k_head_num_ = -1
        self.tp_v_head_num_ = -1
        self.tp_o_head_num_ = -1
        return
    
    def _pre_cache_kv(self, infer_state:InferStateInfo)->Tuple[torch.Tensor, torch.Tensor]:
        if infer_state.mem_is_contiguous:
            cache_k = infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
            cache_v = infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
        else:
            cache_k = infer_state.key_buffer
            cache_v = infer_state.value_buffer 
        return cache_k, cache_v
    
    def _copy_kv_to_kvcache(self, k, v, cache_k, cache_v):
        '''
        copy k, v we generated in "qkv_proj" to cache_k, cache_v
        '''
        cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_).copy_(k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_).copy_(v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return
    
    def _post_cache_kv(self, cache_k, cache_v, infer_state:InferStateInfo):
        mem_manager = infer_state.mem_manager
        if not infer_state.mem_is_contiguous:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.mem_index, mem_manager)
            return
    
    def _copy_kv_to_mem_cache(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
        return
    
    def _context_attention_kernel(self, q, k, v, infer_state:InferStateInfo, out=None)->torch.Tensor:
        raise Exception("need to impl")
    
    def _token_attention_kernel(self, q, infer_state:InferStateInfo, out=None)->torch.Tensor:
        raise Exception("need to impl")


    def _context_attention(self, q, k, v, infer_state: InferStateInfo):
        cache_k, cache_v = self._pre_cache_kv(infer_state)
        self._copy_kv_to_kvcache(k, v, cache_k, cache_v)
        self._post_cache_kv(cache_k, cache_v, infer_state)
        o = self._context_attention_kernel(q, cache_k, cache_v, infer_state)
        return o

    def _token_attention(self, q, k, v, infer_state: InferStateInfo):
        cache_k, cache_v = self._pre_cache_kv(infer_state)
        self._copy_kv_to_kvcache(k, v, cache_k, cache_v)
        self._post_cache_kv(cache_k, cache_v, infer_state)
        o = self._token_attention_kernel(q, infer_state)
        return o
    
    def context_forward(self, q, k, v, infer_state: InferStateInfo):
        o = self._context_attention(q, k, v, infer_state)
        return o

    def token_forward(self, q, k, v, infer_state: InferStateInfo):
        o = self._token_attention(q, k, v, infer_state)
        return o
    
    def forward(self, q, k, v, infer_state: InferStateInfo):
        o = None
        if infer_state.is_prefill:
            o = self.context_forward(q, k, v, infer_state)
        else:
            o = self.token_forward(q, k, v, infer_state)
        return o
    



class LlamaTransformerLayerInfer(TransformerLayerInferTpl):
    """
    """

    def __init__(self, layer_num, num_attention_heads, num_key_value_heads, head_dim):
        super().__init__(layer_num, num_attention_heads, num_key_value_heads, head_dim)
        self.tp_q_head_num_ = num_attention_heads
        self.tp_k_head_num_ = num_key_value_heads
        self.tp_v_head_num_ = num_key_value_heads
        self.tp_o_head_num_ = self.tp_q_head_num_
        self._bind_func()
        return
    
    def _bind_func(self):
        self._bind_attention()
        return
    
    def _bind_attention(self):
        self._context_attention_kernel = partial(LlamaTransformerLayerInfer._context_attention_kernel, self)
        self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_gqa_flashdecoding, self)
        self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        
    
    def _context_attention_kernel(self, q, k, v, infer_state:LlamaInferStateInfo, out=None)->torch.Tensor:
        o_tensor = torch.empty_like(q) if out is None else out
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor
    
    
    def _copy_kv_to_mem_cache_normal(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
        return
    
   
    def _token_decode_attention_gqa_flashdecoding(self, q, infer_state: LlamaInferStateInfo, out=None):
        # 对 gqa 模型进行推理优化的代码
        from ..triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding
        cache_k = infer_state.mem_manager.key_buffer[self.layer_num_]
        cache_v = infer_state.mem_manager.value_buffer[self.layer_num_]
        return gqa_token_decode_attention_flash_decoding(q, infer_state, self.tp_q_head_num_, self.head_dim_, cache_k, cache_v, out=out)

    