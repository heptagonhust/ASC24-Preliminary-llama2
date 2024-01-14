# coding=utf-8
# Adapted from
# Copyright 2024 Hust-heptagon team
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
#
# This code is based on vllm library, EleutherAI's GPT-NeoX library and the GPT-NeoX
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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, List, Optional, Tuple
import einops

import torch
import os
import json
from torch import nn
from transformers import LlamaConfig
import torch.distributed as dist

from model.model_metadata import InputMetadata
from model.layers.activate import SiluAndMul
from model.layers.attention import GroupAttention
from model.layers.layernorm import RMSNorm
from model.layers.linear import (LinearMethodBase, 
                          MergedColumnParallelLinear,
                          QKVParallelLinear,
                          RowParallelLinear)
from model.layers.embedding import RotaryEmbedding
from sampler.sampler import Sampler
from model.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from model.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from sampler.sampling_metadata import SamplingMetadata
from model.weight_utils import (default_weight_loader, hf_model_weights_iterator)
from sequence.sequence import SamplerOutput
from model.infer_state_info import InferStateInfo, LlamaInferStateInfo
from model.layers.layer_infer_lightllm.transformer_layer_infer import LlamaTransformerLayerInfer
from model.layers.triton_kernel import destindex_copy_kv
from model.parallel_utils.parallel_state import get_tensor_model_parallel_rank
from utils.utils import repair_config,init_req_to_token_indexes
from manager.memory_manager import MemoryManager
from manager.request_manager import RequestManager
from model.layers.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req


from utils.log_utils import init_logger
logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


#! 64 heads, 8 kv heads in llama 70B
class LlamaAttention(nn.Module):

    def __init__(
        self,
        layer_num: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        #! heads per tensor paralleled GPU
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        #! kv heads per tensor paralleled GPU
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        #! dimension of token vector of each head
        self.head_dim = hidden_size // self.total_num_heads
        #! dimension of sum of query vectors of each tp GPU
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        #! project raw token vec to qkv vector
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        self.attn = LlamaTransformerLayerInfer(
            self.layer_num,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        infer_state: LlamaInferStateInfo,
    ) -> torch.Tensor:
        """Attention Layer forward pass.

        Args:
            positions: shape = [batch_size, seq_len]
            hidden_states: shape = [batch_size, seq_len, hidden_size]
            infer_state: LlamaInferStateInfo 

        Returns:
            torch.Tensor: _description_
        """
        #! qkv: [batch_size, seq_len, tp_q_size + 2 * tp_kv_size]
        qkv = self.qkv_proj(hidden_states)
        #! q: [batch_size, seq_len, tp_q_size]
        #! k,v: [batch_size, seq_len, tp_kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # logger.info(f"qkv shape: {q.shape}, {k.shape}, {v.shape}")
        #! rotary embedding encoding for key & value
        # logger.info(f"positions: {positions}")
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn.forward(q, k, v, infer_state)
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        layer_num: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = LlamaAttention(
            layer_num=layer_num,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
        infer_state: LlamaInferStateInfo,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("LlamaDecoderLayer forwarding")
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            infer_state=infer_state
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i, linear_method)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        
        print("LlamaModel forwarding")

        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                None,
                input_metadata,
                residual,
                infer_state,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        **kwargs
        # linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = None
        self.model = LlamaModel(config, self.linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sampler = Sampler(config.vocab_size)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = dist.get_world_size()
        # Use the kwargs
        self.weight_dir = kwargs.get("weight_dir")
        self.max_total_token_num = kwargs.get("max_total_token_num")
        self.max_req_num = kwargs.get("max_req_num", 1000)
        self.max_seq_length = kwargs.get("max_seq_length", 1024 * 5)
        self.return_all_prompt_logprobs = kwargs.get("return_all_prompt_logprobs", False)

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_mem_manager()
        self._init_req_manager()
        self._init_some_value()
        self._init_to_get_rotary()
        
    def _init_config(self):
        with open(os.path.join(self.weight_dir, "config.json"), 'r') as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if "num_key_value_heads" not in self.config:
            self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return

    def _verify_must(self):
        assert self.config["num_attention_heads"] % self.world_size == 0
        return
    
    def _verify_params(self):
        assert self.config["num_key_value_heads"] % self.world_size == 0
        return
    
    def _init_mem_manager(self):
        self.mem_manager = MemoryManager(self.max_total_token_num,dtype=torch.float16,
                                         head_num=self.config["num_key_value_heads"] // self.world_size,
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])
        return
    
    def _init_req_manager(self):
        self.req_manager = RequestManager(self.max_req_num,
                                          self.max_seq_length,
                                          self.mem_manager)
        return
    
    def _init_some_value(self):
        self.head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return
    
    def _init_to_get_rotary(self, default_base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings",
                2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor
            
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
        return
    
    def _prefill(self, batch_size, total_token_num,
                 max_len_in_batch, input_ids,
                 b_req_idx, b_start_loc, b_seq_len,
                 multimodal_params, positions, input_metadata):
        infer_state = LlamaInferStateInfo()
        infer_state.is_prefill = True
        infer_state.return_all_prompt_logprobs = self.return_all_prompt_logprobs
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        # assert (input_ids.shape[0] == total_token_num)
        assert (b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.multimodal_params = multimodal_params

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_mem = self.mem_manager.alloc_contiguous(infer_state.total_token_num)
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]

        else:
            infer_state.mem_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(infer_state.total_token_num)
            infer_state.mem_index = alloc_mem
            infer_state.key_buffer = torch.empty((infer_state.total_token_num, self.tp_k_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            infer_state.value_buffer = torch.empty((infer_state.total_token_num, self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
        
        init_req_to_token_indexes(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len,
                            max_len_in_batch, infer_state.mem_index)

        infer_state.init_some_extra_state(self, input_ids)
        
        hidden_states = self.model(input_ids,positions,infer_state,input_metadata)

        hidden_states = self.postlayer_get_embedding(hidden_states, infer_state, return_logits=False)

        return hidden_states
    
    def _decode(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len, multimodal_params,positions,input_metadata):
        infer_state = LlamaInferStateInfo()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.multimodal_params = multimodal_params
        
        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_mem = self.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]
            copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)
        else:
            infer_state.mem_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(batch_size)
            infer_state.mem_index = alloc_mem
            infer_state.key_buffer = torch.empty((batch_size, self.tp_k_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            infer_state.value_buffer = torch.empty((batch_size, self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)

        infer_state.init_some_extra_state(self, input_ids)
        # hidden_state = self._token_forward(input_ids, infer_state)
        hidden_states = self.model(input_ids, positions, infer_state, input_metadata)

        hidden_states = self.postlayer_get_embedding(hidden_states, infer_state, return_logits=False)
        
        return hidden_states
    
    def forward(
        self,
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        is_prefill: bool,
        # kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        
        print("LlamaForCausalLM forwarding")

        if is_prefill:
            return self._prefill(
                batch_size=batch_size,
                total_token_num=total_token_num,
                max_len_in_batch=max_len_in_batch,
                input_ids=input_ids,
                b_req_idx=b_req_idx,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                multimodal_params=None,
                positions=positions,
                input_metadata=input_metadata
            )
        else:
            return self._decode(
                batch_size=batch_size,
                total_token_num=total_token_num,
                max_len_in_batch=max_len_in_batch,
                input_ids=input_ids,
                b_req_idx=b_req_idx,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                multimodal_params=None,
                positions=positions,
                input_metadata=input_metadata
            )   
            
        # hidden_states = self.model(input_ids, positions, kv_caches,
        #                            input_metadata)
        # return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens
    
    def _slice_get_last_input(self, input_embdings: torch.Tensor, infer_state: LlamaInferStateInfo):
        if infer_state.is_prefill and not infer_state.return_all_prompt_logprobs:
            batch_size = infer_state.batch_size
            last_input = torch.empty((batch_size, input_embdings.shape[-1]),
                                     device=input_embdings.device,
                                     dtype=torch.float16)
            last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[0, last_index, :]
            return last_input, batch_size

        if infer_state.is_prefill and infer_state.return_all_prompt_logprobs:
            total_tokens = infer_state.total_token_num
            return input_embdings, total_tokens
        
        if not infer_state.is_prefill:
            batch_size = infer_state.batch_size
            return input_embdings[0, -batch_size:, :], batch_size
        
        assert False, "Error State"
    
    def postlayer_get_embedding(
        self,
        input_embedding: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        return_logits: bool
    ) -> torch.Tensor:
        logger.info(f"input_embdings: {input_embedding.shape}")
        last_input, token_num = self._slice_get_last_input(input_embedding, infer_state)
        logger.info(f"last_input: {last_input.shape}")
        input_embedding = None
        last_input = self.norm(last_input)
        last_input = einops.rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, token_num)
        logic_batch = torch.mm(self.lm_head.weight, last_input)
        gather_data = logic_batch

        logic_batch = None

        print(f"gather_data: {gather_data.shape}")

        if not return_logits:
            prob_out = torch.softmax(gather_data.permute(1, 0).float(), dim=-1)
            gather_data = None
            return prob_out
        else:
            ans_logits = gather_data.permute(1, 0).float()
            gather_data = None
            return ans_logits

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
