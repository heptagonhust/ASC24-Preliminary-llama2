# coding=utf-8
# Adapted from
# Copyright 2024 Hust-heptagon team
# Copyright 2023 The vLLM team.
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
# This code is based on vLLM library
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
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from model.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)

from utils.utils import divide, set_weight_attrs


def pad_vocab_size(vocab_size: int, pad_to: int = 64) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int,
                                              rank: int) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l



#! world_size: only tp parallelism scale
def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int,
                                       world_size: int) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                     rank)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 params_dtype: Optional[torch.dtype] = None):
        super().__init__()

        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        #! length after padding
        self.num_embeddings_padded = pad_vocab_size(num_embeddings)
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.tp_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        #! vocab_range_from_global_vocab_size: return first and last index
        self.vocab_start_index, self.vocab_end_index = (
            vocab_range_from_global_vocab_size(
                self.num_embeddings_padded, get_tensor_model_parallel_rank(),
                self.tp_size))
        self.num_embeddings_per_partition = (self.vocab_end_index -
                                             self.vocab_start_index)
        #! output dim: num_embeddings_per_partition
        #! input dim: embedding_dim
        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition,
                        self.embedding_dim,
                        device=torch.cuda.current_device(),
                        dtype=params_dtype))
        set_weight_attrs(self.weight, {
            "parallel_dim": 0,
            "weight_loader": self.weight_loader
        })

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        parallel_dim = param.parallel_dim
        assert loaded_weight.shape[parallel_dim] == self.num_embeddings
        loaded_weight = loaded_weight[self.vocab_start_index:self.
                                      vocab_end_index]
        param[:loaded_weight.shape[0]].data.copy_(loaded_weight)

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            input_mask = ((input_ < self.vocab_start_index) |
                          (input_ >= self.vocab_end_index))
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight)
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output


class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 bias: bool = False,
                 params_dtype: Optional[torch.dtype] = None):
        super().__init__(num_embeddings, embedding_dim, params_dtype)
        #! without bias in llama case
        if bias:
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition,
                            device=torch.cuda.current_device(),
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "parallel_dim": 0,
                "weight_loader": self.weight_loader
            })
        else:
            self.register_parameter("bias", None)
    
    #! can not call ParallelLMHead() directly, use ParallelLMHead.weight instead
    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
