from utils.transformers_utils import get_config_from_hf, get_max_len_from_hf
from typing import List, Optional
import torch

class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Whether to use Ray for model workers. Will be set to
            True if either pipeline_parallel_size or tensor_parallel_size is
            greater than 1.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        max_parallel_loading_workers: Optional[int] = None,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.max_parallel_loading_workers = max_parallel_loading_workers

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True


class ModelConfig:
    def __init__(
        self,
        model: str,
        tokenizer: str,
        trust_remote_code: bool,
        seed: int,
        max_model_len: Optional[int]
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.trust_remote_code = trust_remote_code
        self.seed = seed

        self.hf_model_config = get_config_from_hf(model, trust_remote_code)
        self.max_model_len = get_max_len_from_hf(self.hf_model_config, max_model_len)
        
        self.dtype = torch.float16

    def get_head_size(self) -> int:
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_model_config.hidden_size // self.hf_model_config.num_attention_heads
    
    def get_total_num_kv_heads(self) -> int:
        return self.hf_model_config.num_key_value_heads

    def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
        """Returns the number of KV heads per GPU."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1,
                   total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_total_layers(self) -> int:
        return self.hf_model_config.num_hidden_layers

    # def get_num_layers(self, parallel_config: ParallelConfig, pp_rank: int) -> int:
    #     total_num_hidden_layers = self.hf_model_config.num_hidden_layers
    #     return total_num_hidden_layers // parallel_config.pipeline_parallel_size


class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    Args:
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
        block_tables: The block tables. (Seq id -> list of physical block)
    """

    def __init__(
        self,
        prompt_lens: List[int],
        slot_mapping: torch.Tensor,
        max_context_len: Optional[int],
        context_lens: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
    ) -> None:
        self.prompt_lens = prompt_lens
        self.max_context_len = max_context_len
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables

        self.is_prompt = len(prompt_lens) > 0
        self.attn_bias = None

    def __repr__(self) -> str:
        return ("InputMetadata("
                f"prompt_lens={self.prompt_lens}, "
                f"max_context_len={self.max_context_len}, "
                f"slot_mapping={self.slot_mapping}, "
                f"context_lens={self.context_lens}, "
                f"block_tables={self.block_tables})")
