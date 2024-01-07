import os
import torch
from utils.transformers_utils import get_config_from_hf, get_max_len_from_hf
from typing import List, Optional

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
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: int = 1,
        workers_per_node: int = 2,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = num_gpus_per_worker
        self.workers_per_node = workers_per_node

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        num_nodes = int(os.environ.get("SLURM_NNODES"))
        assert num_nodes is not None, \
            "SLURM_NNODES is None, cannot check args validity"
        assert self.world_size // self.workers_per_node == num_nodes, \
            "num of placement groups != num of nodes"
        
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet.")
            

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
