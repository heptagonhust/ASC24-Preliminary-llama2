from utils.transformers_utils import get_config_from_hf, get_max_len_from_hf
from typing import List, Optional
import torch


class PortConfig:
    def __init__(
        self,
        router_port,
        req_server_port,
    ) -> None:
        self.router_port = router_port
        self.req_server_port = req_server_port


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
        self._verify_args()

    def _verify_args(self) -> None:
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


class ReqConfig:
    """Configuration for the request server and router.

    Args:
        batch_size: 两个Batch可能会合并，这是用来控制每个被合并Batch占用token大小的
        max_total_token_num: kvcache里面的最大token数目，越大越容易OOM
        max_req_num: 一个Batch里面的最大请求数
        max_req_total_len: 输入+输出最大长度
        router_token_ratio: router的token占用率
        router_max_new_token_len: router最大的新token长度
    """
    def __init__(
        self,
        batch_size: int,
        max_total_token_num: int,
        max_req_num: int,
        max_req_total_len: int,
        router_token_ratio: float,
        router_max_new_token_len: int,
    ) -> None:
        self.batch_size = batch_size
        self.max_req_num = max_req_num
        self.max_req_total_len = max_req_total_len
        self.max_total_token_num = max_total_token_num
        self.router_token_ratio = router_token_ratio
        self.router_max_new_token_len = router_max_new_token_len

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
