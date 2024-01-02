from utils.transformers_utils import get_config_from_hf, get_max_len_from_hf
from typing import List, Optional
import torch


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
