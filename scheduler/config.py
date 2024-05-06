from typing import Optional

class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
    """

    def __init__(
        self,
        block_size: int = 16,
        num_gpu_blocks: int = 16384,
        num_cpu_blocks: int = 0,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.sliding_window = sliding_window

        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks


#! max_num_batched_tokens to tune the batches
class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        max_paddings: Maximum number of paddings to be added to a batch.
    """

    def __init__(
        self,
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: int = 256,
        max_model_len: int = 2048,
        max_paddings: int = 256,
        max_num_batches: int = 8,
        merge_threshold_min_seq: int = 16,
    ) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # If max_model_len is too short, use 2048 as the default value for
            # higher throughput.
            self.max_num_batched_tokens = max(max_model_len, 2048)
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_seqs = max_num_seqs // max_num_batches
        self.max_model_len = max_model_len
        self.max_paddings = max_paddings
        self.max_num_batches = max_num_batches
        self.merge_threshold_min_seq = merge_threshold_min_seq
        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")