import os
import contextlib
from typing import Dict

import torch
from ray.air.util.torch_dist import TorchDistributedWorker

from utils.utils import set_random_seed
from model.model_metadata import ParallelConfig, ModelConfig
from model.parallel_utils.parallel_state import initialize_model_parallel
from model.llama import LlamaForCausalLM
from sampler.sampling_metadata import SamplingMetadata, _prepare_sample
from sampler.sampling_params import SamplingParams
from sequence.sequence import Sequence

@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

'''
    class Worker:
        specifiy the jobs that should be placed to all the nodes to run
'''
class Worker(TorchDistributedWorker):
    def __init__(self,
                 model_config: ModelConfig, 
                 parallel_config:ParallelConfig,
                 sampling_params: SamplingParams,
                 rank: int):
        #! os.environ vairable must be set before the worker use torch
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.sampling_params = sampling_params
        self.rank = rank
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sampling_metadata_table: Dict[int: SamplingMetadata] = {}
    
    def init_distributed(self):
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")

        torch.cuda.set_device(self.device)
        initialize_model_parallel(self.parallel_config.tensor_parallel_size,
                                  self.parallel_config.pipeline_parallel_size)

    def init_model(self):
        set_random_seed(self.model_config.seed)
        with _set_default_torch_dtype(self.model_config.dtype):
            model = LlamaForCausalLM(self.model_config.hf_model_config)  
            model.to(device=self.device)
            # Load the weights from the cached or downloaded files.
            model.load_weights(self.model_config.model)
        self.model = model

    def run_model(self,
                  seq_id: int,
                  max_output_len: int,
                  eos_token_id: int,
                  seq: Sequence = None):
        assert seq is not None, "seq can not be None in the first run"
        sampling_metadata = _prepare_sample(seq, self.sampling_params)
        for _ in range(max_output_len):
            position = torch.arange(0, sampling_metadata.seq_data[seq_id].get_len())
            seq_token_ids = sampling_metadata.seq_data[seq_id].get_token_ids()
            seq_token_ids = torch.tensor(seq_token_ids, dtype=torch.long, device=self.device)

            hidden_state = self.model(seq_token_ids, position, None, None)
            sample_output = self.model.sample(hidden_state, sampling_metadata)
            new_token_id = sample_output[-1].samples[-1].output_token
            new_token_logprob = sample_output[-1].samples[-1].logprobs[new_token_id]
            sampling_metadata._update(seq_id, new_token_id, new_token_logprob)
            if (new_token_id == eos_token_id):
                break
        
        output_token_ids = sampling_metadata.seq_data[seq_id].output_token_ids
        return output_token_ids

    def execute_method(self, method, *args, **kargs):
        return getattr(self, method)(*args, **kargs)
