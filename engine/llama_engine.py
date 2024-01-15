from typing import List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.distributed as dist

from utils.distributed_utils import (
    init_distributed,
)
from model.model_metadata import ModelConfig,ParallelConfig
from model.llama import LlamaForCausalLM
from sequence.sequence import Sequence
from sequence.scheduler import SequenceScheduler, SequenceBatch
from sequence.config import SchedulerConfig
from sampler.sampling_metadata import SamplingParams, prepare_sample
from utils.utils import (
    set_default_torch_dtype
)
from model.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_first_rank,
    get_pipeline_model_parallel_last_rank
)
from engine.worker import Worker
from engine.master import Master

class LLamaEngine():
    def __init__(
        self, 
        model_config: ModelConfig, 
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig = SchedulerConfig(),
    ) -> nn.Module:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_config = model_config
        self.rank = init_distributed(model_config, parallel_config)
        self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_config.tokenizer, 
                        trust_remote_code=self.model_config.trust_remote_code)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_scheduler = SequenceScheduler(scheduer_config=scheduler_config,
                                               rank=self.rank,
                                               tokenizer=self.tokenizer,
                                               device=self.device)
        self._init_model()
        self._init_worker()

    def _init_model(self):
        with set_default_torch_dtype(self.model_config.dtype):
            model = LlamaForCausalLM(self.model_config.hf_model_config)  
            model.to(device=self.device)
            model.load_weights(self.model_config.model)
        self.model = model
    
    def _init_worker(self):
        if self.rank == get_pipeline_model_parallel_first_rank():
            self.worker = Master(model=self.model,
                                 model_config=self.model_config,
                                 parallel_config=self.parallel_config,
                                 scheduler=self.seq_scheduler)

        else:
            self.worker = Worker(model=self.model,
                                 model_config=self.model_config,
                                 parallel_config=self.parallel_config)
    
    def generate(self, 
                 requests: List[Tuple[str, int, int]], 
                 sampling_params: SamplingParams = None):
        if (self.rank == get_pipeline_model_parallel_first_rank()):
            self.seq_scheduler.add_requests(requests)
            self.worker.run(sampling_params)
        else:
            self.worker.run()
