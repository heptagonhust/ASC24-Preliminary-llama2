from typing import List, Tuple

import torch
from transformers import AutoTokenizer

from model.llama import LlamaForCausalLM
from model.model_metadata import (
    ModelConfig, 
    ParallelConfig
)
from engine.sender import Sender
from engine.receiver import Receiver
from sequence.scheduler import SequenceScheduler
from sequence.config import SchedulerConfig
from sampler.sampling_params import SamplingParams
from utils.utils import set_default_torch_dtype
from utils.distributed_utils import (
    initialize_calculator_distributed,
)


class Master():
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer, 
            trust_remote_code=self.model_config.trust_remote_code
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.scheduler = SequenceScheduler(scheduer_config=scheduler_config,
                                               rank=self.rank,
                                               tokenizer=self.tokenizer,
                                               device=self.device)
        self.sender = Sender(
            parallel_config=self.parallel_config
        )
        self.receiver = Receiver(
            model_config=self.model_config,
            parallel_config=self.parallel_config
        )
    
    def start_master(self):
        self.send_queue = self.sender.start_loop()
        self.recv_queue = self.receiver.start_loop()
        self.rank = initialize_calculator_distributed(self.model_config, self.parallel_config)
        self._init_model()
    
    def add_requests(self, requests: List[Tuple[str, int, int]]):
        self.scheduler.add_requests(requests)

    def run(self,
            sampling_params: SamplingParams):
        self.sampling_params = sampling_params
        hidden_state = None
        positions = None
        seqs_id = None
        while not self.scheduler.is_finished():
            if self.recv_queue.empty():
                hidden_state, positions, seqs_id = self.scheduler.get_new_batch()
            else:
                recv_hidden_state, recv_positions, recv_seqs_id = self.recv_queue.get()
                hidden_state, positions, seqs_id = \
                    self._do_sample(recv_hidden_state, recv_seqs_id)
                del recv_hidden_state
                del recv_positions
                del recv_seqs_id
            hidden_state = self.model(input_ = hidden_state,
                                      positions = positions,
                                      kv_caches = None,
                                      input_metadata = None)
            self.send_queue.put((hidden_state, positions, seqs_id))
                
        #! end of work
        self.send_queue.put((None, None, None))
        self.receiver.receiver.kill()
        return

    def _do_sample(self,
                  hidden_state: torch.Tensor,
                  seqs_id: torch.Tensor):
        sampling_metadata = \
            self.scheduler.get_sampling_metadata_from_seqs_id(seqs_id,
                                                              self.sampling_params)
        sampler_output = self.model.sample(hidden_state, sampling_metadata)
        return self.scheduler.update_batch(seqs_id, sampler_output)

        
    def _init_model(self):
        with set_default_torch_dtype(self.model_config.dtype):
            model = LlamaForCausalLM(self.model_config.hf_model_config)
            model.to(device=self.device)
            model.load_weights(self.model_config.model)
        self.model = model