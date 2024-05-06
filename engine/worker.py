import logging
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from engine.sender import Sender
from engine.receiver import Receiver
from model.llama import LlamaForCausalLM
from model.model_metadata import (
    ModelConfig, 
    ParallelConfig
)
from utils.utils import set_default_torch_dtype
from utils.distributed_utils import (
    initialize_calculator_distributed,
)
from engine.cache_engine import CacheEngine
from scheduler.config import CacheConfig
from scheduler.input_metadata import InputMetadata

class Worker():
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        device: str = "cuda",
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.device = device
        mp.set_start_method('spawn')
        self.sender = Sender(
            parallel_config=self.parallel_config
        )
        self.receiver = Receiver(
            model_config=self.model_config,
            parallel_config=self.parallel_config
        )
    
    def start_worker(self):
        self.send_queue = self.sender.start_loop()
        self.recv_queue = self.receiver.start_loop()
        self.rank = initialize_calculator_distributed(self.model_config, self.parallel_config)
        self._init_model()
        self._init_cache_engine()
    
    @torch.inference_mode()
    def run(self):
        logging.info("Worker started")
        idx = 0
        input_metadata: InputMetadata = None
        recv_input_metadata: InputMetadata = None
        while True:
            recv_hidden_states, recv_input_positions, recv_input_metadata = self.recv_queue.get()
            input_positions = recv_input_positions.clone()
            del recv_input_positions
            input_metadata = recv_input_metadata.clone()
            del recv_input_metadata

            if recv_hidden_states is None:
                break
            
            hidden_states = self.model(input_ = recv_hidden_states,
                                      positions = input_positions,
                                      kv_caches = self.gpu_cache,
                                      input_metadata = input_metadata)
            del recv_hidden_states
            self.send_queue.put((
                hidden_states, 
                input_positions, 
                input_metadata, 
            ))
            idx += 1
                
        #! end of work
        #! sender will stop looping after receiving None
        self.send_queue.put((None, None, None))
        self.receiver.receiver.kill()
        return

    def _init_model(self):
        with set_default_torch_dtype(self.model_config.dtype):
            model = LlamaForCausalLM(self.model_config.hf_model_config)  
            model.to(device=self.device)
            model.load_weights(self.model_config.model)
        self.model = model

    def _init_cache_engine(self) -> None:
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        #! for cuda graph
        self.block_size = self.cache_engine.block_size