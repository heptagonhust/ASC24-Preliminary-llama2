import torch
import torch.distributed as dist

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


class Worker():
    def __init__(self,
                 model_config: ModelConfig,
                 parallel_config: ParallelConfig):
        self.model_config = model_config
        self.parallel_config = parallel_config
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
    
    def run(self):
        while True:
            recv_hidden_state, recv_positions, recv_seqs_id = self.recv_queue.get()
            if recv_hidden_state is None:
                break
            hidden_state = self.model(input_ = recv_hidden_state,
                                      positions = recv_positions,
                                      kv_caches = None,
                                      input_metadata = None)
            positions = recv_positions.clone()
            seqs_id = recv_seqs_id.clone()
            del recv_hidden_state
            del recv_positions
            del recv_seqs_id
            self.send_queue.put((hidden_state, positions, seqs_id))
                
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