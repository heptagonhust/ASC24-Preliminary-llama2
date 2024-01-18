import logging
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
from manager.tiny_batch_manager import TinyBatchManager
from model.infer_state_info import InferStateInfoForTransfer
from manager.tiny_batch_manager_metadata import TinyBatchManagerOpKind as OpKind
from manager.tiny_batch_manager import TinyBatchManager

class Worker():
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_batch_size: int = 1024,
        device: str = "cuda",
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config

        #! "max_req_num" is for fixed-length metadata transfer,
        #! the value can be obtained from ModelRpcServer
        #! or cmd line arguments
        self.device = device
        mp.set_start_method('spawn')
        self.sender = Sender(
            parallel_config=self.parallel_config
        )
        self.receiver = Receiver(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            max_batch_size=max_batch_size
        )
    
    def start_worker(self):
        self.send_queue = self.sender.start_loop()
        self.recv_queue = self.receiver.start_loop()
        self.rank = initialize_calculator_distributed(self.model_config, self.parallel_config)
        self._init_model()
        self._init_tiny_batch_manager()
    
    @torch.inference_mode()
    def run(self):
        logging.info("Worker started")
        # idx = 0
        while True:
            recv_hidden_state, recv_infer_state = self.recv_queue.get()
            if recv_hidden_state is None:
                break

            infer_state_tensor = recv_infer_state.clone()
            infer_state: InferStateInfoForTransfer = \
                InferStateInfoForTransfer.from_transferred_tensor(infer_state_tensor)

            self.tiny_batch_manager.perform_op(infer_state.infer_state_op)
            if infer_state.infer_state_op.batch_op_kind != OpKind.PAUSE:
                # TODO: adjust model forward args
                hidden_state = self.model(
                    input_ = recv_hidden_state,
                    infer_state = infer_state,
                )

            del recv_hidden_state
            del recv_infer_state
            # print(f"idx: {idx}, rank: {self.rank}, req_id: {infer_state.b_req_idx}")
            # idx += 1
            self.send_queue.put((hidden_state, infer_state_tensor))

        #! end of work
        #! sender will stop looping after receiving None
        self.send_queue.put((None, None))
        self.receiver.receiver.kill()
        return

    def _init_model(self):
        with set_default_torch_dtype(self.model_config.dtype):
            model = LlamaForCausalLM(self.model_config.hf_model_config)  
            model.to(device=self.device)
            model.load_weights(self.model_config.model)
        self.model = model

    def _init_tiny_batch_manager(self):
        self.tiny_batch_manager = TinyBatchManager(
            req_manager=self.model.req_manager
        )