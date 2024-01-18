import logging
from typing import List, Tuple

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from manager.tiny_batch_manager import TinyBatchManager
from model.infer_state_info import InferStateInfoForTransfer
from manager.tiny_batch_manager_metadata import TinyBatchManagerOpKind as OpKind


from model.llama import LlamaForCausalLM
from model.model_metadata import (
    ModelConfig, 
    ParallelConfig
)
from engine.sender import Sender
from engine.receiver import Receiver
from scheduler.config import SchedulerConfig
from scheduler.scheduler import Scheduler
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
        device: str = "cuda",
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device = device
        self.batches = 0
        self.max_batch_num = scheduler_config.max_req_num
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer, 
            trust_remote_code=self.model_config.trust_remote_code
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        mp.set_start_method('spawn')
        self.sender = Sender(
            parallel_config=self.parallel_config
        )
        self.receiver = Receiver(
            model_config=self.model_config,
            parallel_config=self.parallel_config
        )
        self.scheduler = Scheduler(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config
        )
    
    def start_master(self):
        self.send_queue = self.sender.start_loop()
        self.recv_queue = self.receiver.start_loop()
        self.scheduler_pipe = self.scheduler.start_loop()
        self.rank = initialize_calculator_distributed(self.model_config, self.parallel_config)
        self._init_model()
        self._init_tiny_batch_manager()
    

    @torch.inference_mode()
    def run(self,
            sampling_params: SamplingParams):
        logging.info("Master started")
        self.sampling_params = sampling_params
        hidden_state = None
        infer_state = None
        # idx = 0
        while True:
            if self._more_batches():
                token_ids, infer_state = self._get_batch()
            else:
                #TODO: Here we should transfer recv_hidden_state and recv_infer_state
                #      to scheduler directly. Also, sample should be done in scheduler, not here.
                recv_hidden_state, recv_infer_state = self.recv_queue.get()
                token_ids, infer_state = \
                    self._do_sample(recv_hidden_state, recv_infer_state)
                del recv_hidden_state
                del recv_infer_state
            if token_ids is not None:
                self.tiny_batch_manager.perform_op(infer_state.infer_state_op)
                if infer_state.infer_state_op.batch_op_kind != OpKind.PAUSE:
                    # TODO: adjust model forward args
                    hidden_state = self.model(
                        input_ = token_ids,
                        infer_state = infer_state,
                    )
                del token_ids
                # print(f"idx: {idx}, rank: {self.rank}, seqs_id: {seqs_id}")
                # idx += 1
                self.send_queue.put((hidden_state, infer_state))
            else:
                break
                
        #! end of work
        self.send_queue.put((None, None, None))
        self.receiver.receiver.kill()
        return

    def _more_batches(self):
        if self.batches < self.max_batch_num:
            self.batches += 1
            return True
        else:
            return False

    def _get_batch(self):
        return self.scheduler_pipe.recv()

    def _do_sample(self,
                  hidden_state: torch.Tensor,
                  infer_state: torch.Tensor):
        logits = self.model.postlayer_get_embedding(hidden_state, infer_state)
        self.scheduler_pipe.send((logits, infer_state))
        return self._get_batch()

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