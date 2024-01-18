import asyncio
import torch
import rpyc
from typing import Dict, List, Optional, Tuple
from transformers.models.llama import LlamaConfig
from manager.tiny_batch_manager_metadata import TinyBatchManagerOp, TinyBatchManagerOpKind, BatchInitMetadata
from router.io_struct import Req
from router.model_infer.infer_batch import InferBatch

from model.llama import LlamaForCausalLM

from router.model_infer.preprocess import prepare_decode_inputs, prepare_prefill_inputs
from sampler.postprocess import sample
from sampler.sampling_metadata import SamplingParams, _prepare_sample
from router.model_infer.infer_batch import requests_mapping
from router.model_infer.infer_batch import InferReq

from model.parallel_utils.parallel_state import setup_distributed
from model.model_metadata import ParallelConfig, ModelConfig
from model.infer_state_info import InferStateInfoForTransfer

import torch.multiprocessing as mp

from utils.log_utils import init_logger
logger = init_logger(__name__)

class InferBatchManager:

    def __init__(
        self,
        model: LlamaForCausalLM,
        parallel_config: ParallelConfig,
        model_config: ModelConfig,
        scheduler_pipe: mp.Pipe,
        return_all_prompt_logprobs: bool = False,
    ) -> None:
        self.model = model
        self.parallel_config = parallel_config
        self.model_config = model_config
        self.return_all_prompt_logprobs = return_all_prompt_logprobs
        self.cache: Dict[int, InferBatch] = {}
        self.scheduler_pipe = scheduler_pipe
        return
    
    def _add_batch(self, batch_id: int, reqs: List[Req]) -> None:
        self.cache[batch_id] = InferBatch(batch_id, reqs)
        return
    

    def put_for_prefill(self, batch_id: int, reqs: List[Req]):
        #! 1. add batch to cache
        batch_data = InferBatch.init_batch(
            batch_id=batch_id,
            requests=reqs,
        )
        self.cache[batch_id] = batch_data

        #! 2. prepare inputs
        batch = self.cache[batch_id]
        kwargs, run_reqs, not_run_reqs = prepare_prefill_inputs(batch=batch)

        #! 3. put into Master node
        if len(run_reqs) >= 1:
            token_ids = kwargs.pop("input_ids")
            infer_state_info = InferStateInfoForTransfer()
            infer_state_info.batch_id = batch_id
            infer_state_info.batch_size = kwargs["batch_size"]
            infer_state_info.total_token_num = kwargs["total_token_num"]
            infer_state_info.max_len_in_batch = kwargs["max_len_in_batch"]
            infer_state_info.b_req_idx = kwargs["b_req_idx"]
            infer_state_info.b_start_loc = kwargs["b_start_loc"]
            infer_state_info.b_seq_len = kwargs["b_seq_len"]
            infer_state_info.is_prefill = True

            infer_state_info.infer_state_op = TinyBatchManagerOp()
            infer_state_info.infer_state_op.batch_op_kind = TinyBatchManagerOpKind.INIT
            infer_state_info.infer_state_op.batch_op_metadata = BatchInitMetadata()
            infer_state_info.infer_state_op.batch_op_metadata.need_alloc_size = \
                len([r for r in reqs if r.request_id not in requests_mapping])
            
            self.scheduler_pipe.send((token_ids, infer_state_info))


        
    def put_for_decode(self, batch_id: int, req_ids: List[int],
                       finished_or_removed_req_ids: List[int],
                       op_kind: TinyBatchManagerOpKind):
        pass
    
    def put_for_pause(self, batch_id: int, req_ids: List[int]):
        pass


