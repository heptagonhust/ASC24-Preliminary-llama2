import asyncio
import torch
import rpyc
from typing import Dict, List, Optional, Tuple
from transformers.models.llama import LlamaConfig
from manager.tiny_batch_manager_metadata import BatchFilterMetadata, BatchPauseMetadata, BatchRemoveMetadata, ReqToFree, TinyBatchManagerOp, TinyBatchManagerOpKind, BatchInitMetadata
from router.io_struct import Req, ReqRunStatus
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


    def _add_batch(self, batch_id: int, reqs: List[Req]):
        new_batch, need_alloc_size = InferBatch.init_batch(batch_id, reqs, self.model.vocab_size)
        return new_batch, need_alloc_size
    

    def _filter_batch(self, batch_id: int, req_id_list: List[int], finished_req_id_list: List[int]):
        batch = self.cache.pop(batch_id)
        filter_batch, req_idx_list, cur_kv_len_list = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return req_idx_list, cur_kv_len_list


    def _remove_batch(self, batch_id: int):
        batch = self.cache.pop(batch_id)
        req_idx_list, cur_kv_len_list =  batch.free_self()
        del batch
        return req_idx_list, cur_kv_len_list
    
    def _pause_reqs(self, batch_id: int, pause_reqs: List[Tuple[int, ReqRunStatus]]):
        batch = self.cache.pop(batch_id)
        pause_batch, req_idx_list, cur_kv_len_list = batch.pause_reqs(pause_reqs)
        del batch
        self.cache[batch_id] = pause_batch
        return req_idx_list, cur_kv_len_list
    

    def put_for_prefill(self, batch_id: int, reqs: List[Req]):
        #! 1. add batch to cache
        new_batch, need_alloc_size = self._add_batch(
            batch_id=batch_id,
            requests=reqs,
        )
        self.cache[batch_id] = new_batch

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
            infer_state_info.infer_state_op.batch_op_metadata.need_alloc_size = need_alloc_size
            
            self.scheduler_pipe.send((token_ids, infer_state_info))


        
    def put_for_decode(self, batch_id: int, old_batch_id: int, req_ids: List[int],
                       finished_or_removed_req_ids: List[int],
                       op_kind: TinyBatchManagerOpKind):
        req_idx_list, cur_kv_len_list = [], []
        if op_kind == TinyBatchManagerOpKind.REMOVE:
            req_idx_list, cur_kv_len_list = self._remove_batch(old_batch_id)
        elif op_kind == TinyBatchManagerOpKind.FILTER:
            req_idx_list, cur_kv_len_list = self._filter_batch(old_batch_id, req_ids, finished_or_removed_req_ids)

        #! We should not pop the batch from cache, because we need to keep the batch for recv
        batch: InferBatch = self.cache[batch_id]
        kwargs, run_reqs, not_run_reqs = prepare_decode_inputs(batch=batch)

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
            infer_state_info.is_prefill = False

            infer_state_info.infer_state_op = TinyBatchManagerOp()
            infer_state_info.infer_state_op.batch_op_kind = op_kind
            infer_state_info.infer_state_op.batch_op_metadata = \
                BatchFilterMetadata() if op_kind == TinyBatchManagerOpKind.FILTER else BatchRemoveMetadata()
            infer_state_info.infer_state_op.batch_op_metadata.req_list_to_free = \
                [ReqToFree(req_idx, cur_kv_len) for (req_idx, cur_kv_len) in zip(req_idx_list, cur_kv_len_list)]
            
            self.scheduler_pipe.send((token_ids, infer_state_info))
        
    
    def put_for_pause(self, batch_id: int, reqs: List[Tuple[int, ReqRunStatus]]):
        req_idx_list, cur_kv_len_list = self._pause_reqs(batch_id, reqs)
        batch: InferBatch = self.cache[batch_id]
        kwargs, run_reqs, not_run_reqs = prepare_decode_inputs(batch=batch)
        if len(run_reqs) >= 1:
            #! make up a dummy input
            #! should i use prepare_decode_inputs()?
            token_ids = kwargs.pop("input_ids")
            infer_state_info = InferStateInfoForTransfer()
            infer_state_info.batch_id = batch_id
            infer_state_info.batch_size = kwargs["batch_size"]
            infer_state_info.total_token_num = kwargs["total_token_num"]
            infer_state_info.max_len_in_batch = kwargs["max_len_in_batch"]
            infer_state_info.b_req_idx = kwargs["b_req_idx"]
            infer_state_info.b_start_loc = kwargs["b_start_loc"]
            infer_state_info.b_seq_len = kwargs["b_seq_len"]
            infer_state_info.is_prefill = False

            infer_state_info.infer_state_op = TinyBatchManagerOp()
            infer_state_info.infer_state_op.batch_op_kind = TinyBatchManagerOpKind.PAUSE
            infer_state_info.infer_state_op.batch_op_metadata = BatchPauseMetadata()
            infer_state_info.infer_state_op.batch_op_metadata.req_list_to_free = \
                [ReqToFree(req_idx, cur_kv_len) for (req_idx, cur_kv_len) in zip(req_idx_list, cur_kv_len_list)]
            
            self.scheduler_pipe.send((token_ids, infer_state_info))

    
    def merge_batch(self, new_batch_id: int, old_batch_id: int):
        new_batch = self.cache.pop(new_batch_id)
        old_batch = self.cache.pop(old_batch_id)
        m_batch = InferBatch.merge(batch1=new_batch, batch2=old_batch)
        del new_batch
        del old_batch
        self.cache[new_batch_id] = m_batch
        return



