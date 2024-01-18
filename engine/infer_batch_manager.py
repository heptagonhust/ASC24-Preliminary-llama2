import asyncio
import torch
import rpyc
from typing import Dict, List, Optional, Tuple
from transformers.models.llama import LlamaConfig
from manager.tiny_batch_manager_metadata import TinyBatchManagerOpKind
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

from utils.log_utils import init_logger
logger = init_logger(__name__)

class InferBatchManager:

    def __init__(self, model: LlamaForCausalLM, parallel_config: ParallelConfig, model_config: ModelConfig, return_all_prompt_logprobs: bool = False) -> None:
        self.model = model
        self.parallel_config = parallel_config
        self.model_config = model_config
        self.return_all_prompt_logprobs = return_all_prompt_logprobs
        self.cache: Dict[int, InferBatch] = {}
        return
    
    def add_batch(self, batch_id, reqs, dtype):
        import torch
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), self.model.req_manager, self.model.vocab_size)
        self.cache[batch_id] = batch_data

        # 将更新后的状态返回给调用方用于router中请求的状态
        ans = {}
        for req_id in batch_data.request_ids:
            req_obj : InferReq  = requests_mapping[req_id]
            ans[req_id] = (req_obj.req_status, req_obj.cur_kv_len)
        return ans
    
    def prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    def decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    def filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    def pause_reqs(self, batch_id, req_list):
        batch1 = self.cache.pop(batch_id)
        batch2 = batch1.pause_reqs(req_list)
        self.cache[batch_id] = batch2
        del batch1
        return

    def merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    def remove_batch(self, batch_id):
        batch = self.cache.pop(batch_id)
        batch.free_self()
        del batch
        return
    
    # @calculate_time(show=True, min_cost_ms=150)
    def forward(self, batch_id, is_prefill):        
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs, not_run_reqs = prepare_prefill_inputs(batch)
        else:
            kwargs, run_reqs, not_run_reqs = prepare_decode_inputs(batch)

        
        if len(run_reqs) >= 1:
            logits = self.model.forward(**kwargs)
            # 原 sample 需要用到 Sequence，与我们的发生了冲突
            # 这里的 sample 姑且沿用 lightllm 的，不知道会出现什么问题
            next_token_ids, next_token_probs = sample(logits, run_reqs)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
                # prefill and decode is same
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    'id': int(next_token_id),
                    'logprob': float(next_token_logprob),
                }
                output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata) # 状态， cur_kv_len, token_id, metadata

        for req_obj in not_run_reqs:
            output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, None, None) # 状态， cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        return output_dict
    
    def put_for_prefill(batch_id: int, reqs: List[Req]):
        pass
        
    def put_for_decode(batch_id: int, req_ids: List[int],
                       finished_or_removed_req_ids: List[int],
                       op_kind: TinyBatchManagerOpKind):
        pass
    
    def put_for_pause(batch_id: int, req_ids: List[int]):
        pass


