import time
import torch
import numpy as np
import collections

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from manager.request_manager import RequestManager
from manager.memory_manager import MemoryManager
from router.io_struct import Req, ReqRunStatus


requests_mapping = {}

class InferSamplingParams:

    def __init__(
        self,
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        vocab_size: int = -1,
    ) -> None:
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if self.top_k == -1:
            self.top_k = vocab_size
        return


class InferReq:

    def __init__(
        self,
        r_id,
        input_token_ids=[],
        out_token_id_count={},
        sampling_param=None,
        req_idx=-1,
        prompt_len=0,
        req_status=None,
        prompt_cache_len=None,
        prompt_cache_req_id=None,
        multimodal_params=None,
    ) -> None:
        self.r_id = r_id
        self.out_token_id_count = out_token_id_count
        self.sampling_param = sampling_param
        self.multimodal_params = multimodal_params
        self.req_idx = req_idx
        self.prompt_len = prompt_len
        self.input_token_ids = input_token_ids
        self.req_status = req_status
        self.cur_kv_len = 0 # 当前已经占用掉 token 现存的 kv len 长度
        self.prompt_cache_len = prompt_cache_len # 可以复用的一些公共 prompt 头对应的 kv cache 长度， prompt cache 目前只会在 splitfuse 模式下使用
        self.prompt_cache_req_id = prompt_cache_req_id # 对应的可复用的请求的 id，方便初始化的时候，将其 kv cache 复制到当前请求中
        return


@dataclass
class InferBatch:
    batch_id: int
    request_ids: List
    
    @classmethod
    @torch.no_grad()
    def init_batch(cls, batch_id, requests: List[Req], vocab_size: int):

        request_ids = []
        
        # need_alloc_size = len([r for r in requests if r.request_id not in requests_mapping])
        # nopad_b_req_idx = req_manager.alloc(need_alloc_size)
        # nopad_b_req_idx = nopad_b_req_idx.cpu().numpy()

        #! I don't know whether it is right.
        #! The problem is that you don't know whether "requests_mapping" is the same with 
        #! RequestManager's "req_state" or not.
        need_alloc_req_idxs = [r.request_id for r in requests if r.request_id not in requests_mapping]
        need_alloc_size = len(need_alloc_req_idxs)
        nopad_b_req_idx = need_alloc_req_idxs
        nopad_b_req_idx = nopad_b_req_idx.cpu().numpy()
        
        index = 0
        for r in requests:
            # request id -> idx in list mapping
            r_id = r.request_id

            if r_id not in requests_mapping.keys():
                tokenized_input = r.prompt_ids
                input_length = len(tokenized_input)
                # postprocessor
                sampling_param = r.sample_params
                multimodal_params = None
                assert r.req_status == ReqRunStatus.WAIT_IN_QUEUE
                r_obj = InferReq(r_id, 
                                input_token_ids=tokenized_input,
                                out_token_id_count=collections.defaultdict(int), 
                                sampling_param=InferSamplingParams(
                                    presence_penalty=sampling_param.presence_penalty,
                                    frequency_penalty=sampling_param.frequency_penalty,
                                    repetition_penalty=sampling_param.repetition_penalty,
                                    temperature=sampling_param.temperature,
                                    top_p=sampling_param.top_p,
                                    top_k=sampling_param.top_k,
                                    vocab_size=vocab_size,
                                ), 
                                multimodal_params=multimodal_params,
                                req_idx=nopad_b_req_idx[index], 
                                prompt_len=input_length,
                                req_status=r.req_status,
                                prompt_cache_len=r.prompt_cache_len,
                                prompt_cache_req_id=r.prompt_cache_req_id)
                requests_mapping[r_id] = r_obj
                index += 1
            else:
                if requests_mapping[r_id].req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
                    r_obj : InferReq = requests_mapping[r_id]
                    r_obj.req_status = ReqRunStatus.RERUNNING_FROM_OFFLOAD
                elif requests_mapping[r_id].req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
                    r_obj : InferReq = requests_mapping[r_id]
                    r_obj.req_status = ReqRunStatus.RERUNNING_FROM_KVKEEP
                else:
                    assert False, f"should not exist {requests_mapping[r_id].req_status}"
            
            request_ids.append(r_id)
            
            # 初始化之后 所有请求状态置换为 RUNNING 状态
            r_obj.req_status = ReqRunStatus.RUNNING

        return cls(
            batch_id=batch_id,
            request_ids=request_ids,
        ), need_alloc_size
    
    @torch.no_grad()
    def free_self(self):
        req_idx_list = []
        cur_kv_len_list = []
        for request_id in self.request_ids:
            req : InferReq = requests_mapping.pop(request_id)
            req_idx_list.append(req.req_idx)
            cur_kv_len_list.append(req.cur_kv_len)
        if len(requests_mapping) == 0:
            requests_mapping.clear()
        return req_idx_list, cur_kv_len_list
    
    @torch.no_grad()
    def filter(self, request_ids: List[int], finished_request_ids: List[int]):
        if len(requests_mapping) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self
        if len(request_ids) == 0:
            return InferBatch(
                batch_id=self.batch_id,
                request_ids=[],
            ), self.free_self()
        
        req_idx_list = []
        cur_kv_len_list = []

        for request_id in finished_request_ids:
            req : InferReq = requests_mapping.pop(request_id)
            req_idx_list.append(req.req_idx)
            cur_kv_len_list.append(req.cur_kv_len)
        
        return InferBatch(
            batch_id=self.batch_id,
            request_ids=request_ids,
        ), req_idx_list, cur_kv_len_list

    @torch.no_grad()
    def pause_reqs(self, pause_reqs: List[Tuple[int, ReqRunStatus]]):
        req_idx_list = []
        cur_kv_len_list = []
        for request_id, pause_way in pause_reqs:
            req : InferReq = requests_mapping[request_id]
            req.req_status = pause_way
            self.request_ids.remove(request_id)
            if pause_way == ReqRunStatus.PAUSED_AND_OFFLOAD:
                # 现在只支持全卸载一个请求的所有 kv 了
                req_idx_list.append(req.req_idx)
                cur_kv_len_list.append(req.cur_kv_len)
                req.cur_kv_len = 0
        return self, req_idx_list, cur_kv_len_list

    @classmethod
    @torch.no_grad()
    def merge(cls, batch1, batch2):
        request_ids = batch1.request_ids + batch2.request_ids
        
        return InferBatch(
            batch_id=batch1.batch_id,
            request_ids=request_ids,
        )

    def __len__(self):
        return len(self.request_ids)
    
