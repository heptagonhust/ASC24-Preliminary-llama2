import torch
import numpy as np
from typing import List, Dict


from .infer_batch import requests_mapping, InferReq, InferBatch
from io_struct import ReqRunStatus

#@calculate_time(show=True, min_cost_ms=1)
def prepare_prefill_inputs(batch: InferBatch):
    run_reqs: List[InferReq] = []
    not_run_reqs: List[InferReq] = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    for request_id in batch.request_ids:
        req : InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        # 当请求已经存在 cur_kv_len 不为 0 的时候，就不需要做全 prefill 操作了，
        # 说明是从 RERUNNING_FROM_KVKEEP 中 恢复的请求
        if req.cur_kv_len != 0: 
            not_run_reqs.append(req)
            continue
        
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        
        seq_len = len(req.input_token_ids)
        input_id = req.input_token_ids
        
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len
    
    if len(run_reqs) >= 1:
        
        input_ids = np.concatenate(input_ids, dtype=np.int64)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
        nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
        nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
        nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
        kwargs = {
            "batch_size": len(batch),
            "total_token_num": nopad_total_token_num,
            "max_len_in_batch": nopad_max_len_in_batch,
            "input_ids": input_ids,
            "positions": nopad_b_seq_len,  # reuse b_seq_len
            "b_req_idx": nopad_b_req_idx,
            "b_start_loc": nopad_b_start_loc,
            "b_seq_len": nopad_b_seq_len,
            "is_prefill": True,
            "input_metadata": None
        }
        return kwargs, run_reqs, not_run_reqs
    else:
        return {}, run_reqs, not_run_reqs
    
#@calculate_time(show=True, min_cost_ms=1)
def prepare_decode_inputs(batch:InferBatch):
    run_reqs, not_run_reqs = [], []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    for request_id in batch.request_ids:
        req : InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        input_id = req.input_token_ids[-1]
        seq_len = len(req.input_token_ids)
        assert req.cur_kv_len == seq_len - 1
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len
    
    if len(run_reqs) >= 1:

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
        nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
        nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
        nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
        kwargs = {
            "batch_size": len(batch),
            "total_token_num": nopad_total_token_num,
            "max_len_in_batch": nopad_max_len_in_batch,
            "input_ids": input_ids,
            "positions": nopad_b_seq_len,  # reuse b_seq_len
            "b_req_idx": nopad_b_req_idx,
            "b_start_loc": nopad_b_start_loc,
            "b_seq_len": nopad_b_seq_len,
            "is_prefill": False,
            "input_metadata": None
        }
        return kwargs, run_reqs, not_run_reqs
    else:
        return {}, run_reqs, not_run_reqs


    
