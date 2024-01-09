'''
TODO: 这一整个文件都需要大改，完全去掉它的 rpc 调用，改成直接调用 model 的函数
      其他的类应该可以先留着
'''


import torch
from typing import Dict, List, Optional, Tuple
from transformers.models.llama import LlamaConfig
from router.model_infer.infer_batch import InferBatch

from model.llama import LlamaForCausalLM

from preprocess import prepare_decode_inputs, prepare_prefill_inputs
from postprocess import sample
from sampler.sampling_metadata import SamplingParams, _prepare_sample
from infer_batch import requests_mapping
from infer_batch import InferReq




class ModelRpcServer():

    def init_model(self, config: LlamaConfig, **kvargs):        

        self.return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)

        self.cache : Dict[int, InferBatch] = {}

        weight_dir = kvargs["weight_dir"]
        max_total_token_num = kvargs["max_total_token_num"]

        model_kvargs = {
            "weight_dir": weight_dir,
            "max_total_token_num": max_total_token_num,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "return_all_prompt_logprobs": self.return_all_prompt_logprobs
        }

        self.model = LlamaForCausalLM(config, **model_kvargs)
        
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
        # special code for return all prompt_logprobs
        if self.return_all_prompt_logprobs and is_prefill:
            return self._prefill_to_return_all_prompt_logprobs(batch_id)
        
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
    
    @torch.no_grad()
    def _prefill_to_return_all_prompt_logprobs(self, batch_id: int):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs, not_run_reqs = prepare_prefill_inputs(batch)
        
        if len(run_reqs) >= 1:
            prompt_all_logits = self.model.forward(**kwargs)
            input_ids = kwargs["input_ids"]
            b_start_loc = kwargs["b_start_loc"]
            b_seq_len = kwargs["b_seq_len"]            
            last_index = torch.cumsum(b_seq_len, dim=0, dtype=torch.long) - 1
            logits = prompt_all_logits[last_index, :]
            # TODO: 这里需要修改传参
            next_token_ids, next_token_probs = self.sampler(logits, run_reqs)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            
            b_start_loc = b_start_loc.cpu().numpy()
            b_seq_len = b_seq_len.cpu().numpy()
            for req_obj, next_token_id, next_token_logprob, start_loc, seq_len in zip(run_reqs, next_token_ids, next_token_logprobs, b_start_loc, b_seq_len):
                # prefill and decode is same
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    'id': int(next_token_id),
                    'logprob': float(next_token_logprob),
                }

                cur_ids: torch.Tensor = input_ids[start_loc : start_loc + seq_len]
                cur_logits = prompt_all_logits[start_loc : start_loc + seq_len]
                cur_logprobs = torch.log_softmax(cur_logits, dim=-1, dtype=torch.float)[0:-1, :]
                cur_logprobs = torch.gather(cur_logprobs, dim=1, index=cur_ids[1:].view(-1, 1)).detach().cpu().numpy()

                cur_ids = cur_ids.cpu().numpy()
                all_prompts = []
                for index in range(len(cur_ids) - 1):
                    tmp_dict = {
                        int(cur_ids[index + 1]) : float(cur_logprobs[index, 0])
                    }
                    all_prompts.append([int(cur_ids[index]), tmp_dict])

                metadata["prompt_logprobs"] = all_prompts
                metadata["prompt_token_ids"] = [int(e) for e in cur_ids]
                output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata) # 状态， cur_kv_len, token_id, metadata

        for req_obj in not_run_reqs:
            output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, None, None) # 状态， cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        return output_dict
