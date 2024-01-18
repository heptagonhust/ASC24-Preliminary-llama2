import asyncio
import torch
import rpyc
from typing import Dict, List, Optional, Tuple
from transformers.models.llama import LlamaConfig
from router.model_infer.infer_batch import InferBatch

from model.llama import LlamaForCausalLM

from router.model_infer.preprocess import prepare_decode_inputs, prepare_prefill_inputs
from router.model_infer.postprocess import sample
from sampler.sampling_metadata import SamplingParams, _prepare_sample
from router.model_infer.infer_batch import requests_mapping
from router.model_infer.infer_batch import InferReq

from model.parallel_utils.parallel_state import setup_distributed
from model.model_metadata import ParallelConfig, ModelConfig
import random
import numpy as np
import contextlib
from rpyc.utils.classic import obtain

import os

from utils.log_utils import init_logger
logger = init_logger(__name__)


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

class ModelRpcServer(rpyc.Service):

    def exposed_init_model(self, config: LlamaConfig, kwargs: Dict[str, any]):
        '''
        things need to be passed by kwargs:
        weight_dir: str
        max_total_token_num: int
        max_req_num: int
        max_seq_length: int
        return_all_prompt_logprobs: bool
        
        seed: int
        parallel_config_llama: ParallelConfig
        dtype: torch.dtype
        '''
        pp_size, tp_size = kwargs["parallel_config_llama"]
        self.parallel_config = ParallelConfig(pipeline_parallel_size=pp_size, tensor_parallel_size=tp_size)
        world_size = self.parallel_config.world_size
        if world_size != 1:
            kwargs = obtain(kwargs)
            config = obtain(config)

        logger.info(f"init_model: {config}, {kwargs}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # set random seeds
        self.seed = kwargs["seed"]
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.return_all_prompt_logprobs = kwargs["return_all_prompt_logprobs"]

        self.rank_id = kwargs["rank_id"]

        self.cache : Dict[int, InferBatch] = {}

        weight_dir = kwargs["weight_dir"]
        max_total_token_num = kwargs["max_total_token_num"]

        model_kwargs = {
            "weight_dir": weight_dir,
            "max_total_token_num": max_total_token_num,
            "max_req_num": kwargs["max_req_num"],
            "max_seq_length": kwargs["max_seq_length"],
            "return_all_prompt_logprobs": self.return_all_prompt_logprobs,
            "parallel_config_llama": self.parallel_config,
        }

        logger.info(f"parallel config: {self.parallel_config.world_size}, {self.parallel_config.pipeline_parallel_size}, {self.parallel_config.tensor_parallel_size}")

        self._setup_distributed(self.parallel_config, self.rank_id)

        self.dtype = torch.float16
        with _set_default_torch_dtype(self.dtype):
            self.model = LlamaForCausalLM(config, **model_kwargs)
            self.model.to(device=device)
            self.model.load_weights(weight_dir)
        
        self.device = device
        
        return
    
    def exposed_add_batch(self, batch_id, reqs, dtype):
        logger.info(f"entering add_batch in rpc server")
        if self.parallel_config.world_size != 1:
            batch_id, reqs, dtype = obtain(batch_id), obtain(reqs), obtain(dtype)
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
    
    def exposed_prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    def exposed_decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    def exposed_filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        logger.info(f"entering filter_batch in rpc server")
        if self.parallel_config.world_size != 1:
            batch_id, req_id_list, finished_req_id_list = obtain(batch_id), obtain(req_id_list), obtain(finished_req_id_list)
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    def exposed_pause_reqs(self, batch_id, req_list):
        if self.parallel_config.world_size != 1:
            batch_id, req_list = obtain(batch_id), obtain(req_list)
        batch1 = self.cache.pop(batch_id)
        batch2 = batch1.pause_reqs(req_list)
        self.cache[batch_id] = batch2
        del batch1
        return

    def exposed_merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    def exposed_remove_batch(self, batch_id):
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
    
    def _setup_distributed(self, parallel_config_llama: ParallelConfig, rank_id: int):
        setup_distributed(parallel_config_llama)
    
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

class ModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: ModelRpcServer = model_rpc
        self.world_size = world_size
        self.rpc_server_process = rpc_server_process
        self.use_rpc = True
        if self.use_rpc:
            def async_wrap(f):
                f = rpyc.async_(f)
                async def _func(*args, **kwargs):
                    ans = f(*args, **kwargs)
                    await asyncio.to_thread(ans.wait)
                    # raise if exception
                    return ans.value
                return _func
            self._init_model = async_wrap(self.model.init_model)
            self._add_batch = async_wrap(self.model.add_batch)
            self._prefill_batch = async_wrap(self.model.prefill_batch)
            self._decode_batch = async_wrap(self.model.decode_batch)
            self._pause_reqs = async_wrap(self.model.pause_reqs)
            self._filter_batch = async_wrap(self.model.filter_batch)
            self._merge_batch = async_wrap(self.model.merge_batch)
            self._remove_batch = async_wrap(self.model.remove_batch)
        else:
            self._init_model = self.model.exposed_init_model
            self._add_batch = self.model.exposed_add_batch
            self._prefill_batch = self.model.exposed_prefill_batch
            self._decode_batch = self.model.exposed_decode_batch
            self._pause_reqs = self.model.exposed_pause_reqs
            self._filter_batch = self.model.exposed_filter_batch
            self._merge_batch = self.model.exposed_merge_batch
            self._remove_batch = self.model.exposed_remove_batch
        return

    async def init_model(self, config_llama, kvargs):
        ans : rpyc.AsyncResult = self._init_model(config_llama, kvargs)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def init_batch(self, batch_id, reqs):
        ans = self._add_batch(batch_id, reqs, "fp16")
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def prefill_batch(self, batch_id):
        ans = self._prefill_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def decode_batch(self, batch_id):
        ans = self._decode_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        logger.info(f"entering filter_batch in rpc client")
        ans = self._filter_batch(batch_id, req_id_list, finished_req_id_list)
        if self.use_rpc:
            await ans
            return
        else:
            return 

    async def pause_reqs(self, batch_id, reqs_list):
        ans = self._pause_reqs(batch_id, reqs_list)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def merge_batch(self, batch_id1, batch_id2):
        ans = self._merge_batch(batch_id1, batch_id2)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def remove_batch(self, batch_id):
        ans = self._remove_batch(batch_id)
        if self.use_rpc:
            await ans
            return
        else:
            return


