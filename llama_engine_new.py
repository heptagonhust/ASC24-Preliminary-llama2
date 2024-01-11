from sampler.sampling_metadata import SamplingParams, _prepare_sample
from model.parallel_utils.parallel_state import setup_distributed
from model.model_metadata import ModelConfig, ParallelConfig, PortConfig, ReqConfig
from model.llama import LlamaForCausalLM
from sequence.sequence import Sequence
import torch.distributed as dist
import numpy as np

from transformers import AutoTokenizer
from typing import List, Tuple
from tqdm import tqdm
import torch.nn as nn
import torch
import contextlib
import random

from router.manager import RouterManager
from req_server import ReqServer

import asyncio
import heapq


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

class RequestEngine():
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config_llama:ParallelConfig,
        port_config:PortConfig,
        sampling_params: SamplingParams = None,
        req_config: ReqConfig = None
    ) -> nn.Module:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # set random seeds
        random.seed(model_config.seed)
        torch.manual_seed(model_config.seed)
        np.random.seed(model_config.seed)
        torch.cuda.manual_seed_all(model_config.seed)
        
        setup_distributed(parallel_config_llama)
        with _set_default_torch_dtype(model_config.dtype):
            pass
        self.model_config = model_config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_config.tokenizer, 
                        trust_remote_code=self.model_config.trust_remote_code)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.requests_queue = asyncio.Queue()
        self.results_queue = []

        self.sampling_params = sampling_params

        self.http_server_manager = ReqServer(
            model_dir=model_config.model,
            max_total_token_num=model_config.max_model_len,

            max_req_total_len=req_config.max_req_total_len,

            router_port=port_config.router_port,
            req_server_port=port_config.req_server_port,
        )

    async def worker(self):
        while True:
            request_id, prompt, prompt_len, output_len = await self.requests_queue.get()
            result = await self.http_server_manager.generate(prompt, output_len, self.sampling_params, request_id)
            result_with_input_prompt = (request_id, prompt, result)
            heapq.heappush(self.results_queue, result_with_input_prompt)
            self.requests_queue.task_done()
            self.progress_bar.update(1)  # 更新进度条
        

    def generate(self, requests: List[Tuple[str, int, int]], 
                 sampling_params: SamplingParams = None):
        print("generating...")
        self.progress_bar = tqdm(total=len(requests), desc="generating")
        asyncio.run(self.process_dataset(requests))
    
    async def process_dataset(self, dataset):
        # 将数据集请求加入队列
        for request_id, (prompt, prompt_len, output_len) in enumerate(dataset):
            await self.requests_queue.put((request_id, prompt, prompt_len, output_len))

        # 启动工作器(只有1个)
        workers = [asyncio.create_task(self.worker()) for _ in range(1)]
        
        # 等待所有请求处理完毕
        await self.requests_queue.join()

        # 取消所有工作器
        for w in workers:
            w.cancel()

        # 关闭进度条
        self.progress_bar.close()

        # 按顺序输出结果
        while self.results_queue:
            print(heapq.heappop(self.results_queue)[1])

        
        

