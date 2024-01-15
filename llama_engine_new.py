import threading
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

import asyncio
import heapq
import zmq
import zmq.asyncio
from model.parallel_utils.parallel_state import setup_distributed
from model.parallel_utils.parallel_state import get_tensor_model_parallel_rank

import utils.log_utils as log_utils
import multiprocessing as mp

logger = log_utils.init_logger(__name__)



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
        self.model_config = model_config
        self.device = device

        self.requests_queue = asyncio.Queue()
        self.results_queue = []

        self.sampling_params = sampling_params
        # setup_distributed(parallel_config_llama)
        # tp_rank = get_tensor_model_parallel_rank()
        tp_rank = 0

        # context = zmq.asyncio.Context(2)
        # self.send_to_router = context.socket(zmq.PUSH)
        # try:
        #     self.send_to_router.connect(f"tcp://127.0.0.1:{port_config.router_port + tp_rank * 2}")
        # except Exception as e:
        #     self.send_to_router.connect(f"tcp://127.0.0.1:{port_config.router_port + 2}")


        # self.recv_from_router = context.socket(zmq.PULL)
        # try:
        #     self.recv_from_router.bind(f"tcp://127.0.0.1:{port_config.req_server_port + tp_rank * 2}")
        # except Exception as e:
        #     self.recv_from_router.bind(f"tcp://127.0.0.1:{port_config.req_server_port + 2}")

        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model,
            trust_remote_code=True
        )

        self.req_id_to_out_map = {}  # value type (out_str, metadata, finished)
        self.req_id_to_out_map_lock = asyncio.Lock()

        self.req_config = req_config

    def setqueue(self,queues):
        self.queue1 = queues[0] # send_to_router
        self.queue2 = queues[1] # recv_from_router

    async def worker(self, request_id, prompt, prompt_len, output_len):
        logger.info("worker")
        prompt_ids = self.tokenizer.encode(prompt)
        # special tokenizer for multimodal_params
        prompt_tokens = len(prompt_ids)

        req_total_len = prompt_tokens + output_len + 1
        
        req_status = ReqStatus(request_id)
        self.req_id_to_out_map[request_id] = req_status

        # 将 sampling_params 的 max_tokens 设置为 output_len
        sampling_params = self.sampling_params
        sampling_params.max_tokens = output_len

        logger.info(f"sending request_id:{request_id}, output_len:{output_len}")
        self.queue1.put((request_id, prompt_ids, sampling_params, 0, None))

    async def handler(self, request_num):
        '''
        handle the request from the router
        '''
        while request_num > 0:
            try:
                batch_str_out = self.queue2.get_nowait()
            except mp.queues.Empty:
                await asyncio.sleep(0.1)
                continue
                
            logger.info(f"batch_str_out: {batch_str_out}")
            req_id, out_ids, out_metadata, finished, aborted = batch_str_out
            logger.info(f"req_id:{req_id}, out_ids:{out_ids}, out_metadata:{out_metadata}, finished:{finished}")
            async with self.req_id_to_out_map_lock:
                req_status = self.req_id_to_out_map.get(req_id)
                req_status.out_token_info_list.append((out_ids, out_metadata, finished))
                if finished:
                    self.progress_bar.update(1)
                    


    def generate(self, requests: List[Tuple[str, int, int]], 
                 sampling_params: SamplingParams = None):
        logger.info("generating...")
        self.progress_bar = tqdm(total=len(requests), desc="generating")
        asyncio.run(self.process_dataset(requests))
    
    async def process_dataset(self, dataset):
        # 将数据集请求加入队列
        for request_id, (prompt, prompt_len, output_len) in enumerate(dataset):
            await self.worker(request_id, prompt, prompt_len, output_len)

        await self.handler(len(dataset))

        # 关闭进度条
        self.progress_bar.close()

        # 按顺序输出结果
        while self.results_queue:
            print(heapq.heappop(self.results_queue)[1])


class ReqStatus:
    def __init__(self, req_id) -> None:
        self.req_id = req_id
        self.out_token_info_list = []
    
    def __repr__(self) -> str:
        return f"ReqStatus(req_id={self.req_id}, out_token_info_list={self.out_token_info_list})"