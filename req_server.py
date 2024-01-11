import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import hashlib

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from transformers import AutoTokenizer
from router.io_struct import BatchStrOut, AbortReq
from typing import Generator, Any

from sampler.sampling_params import SamplingParams

class ReqServer:
    def __init__(
        self,
        model_dir,
        max_total_token_num,
        max_req_total_len,
        router_port,
        req_server_port,
    ):
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")


        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"tcp://127.0.0.1:{req_server_port}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )

        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)

        self.total_token_num = max_total_token_num
        self.max_req_total_len = max_req_total_len

        self._init_prompt_cache()
        return
    
    def _init_prompt_cache(self):
        """
        初始化 prompt cache 特性, 这个地方的id 分配要于 router 中 的id 分配对齐
        """
        self.prompt_cache_reqs = []
        return
    
    def _find_prompt_cache_req(self, token_ids):
        prompt_cache_len = 0
        prompt_cache_req_id = None
        for (req_id, prompt_ids) in self.prompt_cache_reqs:
            prompt_len = len(prompt_ids)
            if len(token_ids) > prompt_len:
                if token_ids[0 : prompt_len] == prompt_ids:
                    prompt_cache_len = prompt_len
                    prompt_cache_req_id = req_id
                    break
        return prompt_cache_len, prompt_cache_req_id

    async def generate(
        self,
        prompt,
        output_len,
        sampling_params: SamplingParams,
        request_id
    ) -> Generator[tuple, Any, None]:
        prompt_ids = self.tokenizer.encode(prompt)
        # special tokenizer for multimodal_params
        prompt_tokens = len(prompt_ids)

        req_total_len = prompt_tokens + output_len + 1
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num:
            raise ValueError(
                f"the req token total len + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}"
            )
        
        req_status = ReqStatus(request_id)
        event = req_status.event
        self.req_id_to_out_inf[request_id] = req_status

        # 寻找是否有可用的prompt cache 可用
        prompt_cache_len, prompt_cache_req_id = self._find_prompt_cache_req(prompt_ids)
  

        self.send_to_router.send_pyobj((request_id, prompt_ids, sampling_params, prompt_cache_len, prompt_cache_req_id))

        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

            async with req_status.lock:
                event.clear()
                if len(req_status.out_token_info_list) == 0:
                    continue

                for out_str, metadata, finished in req_status.out_token_info_list:
                    metadata["prompt_tokens"] = prompt_tokens
                    yield out_str, metadata, finished

                    if finished:
                        try:
                            del self.req_id_to_out_inf[request_id]
                        except:
                            pass
                        return
                req_status.out_token_info_list.clear()
        return

    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.send_to_router.send_pyobj(abort_req)
        try:
            req = self.req_id_to_out_inf[request_id]
            del self.req_id_to_out_inf[request_id]
        except:
            pass
        return

    async def handle_loop(self):
        while True:
            recv_ans: BatchStrOut = await self.recv_from_router.recv_pyobj()
            assert isinstance(
                recv_ans, BatchStrOut
            ), f"error recv type {type(recv_ans)}"
            for req_id, text, metadata, finished, abort in recv_ans.reqs_infs:
                try:
                    if not abort:
                        req_status : ReqStatus = self.req_id_to_out_inf[req_id]
                        async with req_status.lock: 
                            req_status.out_token_info_list.append((text, metadata, finished))
                            req_status.event.set()
                    else:
                        del self.req_id_to_out_inf[req_id]
                except:
                    pass
        return

class ReqStatus:
    def __init__(self, req_id) -> None:
        self.req_id = req_id
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.out_token_info_list = []
