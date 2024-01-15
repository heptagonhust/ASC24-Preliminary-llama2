from typing import Dict, List, Optional
from model.model_metadata import ParallelConfig
from sampler.sampling_params import SamplingParams
from router.io_struct import Req, NormalReq, Batch
from router.model_infer.model_rpc import ModelRpcServer
from router.req_queue import ReqQueue
from router.io_struct import BatchTokenIdOut, AbortReq, ReqRunStatus, ReqDetokenizationState, BatchStrOut
from router.pause_strategy import Fcfs, select_paused_reqs

from transformers import AutoTokenizer


import zmq
import zmq.asyncio
import asyncio

from utils.log_utils import init_logger

logger = init_logger(__name__)


class RouterManager:
    '''RouterManager manages batch scheduling and token management.

    Args:
        model_dir: 模型路径
        batch_size: 批处理最大的大小
        max_total_token_num: 模型最大长度
        max_req_num: 同时到来的最大请求数
        max_req_total_len: 输入+输出最大长度
        router_token_ratio: router的token占用率
        router_max_new_token_len: router最大的新token长度
        router_port: router的端口
        req_server_port: req_server的端口
    '''

    def __init__(
            self,
            model_dir,
            model_llama_config,
            batch_size,
            max_total_token_num,
            max_req_num,
            max_req_total_len,
            router_token_ratio,
            router_max_new_token_len,
            router_port,
            req_port,
            parallel_config_llama: ParallelConfig,
        ):
        self.batch_size = batch_size
        self.weight_dir = model_dir
        self.max_total_token_num = max_total_token_num
        self.max_req_num = max_req_num
        self.max_req_total_len = max_req_total_len
        self.router_token_ratio = router_token_ratio
        self.router_max_new_token_len = router_max_new_token_len
        
        self.pause_strategy = Fcfs()
        self.running_batch: Batch = None
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        self.req_id_to_out = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )

        self.model_llama_config = model_llama_config

        self.eos_id = self.tokenizer.eos_token_id

        self.prompt_cache_strs = []

        self.model_rpc_server = ModelRpcServer()

        self.parallel_config_llama = parallel_config_llama

        context = zmq.asyncio.Context(2)

        self.recv_from_req_server = context.socket(zmq.PULL)
        self.recv_from_req_server.bind(f"ipc:///tmp/router.ipc")
        # self.recv_from_req_server.bind(f"tcp://127.0.0.1:{router_port}")

        self.send_to_req_server = context.socket(zmq.PUSH)
        self.send_to_req_server.connect(f"ipc:///tmp/req_server.ipc")
        # self.send_to_req_server.connect(f"tcp://127.0.0.1:{req_port}")

        return

    def wait_to_model_ready(self):
        # 初始化模型
        model_args = {
            "weight_dir" : self.weight_dir,
            "max_total_token_num" : self.max_total_token_num,
            "max_req_num" : self.max_req_num + 8, # 最大同时发起的请求数
            "max_seq_length" : self.max_req_total_len + 8, # 最大的请求长度
            "return_all_prompt_logprobs" : False,
            "parallel_config_llama" : self.parallel_config_llama,
        }
        self.model_rpc_server.init_model(self.model_llama_config, model_args)

        self._init_prompt_cache()

        args = {
            "max_total_token_num" : self.max_total_token_num,
            "batch_max_tokens" : self.batch_size,
            "running_max_req_size" : self.max_req_num,
            "router_token_ratio" : self.router_token_ratio,
            "router_max_new_token_len" : self.router_max_new_token_len,
        }
        
        self.req_queue = ReqQueue(args, 
                                  self.prompt_cache_used_tokens, 
                                  self.prompt_cache_req_num)   
        return
    
    def _init_prompt_cache(self):
        """
        初始化 prompt cache 特性, 这个地方的id 分配要于 httpserver 中的id 分配对齐
        """
        # 初始化 prompt cahce， 然后初始化请求队列
        self.prompt_cache_used_tokens = 0
        self.prompt_cache_req_num = len(self.prompt_cache_strs)
        return

    def add_req(
        self,
        request_id: int,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        prompt_cache_len, 
        prompt_cache_req_id
    ):
        # logger.info(f"add_req: {request_id}")
        req = NormalReq(request_id, prompt_ids, sampling_params,
                            prompt_cache_len, prompt_cache_req_id)
        self.req_queue.append(req)
        return

    def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True

        # deleting req_id from req_id_to_out
        # it's for detokenization originally
        if request_id in self.req_id_to_out:
            del self.req_id_to_out[request_id]
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        logger.info("start loop_for_fwd")
        while True:
            self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    total_used_tokens = self.prompt_cache_used_tokens + self.running_batch.batch_used_tokens + self.req_queue.pause_req_used_tokens
                    token_ratio = total_used_tokens / self.max_total_token_num
                    print(
                        f"current batch size: {len(self.running_batch.reqs)} " 
                        f"paused req num: {len(self.req_queue.pause_req_dict)} "
                        f"token used ratio: {token_ratio} "
                    )
            
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms
                

    def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                logger.info(f"new_batch: {new_batch.batch_id}, requests count: {len(new_batch.reqs)}")
                self.running_batch = new_batch
                self._prefill_batch(self.running_batch)
                self._filter_running_batch()
                self.has_wait_tokens = 0
            return

        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.has_wait_tokens >= self.max_wait_tokens:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                return

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        if self._can_decode(self.running_batch):
            self._decode_batch(self.running_batch)
            self._filter_running_batch()
            self.has_wait_tokens += 1
            return
        else:
            # pause strategy
            paused_reqs = select_paused_reqs(self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num)
            self._pause_reqs(self.running_batch, paused_reqs)
            self.has_wait_tokens = 0   
            return
        return

    def _init_batch(self, batch: Batch):
        reqs = batch.reqs
        ret = self.model_rpc_server.add_batch(batch.batch_id, reqs, "fp16")
        req_to_req_status = ret
        
        self._update_init_status_to_batch(batch, req_to_req_status)
        return

    def _prefill_batch(self, batch:Batch):
        self._init_batch(batch)
        # 在 非 splitfuse 模式下，才需要真的执行 prefill 的操作。
        ret = self.model_rpc_server.prefill_batch(batch.batch_id)
        req_to_out_status = ret

        self._update_out_status_to_batch(batch, req_to_out_status)
        unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id)
        
        logger.info(f"batch: {batch.batch_id}, req_id and has_generate_finished: {[(req.request_id, req.has_generate_finished) for req in batch.reqs]}")
         
        # self._send_to_detokenization_proc(batch, req_to_out_status)
        detokenize_res = self._detokenize(batch, req_to_out_status)
        logger.info(f"detokenize_res: {detokenize_res}")
        for res in detokenize_res:
            self.send_to_req_server.send_pyobj(res)
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
        self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        return


    def _decode_batch(self, batch:Batch):
        req_to_out_status = self.model_rpc_server.decode_batch(batch.batch_id)

        self._update_out_status_to_batch(batch, req_to_out_status)
        unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id)

        logger.info(f"batch: {batch.batch_id}, req_id and has_generate_finished: {[(req.request_id, req.has_generate_finished) for req in batch.reqs]}")

        detokenize_res = self._detokenize(batch, req_to_out_status)
        logger.info(f"detokenize_res: {detokenize_res}")
        for res in detokenize_res:
            self.send_to_req_server.send_pyobj(res)
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
        self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        return

    def _filter_batch(self, batch: Batch, unfinished_req_ids, finished_req_ids: List):
        self.model_rpc_server.filter_batch(batch.batch_id, unfinished_req_ids, finished_req_ids)
        return

    def _merge_batch(self, batch1: Batch, batch2: Batch):
        self.model_rpc_server.merge_batch(batch1.batch_id, batch2.batch_id)
        return

    def _remove_batch(self, batch: Batch):
        self.model_rpc_server.remove_batch(batch.batch_id)
        return
    
    def _pause_reqs(self, batch: Batch, pasue_reqs: List[Req]):
        pasue_reqs_info = [(r.request_id, r.req_status) for r in pasue_reqs]
        self.model_rpc_server.pause_reqs(batch.batch_id, pasue_reqs_info)
        return

    def _handle_finish_req(self, batch: Batch, unfinished_req_ids, finished_req_ids):
        if len(finished_req_ids) != 0:
            if batch.is_clear():
                self._remove_batch(batch)
            else:
                self._filter_batch(batch, unfinished_req_ids, finished_req_ids)
        return

    def _filter_running_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return
    
    def _update_init_status_to_batch(self, batch: Batch, req_to_req_status):
        # 更新请求状态
        new_batch_used_tokens = 0
        new_batch_decode_need_tokens = 0 # 只有在 splitfuse 模式下有意义
        for req_id, (req_status, cur_kv_len) in req_to_req_status.items():
            r_obj = batch.id_to_reqs[req_id]
            r_obj.req_status = req_status
            r_obj.cur_kv_len = cur_kv_len
            new_batch_used_tokens += r_obj.get_used_tokens()
            new_batch_decode_need_tokens += r_obj.get_decode_need_tokens()
        
        batch.batch_used_tokens = new_batch_used_tokens
        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return
    
    def _update_out_status_to_batch(self, batch: Batch, req_to_out_status):
        new_batch_used_tokens = 0
        new_batch_decode_need_tokens = 0 # 只有在 splitfuse 模式下有意义
        for req_id, (req_status, cur_kv_len, new_token_id, new_gen_metadata) in req_to_out_status.items():
            req : Req = batch.id_to_reqs[req_id]
            req.req_status = req_status
            req.cur_kv_len = cur_kv_len
            if new_token_id is not None:
                req.output_ids.append(new_token_id)
                req.output_metadata_list.append(new_gen_metadata)
            new_batch_used_tokens += req.get_used_tokens()
            new_batch_decode_need_tokens += req.get_decode_need_tokens()
        
        batch.batch_used_tokens = new_batch_used_tokens
        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return
        
    def _can_decode(self, batch: Batch):
        total_used_tokens = self.prompt_cache_used_tokens + batch.batch_used_tokens + self.req_queue.pause_req_used_tokens
        remaining_tokens = self.max_total_token_num - total_used_tokens
        return batch.batch_decode_need_tokens <= remaining_tokens
    
    def _detokenize(self, batch: Batch, req_to_out_status):
        '''
        fuck it, i should choose finished reqs and detokenize them and return them only
        '''
        batch_out = BatchTokenIdOut()
        for req_id, (_, _, new_token_id, new_gen_metadata) in req_to_out_status.items():
            req = batch.id_to_reqs[req_id]
            if new_token_id is not None:
                batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
        
        new_batch_str_out = BatchStrOut()
        for req_id, req in batch.id_to_reqs.items():

            if not req.has_generate_finished:
                continue

            req_status, req_cur_kv_len, next_token_id, metadata = req_to_out_status[req.request_id]

            out_text = req.output_ids
            out_text.append(next_token_id)

            out_text = self.tokenizer.decode(
                out_text,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            if out_text.endswith(u'\ufffd'):
                new_text = ''
            else:
                new_text = out_text
            
            new_batch_str_out.reqs_infs.append(
                (req.request_id, new_text, new_gen_metadata, True if req.aborted else req.has_generate_finished, req.aborted))
            if req.has_generate_finished or req.aborted:
                try:
                    del self.req_id_to_out[req_id]
                except:
                    pass
        return new_batch_str_out.reqs_infs

        
    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_req_server.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 5:
                request_id, prompt_ids, sampling_params, prompt_cache_len, prompt_cache_req_id = recv_req
                self.add_req(request_id, prompt_ids, sampling_params, prompt_cache_len, prompt_cache_req_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                self.abort(request_id)
            else:
                assert False, f"Error Req Inf {recv_req}"


def start_router_process(
        model_dir,
        model_llama_config,
        batch_size,
        max_total_token_num,
        max_req_num,
        max_req_total_len,
        router_token_ratio,
        router_max_new_token_len,
        router_port,
        req_port,
        parallel_config_llama,
        pipe_writer
    ):
    '''Helper function to start router process.

    Args:
        model_dir: 模型路径
        model_llama_config: llama 模型配置
        batch_size: 批处理最大的大小
        max_total_token_num: 模型最大长度
        max_req_num: 同时到来的最大请求数
        max_req_total_len: 输入+输出最大长度
        router_token_ratio: router的token占用率
        router_max_new_token_len: router最大的新token长度
        router_port: router的端口
        req_server_port: req_server的端口
    '''
    try:
        router = RouterManager(
            model_dir,
            model_llama_config,
            batch_size,
            max_total_token_num,
            max_req_num,
            max_req_total_len,
            router_token_ratio,
            router_max_new_token_len,
            router_port,
            req_port,
            parallel_config_llama
        )
        router.wait_to_model_ready()
    except Exception as e:
        import traceback
        import sys
        etype, evalue, tb = sys.exc_info()
        err_str = '\n'.join(traceback.format_exception(etype, evalue, tb))
        pipe_writer.send(err_str)
        raise
    

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return