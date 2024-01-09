'''
TODO: 这一整个文件都需要大改，完全去掉它的 rpc 调用，改成直接调用 model 的函数
      其他的类应该可以先留着
'''


import uuid
from typing import Dict, List, Optional
from sampler.sampling_params import SamplingParams
from io_struct import Req, NormalReq, SplitFuseReq, Batch
from model_infer.model_rpc import start_model_process, ModelRpcClient
from req_queue import ReqQueue
from io_struct import BatchTokenIdOut, AbortReq, ReqRunStatus
from .pause_strategy import Fcfs, select_paused_reqs
from tokenizer import get_tokenizer



class RouterManager:

    def __init__(self, args):
        self.args = args
        self.model_weightdir = args.model_dir
        self.world_size = args.tp
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        
        self.pause_strategy = Fcfs()
        self.running_batch: Batch = None
        self.eos_id = args.eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10

        return

    async def wait_to_model_ready(self):
        # 初始化模型
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            kvargs = {
                "rank_id" : rank_id,
                "world_size" : self.world_size,
                "weight_dir" : self.model_weightdir,
                "load_way" : self.load_way,
                "max_total_token_num" : self.max_total_token_num,
                "mode" : self.mode,
                "max_req_num" : self.args.running_max_req_size + 8,
                "max_seq_length" : self.args.max_req_total_len + 8, # 留一点余量
                "return_all_prompt_logprobs" : self.args.return_all_prompt_logprobs
            }
            init_model_ret.append(self.model_rpcs[rank_id].init_model(kvargs))

        await asyncio.gather(*init_model_ret)

        await self._init_prompt_cache()
        
        self.req_queue = ReqQueue(self.args, 
                                  self.prompt_cache_used_tokens, 
                                  self.prompt_cache_req_num)   
        return
    
    async def _init_prompt_cache(self):
        """
        初始化 prompt cache 特性, 这个地方的id 分配要于 httpserver 中的id 分配对齐
        """
        # 初始化 prompt cahce， 然后初始化请求队列
        self.prompt_cache_used_tokens = 0
        self.prompt_cache_req_num = len(self.args.prompt_cache_strs)
        if self.is_splitfuse_mode:
            reqs = []
            id = -1 # id 从 -1， -2， .... 避免和正常的 id 占用
            for prompt_cache_str in self.args.prompt_cache_strs:
                prompt_ids = self.tokenizer.encode(prompt_cache_str)
                req = NormalReq(id, prompt_ids, SamplingParams(stop_sequences=[]))
                self.prompt_cache_used_tokens += len(prompt_ids)
                reqs.append(req)
                id -= 1
            if len(reqs) != 0:
                self.prompt_cache_batch = Batch(uuid.uuid4().hex, reqs)
                await self._prefill_to_init_prompt_cache(self.prompt_cache_batch)
        return

    def add_req(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_cache_len, 
        prompt_cache_req_id
    ):  
        req = NormalReq(request_id, prompt_ids, sampling_params,
                            prompt_cache_len, prompt_cache_req_id)
        self.req_queue.append(req)
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    total_used_tokens = self.prompt_cache_used_tokens + self.running_batch.batch_used_tokens + self.req_queue.pause_req_used_tokens
                    token_ratio = total_used_tokens / self.max_total_token_num

                    pass
                

    async def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                self.running_batch = new_batch
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.has_wait_tokens >= self.max_wait_tokens:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                await self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                return

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        if self._can_decode(self.running_batch):
            await self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            # pause strategy
            paused_reqs = select_paused_reqs(self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num)
            await self._pause_reqs(self.running_batch, paused_reqs)
            self.has_wait_tokens = 0
            return
        return

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_req_status = obtain(ans[0])
        else:
            req_to_req_status = ans[0]
        
        self._update_init_status_to_batch(batch, req_to_req_status)
        return

    async def _prefill_batch(self, batch:Batch):
        await self._init_batch(batch)
        # 在 非 splitfuse 模式下，才需要真的执行 prefill 的操作。
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_status = obtain(ans[0])
        else:
            req_to_out_status = ans[0]

        self._update_out_status_to_batch(batch, req_to_out_status)
        unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_status)
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
        await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        return
    
    async def _prefill_to_init_prompt_cache(self, batch:Batch):
        """
        专用于初始化prompt cahce 请求的接口, 只在 splitfuse + prompt cache 模式下调用
        """
        await self._init_batch(batch)
        # 在 splitfuse 模式下，才需要真的执行 prefill 的操作。
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_status = obtain(ans[0])
        else:
            req_to_out_status = ans[0]

        self._update_out_status_to_batch(batch, req_to_out_status)
        return

    async def _decode_batch(self, batch:Batch):
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_status = obtain(ans[0])
        else:
            req_to_out_status = ans[0]

        self._update_out_status_to_batch(batch, req_to_out_status)
        unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_status)
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
        await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        return

    async def _filter_batch(self, batch: Batch, unfinished_req_ids, finished_req_ids: List):
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, unfinished_req_ids, finished_req_ids) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return
    
    async def _pause_reqs(self, batch: Batch, pasue_reqs):
        pasue_reqs_info = [(r.request_id, r.req_status) for r in pasue_reqs]
        rets = [self.model_rpcs[tp_rank].pause_reqs(batch.batch_id, pasue_reqs_info) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, unfinished_req_ids, finished_req_ids):
        if len(finished_req_ids) != 0:
            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch, unfinished_req_ids, finished_req_ids)
        return

    def _filter_runing_batch(self):
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
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (_, _, new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            if new_token_id is not None:
                batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
            return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 6:
                prompt_ids, sampling_params, multimodal_params, request_id, prompt_cache_len, prompt_cache_req_id = recv_req
                self.add_req(prompt_ids, sampling_params, multimodal_params, request_id, prompt_cache_len, prompt_cache_req_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return

def start_router_process(args, router_port, detokenization_port, model_rpc_ports, pipe_writer):
    try:
        router = RouterManager(
            args,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports)
    
        asyncio.run(router.wait_to_model_ready())
    except Exception as e:
        import traceback
        import sys
        etype, evalue, tb = sys.exc_info()
        err_str = '\n'.join(traceback.format_exception(etype, evalue, tb))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
