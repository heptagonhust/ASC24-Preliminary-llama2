from manager.request_manager import RequestManager
from manager.tiny_batch_manager_metadata import *

class TinyBatchManager:
    """
    在每个 Worker pp 节点上，用于管理 ReqManager 的类。其作用对应了 lightllm 的
    ModelRpcServer 和 InferBatch，主要涉及对每个请求 kvcache 的管理和释放。

    由于 Worker 节点只应该涉及对请求 kvcache 的释放操作，
    这里的 TinyBatchManager 只需要知道哪些请求需要分配/释放 kvcache，
    分配/释放的量又有多少。这些由 BatchFreeMetadata 给出。

    Master 节点上仍然使用 ModelRpcServer 和 InferBatch，不使用 TinyBatchManager。
    """
    def __init__(self, req_manager: RequestManager) -> None:
        self.req_manager = req_manager

    def _init_batch(self, batch_init_metadata: BatchInitMetadata):
        """
        为一个 batch 分配 kvcache
        """
        self.req_manager.alloc(batch_init_metadata.need_alloc_size)
        return
    
    def _remove_batch(self, batch_free_metadata: BatchFreeMetadata):
        """
        释放一个 batch 的所有 kvcache
        """
        free_req_idx = []
        free_token_idx = []
        for req_idx, cur_kv_len in batch_free_metadata.req_list_to_free:
            free_req_idx.append(req_idx)
            free_token_idx.append(self.req_manager.req_to_token_indexs[req_idx][:cur_kv_len])
        free_token_idx = torch.cat(free_token_idx, dim=-1)
        self.req_manager.free(free_req_idx, free_token_idx)
        return
    
    def _filter_batch(self, batch_filter_metadata: BatchFilterMetadata):
        """
        释放一个 batch 已完成请求的 kvcache
        """
        free_req_idx = []
        free_token_idx = []
        for req_idx, cur_kv_len in batch_filter_metadata.req_list_to_free:
            free_req_idx.append(req_idx)
            free_token_idx.append(self.req_manager.req_to_token_indexs[req_idx][:cur_kv_len])
        free_token_idx = torch.cat(free_token_idx, dim=-1)
        self.req_manager.free(free_req_idx, free_token_idx)
        return
    
    def _pause_batch(self, batch_pause_metadata: BatchPauseMetadata):
        """
        释放一个 batch 已完成请求的 kvcache
        """
        for req_idx, cur_kv_len in batch_pause_metadata.req_list_to_free:
            self.req_manager.free_token(self.req_manager.req_to_token_indexs[req_idx][:cur_kv_len])
        return
    
    def perform_op(self, tiny_batch_manager_op: TinyBatchManagerOp):
        """
        执行一个 TinyBatchManagerOp
        """
        if tiny_batch_manager_op.batch_op_kind == TinyBatchManagerOpKind.INIT:
            self._init_batch(tiny_batch_manager_op.batch_op_metadata)
        elif tiny_batch_manager_op.batch_op_kind == TinyBatchManagerOpKind.FILTER:
            self._filter_batch(tiny_batch_manager_op.batch_op_metadata)
        elif tiny_batch_manager_op.batch_op_kind == TinyBatchManagerOpKind.PAUSE:
            self._pause_batch(tiny_batch_manager_op.batch_op_metadata)
        elif tiny_batch_manager_op.batch_op_kind == TinyBatchManagerOpKind.REMOVE:
            self._remove_batch(tiny_batch_manager_op.batch_op_metadata)
        elif tiny_batch_manager_op.batch_op_kind == TinyBatchManagerOpKind.FORWARD:
            pass
        else:
            raise ValueError(f"Unknown batch_op_kind: {tiny_batch_manager_op.batch_op_kind}")
        return
    