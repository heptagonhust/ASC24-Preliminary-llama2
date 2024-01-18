from typing import List, Tuple, Union, Dict
import enum

import torch

import torch.nn.functional as F


class TinyBatchManagerOpKind(enum.Enum):
    """
    TinyBatchManager 的操作类型；
    只有涉及 ReqManager 的 free 操作才需要这个。
    """

    """
    FORWARD 包括 prefill 和 decode 操作，
    当 pp 节点收到 FORWARD 操作时，直接进行模型前向。
    """
    FORWARD = 0

    """
    以下操作都只对 ReqManager 进行相应的 alloc 和 free 操作，不进行模型前向。
    元数据会在下面以类的形式给出。

    一个需要注意的是，MERGE 不需要在 pp 节点间传递，因为它没有对 ReqManager
    进行操作，因此也不涉及 kvcache 的 free。
    """
    INIT    = 1
    FILTER  = 2
    PAUSE   = 3
    # MERGE   = 4
    REMOVE  = 5


class ReqToFree:
    """
    用于表示 ReqManager 中的一个 Req 需要被 free 掉
    这里的元数据足以让 TinyBatchManager 确定 free 操作需要的信息

    Args:
        req_idx: 要 free 的 Req 的索引
        cur_kv_len: 要 free 的 Req 的当前 kvcache 的长度
    """
    def __init__(self, req_idx: int, cur_kv_len: int) -> None:
        self.req_idx: int = req_idx
        self.cur_kv_len: int = cur_kv_len

class BatchFreeMetadata:
    """
    用于所有 free 操作的元数据
    """
    def __init__(self):
        self.req_list_to_free: List[ReqToFree] = None
    
    def to_tensor_for_transfer(self, max_tensor_size: int):
        """
        将 BatchFreeMetadata 转换为一个 Tensor 和 tensor 内有用信息的长度，用于通信
        """
        req_list_to_free_size = len(self.req_list_to_free)
        req_list_to_free_req_idx = [req_to_free.req_idx \
                                    for req_to_free in self.req_list_to_free]
        req_list_to_free_cur_kv_len = [req_to_free.cur_kv_len \
                                       for req_to_free in self.req_list_to_free]
        
        req_list_to_free_req_idx_tensor = \
            torch.Tensor(req_list_to_free_req_idx).cuda()
        req_list_to_free_cur_kv_len_tensor = \
            torch.Tensor(req_list_to_free_cur_kv_len).cuda()
        
        req_list_to_free_tensor = torch.stack(
            [req_list_to_free_req_idx_tensor, req_list_to_free_cur_kv_len_tensor], dim=0)
        req_list_to_free_tensor = \
            F.pad(req_list_to_free_tensor, (0, max_tensor_size - req_list_to_free_size))
        return req_list_to_free_size, req_list_to_free_tensor
    
    def from_tensor_and_size(self,
                             req_list_to_free_size: int,
                             req_list_to_free_tensor: torch.Tensor):
        """
        从一个 Tensor 和一个 size 重构 BatchFreeMetadata
        """
        req_list_to_free_req_idx_tensor = \
            req_list_to_free_tensor[0, :req_list_to_free_size]
        req_list_to_free_cur_kv_len_tensor = \
            req_list_to_free_tensor[1, :req_list_to_free_size]
        
        req_list_to_free_req_idx = \
            req_list_to_free_req_idx_tensor.cpu().numpy().tolist()
        req_list_to_free_cur_kv_len = \
            req_list_to_free_cur_kv_len_tensor.cpu().numpy().tolist()
        
        self.req_list_to_free = [
            ReqToFree(
                req_list_to_free_req_idx[i],
                req_list_to_free_cur_kv_len[i]
            ) for i in range(req_list_to_free_size)
        ]
        return

class BatchInitMetadata:
    """
    用于 INIT 操作的元数据
    """
    def __init__(self):
        self.need_alloc_size = None

class BatchFilterMetadata(BatchFreeMetadata):
    """
    用于 FILTER 操作的元数据
    """
    def __init__(self):
        super().__init__()

class BatchPauseMetadata(BatchFreeMetadata):
    """
    用于 FILTER 操作的元数据
    """
    def __init__(self):
        super().__init__()

class BatchRemoveMetadata(BatchFreeMetadata):
    """
    用于 FILTER 操作的元数据
    """
    def __init__(self):
        super().__init__()

class TinyBatchManagerOp:
    """
    用于 TinyBatchManager 的元数据，
    指定 TinyBatchManager 需要进行的操作和相应的元数据
    """
    def __init__(self):
        self.batch_op_kind: TinyBatchManagerOpKind = None
        self.batch_op_metadata: Union[
            BatchInitMetadata,
            BatchFilterMetadata,
            BatchPauseMetadata,
            BatchRemoveMetadata,
            None # for FORWARD
        ] = None
    
    def to_tensor_for_transfer(self, max_tensor_size: int):
        """
        将 TinyBatchManagerOp 转换为一个枚举和一个 Tensor，用于通信

        通信格式：(batch_op_kind, batch_op_metadata_size, batch_op_metadata_tensor)
        """
        if self.batch_op_kind is TinyBatchManagerOpKind.FORWARD:
            return (
                self.batch_op_kind,
                0,
                torch.zeros([2, max_tensor_size], device="cuda")
            )
        elif self.batch_op_kind is TinyBatchManagerOpKind.INIT:
            return (
                self.batch_op_kind,
                self.batch_op_metadata.need_alloc_size,
                torch.zeros([2, max_tensor_size], device="cuda")
            )
        else:
            batch_op_metadata_size, batch_op_metadata_tensor = \
                self.batch_op_metadata.to_tensor_for_transfer(max_tensor_size)
            return (
                self.batch_op_kind,
                batch_op_metadata_size,
                batch_op_metadata_tensor
            )
        
    def from_transferred_tensor(
        self,
        batch_op_kind: TinyBatchManagerOpKind,
        batch_op_metadata_size: int,
        batch_op_metadata_tensor: torch.Tensor
    ):
        self.batch_op_kind = batch_op_kind
        if self.batch_op_kind is TinyBatchManagerOpKind.FORWARD:
            self.batch_op_metadata = None
        elif self.batch_op_kind is TinyBatchManagerOpKind.INIT:
            self.batch_op_metadata = BatchInitMetadata()
            self.batch_op_metadata.need_alloc_size = batch_op_metadata_size
        else:
            if self.batch_op_kind is TinyBatchManagerOpKind.FILTER:
                self.batch_op_metadata = BatchFilterMetadata()
            elif self.batch_op_kind is TinyBatchManagerOpKind.PAUSE:
                self.batch_op_metadata = BatchPauseMetadata()
            elif self.batch_op_kind is TinyBatchManagerOpKind.REMOVE:
                self.batch_op_metadata = BatchRemoveMetadata()
            self.batch_op_metadata.from_tensor_and_size(
                batch_op_metadata_size,
                batch_op_metadata_tensor
            )