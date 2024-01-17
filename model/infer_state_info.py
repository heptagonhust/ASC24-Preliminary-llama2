import torch
import numpy as np

from manager.memory_manager import MemoryManager
from manager.request_manager import RequestManager
from manager.tiny_batch_manager_metadata import TinyBatchManagerOp

class InferStateInfo:
    """
    推理时用的信息结构体
    注意：pp通信时不应该直接传递这个类，而应该传递InferStateInfoForTransmission
    """

    def __init__(self):
        self.batch_size = None
        self.total_token_num = None
        self.b_req_idx = None
        self.b_start_loc = None
        self.b_seq_len = None
        self.max_len_in_batch = None
        self.is_prefill = None
        
        self.mem_manager: MemoryManager = None
        self.req_manager: RequestManager = None
        
        self.mem_is_contiguous = None
        self.mem_index = None
        self.mem_start = None 
        self.mem_end = None
        self.key_buffer = None
        self.value_buffer = None

        self.is_splitfuse = False
        self.return_all_prompt_logprobs = False
        self.multimodal_params = None
    

class LlamaInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.position_cos = None
        self.position_sin = None
        self.other_kv_index = None
        self.position_ids = None
    
    def init_some_extra_state(self, model, input_ids : torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                            for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_ids = position_ids
            position_ids = None
        else:
            position_ids = self.b_seq_len - 1
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(self.b_seq_len.shape[0], -1)
            self.other_kv_index = self.req_manager.req_to_token_indexs[self.b_req_idx[0], 0].item()
            self.position_ids = position_ids
            position_ids = None
            # b_loc[0, max_len_in_batch - 1].item()
        return


class InferStateInfoForTransfer:
    """
    pp通信的元数据
    """

    def __init__(self):
        self.batch_size = None
        self.total_token_num = None
        self.b_req_idx = None
        self.b_start_loc = None
        self.b_seq_len = None
        self.max_len_in_batch = None
        self.is_prefill = None
        
        self.mem_is_contiguous = None
        self.mem_index = None
        self.mem_start = None 
        self.mem_end = None

        self.is_splitfuse = False
        self.return_all_prompt_logprobs = False
        self.multimodal_params = None

        '''
        members for ReqManager operations in a batch
        '''
        self.infer_state_op: TinyBatchManagerOp = None