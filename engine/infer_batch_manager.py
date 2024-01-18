import asyncio
import torch
import rpyc
from typing import Dict, List, Optional, Tuple
from transformers.models.llama import LlamaConfig
from manager.tiny_batch_manager_metadata import TinyBatchManagerOpKind
from router.io_struct import Req
from router.model_infer.infer_batch import InferBatch

from model.llama import LlamaForCausalLM

from router.model_infer.preprocess import prepare_decode_inputs, prepare_prefill_inputs
from sampler.postprocess import sample
from sampler.sampling_metadata import SamplingParams, _prepare_sample
from router.model_infer.infer_batch import requests_mapping
from router.model_infer.infer_batch import InferReq

from model.parallel_utils.parallel_state import setup_distributed
from model.model_metadata import ParallelConfig, ModelConfig

from utils.log_utils import init_logger
logger = init_logger(__name__)

class InferBatchManager:

    def __init__(self, model: LlamaForCausalLM, parallel_config: ParallelConfig, model_config: ModelConfig, return_all_prompt_logprobs: bool = False) -> None:
        self.model = model
        self.parallel_config = parallel_config
        self.model_config = model_config
        self.return_all_prompt_logprobs = return_all_prompt_logprobs
        self.cache: Dict[int, InferBatch] = {}
        return
    
    
    def put_for_prefill(self, batch_id: int, reqs: List[Req]):
        pass
        
    def put_for_decode(self, batch_id: int, req_ids: List[int],
                       finished_or_removed_req_ids: List[int],
                       op_kind: TinyBatchManagerOpKind):
        pass
    
    def put_for_pause(self, batch_id: int, req_ids: List[int]):
        pass


