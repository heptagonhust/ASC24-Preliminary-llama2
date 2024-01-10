from sampler.sampling_metadata import SamplingParams, _prepare_sample
from model.parallel_utils.parallel_state import setup_distributed
from model.model_metadata import ModelConfig,ParallelConfig
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



@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

class LLamaEngine():
    def __init__(self, model_config: ModelConfig, parallel_config_llama:ParallelConfig) -> nn.Module:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # set random seeds
        random.seed(model_config.seed)
        torch.manual_seed(model_config.seed)
        np.random.seed(model_config.seed)
        torch.cuda.manual_seed_all(model_config.seed)
        
        setup_distributed(parallel_config_llama)
        with _set_default_torch_dtype(model_config.dtype):
            model = LlamaForCausalLM(
                model_config.hf_model_config,
                weight_dir="/data/7B-chat-hf",
                max_total_token_num=2048, # TODO:确定真实值？
                max_req_num=1
                                     )  
            model.to(device=device)
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model)
        self.model_config = model_config
        self.model = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_config.tokenizer, 
                        trust_remote_code=self.model_config.trust_remote_code)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, requests: List[Tuple[str, int, int]], 
                 sampling_params: SamplingParams = None):
        for i in tqdm(range(len(requests))):
            prompt, prompt_len, output_len = requests[i] 
            output = self.run_seq(prompt, i, output_len, sampling_params)
            if dist.get_rank() == 0:
                print(f'input: {prompt}\ninput_len: {prompt_len}\n')
                print(f'output: {output}\noutput_len: {len(output)}\n\n')

    def run_seq(self, request, request_id, max_output_len, 
                sampling_params: SamplingParams = None,batch_size:int = 1):
        input_id = self.tokenizer.encode(request)
        seq = Sequence(request_id, request, input_id, block_size=0)   
        
        input_len = seq.get_len()
        total_token_num = input_len * batch_size
        b_req_idx = self.model.req_manager.alloc(batch_size).int()
        b_start_loc = torch.zeros(batch_size,dtype=torch.int32,device="cuda")
        b_seq_len = torch.zeros(batch_size,dtype=torch.int32,device="cuda")
        for i in range(batch_size):
            b_start_loc[i] = i * input_len
            b_seq_len[i] = input_len
        
        # 开始进行prefill
        sampling_metadata = _prepare_sample(seq, sampling_params)
        position = torch.arange(0, seq.get_len())
        seq_token_ids = seq.get_token_ids()
        seq_token_ids = torch.tensor(seq_token_ids, dtype=torch.long, device=self.device)
        # print(seq_token_ids.shape[0])
        # print(total_token_num)
        hidden_state = self.model(batch_size,total_token_num,input_len,seq_token_ids,position,b_req_idx,b_start_loc,b_seq_len,True,None)
        sample_output = self.model.sample(hidden_state, sampling_metadata)
        new_token_id = sample_output[-1].samples[-1].output_token
        tokens_logprob = sample_output[-1].samples[-1].logprobs
        seq.append_token_id(new_token_id, tokens_logprob)

        for i in range(max_output_len):
            sampling_metadata = _prepare_sample(seq, sampling_params)

            position = torch.arange(0, seq.get_len())
            seq_token_ids = seq.get_token_ids()
            seq_token_ids = torch.tensor(seq_token_ids, dtype=torch.long, device=self.device)
            new_token_id_tensor = torch.tensor([new_token_id], dtype=torch.long, device=self.device)
            # TODO:修改传入参数
            print('new_token_id_tensor:')
            print(new_token_id_tensor.shape)
            hidden_state = self.model(batch_size, total_token_num, input_len + i,
                                      new_token_id_tensor, position, b_req_idx,
                                      b_start_loc, b_seq_len, False, None)
            # hidden_state = self.model(seq_token_ids, position, None, None)
            sample_output = self.model.sample(hidden_state, sampling_metadata)
            new_token_id = sample_output[-1].samples[-1].output_token
            tokens_logprob = sample_output[-1].samples[-1].logprobs
            seq.append_token_id(new_token_id, tokens_logprob)

            if (seq.get_last_token_id() == self.tokenizer.eos_token_id):
                break

        output_token_ids = seq.get_output_token_ids()
        output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
        return output

