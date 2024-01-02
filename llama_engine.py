from sampler.sampling_metadata import SamplingParams, _prepare_sample
from model.model_metadata import ModelConfig
from model.llama import LlamaForCausalLM
from sequence.sequence import Sequence 

from transformers import AutoTokenizer
from typing import List, Tuple
from tqdm import tqdm
import torch.nn as nn
import torch
import contextlib
import random
import json


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

class LLamaEngine():
    def __init__(self, model_config: ModelConfig) -> nn.Module:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        random.seed(model_config.seed)
        with _set_default_torch_dtype(model_config.dtype):
            # 在GPU中建立模型，同时根据TP将模型进行切分，因此开辟的显存空间是切分后的模型大小
            model = LlamaForCausalLM(model_config.hf_model_config)  
            # print("debug: device")
            # print(device)
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
            print(f'input: {prompt}\ninput_len: {prompt_len}\n')
            print(f'output: {output}\noutput_len: {len(output)}\n\n')

    def run_seq(self, request, request_id, output_len, 
                sampling_params: SamplingParams = None):
        input_id = self.tokenizer.encode(request)
        seq = Sequence(request_id, request, input_id, block_size=0)   

        for i in range(output_len):
            sampling_metadata = _prepare_sample(seq, sampling_params)

            position = torch.arange(0, seq.get_len())
            seq_token_ids = seq.get_token_ids()
            seq_token_ids = torch.tensor(seq_token_ids, dtype=torch.long, device=self.device)
            hidden_state = self.model(seq_token_ids, position, None, None)
            sample_output = self.model.sample(hidden_state, sampling_metadata)
            new_token_id = sample_output[-1].samples[-1].output_token
            tokens_logprob = sample_output[-1].samples[-1].logprobs
            seq.append_token_id(new_token_id, tokens_logprob)

            if (seq.get_last_token_id() == self.tokenizer.eos_token_id):
                break
        output_token_ids = seq.get_output_token_ids()
        output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
        return output