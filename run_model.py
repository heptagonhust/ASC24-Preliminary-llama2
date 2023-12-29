import torch
import torch.nn as nn
from config import ModelConfig
import contextlib
from transformers import AutoTokenizer
from model.llama import LlamaForCausalLM
import random
from typing import List, Tuple
from utils.sampling_metadata import SamplingMetadata
import json
from tqdm import tqdm
from model_sample_metadata import _prepare_sample
@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

class LLamaEngine():
    def __init__(self, model_config: ModelConfig, sample_config: SamplingMetadata) -> nn.Module:
        random.seed(model_config.seed)
        with _set_default_torch_dtype(model_config.dtype):
            with torch.device("cuda"):
                model = LlamaForCausalLM(model_config.hf_model_config)  # 在GPU中建立模型，同时根据TP将模型进行切分，因此开辟的显存空间是切分后的模型大小
                # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model)
        self.model_config = model_config
        self.model = model
        self.sample_config = sample_config

    def generate(self, requests: List[Tuple[str, int, int]]):
        self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_config.tokenizer, 
                        trust_remote_code=self.model_config.trust_remote_code)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for i in requests:
            prompt, prompt_len, output_len = i
            self.run_seq(prompt, prompt_len, output_len)

    def run_seq(self, request, prompt_len, output_len):
        input_id = self.tokenizer(request, return_tensors="pt", padding=True).input_ids
        for i in range(output_len):
            position = torch.arange(0, input_id.shape[-1])
            hidden_state = self.model(input_id, position, None, None)
            sample_output = self.model.sample(hidden_state, self.sample_config)
            new_token = sample_output[-1].samples[-1].output_token
            input_id = torch.cat([input_id, new_token.unsqueeze(0)], dim=-1)
            if (new_token == self.tokenizer.eos_token_id):
                break
        print(self.tokenizer.decode(input_id, skip_special_tokens=True))


if __name__ == "__main__":
    model_config_llama = ModelConfig("/data/7B-chat-hf", "/data/7B-chat-hf", True, 1)
    # Todo 这里seq_group_metadata_list 应该是从_schedule中调度出来的，如果我们不需要调度，直接建立就好
    seq_group_metadata_list = self._schedule()
    sampling_metadata = _prepare_sample(seq_group_metadata_list, input_metadata.prompt_lens)
    LLama = LLamaEngine(model_config_llama, sampling_metadata)
    with open('scrambled_sampled_dataset.json') as f:
        requests = json.load(f)
    for i in tqdm(range(len(requests))):
        prompt, prompt_len, output_len = requests[i]
        LLama.generate([prompt, prompt_len, output_len])
        
        