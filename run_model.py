import torch
import torch.nn as nn
from config import ModelConfig
import contextlib
from transformers import AutoTokenizer
from model.llama import LlamaForCausalLM
import random
from typing import List, Tuple,Dict
from utils.sampling_metadata import SamplingMetadata
import json
from tqdm import tqdm
from model_sample_metadata import _prepare_sample
from utils.sequence import SequenceGroupMetadata,SequenceGroup
from utils.sampling_metadata import SamplingParams
from utils.sequence import Sequence, SchedulerOutputs,SequenceData,SequenceStatus
@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)

class LLamaEngine():
    def __init__(self, model_config: ModelConfig) -> nn.Module:
        random.seed(model_config.seed)
        with _set_default_torch_dtype(model_config.dtype):
            with torch.device("cuda"):
                model = LlamaForCausalLM(model_config.hf_model_config)  # 在GPU中建立模型，同时根据TP将模型进行切分，因此开辟的显存空间是切分后的模型大小
                # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model)
        self.model_config = model_config
        self.model = model

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
        sampling_params = SamplingParams()
        token_ids = None
        request_id = 0
        seq = Sequence(request_id, request, input_id, block_size=0)   
        seq.status = SequenceStatus.RUNNING
        seq_group = SequenceGroup(request_id, [seq], sampling_params, arrival_time = 0)
        scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=[seq_group],
                    prompt_run=True,
                    num_batched_tokens=1 ,
                    blocks_to_swap_in=0,
                    blocks_to_swap_out=0,
                    blocks_to_copy=0,
                    ignored_seq_groups=0,
        )
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            print('debug: seq_group')
            print(seq_group)
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                print('debug: seq_id')
                print(seq_id)
                seq_data[seq_id] = seq.data
                #block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=None,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        print('debug: seq_group_metadata')
        print(len(seq_group_metadata_list))
        
        sampling_metadata = _prepare_sample(seq_group_metadata_list, [prompt_len])
        self.sample_config = sampling_metadata
        for i in range(output_len):
            print(output_len)
            position = torch.arange(0, input_id.shape[-1])
            hidden_state = self.model(input_id, position, None, None)
            sample_output = self.model.sample(hidden_state, self.sample_config)
            new_token = sample_output[-1].samples[-1].output_token
            input_id = torch.cat([input_id, new_token.unsqueeze(0)], dim=-1)
            if (new_token == self.tokenizer.eos_token_id):
                break
        print(self.tokenizer.decode(input_id, skip_special_tokens=True))


if __name__ == "__main__":
    model_config_llama = ModelConfig("/data/7B-chat-hf", "/data/7B-chat-hf", True, 1, None)
    LLama = LLamaEngine(model_config_llama)
    with open('scrambled_sampled_dataset.json') as f:
        requests = json.load(f)
    # for i in tqdm(range(len(requests))):
    #     prompt, prompt_len, output_len = requests[i] 
    LLama.generate(requests)
        
        