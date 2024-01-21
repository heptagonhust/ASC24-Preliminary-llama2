from typing import List, Dict, Tuple, Union
from collections import deque

import torch
import torch.distributed as dist
from tqdm import tqdm
from sequence.sequence import (
    SequenceData,
    SamplerOutput,
)
from sequence.config import SchedulerConfig
from sampler.sampling_metadata import prepare_sample
from sampler.sampling_params import SamplingParams


#! schedule unit
class SequenceBatch:
    def __init__(self, 
                 device: str = "cuda"):
        self.device = device
        self.seqs_id: List[int] = []
        self.seqs_data: List[SequenceData] = []
        self.input_token: torch.Tensor = None
        self.input_position: torch.Tensor = None
        self.seqs_max_output_len: List[int] = []
        self.tokenizer = None
    
    @classmethod
    def from_requests(cls, 
                      requests: List[Tuple[str, int, int]], 
                      requests_id: List[int],
                      tokenizer,
                      device: str = "cuda"):
        batch = cls(device=device)
        batch.tokenizer = tokenizer
        for seq_id, request in zip(requests_id, requests):
            prompt, prompt_len, output_len = request
            batch.seqs_id.append(seq_id)
            batch.seqs_max_output_len.append(output_len)
            prompt_token_ids = tokenizer.encode(prompt)
            seq_data = SequenceData(prompt_token_ids)
            batch.seqs_data.append(seq_data)
        batch._update_seqs_token_ids_position_tensor()
        return batch

    def get_batch_ids(self) -> torch.Tensor:
        return torch.tensor(self.seqs_id, device="cuda")
    
    def get_seq_id_from_idx(self, idx: int):
        return self.seqs_id[idx]

    def get_seq_data_from_idx(self, idx: int):
        return self.seqs_data[idx]

    def get_seqs_token_ids_position(self):
        return self.input_token, self.input_position
    
    def update_seqs_from_sample_output(self, sampler_output: SamplerOutput):
        assert len(sampler_output) == len(self.seqs_data), \
            "The number of sequences in the sampler output should be \
                the same as the number of sequences in the batch"
        finished_idx: List[int] = []
        has_finished: bool = False
        for idx, seq_info in \
            enumerate(zip(self.seqs_data, self.seqs_max_output_len, sampler_output)):
            seq_data, seq_max_output_len, seq_group_output = seq_info
            #! len(seq_output) must be 1, beam search not supported
            for seq_output in seq_group_output.samples:
                seq_data.append_token_id(seq_output.output_token,
                                         seq_output.logprobs[seq_output.output_token])
                if seq_output.output_token == self.tokenizer.eos_token_id or \
                   seq_data.get_output_len() >= seq_max_output_len:
                    finished_idx.append(idx)
                    has_finished = True
        if not has_finished:
            self._update_seqs_token_ids_position_tensor()
        return finished_idx, has_finished

    def _update_seqs_token_ids_position_tensor(self):
        input_token = [
            seq.get_token_ids() for seq in self.seqs_data
        ]
        token_ids_len = [len(token_id) for token_id in input_token]
        input_position = [
            list(range(token_len)) for token_len in token_ids_len]
        input_token = _make_tensor_with_pad(input_token, 
                                            max_len=max(token_ids_len), 
                                            pad=0, 
                                            dtype=torch.long, 
                                            device=self.device)
        input_position = _make_tensor_with_pad(input_position,
                                               max_len=max(token_ids_len),
                                               pad=0,
                                               dtype=torch.long,
                                               device=self.device)
        self.input_token = input_token
        self.input_position = input_position

    
    def replace_seqs(self,
                     requests_id: List[int], 
                     requests: List[Tuple[str, int, int]],
                     idx: List[int]):
        for i, seq_id, request in zip(idx, requests_id, requests):
            prompt, prompt_len, output_len = request
            if (seq_id != -1):
                prompt_token_ids = self.tokenizer.encode(prompt)
                seq_data = SequenceData(prompt_token_ids)
                self.seqs_data[i] = seq_data
                self.seqs_id[i] = seq_id
                self.seqs_max_output_len[i] = output_len
            else:
                self.seqs_data.pop(i)
                self.seqs_id.pop(i)
                self.seqs_max_output_len.pop(i)
        self._update_seqs_token_ids_position_tensor()

class SequenceScheduler:
    def __init__(self,
                 scheduer_config: SchedulerConfig,
                 rank: int,
                 tokenizer,
                 device: str = "cuda"):
        self.config = scheduer_config
        self.rank = rank
        self.tokenizer = tokenizer
        self.device = device
        self.waiting_queue: deque[Tuple[str, int, int]] = deque()
        self.running_queue: Dict[int, Tuple[str, int, int]] = {}
        self.finished_queue: Dict[int, SequenceData] = {}
        self.batches: List[SequenceBatch] = []
        self.request_id = 0
    
    def add_requests(self,
                     requests: List[Tuple[str, int, int]]):
        self.waiting_queue.extend(requests)
        if (self.config.use_tqdm and self.rank == 0 and
            self.config.show_progress):
            self.pbar = tqdm(total=len(self.waiting_queue)
                                   +len(self.running_queue)
                                   +len(self.finished_queue), 
                             desc="Processed prompts")
    
    def more_batches(self):
        if len(self.batches) < self.config.max_batches and \
            len(self.waiting_queue) > 0:
            return True
        else:
            return False
    
    def get_n_requests(self, 
                       n: int
    ) -> Tuple[List[Tuple[str, int, int]], List[int], int]:
        requests = []
        requests_id = []
        for i in range(n):
            if (len(self.waiting_queue) == 0):
                return requests, i
            req = self.waiting_queue.popleft()
            self.running_queue[self.request_id] = req
            requests.append(req)
            requests_id.append(self.request_id)
            self.request_id += 1
        return requests, requests_id, n
    
    def get_new_batch(self):
        requests, requests_id, n = self.get_n_requests(self.config.max_batch_size)
        batch = SequenceBatch.from_requests(
            requests=requests,
            requests_id=requests_id,
            tokenizer=self.tokenizer,
            device=self.device
        )
        self.batches.append(batch)
        return *batch.get_seqs_token_ids_position(), batch.get_batch_ids()

    def get_sampling_metadata_from_seqs_id(self,
                                           seqs_id: List[int],
                                           sampling_params: SamplingParams):
        for batch in self.batches:
            if seqs_id[0] in batch.get_batch_ids():
                return prepare_sample(
                    seqs_id=seqs_id,
                    seqs_data=batch.seqs_data,
                    sampling_params=sampling_params,
                )
        raise ValueError(f"seqs_id {seqs_id} not found in any batch")

    def update_batch(self,
                     seqs_id: List[int],
                     sampler_output: SamplerOutput):
        for batch in self.batches:
            if seqs_id[0] in batch.get_batch_ids():
                finished_idx, has_finished = \
                    batch.update_seqs_from_sample_output(sampler_output)
                if has_finished:
                    for idx in finished_idx:
                        seq_id = batch.get_seq_id_from_idx(idx)
                        seq_data = batch.get_seq_data_from_idx(idx)
                        self.finished_queue[seq_id] = self.running_queue.pop(seq_id)
                        if self.config.show_progress:
                            self.show_progress(seq_data)
                    if self.is_finished():
                        return None, None, None
                        
                    requests, requests_id, n = self.get_n_requests(len(finished_idx))
                    if n < len(finished_idx):
                        requests += [(None, None, None)] * (len(finished_idx) - n)
                        requests_id += [-1] * (len(finished_idx) - n)
                    batch.replace_seqs(requests_id=requests_id, 
                                       requests=requests,
                                       idx=finished_idx)
                token_id, positions = batch.get_seqs_token_ids_position()
                return token_id, positions, batch.get_batch_ids()
        raise ValueError(f"seqs_id {seqs_id} not found in any batch")
    
    def show_progress(self, seq_data: SequenceData):
        if self.rank == 0:
            prompt = self.tokenizer.decode(seq_data.prompt_token_ids)
            output = self.tokenizer.decode(seq_data.output_token_ids)
            print(f'input: {prompt}\ninput_len: {seq_data.get_prompt_len()}\n')
            print(f'output: {output}\noutput_len: {seq_data.get_output_len()}\n\n')
            if self.config.use_tqdm:
                self.pbar.update(1)

    
    def is_finished(self):
        return len(self.waiting_queue) == 0 and \
               len(self.running_queue) == 0

    
    
