import logging
import time
from typing import List, Tuple, Union, Dict
from queue import Queue

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from tqdm import tqdm

from model.llama import LlamaForCausalLM
from model.model_metadata import (
    ModelConfig, 
    ParallelConfig
)
from engine.sender import Sender
from engine.receiver import Receiver
from sequence.sequence import Sequence, SequenceGroup, SequenceData, SequenceGroupMetadata, SequenceGroupOutput, SequenceStatus, SequenceOutput, SamplerOutput
from sampler.sampling_params import SamplingParams, SamplingType
from utils.utils import set_default_torch_dtype, _make_tensor_with_pad, _async_h2d, _PAD_SLOT_ID
from utils.distributed_utils import (
    initialize_calculator_distributed,
)
from utils.outputs import RequestOutput
from engine.cache_engine import CacheEngine
from scheduler.config import CacheConfig, SchedulerConfig
from scheduler.scheduler import Scheduler, SchedulerOutputs
from sampler.sampling_metadata import SamplingMetadata
from scheduler.input_metadata import InputMetadata


class Master():
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        device: str = "cuda",
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer, 
            trust_remote_code=self.model_config.trust_remote_code
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sliding_window = None

        mp.set_start_method('spawn')
        self.sender = Sender(
            parallel_config=self.parallel_config
        )
        self.receiver = Receiver(
            model_config=self.model_config,
            parallel_config=self.parallel_config
        )
        self.scheduler = Scheduler(
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config,
        )
        self.req_id = 0

    def start_master(self):
        self.send_queue = self.sender.start_loop()
        self.recv_queue = self.receiver.start_loop()
        self.rank = initialize_calculator_distributed(self.model_config, self.parallel_config)
        self._init_model()
        self._init_cache_engine()
    
    def add_requests(
        self, 
        requests: List[Tuple[str, int, int]], 
        sampling_params: SamplingParams = None
    ):
        self.sampling_params = sampling_params
        for req in requests:
            prompt, prompt_len, output_len = req
            prompt_token_ids = self.tokenizer.encode(prompt)
            block_size = self.cache_config.block_size
            seq_id = self.req_id
            seq = Sequence(
                seq_id=seq_id,
                prompt=prompt, 
                prompt_token_ids=prompt_token_ids, 
                block_size=block_size,
            )
            seq_group = SequenceGroup(
                request_id=seq_id, 
                seqs=[seq], 
                sampling_params=self.sampling_params, 
                arrival_time=time.monotonic(),
                max_output_len=output_len,
            )
            self.req_id += 1
            self.scheduler.add_seq_group(seq_group)
        self.pbar = tqdm(total=len(requests), desc="Processed prompts")
    
    @torch.inference_mode()
    def run(self):
        logging.info("Master started")
        self.running_batches = 0
        scheduled_batches = Queue()
        hidden_states: torch.Tensor = None
        input_positions: torch.Tensor = None
        input_metadata: InputMetadata = None
        outputs = []
        idx = 0
        while self.scheduler.has_unfinished_seqs():
            last_seq_group_metadata_list: List[SequenceGroup] = None
            if not self._more_batches():
                hidden_states, input_positions, input_metadata = self.recv_queue.get()
                del input_positions
                last_seq_group_metadata_list, last_scheduler_outputs = scheduled_batches.get()
                sampling_metadata = self._prepare_sample(last_seq_group_metadata_list, input_metadata.prompt_lens)
                sampling_output = self.model.sample(
                    hidden_states=hidden_states,
                    sampling_metadata=sampling_metadata,
                )
                del hidden_states
                self._process_model_outputs(sampling_output, last_scheduler_outputs)

            seq_group_metadata_list, scheduler_outputs, ignore = self._schedule(last_seq_group_metadata_list)
            if not scheduler_outputs.is_empty():
                scheduled_batches.put((seq_group_metadata_list, scheduler_outputs))
                if last_seq_group_metadata_list is None:
                    self.running_batches += 1
            else:
                for output in ignore:
                    if output.finished:
                        outputs.append(output)
                        self.pbar.update(1)
                continue

            if seq_group_metadata_list is not None and \
                    len(seq_group_metadata_list) != 0:
                is_prompt = seq_group_metadata_list[0].is_prompt
                # Prepare input tensors.
                if is_prompt:
                    inputs = self._prepare_prompt(seq_group_metadata_list)
                    input_tokens, input_positions, input_metadata = inputs
                else:
                    #! include only one element in input_tokens, input_positions
                    inputs = self._prepare_decode(seq_group_metadata_list)
                    input_tokens, input_positions, input_metadata = inputs

                hidden_states = self.model(input_ = input_tokens,
                                          positions = input_positions,
                                          kv_caches = self.gpu_cache,
                                          input_metadata = input_metadata)
                print(f"idx: {idx}, rank: {self.rank}, hidden_state shape: {hidden_states.shape}")
                self.send_queue.put((
                    hidden_states, 
                    input_positions, 
                    input_metadata, 
                ))
                idx += 1

        #! end of work
        self.send_queue.put((None, None, None))
        self.receiver.receiver.kill()
        return outputs

    def _do_sample(self,
                  hidden_state: torch.Tensor,
                  seqs_id: torch.Tensor,
                  idx: int):
        sampling_metadata = \
            self.scheduler.get_sampling_metadata_from_seqs_id(seqs_id.tolist(),
                                                              self.sampling_params)
        # print(f"idx: {idx}, sampler, rank: {self.rank}, seqs_id: {seqs_id} \n \
        #         sampling metadata: {sampling_metadata}")
        sampler_output = self.model.sample(hidden_state, sampling_metadata)
        return self.scheduler.update_batch(seqs_id, sampler_output)


    def _init_model(self):
        with set_default_torch_dtype(self.model_config.dtype):
            model = LlamaForCausalLM(self.model_config.hf_model_config)
            model.to(device=self.device)
            model.load_weights(self.model_config.model)
        self.model = model
    
    def _init_cache_engine(self) -> None:
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        #! for cuda graph
        self.block_size = self.cache_engine.block_size
        # self.set_block_size(self.cache_engine.block_size)

    def _more_batches(self) -> bool:
        return self.scheduler.has_unfinished_seqs() and \
            self.running_batches < self.scheduler_config.max_num_batches

    def _schedule(
        self, 
        recv_seq_group_metadata_list: List[SequenceGroupMetadata] = None,
    ):
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule(
            recv_seq_group_metadata_list=recv_seq_group_metadata_list
        )
        return seq_group_metadata_list, scheduler_outputs, [
            RequestOutput.from_seq_group(seq_group)
            for seq_group in scheduler_outputs.ignored_seq_groups
        ]
    
    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []

        #! length of all seqs in seq_group_metadata_list
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            #! only one seq in a seq_group cause it's prompt stage
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(prompt_len)))

            #! only happens when profiling
            #! block_tables: Dict[int, List[int]]: seq_id -> block_table
            #! a list of block_number of specified seq
            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            #! slot mapping list for a single seq
            slot_mapping.append([])
            #! block_table: List[int] = [block_number, ...]
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, prompt_len - self.sliding_window)
            #! arrange all tokens in prompt to physical slots
            for i in range(prompt_len):
                if i < start_idx:
                    slot_mapping[-1].append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        #! get the maximum length of all seqs in seq_group_metadata_list
        #!      in this schedule round to padding
        max_prompt_len = max(prompt_lens)
        #! padding input tokens to maximum length of seqs in this schedule round
        input_tokens = _make_tensor_with_pad(input_tokens,
                                             max_prompt_len,
                                             pad=0,
                                             dtype=torch.long)
        input_positions = _make_tensor_with_pad(input_positions,
                                                max_prompt_len,
                                                pad=0,
                                                dtype=torch.long)
        slot_mapping = _make_tensor_with_pad(slot_mapping,
                                             max_prompt_len,
                                             pad=_PAD_SLOT_ID,
                                             dtype=torch.long)

        input_metadata = InputMetadata(
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping,
            max_context_len=None,
            context_lens=None,
            block_tables=None,
            use_cuda_graph=False,
        )
        return input_tokens, input_positions, input_metadata

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []

        #! traverse seq_groups
        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())
            #! traverse seqs in a seq_group
            for seq_id in seq_ids:
                #! put last generated token to input
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])
                #! put last generated token position index to input
                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])
                #! get length to process in this round of forward
                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])

                #! add one token, move sliding window
                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        batch_size = len(input_tokens)
        max_context_len = max(context_lens)
        use_captured_graph = False
        # use_captured_graph = (
        #     not self.model_config.enforce_eager
        #     and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
        #     and max_context_len <= self.max_context_len_to_capture)
        # if use_captured_graph:
        #     # Pad the input tokens, positions, and slot mapping to match the
        #     # batch size of the captured graph.
        #     graph_batch_size = _get_graph_batch_size(batch_size)
        #     assert graph_batch_size >= batch_size
        #     for _ in range(graph_batch_size - batch_size):
        #         input_tokens.append([])
        #         input_positions.append([])
        #         slot_mapping.append([])
        #         context_lens.append(1)
        #         block_tables.append([])
        #     batch_size = graph_batch_size

        # When using CUDA graph, we don't need to make the tensors on the GPU
        # because they will be eventually copied to the designated GPU buffer.
        device = "cpu" if use_captured_graph else "cuda"
        pin_memory = False
        #! generate one token in each seq in a seq_group
        input_tokens = _make_tensor_with_pad(input_tokens,
                                             max_len=1,
                                             pad=0,
                                             dtype=torch.long,
                                             device=device,
                                             pin_memory=pin_memory)
        input_positions = _make_tensor_with_pad(input_positions,
                                                max_len=1,
                                                pad=0,
                                                dtype=torch.long,
                                                device=device,
                                                pin_memory=pin_memory)
        slot_mapping = _make_tensor_with_pad(slot_mapping,
                                             max_len=1,
                                             pad=_PAD_SLOT_ID,
                                             dtype=torch.long,
                                             device=device,
                                             pin_memory=pin_memory)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=device,
                                    pin_memory=pin_memory)

        if use_captured_graph:
            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device=device)
        else:
            block_tables = _make_tensor_with_pad(
                block_tables,
                max_len=max_context_len,
                pad=0,
                dtype=torch.int,
            )

        input_metadata = InputMetadata(
            prompt_lens=[],
            slot_mapping=slot_mapping,
            max_context_len=max_context_len,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )
        return input_tokens, input_positions, input_metadata

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> SamplingMetadata:
        #! storn sampling param of all seq_groups in seq_group_metadata_list
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        #! categorized_sample_indices: Dict[SamplingType, List[int]]
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0

        max_prompt_len = max(prompt_lens) if prompt_lens else 1
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            #! all seqs in a seq_group share one sampling_params
            #! traverse every seq_group in seq_group_metadata_list
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))
            #! if in prompt run stage
            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += prompt_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        categorized_sample_indices_start_idx)
                categorized_sample_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                           selected_token_start_idx + prompt_len - 1))
                '''
                    output before sampling: hidden_states [batch_size, seq_len, hidden_size]
                    hidden_state will be reshaped to [batch_size * seq_len, hidden_size]
                    select_token_indices is used to decide the index of hidden_state to sample
                '''
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += max_prompt_len
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        range(categorized_sample_indices_start_idx,
                              categorized_sample_indices_start_idx + num_seqs))
                categorized_sample_indices_start_idx += num_seqs

        selected_token_indices = _async_h2d(selected_token_indices,
                                            dtype=torch.long,
                                            pin_memory=True)
        categorized_sample_indices = {
            t: _async_h2d(seq_ids, dtype=torch.int, pin_memory=True)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
        )
        return sampling_metadata


    def _check_stop(self, 
                    seq: Sequence,
                    sampling_params: SamplingParams,
                    max_output_len: int) -> None:
        """Stop the finished sequences."""
        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                if not sampling_params.include_stop_str_in_output:
                    # Truncate the output text so that the stop string is
                    # not included in the output.
                    seq.output_text = seq.output_text[:-len(stop_str)]
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
        if seq.get_last_token_id() in sampling_params.stop_token_ids:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() >= max_output_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == self.tokenizer.eos_token_id):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput) -> None:
        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        sample = outputs.samples[0]
        seq = seq_group.get_seqs()[0]
        seq.append_token_id(sample.output_token, sample.logprobs)

        self._check_stop(seq, seq_group.sampling_params, seq_group.max_output_len)

        if seq.is_finished():
            self.scheduler.free_seq(seq)
            print("")
            if self.rank == 0:
                prompt = seq.prompt
                output = self.tokenizer.decode(seq.get_output_token_ids())
                print(f'input: {prompt}\ninput_len: {seq.get_prompt_len()}\n')
                print(f'output: {output}\noutput_len: {seq.get_output_len()}\n\n')
                self.pbar.update(1)
        return

    def _process_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs(seq_group, outputs)
        
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()
        return
