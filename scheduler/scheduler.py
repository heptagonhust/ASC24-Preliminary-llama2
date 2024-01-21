import enum
import time
import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union
from queue import Queue

from scheduler.config import CacheConfig, SchedulerConfig
from scheduler.block_manager import AllocStatus, BlockSpaceManager
from scheduler.policy import PolicyFactory
from sequence.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []
        self.merging: Queue[List[SequenceGroup]] = Queue()


    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule_prompt_batch(
        self,
        num_merging_seqs: int = 0,
    ) -> SchedulerOutputs:
        ignored_seq_groups: List[SequenceGroup] = []
        scheduled: List[SequenceGroup] = []
        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        #! .get_max_num_running_seqs() returns seqs of a seq_group
        #!      beam search case or return multi-seqs case
        num_curr_seqs = 0
        seq_lens: List[int] = []
        while self.waiting:
            seq_group = self.waiting[0]

            assert seq_group.num_seqs() == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_prompt_tokens = seq_group.get_seqs()[0].get_len()
            #! if too many prompt token, move to ignored
            if num_prompt_tokens > self.prompt_limit:
                logging.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds limit of {self.prompt_limit}")
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.pop(0)
                continue

            # If the sequence group cannot be allocated, stop.
            #! check whether KVcache of a seq_group can be allocated
            can_allocate = self.block_manager.can_allocate(seq_group)
            #! if not, stop processing seq_groups in waiting queue
            if can_allocate == AllocStatus.LATER:
                print("153")
                break
            #! if this seq_group is too big, and cannot be placed in all the cache space, ignore
            elif can_allocate == AllocStatus.NEVER:
                logging.warning(
                    f"Input prompt ({num_prompt_tokens} tokens) is too long"
                    f" and exceeds the capacity of block_manager")
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.pop(0)
                continue

            # If the number of batched tokens exceeds the limit, stop.
            new_seq_lens = seq_lens + [num_prompt_tokens]
            #! check number of batched tokens
            #! all the batched seqs will be padded to the maximum length of all the seqs
            num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
            if (num_batched_tokens >
                    self.scheduler_config.max_num_batched_tokens):
                print("172")
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            #! check number of running seqs
            if (num_curr_seqs + num_new_seqs + num_merging_seqs >
                    self.scheduler_config.max_num_batched_seqs):
                print(f"182, {num_curr_seqs}, {num_new_seqs}, {num_merging_seqs}")
                break
            #! check number of paddings
            num_paddings = num_batched_tokens - sum(new_seq_lens)
            if num_paddings > self.scheduler_config.max_paddings:
                print("187")
                break

            #! pass all checks, remove it from waiting queue, append it to running queue
            seq_lens = new_seq_lens
            seq_group = self.waiting.pop(0)
            #! allocate KVcache for this seq_group
            self._allocate(seq_group)
            self.running.append(seq_group)
            num_curr_seqs += num_new_seqs
            scheduled.append(seq_group)

            #! if a seq_group is moved to running state or ignored state
            #!     generate a scheduler_outputs to specific the change of state
            '''
                all the seq_groups in waiting queue are lack of prompts K & V,
                    so prompt_run should be set to True 
                seq_groups preemptted by recompute are moved to waiting queue, which
                    also have to run prompt first
            '''
        if scheduled or ignored_seq_groups:
            scheduler_outputs = SchedulerOutputs(
                scheduled_seq_groups=scheduled,
                prompt_run=True,
                num_batched_tokens=len(seq_lens) *
                max(seq_lens) if seq_lens else 0,
                blocks_to_swap_in={},
                blocks_to_swap_out={},
                blocks_to_copy={},
                ignored_seq_groups=ignored_seq_groups,
            )
            return scheduler_outputs
        else:
            return None
    
    def _schedule_decode_batch(
        self,
        batch_request_ids: List[int],
        merging_batch: List[SequenceGroup] = None,
    ) -> SchedulerOutputs:
        now = time.monotonic()

        '''
            schedule of sequence groups in running queue:
                traverse running queue from the highest priority to the lowest to check if 
                    there are enough KVcache space left for current seq_groups
                if there is not enough space, preempt the lowest-priority sequence groups
                    preempt:
                        1). if seq_group has single seq
                                free KVcache space directly, recompute later 
                                move seq_group to waiting queue
                        2). if seq_group has multiple seqs
                                swap out the blocks of the preempted sequences to CPU memory
                                move seq_group to swapped queue
                         
        '''
        # Reserve new token slots for the running sequence groups.
        #! changes needed below
        running_batch: List[SequenceGroup] = []
        next_running_batch: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        for idx in range(len(self.running)-1, -1, -1):
            if self.running[idx].request_id in batch_request_ids:
                running_batch.append(self.running.pop(idx))
        if merging_batch is not None:
            running_batch.extend(merging_batch)
        running_batch = self.policy.sort_by_priority(now, running_batch)

        while running_batch:
            seq_group = running_batch.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if running_batch:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_batch.pop(-1)
                    self._preempt(victim_seq_group)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group)
                self.running.append(seq_group)
                next_running_batch.append(seq_group)

        #! changes needed above
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in next_running_batch
        )

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=next_running_batch,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            ignored_seq_groups=[],
        )
        return scheduler_outputs


    def _schedule(
        self, 
        seq_group_metadata_list: List[SequenceGroupMetadata] = None,
    ) -> SchedulerOutputs:
        if seq_group_metadata_list is None:
            self.merging.put(None)
            scheduler_outputs = self._schedule_prompt_batch()
            return scheduler_outputs

        batch_request_ids: List[int] = \
            [seq_group_metadata.request_id for seq_group_metadata in seq_group_metadata_list]
        #! a decode batch coming
        if not seq_group_metadata_list[0].is_prompt:
            if len(seq_group_metadata_list) < self.scheduler_config.merge_threshold_min_seq:
                merging_batch: List[SequenceGroup] = []
                num_merging_seqs = 0
                for idx in range(len(self.running)-1, -1, -1):
                    if self.running[idx].request_id in batch_request_ids:
                        #! pop seq_group from running queue to avoid been scheduled again
                        #!  but do not free the space cause it will run after the new batch
                        #!  prefill phase is finished
                        seq_group = self.running.pop(idx)
                        num_merging_seqs += seq_group.get_max_num_running_seqs()
                        merging_batch.append(seq_group)
                self.merging.put(merging_batch)
                scheduler_outputs = self._schedule_prompt_batch(
                    num_merging_seqs=num_merging_seqs,
                )
            else:
                scheduler_outputs = self._schedule_decode_batch(
                    batch_request_ids=batch_request_ids,
                )
        #! a prompt batch coming
        else:
            merging_batch: List[SequenceGroup] = self.merging.get()
            scheduler_outputs = self._schedule_decode_batch(
                batch_request_ids=batch_request_ids,
                merging_batch=merging_batch,
            )
        return scheduler_outputs
                
            

    def schedule(
        self, 
        recv_seq_group_metadata_list: List[SequenceGroupMetadata] = None,
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule(
            seq_group_metadata_list=recv_seq_group_metadata_list
        )

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        #! create a SequenceGroupMetadata for each seq_group in running queue
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            #! generate seq_group_metadata from seqs in seq_group
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]] = None,
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                if blocks_to_copy is None:
                    raise ValueError(
                        "blocks_to_copy must be specified for appending slots.")
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int] = None,
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            #! insert seq_group to the front of waiting queue
            #!      and free KVcache space
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            #! insert seq_group to swapped queue
            if blocks_to_swap_out is None:
                raise ValueError(
                    "blocks_to_swap_out must be specified for swapping.")
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
