
from typing import Dict, List, Tuple, Union
from utils.sequence import SequenceGroupMetadata
from utils.sampling_metadata import SamplingMetadata
from utils.sampling_params import SamplingParams
from utils.sampling_params import SamplingType
from utils.sequence import SequenceData
import torch
    
def _async_h2d(data: list, dtype, pin_memory):
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory)
    return t.to(device="cuda", non_blocking=True)

def _prepare_sample(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    prompt_lens: List[int],
) -> SamplingMetadata:
    seq_groups: List[Tuple[List[int], SamplingParams]] = []
    selected_token_indices: List[int] = []
    selected_token_start_idx = 0
    categorized_sample_indices = {t: [] for t in SamplingType}
    categorized_sample_indices_start_idx = 0

    max_prompt_len = max(prompt_lens) if prompt_lens else 1
    for i, seq_group_metadata in enumerate(seq_group_metadata_list):
        seq_ids = list(seq_group_metadata.seq_data.keys())
        sampling_params = seq_group_metadata.sampling_params
        seq_groups.append((seq_ids, sampling_params))

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