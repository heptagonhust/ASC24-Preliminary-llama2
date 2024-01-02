
from typing import Dict, List, Tuple, Union
from utils.sequence import SequenceGroupMetadata
from utils.sampling_metadata import SamplingMetadata
from utils.sampling_params import SamplingParams
from utils.sampling_params import SamplingType
from utils.sequence import SequenceData, Sequence
import torch
    
def _async_h2d(data: list, dtype, pin_memory):
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory)
    return t.to(device="cuda", non_blocking=True)

def _prepare_sample(
    seq: Sequence,
    sampling_params: SamplingParams,
) -> SamplingMetadata:
    seq_groups: List[Tuple[List[int], SamplingParams]] = []
    selected_token_indices: List[int] = []
    categorized_sample_indices = {t: [] for t in SamplingType}

    selected_token_indices.append(seq.get_prompt_len() - 1)
    selected_token_indices = _async_h2d(selected_token_indices,
                                        dtype=torch.long,
                                        pin_memory=True)
    seq_groups.append(([seq.seq_id], sampling_params))

    #! put sequence id of a specified categories to device
    categorized_sample_indices[sampling_params.sampling_type] = [seq.seq_id]
    categorized_sample_indices = {
        t: _async_h2d(seq_ids, dtype=torch.int, pin_memory=True)
        for t, seq_ids in categorized_sample_indices.items()
    }

    seq_data: Dict[int, SequenceData] = {}
    seq_data[seq.seq_id] = seq.data

    sampling_metadata = SamplingMetadata(
        seq_groups=seq_groups,
        seq_data=seq_data,
        prompt_lens=[seq.get_prompt_len()],
        selected_token_indices=selected_token_indices,
        categorized_sample_indices=categorized_sample_indices,
    )
    return sampling_metadata