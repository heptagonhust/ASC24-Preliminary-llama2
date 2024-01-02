"""Sequence and its related classes."""
import copy
import enum
from typing import Dict, List, Optional, Union


PromptLogprobs = List[Optional[Dict[int, float]]]
SampleLogprobs = List[Dict[int, float]]


class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceData:
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")


class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size

        self.data = SequenceData(prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        # Initialize the logical token blocks with the prompt token ids.
        # self._append_tokens_to_blocks(prompt_token_ids)
        self.status = SequenceStatus.WAITING

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        assert token_id in logprobs
        # self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id])

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def get_beam_search_score(self,
                              length_penalty: float = 0.0,
                              seq_len: Optional[int] = None,
                              eos_token_id: Optional[int] = None) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
        if seq_len is None:
            seq_len = self.get_len()
            # NOTE: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if (eos_token_id is not None
                    and self.get_last_token_id() == eos_token_id):
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len**length_penalty)

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")


class SequenceOutput:
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
        self,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, float],
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        return (self.parent_seq_id == other.parent_seq_id
                and self.output_token == other.output_token
                and self.logprobs == other.logprobs)


class SequenceGroupOutput:
    """The model output associated with a sequence group."""

    def __init__(
        self,
        samples: List[SequenceOutput],
        prompt_logprobs: Optional[PromptLogprobs],
    ) -> None:
        self.samples = samples
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return (f"SequenceGroupOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceGroupOutput):
            raise NotImplementedError()
        return (self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


# For each sequence group, we generate a list of SequenceOutput object,
# each of which contains one possible candidate for the next token.
SamplerOutput = List[SequenceGroupOutput]
