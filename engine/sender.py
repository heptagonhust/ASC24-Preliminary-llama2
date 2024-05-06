import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.model_metadata import ParallelConfig
from utils.distributed_utils import (
    initialize_communicator_distributed
)
from model.parallel_utils.communication_op import (
    send_to_next_pp_stage,
    pp_batch_send_or_recv,
)
from scheduler.input_metadata import InputMetadata

#! CUDA tensor consumer, required to release sent tensors
#!  reference count
class Sender():
    def __init__(
        self,
        parallel_config: ParallelConfig,
        calculator_rank: int = int(os.environ["RANK"]),
        calculator_world_size: int = int(os.environ["WORLD_SIZE"]),
    ):
        self.send_queue = mp.Queue()
        self.sender = mp.Process(
            target=_send, 
            args=(
                calculator_rank,
                calculator_world_size,
                self.send_queue, 
                parallel_config,
            )
        )
    
    def start_loop(self):
        self.sender.start()
        return self.send_queue
        
def _send(
    calculator_rank: int,
    calculator_world_size: int,
    send_queue: mp.Queue,
    parallel_config: ParallelConfig,
    backend: str = "nccl",
):
    initialize_communicator_distributed(
        calculator_rank=calculator_rank,
        calculator_world_size=calculator_world_size,
        parallel_config=parallel_config,
        communicator="send",
        backend=backend
    )

    send_stream = torch.cuda.Stream()
    hidden_state: torch.Tensor = None
    input_positions: torch.Tensor = None
    input_metadata: InputMetadata = None
    idx = 0
    while True:
        with torch.cuda.stream(send_stream):
            hidden_state, input_positions, input_metadata = send_queue.get()
            #! send_tensors include hidden_state, positions, seqs_id
            if hidden_state == None:
                send_to_next_pp_stage(torch.tensor([-1, -1, -1], device='cuda'))
                break
            else:
                #! sending prompt metadata
                send_to_next_pp_stage(torch.tensor(hidden_state.shape, device='cuda'))
                if input_metadata.is_prompt:
                    send_to_next_pp_stage(torch.tensor([0, 0, 0], device='cuda'))
                    send_to_next_pp_stage(hidden_state)
                    del hidden_state
                    send_to_next_pp_stage(input_positions)
                    del input_positions
                    send_to_next_pp_stage(input_metadata.slot_mapping)
                    send_to_next_pp_stage(
                        torch.tensor(input_metadata.prompt_lens, device='cuda')
                    )
                    del input_metadata
                #! sending decode metadata
                else:
                    send_to_next_pp_stage(torch.tensor(input_metadata.block_tables.shape, device='cuda'))
                    send_to_next_pp_stage(hidden_state)
                    del hidden_state
                    send_to_next_pp_stage(input_positions)
                    del input_positions
                    send_to_next_pp_stage(input_metadata.slot_mapping)
                    send_to_next_pp_stage(input_metadata.context_lens)
                    send_to_next_pp_stage(input_metadata.block_tables)
                    del input_metadata

            idx += 1
        torch.cuda.current_stream().wait_stream(send_stream)
