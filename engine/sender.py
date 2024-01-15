import os
import torch
import torch.multiprocessing as mp

from model.model_metadata import ParallelConfig
from utils.distributed_utils import (
    initialize_communicator_distributed
)
from model.parallel_utils.communication_op import (
    send_to_next_pp_stage,
    pp_batch_send_or_recv,
)

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

    send_stream = torch.cuda.Stream(device="cuda")
    while True:
        with torch.cuda.stream(send_stream):
            #! send_tensors include hidden_state, positions, seqs_id
            hidden_state, positions, seqs_id = send_queue.get()
            if hidden_state == None:
                send_to_next_pp_stage(torch.tensor([-1, -1, -1]))
                break
            else:
                send_to_next_pp_stage(torch.tensor(hidden_state.shape))
                pp_batch_send_or_recv(
                    ops=["send", "send", "send"],
                    tensors=[hidden_state, 
                             positions,
                             seqs_id],
                    is_shape=False
                ).wait()
        torch.cuda.current_stream().wait_stream(send_stream)
        del hidden_state
        del hidden_positions
        del seqs_id
