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
    while True:
        with torch.cuda.stream(send_stream):
            #! send_tensors include hidden_state, positions, seqs_id
            hidden_state, infer_state_info_tensor = send_queue.get()
            print(f"rank: {dist.get_rank()}, send start", flush=True)
            if hidden_state == None:
                send_to_next_pp_stage(torch.tensor([-1, -1, -1], device='cuda'))
                break
            else:
                send_to_next_pp_stage(torch.tensor(hidden_state.shape, device='cuda'))
                send_to_next_pp_stage(hidden_state)
                send_to_next_pp_stage(infer_state_info_tensor)
            print(f"rank: {dist.get_rank()}, send finished", flush=True)
        torch.cuda.current_stream().wait_stream(send_stream)
        del hidden_state
        del infer_state_info_tensor
