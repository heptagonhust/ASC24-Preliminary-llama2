import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.model_metadata import ModelConfig, ParallelConfig
from utils.distributed_utils import (
    initialize_communicator_distributed
)
from model.parallel_utils.communication_op import (
    receive_from_prev_pp_stage,
    pp_batch_send_or_recv,
)

#! CUDA tensor productor, class users are required to 
#! release received tensors reference count after using them
class Receiver():
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        calculator_rank: int = int(os.environ["RANK"]),
        calculator_world_size: int = int(os.environ["WORLD_SIZE"]),
        max_req_num: int = 10000,
    ):
        self.recv_queue = mp.Queue()
        self.receiver = mp.Process(
            target=_recv, 
            args=(
                calculator_rank,
                calculator_world_size,
                self.recv_queue, 
                model_config,
                parallel_config,
                max_req_num,
            )
        )
    
    def start_loop(self):
        self.receiver.start()
        return self.recv_queue
        
def _recv(
    calculator_rank: int,
    calculator_world_size: int,
    recv_queue: mp.Queue,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    max_req_num: int,
    backend: str = "nccl",
):
    initialize_communicator_distributed(
        calculator_rank=calculator_rank,
        calculator_world_size=calculator_world_size,
        parallel_config=parallel_config,
        communicator="recv",
        backend=backend
    )

    recv_stream = torch.cuda.Stream()
    recv_shape = torch.zeros([3], dtype=torch.long, device='cuda')
    while True:
        with torch.cuda.stream(recv_stream):
            receive_from_prev_pp_stage(tensor=recv_shape, tensor_dtype=torch.long)
            print(f"rank: {dist.get_rank()}, receive start", flush=True)
            if torch.equal(recv_shape, torch.tensor([-1, -1, -1], device='cuda')):
                #! end of work, waiting to be killed
                hidden_state = None
                infer_state_tensor = None
            else:
                hidden_state = receive_from_prev_pp_stage(tensor_shape=recv_shape, 
                                           tensor_dtype=model_config.dtype)
                #! the shape of InferStateInfoForTransfer tensor is [11, max_req_num]
                infer_state_tensor = receive_from_prev_pp_stage(
                    tensor_shape=torch.Tensor([11, max_req_num]),
                    tensor_dtype=torch.long
                )
            print(f"rank: {dist.get_rank()}, receive end", flush=True)
        torch.cuda.current_stream().wait_stream(recv_stream)
        recv_queue.put((hidden_state, infer_state_tensor))
