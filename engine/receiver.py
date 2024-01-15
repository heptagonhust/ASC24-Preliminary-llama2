import os
import torch
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
    backend: str = "nccl",
):
    initialize_communicator_distributed(
        calculator_rank=calculator_rank,
        calculator_world_size=calculator_world_size,
        parallel_config=parallel_config,
        communicator="recv",
        backend=backend
    )

    recv_stream = torch.cuda.Stream(device="cuda")
    recv_shape = torch.zeros([3], dtype=torch.long, device='cuda')
    while True:
        with torch.cuda.stream(recv_stream):
            receive_from_prev_pp_stage(tensor=recv_shape, tensor_dtype=torch.long)
            if torch.equal(recv_shape, torch.tensor([-1, -1, -1])):
                #! end of work, waiting to be killed
                hidden_state = None
                positions = None
                seqs_id = None
            else:
                hidden_state, positions, seqs_id = \
                    pp_batch_send_or_recv(
                        ops=["recv", "recv", "recv"],
                        tensors=[recv_shape, 
                                 recv_shape,
                                 recv_shape[0]],
                        dtypes=[model_config.dtype, 
                                torch.long, 
                                torch.long],
                        is_shape=True
                    ).wait()
        torch.cuda.current_stream().wait_stream(recv_stream)
        recv_queue.put((hidden_state, positions, seqs_id))
