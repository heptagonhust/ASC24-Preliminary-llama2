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
from scheduler.input_metadata import InputMetadata

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

    recv_stream = torch.cuda.Stream()
    recv_hidden_shape = torch.zeros([3], dtype=torch.long, device='cuda')
    recv_meta_shape = torch.zeros([2], dtype=torch.long, device='cuda')
    hidden_state: torch.Tensor = None
    input_positions: torch.Tensor = None
    input_metadata: InputMetadata = None
    idx = 0
    while True:
        with torch.cuda.stream(recv_stream):
            receive_from_prev_pp_stage(tensor=recv_hidden_shape, tensor_dtype=torch.long)
            if torch.equal(recv_hidden_shape, torch.tensor([-1, -1, -1], device='cuda')):
                #! end of work, waiting to be killed
                hidden_state = None
                input_positions = None
                input_metadata = None
            else:
                receive_from_prev_pp_stage(tensor=recv_meta_shape, tensor_dtype=torch.long)
                hidden_state = receive_from_prev_pp_stage(
                    tensor_shape=recv_hidden_shape,
                    tensor_dtype=model_config.dtype,
                )
                input_positions = receive_from_prev_pp_stage(
                    tensor_shape=recv_hidden_shape[:2], 
                    tensor_dtype=torch.long,
                )
                slot_mapping = receive_from_prev_pp_stage(
                    tensor_shape=recv_hidden_shape[:2], 
                    tensor_dtype=torch.long
                )

                if torch.equal(recv_meta_shape, torch.tensor([0, 0], device='cuda')):
                    prompt_len = receive_from_prev_pp_stage(
                        tensor_shape=recv_hidden_shape[:1],
                        tensor_dtype=torch.int,
                    )
                    input_metadata = InputMetadata(
                        prompt_lens=prompt_len.tolist(),
                        slot_mapping=slot_mapping,
                        max_context_len=None,
                        context_lens=None,
                        block_tables=None,
                        use_cuda_graph=False
                    )

                else:
                    context_lens = receive_from_prev_pp_stage(
                        tensor_shape=recv_hidden_shape[:1],
                        tensor_dtype=torch.int,
                    )
                    block_tables = receive_from_prev_pp_stage(
                        tensor_shape=recv_meta_shape,
                        tensor_dtype=torch.int
                    )
                    input_metadata = InputMetadata(
                        prompt_lens=[],
                        slot_mapping=slot_mapping,
                        max_context_len=recv_meta_shape.tolist()[-1],
                        context_lens=context_lens,
                        block_tables=block_tables,
                        use_cuda_graph=False
                    )

        torch.cuda.current_stream().wait_stream(recv_stream)
        recv_queue.put((hidden_state, input_positions, input_metadata))
        idx += 1
