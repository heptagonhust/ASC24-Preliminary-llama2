import os
import torch
import torch.distributed as dist

from model.model_metadata import ParallelConfig, ModelConfig
from model.parallel_utils.parallel_state import (
    initialize_calculator_parallel,
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_group,
)
from model.parallel_utils.communicator_state import (
    initialize_communicator_state,
    get_pp_communicator_prev_rank,
    get_pp_communicator_next_rank,
)
from utils.utils import set_random_seed


def initialize_calculator_distributed(
    model_config:ModelConfig,
    parallel_config:ParallelConfig,
    backend="nccl"
) -> int:
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    set_random_seed(model_config.seed)
    num_gpus = torch.cuda.device_count()

    rank = int(os.environ["RANK"])
    calculator_world_size = int(os.environ["WORLD_SIZE"])
    world_size = calculator_world_size * 3
    print(f'node: {os.environ["SLURM_NODEID"]}, rank: {rank}')
    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

    initialize_calculator_parallel(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size
    )

    # A small all_reduce for warmup the tp process group
    dist.all_reduce(torch.zeros(1).cuda(), group=get_tensor_model_parallel_group())
    # A small all_reduce for warmup the pp process group
    dist.all_reduce(torch.zeros(1).cuda(), group=get_pipeline_model_parallel_group())
    return rank


def initialize_communicator_distributed(
    calculator_rank: int, 
    calculator_world_size: int,
    parallel_config: ParallelConfig,
    communicator: str = "recv",
    backend: str = "nccl",
) -> int:
    """Initialize peer-to-peer communication for pipeline parallelism"""
    if (calculator_world_size != parallel_config.tensor_parallel_size * \
            parallel_config.pipeline_parallel_size):
        raise ValueError(
            f"calculator_world_size ({calculator_world_size}) != "
            f"tensor_model_parallel_size ({parallel_config.tensor_parallel_size}) * "
            f"pipeline_model_parallel_size ({parallel_config.pipeline_parallel_size})")
    
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(calculator_rank % num_gpus)

    rank = calculator_world_size + calculator_rank * 2
    if communicator == "recv":
        rank = rank
    elif communicator == "send":
        rank += 1
    else:
        raise ValueError(f"Invalid communicator: {communicator}")
    
    world_size = calculator_world_size * 3
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    initialize_communicator_state(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size
    )

    if communicator == "recv":
        dist.recv(torch.zeros(1).cuda(), get_pp_communicator_prev_rank())
    else:
        dist.send(torch.zeros(1).cuda(), get_pp_communicator_next_rank())
    return rank
