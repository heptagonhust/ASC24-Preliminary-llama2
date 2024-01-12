import os
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from model.model_metadata import ParallelConfig, ModelConfig
from model.parallel_utils.parallel_state import (
    initialize_model_parallel,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group
)
from utils.utils import set_random_seed


def init_distributed(model_config:ModelConfig,
                     parallel_config:ParallelConfig,
                     backend="nccl") -> int:
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    set_random_seed(model_config.seed)
    num_gpus = torch.cuda.device_count()
    # os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f'node: {os.environ["SLURM_NODEID"]}, rank: {rank}')

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

    # A small all_reduce for warmup the default process group
    dist.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)
    # A small all_reduce for warmup the tp process group
    dist.all_reduce(torch.zeros(1).cuda(), group=get_tensor_model_parallel_group())
    # A small all_reduce for warmup the pp process group
    dist.all_reduce(torch.zeros(1).cuda(), group=get_pipeline_model_parallel_group())
    return rank

def init_distributed_rpc(rank: int):
    world_size = dist.get_world_size()
    if get_pipeline_model_parallel_rank() == 0:
        rpc.init_rpc(
            name=f"master_{rank}",
            rank=rank,
            world_size=world_size,
        )
    else:
        rpc.init_rpc(
            name=f"worker_{rank}",
            rank=rank,
            world_size=world_size,
        )