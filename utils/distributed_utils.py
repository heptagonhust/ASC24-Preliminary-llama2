import os
import torch
import torch.distributed as dist
from model.model_metadata import ParallelConfig
from model.parallel_utils.parallel_state import initialize_model_parallel


def init_torch_dist(rank: int, 
                    parallel_config:ParallelConfig, 
                    backend="nccl"):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=parallel_config.world_size,
        rank=rank,
    )

    # A small all_reduce for warmup.
    dist.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)