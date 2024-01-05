import os
from typing import List, Dict
from datetime import timedelta

import torch
import torch.distributed as dist
import ray
from ray.util.placement_group import (
    PlacementGroup
)
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy
)

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
    
    node_id = os.environ.get("SLURM_NODEID")
    assert node_id is not None, \
        "SLURM_NODEID is None, cannot initialize distributed environment"


    torch.cuda.set_device(rank % num_gpus)
    # print(f"rank:{rank}, node_id: {node_id}", flush=True)

    dist.init_process_group(
        backend=backend,
        world_size=parallel_config.world_size,
        rank=rank,
        timeout=timedelta(seconds=120),
    )

    # A small all_reduce for warmup.
    dist.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)

def get_node_rank():
    node_id = os.environ.get("SLURM_NODEID")
    assert node_id is not None, "SLURM_NODEID is None"
    return node_id

def get_node_rank_of_pgs(pg_list: List[PlacementGroup]):
    node_rank_table: Dict[int: int] = {}
    for i, pg in enumerate(pg_list):
        node_rank = ray.remote(num_cpus=1,
                               scheduling_strategy=PlacementGroupSchedulingStrategy(
                                   placement_group=pg,
                                   placement_group_bundle_index=0,
                               ))(get_node_rank).remote()
        node_rank = int(ray.get(node_rank))
        node_rank_table[node_rank] = i
    return node_rank_table
