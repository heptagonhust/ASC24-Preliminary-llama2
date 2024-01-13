# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Tensor and pipeline parallel groups."""

import torch.distributed as dist
# Tensor model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Pipeline model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None

# A list of global ranks for each pipeline group to ease calculation of the
# source rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert dist.is_initialized()
    world_size: int = dist.get_world_size()

    if (world_size !=
            tensor_model_parallel_size * pipeline_model_parallel_size):  # 2 * 3
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

    num_tensor_model_parallel_groups: int = (world_size //
                                             tensor_model_parallel_size)  # 3
    num_pipeline_model_parallel_groups: int = (world_size //
                                               pipeline_model_parallel_size)  # 2
    rank = dist.get_rank()

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, (
        "tensor model parallel group is already initialized")
    for i in range(num_tensor_model_parallel_groups):  # 3
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)  # (0,1) (2,3) (4,5)
        group = dist.new_group(ranks)  # 划分了三个group
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups.
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, (
        "pipeline model parallel group is already initialized")
    for i in range(num_pipeline_model_parallel_groups):  # 2
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)  # (0,2,4) (1,3,5)
        group = dist.new_group(ranks)  # 划分了两个group
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (_TENSOR_MODEL_PARALLEL_GROUP is not None
            and _PIPELINE_MODEL_PARALLEL_GROUP is not None)


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, (
        "tenosr model parallel group is not initialized")
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, (
        "pipeline model parallel group is not initialized")
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return dist.get_world_size(
        group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return dist.get_world_size(
        group=get_pipeline_model_parallel_group())


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return dist.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return dist.get_rank(
        group=get_pipeline_model_parallel_group())


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = dist.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, (
        "Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None

def get_pipeline_model_parallel_layer_num(total_layers: int):
    pp_world_size = get_pipeline_model_parallel_world_size()
    pp_rank = get_pipeline_model_parallel_rank()
    return (total_layers // pp_world_size) \
           + (pp_rank >= pp_world_size - total_layers % pp_world_size)
    
def get_pipeline_model_parallel_first_last_layer_idx(total_layers: int):
    pp_rank = get_pipeline_model_parallel_rank()
    pp_world_size = get_pipeline_model_parallel_world_size()
    num_layers = total_layers // pp_world_size
    if pp_rank < pp_world_size - total_layers % pp_world_size:
        first_layer_idx = num_layers * pp_rank
        last_layer_idx = first_layer_idx * (pp_rank + 1)
    else:
        first_layer_idx = total_layers - (num_layers + 1) * (pp_world_size - pp_rank)
        last_layer_idx = total_layers - (num_layers + 1) * (pp_world_size - pp_rank - 1)
    return first_layer_idx, last_layer_idx