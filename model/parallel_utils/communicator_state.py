from typing import List
import torch.distributed as dist

_PP_COMMUNICATOR_GROUP = None
_PP_COMMUNICATOR_GLOBAL_RANKS = None

def initialize_communicator_state(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> None:
    """
    Initialize async pipeline parallelism sender and receiver groups.
    ! NOTE: only be called in sender / receiver processes

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
    rank0, rank1, ... , rank7 in the default process group will be mapped to 
        g0, g1, ... , g7, respectively. The processes hold those rank are
        compute processes
    rank8, rank10, rank12, ... , rank22 in the default process group will be also mapped
        to g0, g1, ... , g7, respectively. The processes hold those rank are
        receiver processes
    rank9, rank11, rank13, ... , rank23 in the default process group will be also mapped
        to g0, g1, ... , g7, respectively. The processes hold those rank are
        sender processes
    Those sender and receiver processes are used for async pipeline parallelism.
    
    ! NOTE:
        rank0, rank8, rank16 are on the same 1CPU-1GPU hardware group
    """
    # Get world size and rank. Ensure some consistencies.
    assert dist.is_initialized()
    world_size: int = dist.get_world_size()
    assert world_size % 3 == 0, \
        "world_size is not divisible by 3"
    calculator_world_size: int = world_size // 3
    communicator_world_size: int = world_size // 3 * 2
    if (communicator_world_size !=
            tensor_model_parallel_size * pipeline_model_parallel_size * 2):  # 2 * 3
        raise RuntimeError(
            f"communicator_world_size ({communicator_world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) * "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size}) * 2")

    rank = dist.get_rank()
    global _PP_COMMUNICATOR_GROUP
    global _PP_COMMUNICATOR_GLOBAL_RANKS
    assert _PP_COMMUNICATOR_GROUP is None, (
        "pipeline parallel communication processes group is already initialized")
    for tp_rank in range(tensor_model_parallel_size):
        ranks: List[int] = []
        for pp_rank in range(pipeline_model_parallel_size):
            communicator_ranks = \
                range(calculator_world_size + pp_rank * tensor_model_parallel_size * 2 \
                        + tp_rank * 2,
                      calculator_world_size + pp_rank * tensor_model_parallel_size * 2 \
                        + tp_rank * 2 + 2)
            ranks.extend(communicator_ranks)
            group = dist.new_group(ranks)
        if rank in ranks:
            _PP_COMMUNICATOR_GROUP = group
            _PP_COMMUNICATOR_GLOBAL_RANKS = ranks
        
def get_pp_communicator_group():
    """Return the pipeline parallel communicator group the process belongs to."""
    assert _PP_COMMUNICATOR_GROUP is not None, (
        "pipeline parallel communicator group is not initialized")
    return _PP_COMMUNICATOR_GROUP

def get_pp_communicator_world_size():
    """Return the world size of the pipeline parallel communicator group."""
    assert _PP_COMMUNICATOR_GROUP is not None, (
        "pipeline parallel communicator group is not initialized")
    return dist.get_world_size(group=_PP_COMMUNICATOR_GROUP)

def get_pp_communicator_local_rank():
    """Return the local rank of the communicator process in current pipeline 
    parallel communicator group."""
    assert _PP_COMMUNICATOR_GROUP is not None, (
        "pipeline parallel communicator group is not initialized")
    return dist.get_rank(group=_PP_COMMUNICATOR_GROUP)

def get_pp_communicator_rank():
    """Return the global rank of the communicator process"""
    assert _PP_COMMUNICATOR_GROUP is not None, (
        "pipeline parallel communicator group is not initialized")
    return _PP_COMMUNICATOR_GLOBAL_RANKS[get_pp_communicator_local_rank()]

def get_pp_communicator_next_rank():
    """Return the global rank of the next process in current pipeline"""
    assert _PP_COMMUNICATOR_GROUP is not None, (
        "pipeline parallel communicator group is not initialized")
    return _PP_COMMUNICATOR_GLOBAL_RANKS[(get_pp_communicator_local_rank() + 1) % \
                                            get_pp_communicator_world_size()]

def get_pp_communicator_prev_rank():
    """Return the global rank of the previous process in current pipeline"""
    assert _PP_COMMUNICATOR_GROUP is not None, (
        "pipeline parallel communicator group is not initialized")
    return _PP_COMMUNICATOR_GLOBAL_RANKS[(get_pp_communicator_local_rank() - 1) % \
                                            get_pp_communicator_world_size()]