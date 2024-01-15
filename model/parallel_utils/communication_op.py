from typing import List, Optional, Union

import torch
import torch.distributed as dist
from model.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from model.parallel_utils.communicator_state import (
    get_pp_communicator_prev_rank,
    get_pp_communicator_next_rank,
)


def tensor_model_parallel_all_reduce(input_):
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation is applied in-place on the input tensor.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    # All-reduce.
    dist.all_reduce(input_,
                    group=get_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_, dim=-1):
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    dist.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor


#! functions and classes below can only be used by communicator processes
def send_to_next_pp_stage(tensor: torch.Tensor) -> None:
    return dist.send(tensor, get_pp_communicator_next_rank())

Shape = Union[List[int], torch.Size]
def receive_from_prev_pp_stage(
    tensor_dtype: torch.dtype,
    tensor_shape: Optional[Shape] = None,
    tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if tensor is None:
        tensor = torch.empty(tensor_shape, dtype=tensor_dtype, device='cuda')
    dist.recv(tensor, get_pp_communicator_prev_rank())
    return tensor

# def receive_from_last_pp_rank(
#     tensor_shape: Shape,
#     tensor_dtype: torch.dtype,
#     tensor: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     if tensor is None:
#         tensor = torch.empty(tensor_shape, dtype=tensor_dtype, device='cuda')
#     dist.recv(tensor, get_pipeline_model_parallel_last_rank())
#     return tensor


class pp_batch_send_or_recv:
    def __init__(self, 
                 ops: str | List[str], 
                 tensors: torch.Tensor | List[torch.Tensor] | torch.Size | List[torch.Size],
                 dtypes: torch.dtype | List[torch.dtype] = None,
                 is_shape: bool = False
                 ):
        print(f"type: {type(tensors)}")
        if isinstance(ops, str):
            ops = [ops]
            if isinstance(tensors, torch.Tensor):
                tensors = [tensors]
            if isinstance(dtypes, torch.dtype):
                dtypes = [dtypes]
        elif isinstance(ops, List):
            assert isinstance(tensors, List) == True, \
                "'tensors' argument should be a list of \
                    torch.Tensor when 'ops' is a list of string"
            assert len(tensors) == len(ops), \
                "'tensors' argument should have the same length as 'ops'"
            if ops == "recv" and is_shape:
                assert isinstance(dtypes, List) == True, \
                    "'dtypes' argument should be a list of \
                        torch.dtype when 'ops' is a list of string \
                        and is_shape is True"
                assert len(dtypes) == len(ops), \
                    "'dtypes' argument should have the same length as 'ops'"

        handlers = []
        self.tensors = []
        for i, op_tensor in enumerate(zip(ops, tensors)):
            op, tensor = op_tensor
            if op == "send":
                handler = dist.P2POp(op=dist.isend, 
                           tensor=tensor, 
                           peer=get_pp_communicator_prev_rank())
                handlers.append(handler)

            elif op == "recv":
                if is_shape:
                    assert isinstance(dtypes[i], torch.dtype), \
                        "dtype should be specified as type torch.dtype \
                            when is_shape is True and op is 'recv'"
                    tensor = torch.empty(torch.Size(tensor), dtype=dtypes[i], device='cuda')
                handler = dist.P2POp(op=dist.irecv, 
                           tensor=tensor, 
                           peer=get_pp_communicator_next_rank())
                handlers.append(handler)

            else:
                raise ValueError(f"Unsupported operation: {op}")

            self.tensors.append(tensor)

        self.handlers = dist.batch_isend_irecv(handlers)
        self.ops = ops
        
    def is_completed(self):
        completed = []
        all_completed = True
        for handler in self.handlers:
            result = handler.is_completed()
            completed.append(result)
            if not result:
                all_completed = False
        return all_completed 
    
    def wait(self):
        for handler in self.handlers:
            handler.wait()
        return self.tensors