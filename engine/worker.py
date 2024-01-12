import torch
import torch.distributed as dist

from model.llama import LlamaForCausalLM
from model.model_metadata import (
    ModelConfig, 
    ParallelConfig
)
from model.parallel_utils.communication_op import (
    pipeline_model_parallel_async_send_and_recv,
    async_send_to_next_pp_rank,
    receive_from_prev_pp_rank,
    send_to_next_pp_rank,
)


class Worker():
    def __init__(self,
                 model: LlamaForCausalLM,
                 model_config: ModelConfig,
                 parallel_config: ParallelConfig):
        self.model = model
        self.model_config = model_config
        self.parallel_config = parallel_config
    
    def run(self):
        hidden_state = None
        hidden_positions = None
        seqs_id = None
        send_stream = torch.cuda.Stream()
        next_hidden_shape = torch.zeros([3], dtype=torch.long, device='cuda')
        idx = 0
        while True:
            print(f"rank: {dist.get_rank()}, forward: {idx}, 34")
            print(f"next_hidden_shape: {next_hidden_shape}")
            next_hidden_shape = receive_from_prev_pp_rank(
                                    tensor=next_hidden_shape,
                                    tensor_dtype=torch.long
                                )
            print(f"rank: {dist.get_rank()}, forward: {idx}, 39")
            print(f"next_hidden_shape: {next_hidden_shape}")
            if next_hidden_shape == torch.tensor([-1, -1, -1]):
                send_to_next_pp_rank(next_hidden_shape)
                break
            print(f"rank: {dist.get_rank()}, forward: {idx}, 44")

            #! receive hidden_state, positions, seqs_id from prev node
            next_hidden_receiver = pipeline_model_parallel_async_send_and_recv(
                                       ops=["recv", "recv", "recv"],
                                       tensors=[next_hidden_shape, 
                                                next_hidden_shape,
                                                next_hidden_shape[0]],
                                       dtypes=[self.model_config.dtype, 
                                               torch.int,
                                               torch.int],
                                       is_shape=True
                                   )
            print(f"rank: {dist.get_rank()}, forward: {idx}, 57")

            if hidden_state is not None:
                #! running model
                print(f"rank: {dist.get_rank()}, forward: {idx}, 61")
                hidden_state = self.model(input_ = hidden_state,
                                          positions = hidden_positions,
                                          kv_caches = None,
                                          input_metadata = None)
                print(f"rank: {dist.get_rank()}, forward: {idx}, 66")

                #! send hidden_state to next node
                # TODO: a better async way
                torch.cuda.default_stream().wait_stream(send_stream)
                last_hidden_state = hidden_state
                last_hidden_positions = hidden_positions
                last_seqs_id = seqs_id
                print(f"rank: {dist.get_rank()}, forward: {idx}, 74")
                with torch.cuda.stream(send_stream):
                    print(f"rank: {dist.get_rank()}, forward: {idx}, 76")
                    shape_sender = async_send_to_next_pp_rank(torch.tensor(last_hidden_state.shape))
                    print(f"rank: {dist.get_rank()}, forward: {idx}, 78")
                    shape_sender.wait()
                    print(f"rank: {dist.get_rank()}, forward: {idx}, 80")
                    hidden_sender = pipeline_model_parallel_async_send_and_recv(
                                        ops=["send", "send", "send"],
                                        tensors=[last_hidden_state, 
                                                 last_hidden_positions, 
                                                 last_seqs_id],
                                        is_shape=False
                                    )
                    print(f"rank: {dist.get_rank()}, forward: {idx}, 88")
                    hidden_sender.wait()
                    print(f"rank: {dist.get_rank()}, forward: {idx}, 90")
            idx += 1

            #! wait for receive complete
            hidden_state, hidden_positions, seqs_id = next_hidden_receiver.wait()
