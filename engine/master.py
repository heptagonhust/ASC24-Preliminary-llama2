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
from sequence.scheduler import SequenceScheduler
from sampler.sampling_params import SamplingParams


class Master():
    def __init__(self,
                 model: LlamaForCausalLM,
                 model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 scheduler: SequenceScheduler):
        self.model = model
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler = scheduler
        
    def run(self,
            sampling_params: SamplingParams):
        self.sampling_params = sampling_params
        hidden_state = None
        hidden_positions = None
        seqs_id = None
        last_hidden_state = None
        last_hidden_positions = None
        last_seqs_id = None
        next_hidden_receiver = None
        
        send_stream = torch.cuda.Stream()
        recv_stream = torch.cuda.Stream()
        next_hidden_shape = torch.zeros([3], dtype=torch.long, device='cuda')
        idx = 0
        while not self.scheduler.is_finished():
            print(f"rank: {dist.get_rank()}, forward: {idx}, 44")

            if hidden_state is not None:
                print(f"rank: {dist.get_rank()}, forward: {idx}, 49")
                if recv_stream.query():
                    print(f"rank: {dist.get_rank()}, forward: {idx}, 51")
                    torch.cuda.default_stream().wait_stream(recv_stream)
                    print(f"rank: {dist.get_rank()}, forward: {idx}, 53")

                    with torch.cuda.stream(recv_stream):
                        print(f"rank: {dist.get_rank()}, forward: {idx}, 55")
                        next_hidden_shape = receive_from_prev_pp_rank(
                                                tensor_dtype=torch.long,
                                                tensor=next_hidden_shape
                                            )
                        print(f"rank: {dist.get_rank()}, forward: {idx}, 59")

                print(f"rank: {dist.get_rank()}, forward: {idx}, 74")
                #! running model
                hidden_state = self.model(input_ = hidden_state,
                                          positions = hidden_positions,
                                          kv_caches = None,
                                          input_metadata = None)

                print(f"rank: {dist.get_rank()}, forward: {idx}, 81")
                if recv_stream.query():
#! recv hidden_state, positions, seqs_id from prev node
                    with torch.cuda.stream(recv_stream):
                        next_hidden_receiver = \
                            pipeline_model_parallel_async_send_and_recv(
                                ops=["recv", "recv", "recv"],
                                tensors=[next_hidden_shape, 
                                         next_hidden_shape,
                                         next_hidden_shape[0]],
                                dtypes=[self.model_config.dtype, 
                                        torch.int,
                                        torch.int],
                                is_shape=True
                            )
                        print(f"rank: {dist.get_rank()}, forward: {idx}, 72")
                        recv_hidden_state, recv_positions, recv_seqs_id = \
                            next_hidden_receiver.wait()

                print(f"rank: {dist.get_rank()}, forward: {idx}, 88")
                
                torch.cuda.default_stream().wait_stream(send_stream)
                last_hidden_state = hidden_state
                last_hidden_positions = hidden_positions
                last_seqs_id = seqs_id
                print(f"rank: {dist.get_rank()}, forward: {idx}, 86")
                with torch.cuda.stream(send_stream):
                    print(f"shape of sending tensor: {last_hidden_state.shape}")
                    shape_sender = async_send_to_next_pp_rank(torch.tensor(last_hidden_state.shape))
                    shape_sender.wait()
                    #! send hidden_state, positions, seqs_id to next node
                    hidden_sender = pipeline_model_parallel_async_send_and_recv(
                                        ops=["send", "send", "send"],
                                        tensors=[last_hidden_state, 
                                                 last_hidden_positions,
                                                 last_seqs_id],
                                        is_shape=False
                                    )
                    hidden_sender.wait()
                print(f"rank: {dist.get_rank()}, forward: {idx}, 99")




            # print(f"rank: {dist.get_rank()}, forward: {idx}, 86")
            if hidden_state is None or not recv_stream.query():
                hidden_state, hidden_positions, seqs_id = self.scheduler.get_new_batch()
            else:
                torch.cuda.default_stream().wait_stream(recv_stream)
                hidden_state, hidden_positions, seqs_id = \
                    recv_hidden_state, recv_positions, recv_seqs_id
                hidden_state, hidden_positions, seqs_id = \
                    self.do_sample(hidden_state, seqs_id)
            print(f"rank: {dist.get_rank()}, forward: {idx}, 104")
            idx += 1
                
        #! end of work
        send_to_next_pp_rank(torch.tensor([-1, -1, -1]))
        receive_from_prev_pp_rank(
            tensor_shape=torch.Size([3]),
            tensor_dtype=torch.long
        )
        return

    def do_sample(self,
                  hidden_state: torch.Tensor,
                  seqs_id: torch.Tensor):
        sampling_metadata = \
            self.scheduler.get_sampling_metadata_from_seqs_id(seqs_id,
                                                              self.sampling_params)
        sampler_output = self.model.sample(hidden_state, sampling_metadata)
        return self.scheduler.update_batch(seqs_id, sampler_output)

        
        