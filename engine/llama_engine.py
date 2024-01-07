import os
from functools import partial
from typing import List, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm
import ray
from ray.util.placement_group import (
    PlacementGroup,
)
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy
)
from ray.air.util.torch_dist import init_torch_dist_process_group

from sampler.sampling_metadata import SamplingParams
from model.model_metadata import ModelConfig,ParallelConfig
from sequence.sequence import Sequence
# from engine.worker import Worker


class LLamaEngine():
    def __init__(self, 
                 model_config: ModelConfig, 
                 parallel_config:ParallelConfig,
                 sampling_params: SamplingParams = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
                        model_config.tokenizer, 
                        trust_remote_code=model_config.trust_remote_code)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.init_worker(model_config, parallel_config, sampling_params)
        self.init_distributed()
        self.run_worker("init_model")


    def generate(self, requests: List[Tuple[str, int, int]]):
        pbar = tqdm(total=len(requests), desc="Processed prompts")
        for i in range(len(requests)):
            prompt, prompt_len, output_len = requests[i] 
            output = self.run_seq(prompt, i, output_len)
            print(f'input: {prompt}\ninput_len: {prompt_len}\n', flush=True)
            print(f'output: {output}\noutput_len: {len(output)}\n\n', flush=True)
            pbar.update(1)
        pbar.close()

    def run_seq(self, request, request_id, max_output_len):
        input_id = self.tokenizer.encode(request)
        seq = Sequence(request_id, request, input_id, block_size=0)   

        output_token_ids = self.run_worker("run_model", 
                                 request_id, 
                                 max_output_len, 
                                 self.tokenizer.eos_token_id, 
                                 seq)
        output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
        return output

    def init_worker(self, 
                    model_config: ModelConfig,
                    parallel_config: ParallelConfig,
                    sampling_params: SamplingParams):
        ray.init()
        num_gpus = ray.cluster_resources().get('GPU', 0)
        num_cpus = ray.cluster_resources().get('CPU', 0)
        num_workers = parallel_config.world_size
        num_gpus_req = num_workers * parallel_config.num_gpus_per_worker
        num_cpus_req = num_workers * parallel_config.num_cpus_per_worker
        assert num_gpus >= num_gpus_req, "Not enough GPUs for workers"
        assert num_cpus >= num_cpus_req, "Not enough CPUs for workers"

        pg_list: List[PlacementGroup] = []
        from engine.worker import Worker
        self.workers: List[Worker] = []

        assert num_workers % parallel_config.workers_per_node == 0, \
            "Number of workers must be divisible by workers_per_node to place workers on nodes"
        for _ in range(num_workers // parallel_config.workers_per_node):
            bundles = [{'CPU': parallel_config.num_cpus_per_worker,
                        'GPU': parallel_config.num_gpus_per_worker}] * parallel_config.workers_per_node
            pg = ray.util.placement_group(bundles, strategy="STRICT_PACK")
            ray.get(pg.ready(), timeout=10)
            pg_list.append(pg)

        #! place workers[0], ..., workers[workers_per_node-1]into pg[0];
        #!       workers[workers_per_node], ...,workers[2workers_per_node-1] into pg[1]
        #!       ...
        print(f"test cuda visible devices(master node): {os.environ['CUDA_VISIBLE_DEVICES']}")
        for rank in range(num_workers):
            node_rank = rank // parallel_config.workers_per_node
            local_rank = rank % parallel_config.workers_per_node
            worker = ray.remote(
                        num_cpus=0,
                        num_gpus=parallel_config.num_gpus_per_worker,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg_list[node_rank],
                            placement_group_bundle_index=local_rank,
                            placement_group_capture_child_tasks=True)
                        )(Worker).remote(model_config = model_config, 
                                        parallel_config = parallel_config,
                                        sampling_params = sampling_params,
                                        rank = rank)
            self.workers.append(worker)
        
    def init_distributed(self):
        init_torch_dist_process_group(self.workers, backend='nccl')
        self.run_worker("init_distributed")
        
        
    def run_worker(self, method, *args, **kargs):
        outputs = []
        for worker in self.workers:
            executor = partial(worker.execute_method.remote, method)
            outputs.append(executor(*args, **kargs))
        output_list = ray.get(outputs)
        
        output = output_list[0]
        for output_node in output_list[1:]:
            assert output_node == output, "The output of all nodes should be the same"
        return output
        