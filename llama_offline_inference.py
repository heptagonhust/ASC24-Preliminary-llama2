from engine.llama_engine import LLamaEngine
from model.model_metadata import ModelConfig,ParallelConfig
from scheduler.config import SchedulerConfig
from sampler.sampling_metadata import SamplingParams
import json


if __name__ == "__main__":
    model_config = ModelConfig(
        model="/data_local/70B-hf", 
        tokenizer="/data_local/70B-hf", 
        trust_remote_code=True, 
        seed=1, 
        max_model_len=None)
    parallel_config = ParallelConfig(
        pipeline_parallel_size = 3, 
        tensor_parallel_size = 2)
    scheduler_config = SchedulerConfig(
        dataset_path = "./scrambled_sampled_dataset.json", 
        batch_size = 1, 
        max_req_num = 1, 
        max_token_num = 512, 
        max_req_total_len = 512, 
        max_new_token_len = 512, 
        router_token_ratio = 0.1)
    LLama = LLamaEngine(
        model_config=model_config, 
        parallel_config=parallel_config,
        scheduler_config=scheduler_config)
    with open('./scrambled_sampled_dataset.json') as f:
        requests = json.load(f)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.00, max_tokens=512)
    LLama.generate(requests[:50], sampling_params=sampling_params)
