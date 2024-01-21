from engine.llama_engine import LLamaEngine
from model.model_metadata import ModelConfig, ParallelConfig
from scheduler.config import SchedulerConfig, CacheConfig
from sampler.sampling_metadata import SamplingParams
import json


if __name__ == "__main__":
    model_config = ModelConfig("/data_local/70B-hf", "/data_local/70B-hf", True, 1, None)
    parallel_config = ParallelConfig(pipeline_parallel_size = 3, tensor_parallel_size = 2)
    scheduler_config = SchedulerConfig()
    LLama = LLamaEngine(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config
    )
    with open('./scrambled_sampled_dataset.json') as f:
        requests = json.load(f)
    sampling_params = SamplingParams(temperature=1e-6, top_p=1.00, max_tokens=512)
    LLama.generate(requests[:500], sampling_params=sampling_params)
