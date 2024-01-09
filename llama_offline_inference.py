from llama_engine import LLamaEngine
from model.model_metadata import ModelConfig,ParallelConfig
from sampler.sampling_metadata import SamplingParams
import json


if __name__ == "__main__":
    model_config_llama = ModelConfig("/data/70B-hf", "/data/70B-hf", True, 1, None)
    parallel_config_llama = ParallelConfig(pipeline_parallel_size = 1, tensor_parallel_size = 4)
    LLama = LLamaEngine(model_config_llama,parallel_config_llama)
    with open('./scrambled_sampled_dataset.json') as f:
        requests = json.load(f)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.00, max_tokens=512)
    LLama.generate(requests, sampling_params=sampling_params)
