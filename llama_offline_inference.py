from engine.llama_engine import LLamaEngine
from model.model_metadata import ModelConfig,ParallelConfig
from sampler.sampling_metadata import SamplingParams
import json


if __name__ == "__main__":
    model_config = ModelConfig("/data_local/70B-hf", "/data_local/70B-hf", True, 1, None)
    parallel_config = ParallelConfig(pipeline_parallel_size = 1, 
                                     tensor_parallel_size = 4)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.00, max_tokens=512)
    LLama = LLamaEngine(model_config, parallel_config, sampling_params)
    with open('./scrambled_sampled_dataset.json') as f:
        requests = json.load(f)
    LLama.generate(requests)
