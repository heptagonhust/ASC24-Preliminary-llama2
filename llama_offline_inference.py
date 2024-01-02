from llama_engine import LLamaEngine
from model.model_metadata import ModelConfig
from sampler.sampling_metadata import SamplingParams
import json


if __name__ == "__main__":
    model_config_llama = ModelConfig("/data/7B-chat-hf", "/data/7B-chat-hf", True, 1, None)
    LLama = LLamaEngine(model_config_llama)
    with open('./scrambled_sampled_dataset.json') as f:
        requests = json.load(f)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.00, max_tokens=512)
    LLama.generate(requests, sampling_params=sampling_params)
