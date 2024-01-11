# from llama_engine import LLamaEngine
from llama_engine_new import RequestEngine
from model.model_metadata import ModelConfig, ParallelConfig, PortConfig
from sampler.sampling_metadata import SamplingParams
import json
from utils.start_utils import start_submodule_processes, kill_submodule_processes
from router.manager import start_router_process


if __name__ == "__main__":
    model_config_llama = ModelConfig("/data/7B-chat-hf", "/data/7B-chat-hf", True, 1, None)
    parallel_config_llama = ParallelConfig(pipeline_parallel_size = 1, tensor_parallel_size = 1)
    port_config = PortConfig(router_port=55555, req_server_port=55556)
    LLama = RequestEngine(model_config_llama, parallel_config_llama)
    with open('./scrambled_sampled_dataset.json') as f:
        requests = json.load(f)
    requests = requests[100:101]
    sampling_params = SamplingParams(temperature=1.0, top_p=1.00, max_tokens=512)


    '''
    the following code is used to start the submodule processes
    Args:
        model_dir: 模型路径
        batch_size: 批处理最大的大小
        max_total_token_num: 模型最大长度
        max_req_num: 同时到来的最大请求数
        max_req_total_len: 输入+输出最大长度
        router_token_ratio: router的token占用率
        router_max_new_token_len: router最大的新token长度
        router_port: router的端口
        req_server_port: req_server的端口
    '''
    submodule_pid = start_submodule_processes(
        start_funcs=[start_router_process],
        start_args=[(model_config_llama.model,
                     20,
                     121060,
                     10000,
                     2048 + 2048,
                     0.5,
                     2048,
                     port_config.router_port,
                     port_config.req_server_port)]
    )

    LLama.generate(requests, sampling_params=sampling_params)

    kill_submodule_processes(submodule_pid)
