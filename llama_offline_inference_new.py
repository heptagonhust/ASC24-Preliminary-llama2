# from llama_engine import LLamaEngine
import os
from llama_engine_new import RequestEngine
from model.model_metadata import ModelConfig, ParallelConfig, PortConfig, ReqConfig
from sampler.sampling_metadata import SamplingParams
import json
from utils.start_utils import start_submodule_processes, kill_submodule_processes
from router.manager import start_router_process

from utils.log_utils import init_logger
logger = init_logger(__name__)

if __name__ == "__main__":

    # 拿到模型进程的主机号
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=str, default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=44444)
    args = parser.parse_args()
    hosts: str = args.hosts
    hosts = hosts.split(",")
    hosts = hosts[:-1]
    rpc_base_port = args.base_port


    model_config_llama = ModelConfig("/data_local/13B-hf", "/data_local/13B-hf", True, 1, None)
    parallel_config_llama = ParallelConfig(pipeline_parallel_size = 1, tensor_parallel_size = 2)
    port_config = PortConfig(router_port=55555, req_server_port=55556, rpc_base_port=rpc_base_port)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.00, max_tokens=512)
    req_config = ReqConfig(batch_size=10000,
                           max_total_token_num=20000,
                           max_req_num=10000,
                           max_req_total_len=2048+512,
                           router_token_ratio=0.5,
                           router_max_new_token_len=2048)


    '''
    the following code is used to start the submodule processes
    Args:
        model_dir: 模型路径
        batch_size: 两个Batch可能会合并，这是用来控制每个被合并Batch占用token大小的
        max_total_token_num: kvcache里面的最大token数目，越大越容易OOM
        max_req_num: 一个Batch里面的最大请求数
        max_req_total_len: 输入+输出最大长度
        router_token_ratio: router的token占用率
        router_max_new_token_len: router最大的新token长度
        router_port: router的端口
        req_server_port: req_server的端口
    '''
    submodule_pid = start_submodule_processes(
        start_funcs=[start_router_process],
        start_args=[(
            model_config_llama.model,
            model_config_llama.hf_model_config,
            req_config.batch_size,
            req_config.max_total_token_num,
            req_config.max_req_num,
            req_config.max_req_total_len,
            req_config.router_token_ratio,
            req_config.router_max_new_token_len,
            port_config,
            parallel_config_llama,
            hosts
        )]
    )

    LLama = RequestEngine(model_config_llama, parallel_config_llama, port_config, sampling_params, req_config)
    with open('./scrambled_sampled_dataset.json') as f:
        requests = json.load(f)
    requests = requests[:500]

    LLama.generate(requests, sampling_params=sampling_params)

    kill_submodule_processes(submodule_pid)
