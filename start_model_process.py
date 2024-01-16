import os
from router.model_infer.model_rpc import ModelRpcServer
from utils.log_utils import init_logger
logger = init_logger(__name__)
logger.info('sb')


def _init_env(port):
    from rpyc.utils.server import ThreadedServer
    print(f"rank: {os.environ['RANK']}, _init_env port:{port}")
    t = ThreadedServer(ModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return

if __name__ == "__main__":
    print("start_model_process")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-port", type=int, default=44444)
    args = parser.parse_args()

    port = int(os.environ['RANK']) % 2 + args.base_port
    _init_env(port)
    print("model process started")