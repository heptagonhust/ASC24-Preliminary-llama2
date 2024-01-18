from model.model_metadata import (
    ModelConfig, 
    ParallelConfig
)


class SchedulerConfig():
    def __init__(
        self, 
        dataset_path: str,
        batch_size: int,
        max_req_num: int, 
        max_token_num: int,
        max_req_total_len: int,
        max_new_token_len: int,
        router_token_ratio: float,
    ):
        self.model_config = None
        self.parallel_config = None
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_req_num = max_req_num
        self.max_token_num = max_token_num
        self.max_req_total_len = max_req_total_len
        self.max_new_token_len = max_new_token_len
        self.router_token_ratio = router_token_ratio