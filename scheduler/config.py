from model.model_metadata import (
    ModelConfig, 
    ParallelConfig
)


class SchedulerConfig():
    def __init__(
        self, 
        model_config: ModelConfig, 
        parallel_config: ParallelConfig, 
        dataset_path: str,
        batch_size: int,
        max_req_num: int, 
        max_token_num: int,
        max_req_total_len: int,
        max_new_token_len: int,
        router_token_ratio: float,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_req_num = max_req_num
        self.max_token_num = max_token_num
        self.max_req_total_len = max_req_total_len
        self.max_new_token_len = max_new_token_len
        self.router_token_ratio = router_token_ratio