import os
from typing import List, Tuple

import torch
import torch.nn as nn

from model.model_metadata import ModelConfig,ParallelConfig
from sampler.sampling_params import SamplingParams
from engine.worker import Worker
from engine.master import Master
from scheduler.config import SchedulerConfig, CacheConfig

class LLamaEngine():
    def __init__(
        self, 
        model_config: ModelConfig, 
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig = SchedulerConfig(),
        cache_config: CacheConfig = CacheConfig(),
    ) -> nn.Module:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_worker()

    def _init_worker(self):
        if (int(os.environ["RANK"]) // self.parallel_config.tensor_parallel_size == 0):
            self.worker = Master(
                model_config=self.model_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                cache_config=self.cache_config,
                device=self.device,
            )
            self.worker.start_master()

        else:
            self.worker = Worker(
                model_config=self.model_config,
                parallel_config=self.parallel_config,
                cache_config=self.cache_config,
                device=self.device,
            )
            self.worker.start_worker()
    
    def generate(self,
                 requests: List[Tuple[str, int, int]], 
                 sampling_params: SamplingParams = None):
        if (int(os.environ["RANK"]) // self.parallel_config.tensor_parallel_size == 0):
            self.worker.add_requests(requests, sampling_params)
            self.worker.run()
        else:
            self.worker.run()
