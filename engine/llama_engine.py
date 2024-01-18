import os
from typing import List, Tuple

import torch
import torch.nn as nn

from model.model_metadata import ModelConfig,ParallelConfig
from sequence.config import SchedulerConfig
from sampler.sampling_metadata import SamplingParams
from engine.worker import Worker
from engine.master import Master

class LLamaEngine():
    def __init__(
        self, 
        model_config: ModelConfig, 
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig = SchedulerConfig(),
    ) -> nn.Module:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_worker()

    def _init_worker(self):
        if (int(os.environ["RANK"]) // self.parallel_config.tensor_parallel_size == 0):
            self.worker = Master(
                model_config=self.model_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                device=self.device,
            )
            self.worker.start_master()

        else:
            self.worker = Worker(
                model_config=self.model_config,
                parallel_config=self.parallel_config,
                device=self.device,
            )
            self.worker.start_worker()
    
    def generate(self,
                 sampling_params: SamplingParams = None):
        if (int(os.environ["RANK"]) // self.parallel_config.tensor_parallel_size == 0):
            self.worker.run(sampling_params)
        else:
            self.worker.run()
