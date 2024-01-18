import torch.multiprocessing as mp
from model.model_metadata import (
    ModelConfig, 
    ParallelConfig,
)
from scheduler.config import SchedulerConfig


class Scheduler():
    def __init__(
        self, 
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank
    ):
        self.scheduler_config = scheduler_config
        self.scheduler_config.model_config = model_config
        self.scheduler_config.parallel_config = parallel_config
        self.pipe_master_side, self.pipe_scheduler_side = mp.Pipe()
        self.scheduler = mp.Process(
            target=self._scheduler_loop, 
            args=(
                self.scheduler_config,
                self.pipe_scheduler_side,
            )
        )

    def start_loop(self):
        self.scheduler.start()
        return self.pipe_master_side


def _scheduler_loop(
    scheduler_config: SchedulerConfig,
    scheduler_pipe: mp.Pipe,
):
    pass