
class SchedulerConfig:
    def __init__(self,
                 max_batch_size: int = 1,
                 max_seq_len: int = 2048,
                 use_tqdm: bool = True,
                 show_progress: bool = True):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.use_tqdm = use_tqdm
        self.show_progress = show_progress