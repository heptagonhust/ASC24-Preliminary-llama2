import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class Receiver():
    def __init__(self, rank):
        self.rank = rank
        