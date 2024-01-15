import sys
import os
import signal
import multiprocessing as mp
from typing import List
import asyncio


def start_submodule_processes(start_funcs=[], start_args=[]):
    assert len(start_funcs) == len(start_args)
    pipe_readers = []
    processes: List[mp.Process] = []
    queues = []
    for start_func, start_arg in zip(start_funcs, start_args):
        mp.set_start_method('spawn')
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        queue1 = mp.Queue()
        queue2 = mp.Queue()
        process = mp.Process(
            target=start_func,
            args=start_arg + (pipe_writer,queue1,queue2,),
        )
        process.start()
        pipe_readers.append(pipe_reader)
        queues.append(queue1)
        queues.append(queue2)
        processes.append(process)
    
    # wait to ready
    for index, pipe_reader in enumerate(pipe_readers):
        init_state = pipe_reader.recv()
        if init_state != 'init ok':
            print(f"init func {start_funcs[index].__name__} : {str(init_state)}")
            for proc in processes:
                proc.kill()
            sys.exit(1)
        else:
            print(f"init func {start_funcs[index].__name__} : {str(init_state)}")
    
    assert all([proc.is_alive() for proc in processes])
    map(lambda p: p.join(), processes)
    return [proc.pid for proc in processes],queues

def kill_submodule_processes(pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass
    return