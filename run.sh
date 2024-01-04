#!/bin/bash

torchrun --nnodes $SLURM_NNODES \
        --nproc_per_node $GPU_PER_NODE \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --node_rank $SLURM_NODEID \
        ./llama_offline_inference.py
