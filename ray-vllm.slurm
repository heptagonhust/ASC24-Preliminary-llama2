#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:2
#SBATCH --time=24:00:00
#SBATCH --job-name=vllm-baseline
#SBATCH --partition=GPU_nodes
#SBATCH --output=outputs/vllm_%j.out
#SBATCH --error=outputs/vllm_%j.err
#SBATCH --export=ALL

# export headnode=`hostname -s`
export headnode='hepgpu1'
export headaddr='192.168.251.101'
export port=6379
export NCCL_DEBUG=INFO 

srun -J start_ray ./start_ray_server.sh
jobID=$(squeue --name start_ray | tail -1 | awk '{print $1}')

srun -J run_vllm --dependency=afterok:${jobID} ./run_vllm.sh
jobID=$(squeue --name run_vllm | tail -1 | awk '{print $1}')

srun -J stop_ray --dependency=afterany:${jobID} ray stop
