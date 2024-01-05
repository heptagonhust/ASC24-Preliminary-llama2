#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:2
#SBATCH --time=24:00:00
#SBATCH --job-name=ray-baseline
#SBATCH --partition=GPU_nodes
#SBATCH --output=outputs/ray_baseline_%j.out
#SBATCH --error=outputs/ray_baseline_%j.err
#SBATCH --export=ALL


export headnode=`hostname -s`
export headaddr=`hostname -i | awk '{print $2}'`
export port=6379
export MASTER_ADDR=$headaddr
export MASTER_PORT="29500"

export RAY_DEDUP_LOGS=0

srun -J start_ray ./start_ray_server.sh
jobID=$(squeue --name start_ray | tail -1 | awk '{print $1}')

srun -J run_baseline --dependency=afterok:${jobID} ./run_model.sh
jobID=$(squeue --name run_baseline | tail -1 | awk '{print $1}')

srun -J stop_ray --dependency=afterany:${jobID} ray stop
