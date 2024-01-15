#! /bin/bash
#SBATCH --job-name=baseline-tp-test
#SBATCH -o outputs/baseline-tp-test-%j.log
#SBATCH -e outputs/baseline-tp-test-%j.err
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --mem 256G 

export MASTER_ADDR=`hostname -i | awk '{print $2}'`
export MASTER_PORT="29500"
export GPU_PER_NODE=2
export NCCL_AVOID_RECORD_STREAMS=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=0

srun ./run.sh