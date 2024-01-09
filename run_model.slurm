#! /bin/bash
#SBATCH --job-name=baseline-tp-test
#SBATCH -o outputs/baseline-tp-test.log
#SBATCH -e outputs/baseline-tp-test.err
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --mem 256G 

export MASTER_ADDR="192.168.251.102"
export MASTER_PORT="29500"
export GPU_PER_NODE=2

srun ./run.sh