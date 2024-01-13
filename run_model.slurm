#! /bin/bash
#SBATCH --job-name=baseline-tp-test
#SBATCH -o outputs2/baseline-tp-test-%j.log
#SBATCH -e outputs2/baseline-tp-test-%j.err
#SBATCH --nodes=2
#SBATCH -w hepgpu3
#SBATCH --gres=gpu:2
#SBATCH --mem 256G 

export MASTER_ADDR="192.168.251.105"
export MASTER_PORT="29501"
export GPU_PER_NODE=2

srun ./run.sh