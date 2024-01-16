#!/bin/bash
#SBATCH --job-name=baseline-tp-test
#SBATCH -o outputs/baseline-tp-test-%j.log
#SBATCH -e outputs/baseline-tp-test-%j.err
#SBATCH --nodes=2
#SBATCH --mem 256G 

export MASTER_ADDR=`hostname -i | awk '{print $2}'`
export MASTER_PORT="29500"
export GPU_PER_NODE=2

export BASE_PORT=44444

NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
IP_ADDRESSES=""

# get ip addresses of nodes
for NODE in $NODELIST; do
    IP=$(srun --nodes=1 --ntasks=1 -w $NODE hostname -i | awk '{print $2}')
    IP_ADDRESSES="${IP},${IP_ADDRESSES}"
done

export IP_ADDRESSES=$IP_ADDRESSES
srun -o outputs/run2-%j.log -e outputs/run2-%j.err --ntasks-per-node=1 ./run2.sh