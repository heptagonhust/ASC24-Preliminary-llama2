#! /bin/bash
#SBATCH --job-name=FT-BLOOM  ##提交的任务名称
#SBATCH -o baseline.log
#SBATCH -e baseline.err
#SBATCH -p GPU_nodes # 任务提交的分区名称
#SBATCH --nodes=1   # 任务申请3个节点
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16  ## 每个任务使用的cpu核数
#SBATCH --gres=gpu:2   ## 每个节点的gpu数
#SBATCH --mem 256G    ## 每个节点的memory 
###SBATCH -w hepgpu1 # 指定运行作业的节点，若不填写系统自动分配节点

python -u run_model.py

# srun --partition GPU_nodes -w hepgpu2,hepgpu3 -n4 --gres=gpu:2 --ntasks-per-node=2 --job-name=slrum_test python -u run_model.py