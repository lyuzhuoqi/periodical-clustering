#!/bin/bash
#SBATCH --job-name=train_bert  # 作业名称
#SBATCH --output=logs/train_bert_%j.log  # 标准输出日志文件，%j表示job ID
#SBATCH --error=logs/train_bert_%j.err # 错误输出日志文件
#SBATCH --partition=gpu_a40  # 作业提交到的队列或分区（可以是你的集群的名字）
#SBATCH --time=6:00:00  # 预计作业运行时间 (格式: HH:MM:SS)
#SBATCH --cpus-per-task=4  # 每个任务的 CPU 核心数，根据你的并行设置来选择合适的值
#SBATCH --gres=gpu:8 # GPU 数量，根据你的需求调整
#SBATCH --mem=100G  # 申请的内存大小（根据你的需求调整）
#SBATCH --ntasks=1  # 任务数量（通常是 1）
#SBATCH --nodes=1  # 节点数量，通常是 1

# 输出作业信息
echo "Job started on $(date)"
echo "Job running on $SLURM_JOB_NODELIST"

# 启动训练脚本
singularity exec --nv -B /home/zqlyu2/scratch:/home/zqlyu2/ containers/train_bert.sif python train_bert.py 

# 结束时输出
echo "Job finished on $(date)"
