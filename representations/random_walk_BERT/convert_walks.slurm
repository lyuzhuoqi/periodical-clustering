#!/bin/bash
#SBATCH --job-name=convert_pid_to_int  # 作业名称
#SBATCH --output=convert_pid_to_int_%j.log  # 标准输出日志文件，%j表示job ID
#SBATCH --error=convert_pid_to_int_%j.err  # 错误日志文件，%j表示job ID
#SBATCH --time=1:00:00  # 预计作业运行时间 (格式: HH:MM:SS)
#SBATCH --ntasks=1  # 任务数量（通常是 1）
#SBATCH --cpus-per-task=64  # 每个任务的 CPU 核心数，根据你的并行设置来选择合适的值
#SBATCH --mem=100G  # 申请的内存大小（根据你的需求调整）
#SBATCH --partition=tiny  # 作业提交到的队列或分区（可以是你的集群的名字）
#SBATCH --nodes=1  # 节点数量，通常是 1

singularity exec --nv -B /home/zqlyu2/scratch:/home/zqlyu2/ containers/pytorch_2.5.1-cuda12.4-cudnn9-runtime.sif python convert_walks.py 
