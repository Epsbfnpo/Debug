#!/bin/bash

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
TIME_LIMIT=86100

echo "========================================================"
echo "🚀 启动 ESDG 训练任务 (Single Config Mode)"
echo "GPU 数量: $NUM_GPUS"
echo "配置来源: configs/defaults.py"
echo "========================================================"

torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 main.py