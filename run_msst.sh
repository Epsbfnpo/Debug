#!/bin/bash

BASE_OUTPUT_DIR="./output_msst_512_h100"
TARGETS=("APTOS" "DEEPDR" "FGADR" "IDRID" "MESSIDOR" "RLDR")
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
TIME_LIMIT=360000

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

echo "========================================================"
echo "🚀 启动 MSST 批量实验"
echo "GPU 数量: $NUM_GPUS"
echo "待运行目标域: ${TARGETS[*]}"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo "========================================================"

for TARGET in "${TARGETS[@]}"
do
    echo ""
    echo "----------------------------------------------------------------"
    echo "▶️  [进度] 正在启动目标域: $TARGET"
    echo "----------------------------------------------------------------"

    torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 main.py \
        --time-limit $TIME_LIMIT \
        --target-domain $TARGET \
        --output $BASE_OUTPUT_DIR

    if [ $? -ne 0 ]; then
        echo "❌ [错误] 目标域 $TARGET 训练失败！"
    else
        echo "✅ [完成] 目标域 $TARGET 训练结束。"
    fi

    sleep 5
done
