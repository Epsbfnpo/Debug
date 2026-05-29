#!/bin/bash

BASE_OUTPUT_DIR=${OUTPUT_DIR:-"./output_esdg_512_h100_profile"}
PROFILE_COMPUTE=${PROFILE_COMPUTE:-1}

PROFILE_ARGS=()
if [ "$PROFILE_COMPUTE" = "1" ]; then
    PROFILE_ARGS+=("--profile-compute")
fi
SOURCES=("APTOS" "DEEPDR" "FGADR" "IDRID" "MESSIDOR" "RLDR")
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
TIME_LIMIT=360000

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

echo "========================================================"
echo "🚀 启动 SSMT / ESDG 批量实验"
echo "GPU 数量: $NUM_GPUS"
echo "待运行源域: ${SOURCES[*]}"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo "Compute profiling: $PROFILE_COMPUTE"
echo "========================================================"

for SOURCE in "${SOURCES[@]}"
do
    echo ""
    echo "----------------------------------------------------------------"
    echo "▶️  [进度] 正在启动源域: $SOURCE"
    echo "----------------------------------------------------------------"

    torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 main.py \
        --time-limit $TIME_LIMIT \
        --source-domain "$SOURCE" \
        --output "$BASE_OUTPUT_DIR" \
        "${PROFILE_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "❌ [错误] 源域 $SOURCE 训练失败！"
    else
        echo "✅ [完成] 源域 $SOURCE 训练结束。"
    fi

    sleep 5
done
