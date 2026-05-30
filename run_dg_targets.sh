#!/bin/bash

BASE_OUTPUT_DIR=${OUTPUT_DIR:-"./output_dg_512_h100_ddr_eyepacs"}
PROFILE_COMPUTE=${PROFILE_COMPUTE:-1}

PROFILE_ARGS=()
if [ "$PROFILE_COMPUTE" = "1" ]; then
    PROFILE_ARGS+=("--profile-compute")
fi

COMMON_SOURCES=("APTOS" "DEEPDR" "FGADR" "IDRID" "MESSIDOR" "RLDR")
TARGETS=("DDR" "EYEPACS")

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
TIME_LIMIT=360000

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

echo "========================================================"
echo "🚀 启动 Multi-Source Single-Target DG 实验"
echo "GPU 数量: $NUM_GPUS"
echo "Sources: ${COMMON_SOURCES[*]}"
echo "Targets: ${TARGETS[*]}"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo "Compute profiling: $PROFILE_COMPUTE"
echo "========================================================"

FAILED=0

for TARGET in "${TARGETS[@]}"
do
    echo ""
    echo "----------------------------------------------------------------"
    echo "▶️  [进度] 正在启动目标域: $TARGET"
    echo "Sources: ${COMMON_SOURCES[*]}"
    echo "Target: $TARGET"
    echo "----------------------------------------------------------------"

    torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 main.py \
        --time-limit $TIME_LIMIT \
        --source-domains "${COMMON_SOURCES[@]}" \
        --target-domains "$TARGET" \
        --output "$BASE_OUTPUT_DIR" \
        "${PROFILE_ARGS[@]}"

    if [ $? -ne 0 ]; then
        echo "❌ [错误] 目标域 $TARGET 训练失败！"
        FAILED=1
    else
        echo "✅ [完成] 目标域 $TARGET 训练结束。"
    fi

    sleep 5
done

if [ "$FAILED" -ne 0 ]; then
    echo "❌ 至少一个目标域实验失败。"
    exit 1
fi

echo "✅ 所有 Multi-Source Single-Target DG 实验完成。"
