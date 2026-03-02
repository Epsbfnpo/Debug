#!/bin/bash

BASE_OUTPUT_DIR="./output_esdg_h100"
DOMAINS=("APTOS" "DEEPDR" "FGADR" "IDRID" "MESSIDOR" "RLDR")
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
TIME_LIMIT=72000

echo "========================================================"
echo "🚀 启动 ESDG 批量实验 (Leave-One-Out 模式)"
echo "GPU 数量: $NUM_GPUS"
echo "待运行目标域(Target): ${DOMAINS[*]}"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo "注意: DDR 和 EYEPACS 将始终作为 Source Domains 参与训练"
echo "========================================================"

for TARGET in "${DOMAINS[@]}"
do
    echo ""
    echo "----------------------------------------------------------------"
    echo "▶️  [进度] 正在启动目标域: $TARGET (其余7个数据集为Source)"
    echo "----------------------------------------------------------------"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 main.py --time-limit $TIME_LIMIT --target-domain $TARGET --output $BASE_OUTPUT_DIR
    if [ $? -ne 0 ]; then
        echo "❌ [错误] 目标域 $TARGET 训练失败！"
    else
        echo "✅ [完成] 目标域 $TARGET 训练结束。"
    fi
    sleep 5
done
echo ""
echo "########################################################"
echo "📊 最终结果汇总 (Running collect_results.py)"
echo "########################################################"
python3 collect_results.py --base_dir "$BASE_OUTPUT_DIR" --domains "${DOMAINS[@]}"
echo "========================================================"
echo "🎉 所有任务执行完毕"
echo "========================================================"