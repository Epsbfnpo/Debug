#!/bin/bash

BASE_OUTPUT_DIR="./output_esdg_h100"
DOMAINS=("APTOS" "DEEPDR" "FGADR" "IDRID" "MESSIDOR" "RLDR")
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
TIME_LIMIT=36000
echo "========================================================"
echo "🚀 启动 ESDG 批量实验 (Bash 循环模式)"
echo "GPU 数量: $NUM_GPUS"
echo "待运行源域: ${DOMAINS[*]}"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo "========================================================"
for SOURCE in "${DOMAINS[@]}"
do
    echo ""
    echo "----------------------------------------------------------------"
    echo "▶️  [进度] 正在启动源域: $SOURCE"
    echo "----------------------------------------------------------------"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 main.py --time-limit $TIME_LIMIT --source-domain $SOURCE --output $BASE_OUTPUT_DIR
    if [ $? -ne 0 ]; then
        echo "❌ [错误] 源域 $SOURCE 训练失败！"
    else
        echo "✅ [完成] 源域 $SOURCE 训练结束。"
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