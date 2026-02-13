#!/bin/bash

# ================= 配置区域 =================
# 数据集根目录
ROOT_DATA="/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/GDR_Formatted_Data"

# 输出目录 (修改：允许外部环境变量覆盖，默认值不变)
OUTPUT_DIR=${OUTPUT_DIR:-"./output_esdg_h100"}

# 使用的算法
ALGORITHM="GDRNet"

# 域设置
SOURCE="MESSIDOR"
TARGETS="APTOS DDR DEEPDR FGADR IDRID RLDR"

# [设置] 时间限制 (秒)
# 6600秒 = 1小时50分
TIME_LIMIT=36000

# ===========================================

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

echo "========================================================"
echo "🚀 启动 ESDG 训练任务 (文件状态检测模式)"
echo "GPU 数量: $NUM_GPUS"
echo "算法: $ALGORITHM"
echo "输出目录: $OUTPUT_DIR"
echo "时间限制: $TIME_LIMIT 秒"
echo "========================================================"

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29505 \
    main.py \
    --root $ROOT_DATA \
    --algorithm $ALGORITHM \
    --dg_mode ESDG \
    --source-domains $SOURCE \
    --target-domains $TARGETS \
    --output $OUTPUT_DIR \
    --batch-size 32 \
    --lr 0.004 \
    --workers 8 \
    --epochs 100 \
    --time-limit $TIME_LIMIT