#!/bin/bash

ROOT_DATA="/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/GDR_Formatted_Data"

OUTPUT_DIR=${OUTPUT_DIR:-"./output_esdg_h100"}

ALGORITHM="GDRNet"

SOURCE="MESSIDOR"
TARGETS="APTOS DDR DEEPDR EYEPACS FGADR IDRID RLDR"

TIME_LIMIT=86100

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
    --config_file configs/datasets/GDRBench_FPT.yaml \
    --root $ROOT_DATA \
    --algorithm $ALGORITHM \
    --dg_mode ESDG \
    --source-domains $SOURCE \
    --target-domains $TARGETS \
    --output $OUTPUT_DIR \
    --batch-size 32 \
    --lr 0.0005 \
    --weight-decay 0.05 \
    --workers 8 \
    --epochs 100 \
    --time-limit $TIME_LIMIT