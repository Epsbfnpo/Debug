#!/bin/bash

# ================= 配置区域 =================
# 基础输出目录 (对应 defaults.py 中的 _C.OUT_DIR)
# 注意：args.py 会自动在此目录下创建子文件夹 (e.g., ./output_esdg_h100/APTOS)
BASE_OUTPUT_DIR="./output_esdg_h100"

# 需要轮流作为源域的列表
# 注意：DDR 和 EYEPACS 被排除在源域之外，只作目标域
DOMAINS=("APTOS" "DEEPDR" "FGADR" "IDRID" "MESSIDOR" "RLDR")

# 硬件设置
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
# 单个实验的时间限制 (秒)
TIME_LIMIT=36000

# ===========================================

echo "========================================================"
echo "🚀 启动 ESDG 批量实验 (Bash 循环模式)"
echo "GPU 数量: $NUM_GPUS"
echo "待运行源域: ${DOMAINS[*]}"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo "========================================================"

# 1. 循环运行实验
for SOURCE in "${DOMAINS[@]}"
do
    echo ""
    echo "----------------------------------------------------------------"
    echo "▶️  [进度] 正在启动源域: $SOURCE"
    echo "----------------------------------------------------------------"

    # 调用 main.py，传入 --source-domain
    # 注意：这里依赖我们在 args.py 中添加的 --source-domain 参数处理逻辑
    torchrun --nproc_per_node=$NUM_GPUS \
        --master_port=29505 \
        main.py \
        --time-limit $TIME_LIMIT \
        --source-domain $SOURCE \
        --output $BASE_OUTPUT_DIR

    # 检查上一条命令的退出状态
    if [ $? -ne 0 ]; then
        echo "❌ [错误] 源域 $SOURCE 训练失败！"
        # 即使失败也继续跑下一个？还是退出？这里选择继续，以免前功尽弃
    else
        echo "✅ [完成] 源域 $SOURCE 训练结束。"
    fi

    # 等待几秒，确保文件写入完成
    sleep 5
done

# 2. 实验结束，计算平均分 (嵌入式 Python 脚本)
echo ""
echo "########################################################"
echo "📊 最终结果汇总 (Calculating Average Metrics)"
echo "########################################################"

python3 -c "
import os
import re

base_dir = '$BASE_OUTPUT_DIR'
sources = '${DOMAINS[*]}'.split()
results = {}
total_auc = 0.0
count = 0

print(f'{'Domain':<15} | {'Test AUC':<10}')
print('-' * 30)

for source in sources:
    # 寻找 output/SOURCE 目录
    domain_dir = os.path.join(base_dir, source)
    if os.path.exists(domain_dir):
        # 找到该目录下最新的实验文件夹
        subdirs = [os.path.join(domain_dir, d) for d in os.listdir(domain_dir) if os.path.isdir(os.path.join(domain_dir, d))]
        if subdirs:
            latest_dir = max(subdirs, key=os.path.getmtime)
            done_file = os.path.join(latest_dir, 'done')

            if os.path.exists(done_file):
                with open(done_file, 'r') as f:
                    content = f.read().strip()
                    # 提取 test=0.xxxx
                    match = re.search(r'test=([0-9.]+)', content)
                    if match:
                        auc = float(match.group(1))
                        results[source] = auc
                        print(f'{source:<15} | {auc:.4f}')
                        total_auc += auc
                        count += 1
                    else:
                        print(f'{source:<15} | Error (Parse)')
            else:
                print(f'{source:<15} | Not Finished')
        else:
            print(f'{source:<15} | No Exp Dir')
    else:
        print(f'{source:<15} | Missing')

print('-' * 30)
if count > 0:
    avg = total_auc / count
    print(f'{'AVERAGE':<15} | {avg:.4f}')

    # 保存汇总文件
    with open(os.path.join(base_dir, 'final_summary.txt'), 'w') as f:
        f.write('Domain,Test_AUC\n')
        for s, v in results.items():
            f.write(f'{s},{v:.4f}\n')
        f.write(f'Average,{avg:.4f}\n')
    print(f'\n📝 汇总结果已保存至: {os.path.join(base_dir, 'final_summary.txt')}')
else:
    print('❌ 没有找到有效的测试结果。')
"

echo "========================================================"
echo "🎉 所有任务执行完毕"
echo "========================================================"