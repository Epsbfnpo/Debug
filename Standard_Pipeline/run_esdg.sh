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
echo "📊 最终结果汇总 (Calculating Average Metrics)"
echo "########################################################"
python3 -c "
import os
import re
base_dir = '$BASE_OUTPUT_DIR'
sources = '${DOMAINS[*]}'.split()
results = {'auc': [], 'acc': [], 'f1': []}
count = 0
print('{:<12} | {:<10} | {:<10} | {:<10}'.format('Domain', 'Test AUC', 'Test Acc', 'Test F1'))
print('-' * 52)
for source in sources:
    domain_dir = os.path.join(base_dir, source)
    if os.path.exists(domain_dir):
        subdirs = [os.path.join(domain_dir, d) for d in os.listdir(domain_dir) if os.path.isdir(os.path.join(domain_dir, d))]
        if subdirs:
            latest_dir = max(subdirs, key=os.path.getmtime)
            done_file = os.path.join(latest_dir, 'done')
            if os.path.exists(done_file):
                with open(done_file, 'r') as f:
                    content = f.read().strip()
                    match_auc = re.search(r'test_auc=([0-9.]+)', content)
                    match_acc = re.search(r'test_acc=([0-9.]+)', content)
                    match_f1 = re.search(r'test_f1=([0-9.]+)', content)
                    if match_auc and match_acc and match_f1:
                        auc = float(match_auc.group(1))
                        acc = float(match_acc.group(1))
                        f1  = float(match_f1.group(1))
                        results['auc'].append(auc)
                        results['acc'].append(acc)
                        results['f1'].append(f1)
                        print('{:<12} | {:.4f}     | {:.4f}     | {:.4f}'.format(source, auc, acc, f1))
                        count += 1
                    else:
                        print('{:<12} | Error (Parse)'.format(source))
            else:
                print('{:<12} | Not Finished'.format(source))
        else:
            print('{:<12} | No Exp Dir'.format(source))
    else:
        print('{:<12} | Missing'.format(source))
print('-' * 52)
if count > 0:
    avg_auc = sum(results['auc']) / count
    avg_acc = sum(results['acc']) / count
    avg_f1  = sum(results['f1']) / count
    print('{:<12} | {:.4f}     | {:.4f}     | {:.4f}'.format('AVERAGE', avg_auc, avg_acc, avg_f1))
    with open(os.path.join(base_dir, 'final_summary.txt'), 'w') as f:
        f.write('Domain,Test_AUC,Test_Acc,Test_F1\n')
        f.write(f'Average,{avg_auc:.4f},{avg_acc:.4f},{avg_f1:.4f}\n')
    print(f'\n📝 汇总结果已保存至: {os.path.join(base_dir, 'final_summary.txt')}')
else:
    print('❌ 没有找到有效的测试结果。')
"
echo "========================================================"
echo "🎉 所有任务执行完毕"
echo "========================================================"