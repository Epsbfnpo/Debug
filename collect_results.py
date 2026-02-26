import os
import re
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="汇总 ESDG 实验结果")
    parser.add_argument('--base_dir', type=str, required=True, help='实验输出的基础目录')
    parser.add_argument('--domains', type=str, nargs='+', required=True, help='参与实验的源域列表')
    args = parser.parse_args()
    base_dir = args.base_dir
    sources = args.domains
    results = {'auc': [], 'acc': [], 'f1': []}
    count = 0
    print(f"\n{'-' * 70}")
    print(f"{'Domain':<16} | {'Test AUC':<10} | {'Test Acc':<10} | {'Test F1':<10}")
    print(f"{'-' * 70}")
    for source in sources:
        domain_dir = os.path.join(base_dir, source)
        if not os.path.exists(domain_dir):
            print(f'{source:<16} | Missing Directory')
            continue
        subdirs = [os.path.join(domain_dir, d) for d in os.listdir(domain_dir) if os.path.isdir(os.path.join(domain_dir, d))]
        if not subdirs:
            print(f'{source:<16} | No Experiment Subdirs')
            continue
        latest_dir = max(subdirs, key=os.path.getmtime)
        possible_files = [('done', ''), ('done_cnn', '(CNN)'), ('done_vit', '(ViT)')]
        found_any = False
        for filename, suffix in possible_files:
            done_file = os.path.join(latest_dir, filename)
            if os.path.exists(done_file):
                found_any = True
                with open(done_file, 'r') as f:
                    content = f.read().strip()
                    match_auc = re.search(r'test_auc=([0-9.]+)', content)
                    match_acc = re.search(r'test_acc=([0-9.]+)', content)
                    match_f1 = re.search(r'test_f1=([0-9.]+)', content)
                    if match_auc and match_acc and match_f1:
                        auc = float(match_auc.group(1))
                        acc = float(match_acc.group(1))
                        f1 = float(match_f1.group(1))
                        results['auc'].append(auc)
                        results['acc'].append(acc)
                        results['f1'].append(f1)
                        count += 1
                        display_name = f"{source} {suffix}".strip()
                        print(f'{display_name:<16} | {auc:.4f}     | {acc:.4f}     | {f1:.4f}')
                    else:
                        print(f'{source:<16} | Error (Parse Failed in {filename})')
        if not found_any:
            print(f'{source:<16} | Not Finished (No done files)')
    print(f"{'-' * 70}")
    if count > 0:
        avg_auc = sum(results['auc']) / count
        avg_acc = sum(results['acc']) / count
        avg_f1 = sum(results['f1']) / count
        print(f"{'AVERAGE':<16} | {avg_auc:.4f}     | {avg_acc:.4f}     | {avg_f1:.4f}")
        print(f"{'-' * 70}")
        summary_path = os.path.join(base_dir, 'final_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('Domain,Test_AUC,Test_Acc,Test_F1\n')
            f.write(f'Average,{avg_auc:.4f},{avg_acc:.4f},{avg_f1:.4f}\n')
        print(f'\n📝 汇总结果已保存至: {summary_path}')
    else:
        print('❌ 没有找到有效的测试结果。')

if __name__ == "__main__":
    main()