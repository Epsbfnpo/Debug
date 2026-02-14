import os
import subprocess
import time
import re

SOURCE_DOMAINS_TO_RUN = ["APTOS", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR"]
NUM_GPUS = 4
TIME_LIMIT = 36000
BASE_OUT_DIR = "/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/Standard_Pipeline/output_esdg_h10"

def get_result_from_file(log_dir):
    done_file = os.path.join(log_dir, 'done')
    if not os.path.exists(done_file):
        return None
    with open(done_file, 'r') as f:
        content = f.read().strip()
        match = re.search(r'test=([0-9.]+)', content)
        if match:
            return float(match.group(1))
    return None

def main():
    results = {}
    print(f"🚀 开始批量实验: {SOURCE_DOMAINS_TO_RUN}")
    print(f"GPUs: {NUM_GPUS}")
    for source in SOURCE_DOMAINS_TO_RUN:
        print(f"\n\n{'=' * 60}")
        print(f"▶️  正在运行源域: {source}")
        print(f"{'=' * 60}")
        cmd = ["torchrun", f"--nproc_per_node={NUM_GPUS}", "--master_port=29505", "main.py", "--time-limit", str(TIME_LIMIT), "--source-domain", source]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ 错误: 源域 {source} 训练失败！停止批量实验。")
            exit(1)
        domain_out_dir = os.path.join(BASE_OUT_DIR, source)
        if os.path.exists(domain_out_dir):
            subdirs = [os.path.join(domain_out_dir, d) for d in os.listdir(domain_out_dir) if os.path.isdir(os.path.join(domain_out_dir, d))]
            if subdirs:
                latest_dir = max(subdirs, key=os.path.getmtime)
                res = get_result_from_file(latest_dir)
                if res is not None:
                    results[source] = res
                    print(f"✅ {source} 完成. Test AUC: {res:.4f}")
                else:
                    print(f"⚠️ {source} 完成，但无法读取结果 (done文件不存在或格式错误)")
            else:
                print(f"⚠️ 找不到实验输出目录: {domain_out_dir}")
        else:
            print(f"⚠️ 找不到实验输出目录: {domain_out_dir}")
    print(f"\n\n{'#' * 60}")
    print(f"📊 最终结果汇总 (Test AUC)")
    print(f"{'#' * 60}")
    total_auc = 0.0
    count = 0
    for source in SOURCE_DOMAINS_TO_RUN:
        res = results.get(source, 0.0)
        print(f"{source:<15}: {res:.4f}")
        if res > 0:
            total_auc += res
            count += 1
    if count > 0:
        avg = total_auc / count
        print(f"{'-' * 30}")
        print(f"Average AUC    : {avg:.4f}")
        print(f"{'-' * 30}")
        with open(os.path.join(BASE_OUT_DIR, 'final_summary_metrics.txt'), 'w') as f:
            f.write("Domain,Test_AUC\n")
            for source in SOURCE_DOMAINS_TO_RUN:
                f.write(f"{source},{results.get(source, 0.0):.4f}\n")
            f.write(f"Average,{avg:.4f}\n")
    else:
        print("❌ 没有成功获取到任何结果。")

if __name__ == "__main__":
    main()