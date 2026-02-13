import os
import subprocess
import time
import re

# ================= 配置区域 =================
# 想要轮流作为源域的列表 (注意：DDR 和 EYEPACS 不在这里，因为它们只做 Target)
SOURCE_DOMAINS_TO_RUN = ["APTOS", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR"]

# GPU 设置
NUM_GPUS = 4
TIME_LIMIT = 36000  # 10小时

# 基础输出目录 (必须与 defaults.py 中的 _C.OUT_DIR 保持一致)
BASE_OUT_DIR = "./output_esdg_h100"


# ===========================================

def get_result_from_file(log_dir):
    """从 done 文件中读取 test result"""
    done_file = os.path.join(log_dir, 'done')
    if not os.path.exists(done_file):
        return None

    with open(done_file, 'r') as f:
        content = f.read().strip()
        # content 格式: "done, best_val=0.xxxx, test=0.xxxx"
        # 使用正则提取 test= 后面的数字
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

        # 1. 构建命令
        # 我们调用 main.py 并传入 --source-domain 参数
        cmd = [
            "torchrun",
            f"--nproc_per_node={NUM_GPUS}",
            "--master_port=29505",
            "main.py",
            "--time-limit", str(TIME_LIMIT),
            "--source-domain", source
        ]

        # 2. 执行命令
        # check=True 会在命令失败(报错)时抛出异常停止脚本，防止错误蔓延
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ 错误: 源域 {source} 训练失败！停止批量实验。")
            exit(1)

        # 3. 获取结果
        # 路径规则: BASE_OUT_DIR / SOURCE / ALGO_MODE_SOURCE
        # 需要构建出 defaults.py 和 args.py 生成的那个路径
        # 假设 defaults.py 里 ALGORITHM="GDRNet", DG_MODE="ESDG"
        # 你可能需要根据实际情况调整这里，或者让 Python 自动去读
        # 最稳妥的方式是去 output/source 文件夹下找最新的文件夹

        domain_out_dir = os.path.join(BASE_OUT_DIR, source)
        # 找到里面唯一的实验文件夹 (或者最新的)
        if os.path.exists(domain_out_dir):
            subdirs = [os.path.join(domain_out_dir, d) for d in os.listdir(domain_out_dir) if
                       os.path.isdir(os.path.join(domain_out_dir, d))]
            if subdirs:
                # 按修改时间排序，取最新的
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

    # ================= 汇总报告 =================
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

        # 保存汇总结果到文件
        with open(os.path.join(BASE_OUT_DIR, 'final_summary_metrics.txt'), 'w') as f:
            f.write("Domain,Test_AUC\n")
            for source in SOURCE_DOMAINS_TO_RUN:
                f.write(f"{source},{results.get(source, 0.0):.4f}\n")
            f.write(f"Average,{avg:.4f}\n")

    else:
        print("❌ 没有成功获取到任何结果。")


if __name__ == "__main__":
    main()