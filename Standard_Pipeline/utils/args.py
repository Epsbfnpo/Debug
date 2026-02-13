import argparse
import os
from configs.defaults import _C as cfg_default


def get_args():
    parser = argparse.ArgumentParser()

    # 分布式训练必须参数
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    # 自动续训时间限制
    parser.add_argument('--time-limit', type=int, default=0, help='time limit in seconds')

    # [新增回] 基础输出目录 (用于覆盖 defaults.py 中的默认路径)
    # run_esdg.sh 会传入这个参数，作为实验的根目录
    parser.add_argument('--output', type=str, default=None, help='base output directory')

    # 指定本次运行的源域
    parser.add_argument('--source-domain', type=str, default=None,
                        help='Specify the single source domain (e.g., APTOS). Others will be targets.')

    args = parser.parse_args()
    return args


def setup_cfg(args):
    """
    配置加载逻辑：
    1. 加载 defaults.py
    2. 如果命令行有 --output，覆盖默认的 OUT_DIR (作为 Base Dir)
    3. 如果指定了 --source-domain，自动计算目标域，并在 Base Dir 下创建子目录
    """
    cfg = cfg_default.clone()
    cfg.defrost()

    # 1. [修复点] 允许命令行覆盖基础路径
    if args.output is not None:
        cfg.OUT_DIR = args.output

    # 2. 自动化核心逻辑
    if args.source_domain is not None:
        # 定义所有可用域 (包含 DDR 和 EYEPACS)
        ALL_DOMAINS = ["APTOS", "DDR", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR", "EYEPACS"]

        current_source = args.source_domain

        # 检查输入合法性
        if current_source not in ALL_DOMAINS:
            raise ValueError(f"Source domain {current_source} not found in {ALL_DOMAINS}")

        # 设置源域
        cfg.DATASET.SOURCE_DOMAINS = [current_source]

        # 自动计算目标域 (所有域 - 当前源域)
        cfg.DATASET.TARGET_DOMAINS = [d for d in ALL_DOMAINS if d != current_source]

        # [关键] 修改输出目录到子文件夹
        # 逻辑：最终路径 = Base_Dir / Source_Name
        # 例如: ./output_esdg_h100 / APTOS
        cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, current_source)

        print(f"================ [Auto Config] ================")
        print(f"Source: {cfg.DATASET.SOURCE_DOMAINS}")
        print(f"Targets: {cfg.DATASET.TARGET_DOMAINS}")
        print(f"Output Dir: {cfg.OUT_DIR}")
        print(f"===============================================")

    # 生成最终实验 ID 路径 (OUT_DIR / Exp_Name)
    sources_str = '_'.join(cfg.DATASET.SOURCE_DOMAINS)
    cfg.OUTPUT_PATH = f"{cfg.ALGORITHM}_{cfg.DG_MODE}_{sources_str}"

    cfg.freeze()

    return cfg