import argparse
import os
# 关键修正：从 defaults 导入默认配置对象 _C
from configs.defaults import _C as cfg_default


def get_args():
    parser = argparse.ArgumentParser()

    # Basic Settings
    parser.add_argument('--root', type=str, default='/datasets/work/hb-nhmrc-dhcp/work/liu275/GDR_Formatted_Data',
                        help='dataset root path')
    parser.add_argument('--algorithm', type=str, default='GDRNet',
                        choices=['GDRNet', 'ERM', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet', 'Fishr', 'DRGen'],
                        help='algorithm name')

    # 模式: ESDG (单源域) 或 DG
    parser.add_argument('--dg_mode', type=str, default='ESDG', choices=['DG', 'ESDG'], help='DG or ESDG setting')

    # Domains
    parser.add_argument('--source-domains', nargs='+', type=str, default=['MESSIDOR'], help='source domains')
    parser.add_argument('--target-domains', nargs='+', type=str,
                        default=['APTOS', 'DDR', 'DEEPDR', 'FGADR', 'IDRID', 'RLDR'], help='target domains')

    # Training Config
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size per gpu')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # Output
    parser.add_argument('--output', type=str, default='./output', help='output directory')
    parser.add_argument('--save-freq', type=int, default=5, help='save frequency')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')

    # Distributed Training (DDP 必需参数)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    # --- [新增] 自动续训相关参数 ---
    # 0 表示不限制。例如 2小时 = 7200秒，建议设为 6900秒 (预留缓冲)
    parser.add_argument('--time-limit', type=int, default=0, help='time limit in seconds for auto-resume')
    # 指定从哪个 checkpoint 恢复，默认 None 表示自动寻找 latest_model.pth
    parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint to resume')

    args = parser.parse_args()
    return args


def setup_cfg(args):
    # 1. 克隆默认配置
    cfg = cfg_default.clone()

    # 2. 确定要加载的 yaml 文件路径
    # 注意：这里我们手动构建路径，而不是尝试 import 它
    if args.dg_mode == 'DG':
        yaml_file = os.path.join('configs', 'datasets', 'GDRBench.yaml')
    else:
        yaml_file = os.path.join('configs', 'datasets', 'GDRBench_ESDG.yaml')

    # 3. 检查文件是否存在并加载
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Config file not found at: {yaml_file}")

    cfg.merge_from_file(yaml_file)

    # 4. 使用命令行参数覆盖配置 (defrost 解锁 -> 修改 -> freeze 锁定)
    cfg.defrost()

    cfg.DATASET.ROOT = args.root
    cfg.ALGORITHM = args.algorithm
    cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    cfg.DATASET.TARGET_DOMAINS = args.target_domains

    # 训练超参数覆盖
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.MOMENTUM = args.momentum

    # 设置输出路径名
    cfg.OUTPUT_PATH = f"{args.algorithm}_{args.dg_mode}_{'_'.join(args.source_domains)}"

    cfg.freeze()

    return cfg