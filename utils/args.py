import argparse
import os
from configs.defaults import _C as cfg_default


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default=None,
                        help='path to specific config file (e.g., configs/datasets/GDRBench_FPT.yaml)')

    parser.add_argument('--root', type=str, default='/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/GDR_Formatted_Data',
                        help='dataset root path')
    parser.add_argument('--algorithm', type=str, default='GDRNet',
                        choices=['GDRNet', 'ERM', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet', 'Fishr', 'DRGen'],
                        help='algorithm name')

    parser.add_argument('--dg_mode', type=str, default='ESDG', choices=['DG', 'ESDG'], help='DG or ESDG setting')

    parser.add_argument('--source-domains', nargs='+', type=str, default=['MESSIDOR'], help='source domains')
    parser.add_argument('--target-domains', nargs='+', type=str,
                        default=['APTOS', 'DDR', 'DEEPDR', 'FGADR', 'IDRID', 'RLDR'], help='target domains')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size per gpu')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--output', type=str, default='./output', help='output directory')
    parser.add_argument('--save-freq', type=int, default=5, help='save frequency')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')

    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    parser.add_argument('--time-limit', type=int, default=0, help='time limit in seconds for auto-resume')
    parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint to resume')

    args = parser.parse_args()
    return args


def setup_cfg(args):
    cfg = cfg_default.clone()

    if args.config_file is not None:
        yaml_file = args.config_file
        print(f"[Config] Loading specified config file: {yaml_file}")
    else:
        if args.dg_mode == 'DG':
            yaml_file = os.path.join('configs', 'datasets', 'GDRBench.yaml')
        else:
            yaml_file = os.path.join('configs', 'datasets', 'GDRBench_ESDG.yaml')
        print(f"[Config] Loading default config file based on mode: {yaml_file}")

    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Config file not found at: {yaml_file}")

    cfg.merge_from_file(yaml_file)

    cfg.defrost()

    cfg.DATASET.ROOT = args.root
    cfg.ALGORITHM = args.algorithm
    cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    cfg.DATASET.TARGET_DOMAINS = args.target_domains

    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.MOMENTUM = args.momentum

    cfg.OUTPUT_PATH = f"{args.algorithm}_{args.dg_mode}_{'_'.join(args.source_domains)}"

    cfg.freeze()

    return cfg