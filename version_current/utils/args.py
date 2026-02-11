import argparse
import os
import sys
from configs.defaults import _C as cfg_default


def get_args():
    parser = argparse.ArgumentParser(description="GDRNet ESDG Training")

    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for DDP')

    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def setup_cfg(args):
    cfg = cfg_default.clone()

    if args.opts:
        print(f">> [Config] Overriding configs with command line opts: {args.opts}")
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    if hasattr(cfg, 'VERBOSE') and cfg.VERBOSE:
        print("=========================================")
        print("🚀 [Final Configuration] (Loaded from defaults.py)")
        print(cfg)
        print("=========================================")

    return cfg