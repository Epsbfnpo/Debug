import argparse
import os
from configs.defaults import _C as cfg_default

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument('--time-limit', type=int, default=0, help='time limit in seconds')
    parser.add_argument('--output', type=str, default=None, help='base output directory')
    parser.add_argument('--source-domain', type=str, default=None, help='Specify the single source domain (e.g., APTOS). Others will be targets.')
    parser.add_argument('--target-domain', type=str, default=None, help='Specify the single target domain for MSST. All other domains will be sources.')
    args = parser.parse_args()
    return args

def setup_cfg(args):
    cfg = cfg_default.clone()
    cfg.defrost()

    if args.output is not None:
        cfg.OUT_DIR = args.output

    all_domains = ["APTOS", "DDR", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR", "EYEPACS"]

    if args.source_domain is not None and args.target_domain is not None:
        raise ValueError("Please specify only one of --source-domain or --target-domain, not both.")

    output_tag = None

    if args.target_domain is not None:
        current_target = args.target_domain
        if current_target not in all_domains:
            raise ValueError(f"Target domain {current_target} not found in {all_domains}")

        cfg.DATASET.TARGET_DOMAINS = [current_target]
        cfg.DATASET.SOURCE_DOMAINS = [d for d in all_domains if d != current_target]
        cfg.DG_MODE = "MSST"

        cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, f"target_{current_target}")
        output_tag = f"target_{current_target}"

        print(f"================ [Auto Config: MSST] ================")
        print(f"Sources: {cfg.DATASET.SOURCE_DOMAINS}")
        print(f"Target: {cfg.DATASET.TARGET_DOMAINS}")
        print(f"Output Dir: {cfg.OUT_DIR}")
        print(f"=====================================================")

    elif args.source_domain is not None:
        current_source = args.source_domain
        if current_source not in all_domains:
            raise ValueError(f"Source domain {current_source} not found in {all_domains}")

        cfg.DATASET.SOURCE_DOMAINS = [current_source]
        cfg.DATASET.TARGET_DOMAINS = [d for d in all_domains if d != current_source]
        cfg.DG_MODE = "ESDG"

        cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, current_source)
        output_tag = current_source

        print(f"================ [Auto Config: SSMT] ================")
        print(f"Source: {cfg.DATASET.SOURCE_DOMAINS}")
        print(f"Targets: {cfg.DATASET.TARGET_DOMAINS}")
        print(f"Output Dir: {cfg.OUT_DIR}")
        print(f"=====================================================")

    if output_tag is None:
        output_tag = '_'.join(cfg.DATASET.SOURCE_DOMAINS)

    cfg.OUTPUT_PATH = f"{cfg.ALGORITHM}_{cfg.DG_MODE}_{output_tag}"
    cfg.freeze()
    return cfg
