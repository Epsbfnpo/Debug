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
    parser.add_argument(
        "--source-domains",
        nargs="+",
        default=None,
        help="Explicitly specify multiple source domains, e.g. APTOS DEEPDR FGADR IDRID MESSIDOR RLDR.",
    )
    parser.add_argument(
        "--target-domains",
        nargs="+",
        default=None,
        help="Explicitly specify one or more target domains, e.g. DDR or EYEPACS.",
    )
    parser.add_argument(
        "--profile-compute",
        action="store_true",
        help="Profile computation cost once and save compute_cost.json.",
    )
    args = parser.parse_args()
    return args

def setup_cfg(args):
    cfg = cfg_default.clone()
    cfg.defrost()
    cfg.PROFILE_COMPUTE = args.profile_compute

    if args.output is not None:
        cfg.OUT_DIR = args.output

    all_domains = ["APTOS", "DDR", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR", "EYEPACS"]

    if args.source_domain is not None and args.target_domain is not None:
        raise ValueError("Please specify only one of --source-domain or --target-domain, not both.")

    explicit_multi_source = args.source_domains is not None or args.target_domains is not None

    if explicit_multi_source:
        if args.source_domain is not None or args.target_domain is not None:
            raise ValueError(
                "Please do not mix --source-domain/--target-domain with "
                "--source-domains/--target-domains."
            )

        if args.source_domains is None or args.target_domains is None:
            raise ValueError(
                "When using explicit multi-source mode, both --source-domains "
                "and --target-domains must be provided."
            )

    output_tag = None

    if explicit_multi_source:
        source_domains = list(args.source_domains)
        target_domains = list(args.target_domains)

        for d in source_domains:
            if d not in all_domains:
                raise ValueError(f"Source domain {d} not found in {all_domains}")

        for d in target_domains:
            if d not in all_domains:
                raise ValueError(f"Target domain {d} not found in {all_domains}")

        overlap = set(source_domains) & set(target_domains)
        if len(overlap) > 0:
            raise ValueError(
                f"Source and target domains must not overlap. Overlap: {sorted(overlap)}"
            )

        cfg.DATASET.SOURCE_DOMAINS = source_domains
        cfg.DATASET.TARGET_DOMAINS = target_domains

        # This is a Multi-Source Single-Target DG setting when one target is given.
        # Use MSST to keep the existing naming convention.
        cfg.DG_MODE = "MSST"

        if len(target_domains) == 1:
            output_tag = f"target_{target_domains[0]}"
        else:
            output_tag = "targets_" + "_".join(target_domains)

        cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, output_tag)

        print(f"================ [Auto Config: Explicit MSST/DG] ================")
        print(f"Sources: {cfg.DATASET.SOURCE_DOMAINS}")
        print(f"Targets: {cfg.DATASET.TARGET_DOMAINS}")
        print(f"Output Dir: {cfg.OUT_DIR}")
        print(f"===============================================================")

    elif args.target_domain is not None:
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
