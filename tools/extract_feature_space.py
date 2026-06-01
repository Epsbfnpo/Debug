"""Extract image-level features for domain/grade feature-space visualization.

This utility loads a trained algorithm checkpoint, evaluates the held-out target
GDRBench domains, balances samples by (domain, grade), and stores a single .npz
file that can be consumed by tools/plot_feature_space.py.
"""

import argparse
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import algorithms
from configs.defaults import _C as cfg_default
from dataset.data_manager import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract balanced feature-space arrays from unseen target domains."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)

    parser.add_argument("--algorithm", type=str, default="CASS_GDRNet")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--source-domain", type=str, default="RLDR")
    parser.add_argument("--vis-domains", nargs="+", required=True)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--dinov3-path", type=str, default=None)
    parser.add_argument("--pretrained-path", type=str, default=None)

    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--feature-type",
        type=str,
        default="fusion",
        choices=["fusion", "proj_concat", "cnn", "vit", "logits"],
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-per-domain-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_cfg(args):
    cfg = cfg_default.clone()
    cfg.defrost()

    cfg.ALGORITHM = args.algorithm
    if args.backbone is not None:
        cfg.BACKBONE = args.backbone
    cfg.DATASET.SOURCE_DOMAINS = [args.source_domain]
    cfg.DATASET.TARGET_DOMAINS = list(args.vis_domains)
    cfg.DG_MODE = "ESDG_VIS"

    if args.dataset_root is not None:
        cfg.DATASET.ROOT = args.dataset_root
    if args.dinov3_path is not None:
        cfg.GDRNET.DINOV3_PATH = args.dinov3_path
    if args.pretrained_path is not None:
        cfg.MODEL.PRETRAINED_PATH = args.pretrained_path

    cfg.TRANSFORM.INPUT_SIZE = args.input_size
    cfg.TRANSFORM.CROP_SIZE = args.input_size
    cfg.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed
    cfg.DROP_LAST = False
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    cfg.freeze()
    return cfg


def _strip_state_prefixes(state):
    prefixes = ("module.", "network.")
    cleaned = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _pick_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in (
        "state_dict",
        "model_state_dict",
        "network",
        "network_state_dict",
        "model",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_checkpoint_flexible(
    algorithm, checkpoint_path, device, classifier_checkpoint=None
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = _pick_state_dict(checkpoint)
    state = _strip_state_prefixes(state) if isinstance(state, dict) else state

    try:
        algorithm.network.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint as strict network state_dict: {checkpoint_path}")
    except Exception as network_error:
        try:
            missing, unexpected = algorithm.network.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint as non-strict network state_dict: {checkpoint_path}")
            if missing:
                print(f"  Missing network keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected network keys: {len(unexpected)}")
        except Exception as algorithm_network_error:
            try:
                missing, unexpected = algorithm.load_state_dict(state, strict=False)
                print(f"Loaded checkpoint as algorithm state_dict: {checkpoint_path}")
                if missing:
                    print(f"  Missing algorithm keys: {len(missing)}")
                if unexpected:
                    print(f"  Unexpected algorithm keys: {len(unexpected)}")
            except Exception as algorithm_error:
                raise RuntimeError(
                    "Failed to load checkpoint as network or algorithm state.\n"
                    f"Strict network error: {network_error}\n"
                    f"Non-strict network error: {algorithm_network_error}\n"
                    f"Algorithm error: {algorithm_error}"
                ) from algorithm_error

    if classifier_checkpoint is not None:
        if not hasattr(algorithm, "classifier"):
            raise ValueError(
                "--classifier-checkpoint was provided, but algorithm has no classifier."
            )
        classifier_state = torch.load(classifier_checkpoint, map_location=device)
        classifier_state = _pick_state_dict(classifier_state)
        classifier_state = (
            _strip_state_prefixes(classifier_state)
            if isinstance(classifier_state, dict)
            else classifier_state
        )
        algorithm.classifier.load_state_dict(classifier_state, strict=True)
        print(f"Loaded classifier checkpoint: {classifier_checkpoint}")


def _forward_cass_like(algorithm, images):
    try:
        return algorithm.network(x_cnn=images, x_vit=images)
    except TypeError:
        return algorithm.network(images, images)


@torch.no_grad()
def extract_feature_from_batch(algorithm, images, feature_type):
    algorithm.eval()

    if algorithm.__class__.__name__ in {"ERM", "GDRNet"}:
        feature = algorithm.network(images)
        logits = algorithm.classifier(feature)
        return feature, logits

    result = _forward_cass_like(algorithm, images)
    if not isinstance(result, dict):
        raise TypeError("Expected the algorithm network to return a dict of features/logits.")

    if feature_type == "fusion":
        if "fusion_feat" in result:
            feature = result["fusion_feat"]
        else:
            feature = torch.cat([result["proj_cnn"], result["proj_vit"]], dim=1)
    elif feature_type == "proj_concat":
        feature = torch.cat([result["proj_cnn"], result["proj_vit"]], dim=1)
    elif feature_type == "cnn":
        feature = result["proj_cnn"]
    elif feature_type == "vit":
        feature = result["proj_vit"]
    elif feature_type == "logits":
        feature = result["logits_fusion"]
    else:
        raise ValueError(feature_type)

    logits = result.get("logits_fusion")
    return feature, logits


def balanced_subsample(features, labels, domains, preds, paths, max_per_domain_class, seed):
    if max_per_domain_class <= 0:
        return features, labels, domains, preds, paths

    rng = np.random.default_rng(seed)
    keep_indices = []

    for domain in np.unique(domains):
        for label in np.unique(labels):
            indices = np.where((domains == domain) & (labels == label))[0]
            if len(indices) == 0:
                continue
            if len(indices) > max_per_domain_class:
                indices = rng.choice(indices, size=max_per_domain_class, replace=False)
            keep_indices.extend(indices.tolist())

    keep_indices = np.array(sorted(keep_indices), dtype=np.int64)
    return (
        features[keep_indices],
        labels[keep_indices],
        domains[keep_indices],
        preds[keep_indices],
        paths[keep_indices],
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cfg = build_cfg(args)

    loader_args = SimpleNamespace(local_rank=-1)
    _, _, test_loader, dataset_size, _ = get_dataset(loader_args, cfg)
    print(f"Dataset sizes [train, val, test]: {dataset_size}")

    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg).to(device)
    load_checkpoint_flexible(algorithm, args.checkpoint, device, args.classifier_checkpoint)
    algorithm.eval()

    all_features = []
    all_labels = []
    all_domains = []
    all_preds = []
    all_paths = []
    dataset = test_loader.dataset

    for batch in test_loader:
        images, labels, domains, indices = batch
        images = images.to(device, non_blocking=True)
        labels = labels.long()
        domains = domains.long()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            features, logits = extract_feature_from_batch(algorithm, images, args.feature_type)

        features = F.normalize(features.float(), dim=1).cpu().numpy()
        if logits is None:
            preds = np.full(labels.shape[0], -1, dtype=np.int64)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()

        batch_paths = []
        for index in indices.tolist():
            if hasattr(dataset, "data"):
                batch_paths.append(dataset.data[index])
            else:
                batch_paths.append(str(index))

        all_features.append(features)
        all_labels.append(labels.numpy())
        all_domains.append(domains.numpy())
        all_preds.append(preds)
        all_paths.extend(batch_paths)

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    domains = np.concatenate(all_domains, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    paths = np.array(all_paths)

    features, labels, domains, preds, paths = balanced_subsample(
        features,
        labels,
        domains,
        preds,
        paths,
        max_per_domain_class=args.max_per_domain_class,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{args.method_name}_{args.feature_type}_features.npz"

    np.savez(
        save_path,
        features=features,
        labels=labels,
        domains=domains,
        preds=preds,
        domain_names=np.array(args.vis_domains),
        paths=paths,
        method_name=args.method_name,
        feature_type=args.feature_type,
        source_domain=args.source_domain,
        vis_domains=np.array(args.vis_domains),
        checkpoint=args.checkpoint,
    )

    print(f"Saved: {save_path}")
    print(f"features: {features.shape}")
    print(f"labels: {labels.shape}")
    print(f"domains: {domains.shape}")
    print(f"preds: {preds.shape}")
    print(f"domain_names: {args.vis_domains}")


if __name__ == "__main__":
    main()
