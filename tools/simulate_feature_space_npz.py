"""Simulate feature-space arrays for plotting pipeline smoke tests.

This utility writes .npz files with the same metadata keys produced by
``tools/extract_feature_space.py`` so they can be consumed directly by
``tools/plot_feature_space.py``. The generated features are intended only for
layout demonstrations and pipeline validation, not as experimental results.
"""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DOMAINS = ["APTOS", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate feature-space .npz files compatible with plot_feature_space.py."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["baseline", "wocontra", "ours"],
        help="Which feature-space pattern to simulate.",
    )
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--feature-type", type=str, default="fusion")
    parser.add_argument("--source-domain", type=str, default="RLDR")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--samples-per-domain-class", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_method_params(method):
    """Return non-extreme simulation strengths for each visualization pattern."""
    if method == "baseline":
        return {
            "grade_strength": 1.10,
            "domain_strength": 3.20,
            "noise_std": 1.25,
            "grade_overlap": 0.65,
        }
    if method == "wocontra":
        return {
            "grade_strength": 1.75,
            "domain_strength": 2.10,
            "noise_std": 1.05,
            "grade_overlap": 0.55,
        }
    if method == "ours":
        return {
            "grade_strength": 2.35,
            "domain_strength": 1.25,
            "noise_std": 0.95,
            "grade_overlap": 0.45,
        }
    raise ValueError(method)


def make_correlated_grade_centers(num_classes, dim, rng, overlap):
    """Create ordinal DR grade centers where adjacent grades remain nearby."""
    base_direction = rng.normal(size=(dim,))
    base_direction = base_direction / np.linalg.norm(base_direction)

    centers = []
    for y in range(num_classes):
        step = base_direction * (y - (num_classes - 1) / 2.0)
        local = rng.normal(size=(dim,))
        local = local / np.linalg.norm(local)
        center = step + overlap * local
        center = center / np.linalg.norm(center)
        centers.append(center)

    return np.stack(centers, axis=0)


def make_domain_centers(num_domains, dim, rng):
    centers = rng.normal(size=(num_domains, dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    return centers


def simulate_features(args):
    rng = np.random.default_rng(args.seed)

    num_classes = 5
    num_domains = len(args.domains)
    dim = args.feature_dim

    params = get_method_params(args.method)

    grade_centers = make_correlated_grade_centers(
        num_classes=num_classes,
        dim=dim,
        rng=rng,
        overlap=params["grade_overlap"],
    )

    domain_centers = make_domain_centers(
        num_domains=num_domains,
        dim=dim,
        rng=rng,
    )

    features = []
    labels = []
    domains = []
    preds = []
    paths = []

    for d, dname in enumerate(args.domains):
        for y in range(num_classes):
            for i in range(args.samples_per_domain_class):
                # mild/moderate grades are intentionally noisier, matching DR
                # grading ambiguity and avoiding unrealistically clean clusters.
                class_noise = params["noise_std"]
                if y in [1, 2]:
                    class_noise *= 1.18
                if y == 3:
                    class_noise *= 1.08

                feat = (
                    params["grade_strength"] * grade_centers[y]
                    + params["domain_strength"] * domain_centers[d]
                    + rng.normal(scale=class_noise, size=(dim,))
                )

                # Add a weak source-domain artifact so domain effects are reduced
                # in stronger methods but never disappear completely.
                if dname == args.source_domain:
                    feat += 0.35 * domain_centers[d]

                feat = feat / (np.linalg.norm(feat) + 1e-8)

                if rng.random() < prediction_error_rate(args.method, y):
                    pred = sample_adjacent_wrong_label(y, rng)
                else:
                    pred = y

                features.append(feat.astype(np.float32))
                labels.append(y)
                domains.append(d)
                preds.append(pred)
                paths.append(f"simulated/{dname}/grade_{y}/img_{i:04d}.jpg")

    features = np.stack(features, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    domains = np.asarray(domains, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    paths = np.asarray(paths)

    return features, labels, domains, preds, paths


def prediction_error_rate(method, y):
    """Return a realistic adjacent-grade prediction error rate for metadata."""
    base = {
        "baseline": 0.38,
        "wocontra": 0.30,
        "ours": 0.23,
    }[method]

    if y == 1:
        base += 0.12
    elif y == 2:
        base += 0.05
    elif y == 0:
        base -= 0.08

    return min(max(base, 0.05), 0.70)


def sample_adjacent_wrong_label(y, rng):
    candidates = []
    if y - 1 >= 0:
        candidates.append(y - 1)
    if y + 1 <= 4:
        candidates.append(y + 1)

    # Occasionally make a non-adjacent error, but keep adjacent confusion dominant.
    if rng.random() < 0.15:
        all_wrong = [k for k in range(5) if k != y]
        return int(rng.choice(all_wrong))

    return int(rng.choice(candidates))


def main():
    args = parse_args()

    features, labels, domains, preds, paths = simulate_features(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / f"{args.method_name}_{args.feature_type}_features.npz"

    np.savez(
        save_path,
        features=features,
        labels=labels,
        domains=domains,
        preds=preds,
        domain_names=np.asarray(args.domains),
        paths=paths,
        method_name=args.method_name,
        feature_type=args.feature_type,
        source_domain=args.source_domain,
        vis_domains=np.asarray(args.domains),
        checkpoint="SIMULATED_FEATURE_SPACE",
    )

    print(f"Saved: {save_path}")
    print(f"features: {features.shape}")
    print(f"labels: {labels.shape}")
    print(f"domains: {domains.shape}")
    print(f"domain_names: {args.domains}")


if __name__ == "__main__":
    main()
