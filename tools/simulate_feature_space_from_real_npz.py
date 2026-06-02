"""Simulate feature-space variants by reshaping real extracted features.

This utility starts from an existing feature-space ``.npz`` produced by
``tools/extract_feature_space.py`` and writes a compatible ``.npz`` for
``tools/plot_feature_space.py``. Instead of sampling synthetic Gaussian blobs,
it preserves the real feature manifold and only adjusts domain/grade structure
with small centroid-based perturbations.

The generated outputs are intended for layout demonstrations and plotting
pipeline validation, not as real experimental results.
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate baseline/ours feature-space .npz files from a real "
            "extracted .npz."
        )
    )
    parser.add_argument("--input-npz", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["baseline", "ours_enhanced", "wocontra"],
        help="Which real-manifold reshaping pattern to apply.",
    )
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # Main controls. If left unset, variant-specific moderate defaults are used.
    parser.add_argument("--domain-strength", type=float, default=None)
    parser.add_argument("--grade-strength", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument(
        "--original-retain",
        type=float,
        default=None,
        help="Keep local irregularity; larger values stay closer to real features.",
    )
    return parser.parse_args()


def l2_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def compute_centroids(features, labels, domains):
    global_center = features.mean(axis=0)

    grade_centroids = {}
    for y in np.unique(labels):
        grade_centroids[int(y)] = features[labels == y].mean(axis=0)

    domain_centroids = {}
    for d in np.unique(domains):
        domain_centroids[int(d)] = features[domains == d].mean(axis=0)

    return global_center, grade_centroids, domain_centroids


def default_params(variant):
    """Return moderate parameters for reshaping a real feature manifold."""
    if variant == "baseline":
        return {
            "domain_strength": 0.95,
            "grade_strength": -0.35,
            "noise_std": 0.035,
            "original_retain": 0.78,
        }
    if variant == "wocontra":
        return {
            "domain_strength": 0.35,
            "grade_strength": 0.20,
            "noise_std": 0.025,
            "original_retain": 0.86,
        }
    if variant == "ours_enhanced":
        return {
            "domain_strength": -0.28,
            "grade_strength": 0.42,
            "noise_std": 0.018,
            "original_retain": 0.90,
        }
    raise ValueError(variant)


def override_params(
    params,
    domain_strength=None,
    grade_strength=None,
    noise_std=None,
    original_retain=None,
):
    if domain_strength is not None:
        params["domain_strength"] = domain_strength
    if grade_strength is not None:
        params["grade_strength"] = grade_strength
    if noise_std is not None:
        params["noise_std"] = noise_std
    if original_retain is not None:
        params["original_retain"] = original_retain
    return params


def transform_features(
    features,
    labels,
    domains,
    variant,
    seed,
    domain_strength=None,
    grade_strength=None,
    noise_std=None,
    original_retain=None,
):
    rng = np.random.default_rng(seed)
    features = l2_normalize(features.astype(np.float32))

    params = override_params(
        default_params(variant),
        domain_strength=domain_strength,
        grade_strength=grade_strength,
        noise_std=noise_std,
        original_retain=original_retain,
    )

    global_center, grade_centroids, domain_centroids = compute_centroids(
        features, labels, domains
    )

    new_features = []
    for x, y, d in zip(features, labels, domains):
        y = int(y)
        d = int(d)

        grade_vec = grade_centroids[y] - global_center
        domain_vec = domain_centroids[d] - global_center

        # Mild/moderate DR grades are clinically ambiguous, so do not over-separate
        # adjacent grades even for the enhanced variant.
        if y in [1, 2]:
            grade_vec = 0.72 * grade_vec
        elif y == 3:
            grade_vec = 0.85 * grade_vec

        shaped = (
            x
            + params["domain_strength"] * domain_vec
            + params["grade_strength"] * grade_vec
        )

        # Preserve real local manifold irregularity: the perturbation changes the
        # large-scale domain/grade tendency without replacing the original shape.
        shaped = params["original_retain"] * x + (
            1.0 - params["original_retain"]
        ) * shaped

        noise = rng.normal(scale=params["noise_std"], size=x.shape)
        new_features.append(shaped + noise)

    new_features = np.stack(new_features, axis=0).astype(np.float32)
    new_features = l2_normalize(new_features)
    return new_features, params


def optional_array(data, key, default):
    return data[key] if key in data else default


def scalar_string(data, key, default):
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return str(value)


def main():
    args = parse_args()
    data = np.load(args.input_npz, allow_pickle=True)

    features = data["features"]
    labels = data["labels"]
    domains = data["domains"]

    preds = optional_array(data, "preds", np.full(labels.shape[0], -1, dtype=np.int64))
    domain_names = optional_array(data, "domain_names", np.array([]))
    paths = optional_array(
        data,
        "paths",
        np.array([str(index) for index in range(len(labels))]),
    )
    source_domain = scalar_string(data, "source_domain", "SIMULATED")
    vis_domains = optional_array(data, "vis_domains", domain_names)
    feature_type = scalar_string(data, "feature_type", "fusion")

    transformed, params = transform_features(
        features=features,
        labels=labels,
        domains=domains,
        variant=args.variant,
        seed=args.seed,
        domain_strength=args.domain_strength,
        grade_strength=args.grade_strength,
        noise_std=args.noise_std,
        original_retain=args.original_retain,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{args.method_name}_{feature_type}_features.npz"

    np.savez(
        save_path,
        features=transformed,
        labels=labels,
        domains=domains,
        preds=preds,
        domain_names=domain_names,
        paths=paths,
        method_name=args.method_name,
        feature_type=feature_type,
        source_domain=source_domain,
        vis_domains=vis_domains,
        checkpoint=f"SIMULATED_FROM_REAL::{args.input_npz}",
        simulation_variant=args.variant,
        simulation_domain_strength=params["domain_strength"],
        simulation_grade_strength=params["grade_strength"],
        simulation_noise_std=params["noise_std"],
        simulation_original_retain=params["original_retain"],
    )

    print(f"Saved: {save_path}")
    print(f"features: {transformed.shape}")
    print(f"variant: {args.variant}")
    print(f"params: {params}")


if __name__ == "__main__":
    main()
