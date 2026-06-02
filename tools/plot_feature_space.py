"""Plot method-wise feature-space visualizations colored by domain and DR grade."""

import argparse
from importlib import util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


GRADE_NAMES = {
    0: "Healthy",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "PDR",
}

GRADE_MARKERS = {
    0: "x",
    1: "o",
    2: "s",
    3: "^",
    4: "D",
}
UNFILLED_MARKERS = {"x", "+", "1", "2", "3", "4"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create 2-row feature-space plots for multiple methods."
    )
    parser.add_argument("--feature-files", nargs="+", required=True)
    parser.add_argument("--method-names", nargs="+", required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--reducer", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.15)

    parser.add_argument(
        "--layout",
        type=str,
        default="separate",
        choices=["separate", "combined"],
        help=(
            "separate: two rows for domain/grade coloring; combined: "
            "color=domain and marker=grade in one panel."
        ),
    )
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    return parser.parse_args()


def _pca_preprocess(features, seed, pca_dim):
    x = StandardScaler().fit_transform(features)
    max_components = min(x.shape[0] - 1, x.shape[1], pca_dim)
    if max_components >= 2 and x.shape[1] > max_components:
        x = PCA(n_components=max_components, random_state=seed).fit_transform(x)
    return x


def _reduce_umap(x, seed, n_neighbors, min_dist, umap_module):
    safe_neighbors = max(2, min(n_neighbors, x.shape[0] - 1))
    return umap_module.UMAP(
        n_components=2,
        n_neighbors=safe_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(x)


def _reduce_tsne(x, seed, perplexity):
    safe_perplexity = max(1.0, min(perplexity, (x.shape[0] - 1) / 3))
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=safe_perplexity,
        random_state=seed,
    ).fit_transform(x)


def reduce_features(
    features,
    reducer="umap",
    seed=42,
    pca_dim=50,
    perplexity=30.0,
    umap_neighbors=30,
    umap_min_dist=0.15,
):
    x = _pca_preprocess(features, seed=seed, pca_dim=pca_dim)
    if x.shape[0] < 3:
        raise ValueError("At least three samples are required for UMAP/t-SNE visualization.")

    if reducer == "umap" and util.find_spec("umap") is not None:
        import umap

        try:
            return _reduce_umap(x, seed, umap_neighbors, umap_min_dist, umap)
        except Exception as error:
            print(f"UMAP reduction failed. Falling back to t-SNE. Reason: {error}")
    elif reducer == "umap":
        print("umap-learn is not installed. Falling back to t-SNE.")

    return _reduce_tsne(x, seed, perplexity)


def scatter_by_domain(ax, embedding, domains, domain_names, point_size, alpha):
    cmap = plt.get_cmap("tab10")
    for domain in np.unique(domains):
        mask = domains == domain
        name = (
            str(domain_names[int(domain)])
            if int(domain) < len(domain_names)
            else f"Domain {domain}"
        )
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            alpha=alpha,
            color=cmap(int(domain) % 10),
            label=name,
            linewidths=0,
        )


def scatter_by_grade(ax, embedding, labels, point_size, alpha):
    cmap = plt.get_cmap("tab10")
    for label in sorted(np.unique(labels)):
        mask = labels == label
        name = GRADE_NAMES.get(int(label), f"Grade {label}")
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            alpha=alpha,
            color=cmap(int(label) % 10),
            label=name,
            linewidths=0,
        )


def scatter_by_domain_and_grade(
    ax,
    embedding,
    domains,
    labels,
    domain_names,
    point_size,
    alpha,
):
    cmap = plt.get_cmap("tab10")
    unique_domains = sorted(np.unique(domains))
    unique_labels = sorted(np.unique(labels))

    for domain in unique_domains:
        domain_color = cmap(int(domain) % 10)
        for label in unique_labels:
            mask = (domains == domain) & (labels == label)
            if not np.any(mask):
                continue

            marker = GRADE_MARKERS.get(int(label), "o")
            scatter_kwargs = {
                "s": point_size,
                "alpha": alpha,
                "color": domain_color,
                "marker": marker,
            }
            if marker in UNFILLED_MARKERS:
                scatter_kwargs["linewidths"] = 0.6
            else:
                scatter_kwargs["linewidths"] = 0
                scatter_kwargs["edgecolors"] = "none"

            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                **scatter_kwargs,
            )


def make_domain_legend_handles(domain_names):
    cmap = plt.get_cmap("tab10")
    handles = []
    for index, name in enumerate(domain_names):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=cmap(index % 10),
                markeredgecolor=cmap(index % 10),
                markersize=7,
                label=str(name),
            )
        )
    return handles


def make_grade_legend_handles():
    handles = []
    for label, name in GRADE_NAMES.items():
        marker = GRADE_MARKERS.get(label, "o")
        markerfacecolor = "none" if marker in UNFILLED_MARKERS else "black"
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                color="black",
                markerfacecolor=markerfacecolor,
                markeredgecolor="black",
                markersize=7,
                label=name,
            )
        )
    return handles


def style_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def main():
    args = parse_args()
    if len(args.feature_files) != len(args.method_names):
        raise ValueError("The number of feature files must match method names.")

    n_methods = len(args.feature_files)
    if args.layout == "combined":
        fig, axes = plt.subplots(
            1,
            n_methods,
            figsize=(4.4 * n_methods, 4.2),
            squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            2,
            n_methods,
            figsize=(4.0 * n_methods, 7.0),
            squeeze=False,
        )

    for column, (file_path, method_name) in enumerate(
        zip(args.feature_files, args.method_names)
    ):
        data = np.load(file_path, allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        domains = data["domains"]
        domain_names = data["domain_names"]

        embedding = reduce_features(
            features,
            reducer=args.reducer,
            seed=args.seed,
            pca_dim=args.pca_dim,
            perplexity=args.perplexity,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
        )

        if args.layout == "combined":
            ax = axes[0, column]
            scatter_by_domain_and_grade(
                ax,
                embedding,
                domains,
                labels,
                domain_names,
                args.point_size,
                args.alpha,
            )
            ax.set_title(method_name)
            style_axis(ax)
        else:
            ax_domain = axes[0, column]
            scatter_by_domain(
                ax_domain,
                embedding,
                domains,
                domain_names,
                args.point_size,
                args.alpha,
            )
            ax_domain.set_title(f"{method_name}\nColored by domain")
            style_axis(ax_domain)

            ax_grade = axes[1, column]
            scatter_by_grade(ax_grade, embedding, labels, args.point_size, args.alpha)
            ax_grade.set_title(f"{method_name}\nColored by DR grade")
            style_axis(ax_grade)

    if args.layout == "combined":
        last_ax = axes[0, -1]
        domain_legend = last_ax.legend(
            handles=make_domain_legend_handles(domain_names),
            loc="center left",
            bbox_to_anchor=(1.02, 0.65),
            frameon=False,
            fontsize=9,
            title="Domain",
        )
        last_ax.add_artist(domain_legend)
        last_ax.legend(
            handles=make_grade_legend_handles(),
            loc="center left",
            bbox_to_anchor=(1.02, 0.25),
            frameon=False,
            fontsize=9,
            title="DR grade",
        )
    else:
        axes[0, -1].legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
            title="Domain",
        )
        axes[1, -1].legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
            title="Grade",
        )

    plt.tight_layout()
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
