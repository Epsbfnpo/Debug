"""Visualize CNN-guided DINO token selection for trained CASS_GDRNet models.

The script monkey-patches the routed bridge modules at runtime so normal model
forward passes also cache router scores and Top-K token indices. It does not
modify training code or require retraining.
"""

import argparse
import math
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import algorithms  # noqa: E402
from configs.defaults import _C as cfg_default  # noqa: E402
from dataset.data_manager import SquarePad, get_normalize  # noqa: E402


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    return patches, plt


ALL_DOMAINS = [
    "APTOS",
    "DDR",
    "DEEPDR",
    "FGADR",
    "IDRID",
    "MESSIDOR",
    "RLDR",
    "EYEPACS",
]

IDRID_LESION_FOLDERS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    # Optic disc is intentionally excluded because it is anatomy, not lesion.
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cfg(args):
    cfg = cfg_default.clone()
    cfg.defrost()

    cfg.DATASET.ROOT = args.data_root
    cfg.DATASET.SOURCE_DOMAINS = [args.source_domain]
    cfg.DATASET.TARGET_DOMAINS = [d for d in ALL_DOMAINS if d != args.source_domain]
    cfg.DG_MODE = "ESDG"

    cfg.TRANSFORM.INPUT_SIZE = args.input_size
    cfg.TRANSFORM.CROP_SIZE = args.input_size

    cfg.freeze()
    return cfg


def normalize_state_dict_keys(state_dict):
    """Normalize common DataParallel/DDP checkpoint key prefixes."""
    new_state = {}
    for key, value in state_dict.items():
        key = key.replace("network.module.", "network.")
        key = key.replace("momentum_network.module.", "momentum_network.")
        key = key.replace("module.", "")
        new_state[key] = value
    return new_state


def _load_state_flexibly(algorithm, state):
    """Load either an Algorithm state dict or a bare network state dict."""
    state = normalize_state_dict_keys(state)

    if any(key.startswith("network.") or key.startswith("momentum_network.") for key in state):
        missing, unexpected = algorithm.load_state_dict(state, strict=False)
        return "algorithm", missing, unexpected

    missing, unexpected = algorithm.network.load_state_dict(state, strict=False)
    return "network", missing, unexpected


def load_algorithm(cfg, checkpoint_path, device):
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "algorithm_state" in ckpt:
        state = ckpt["algorithm_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    loaded_as, missing, unexpected = _load_state_flexibly(algorithm, state)

    print("========================================================")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded as: {loaded_as} state")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print("First missing keys:", list(missing)[:10])
    if unexpected:
        print("First unexpected keys:", list(unexpected)[:10])
    print("========================================================")

    algorithm.eval()
    return algorithm


def attach_router_cache(bridge, name: str):
    """Monkey-patch RoutedBridgeModule._select_tokens to cache router outputs."""

    def _select_tokens_logged(self, feat_cnn, feat_vit):
        batch_size, num_tokens, dim = feat_vit.shape
        del batch_size

        if num_tokens <= self.num_special_tokens + 1:
            self.vis_cache = {
                "bridge": name,
                "scores": None,
                "topk_idx": None,
                "num_special": min(self.num_special_tokens, num_tokens),
                "N": num_tokens,
            }
            return feat_vit

        cnn_summary = feat_cnn.mean(dim=(2, 3))
        query = self.router_q(cnn_summary)
        key = self.router_k(feat_vit)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        scores = torch.einsum("bd,bnd->bn", query, key)

        num_special = min(self.num_special_tokens, num_tokens)
        special_tokens = feat_vit[:, :num_special, :]
        patch_tokens = feat_vit[:, num_special:, :]
        patch_scores = scores[:, num_special:]

        num_patch_tokens = patch_tokens.size(1)
        k_keep = max(1, int(num_patch_tokens * self.topk_ratio))
        k_keep = min(k_keep, num_patch_tokens)

        topk = patch_scores.topk(k=k_keep, dim=1)
        topk_idx = topk.indices
        topk_scores = topk.values

        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, dim)
        selected_patch_tokens = torch.gather(patch_tokens, 1, gather_idx)

        self.vis_cache = {
            "bridge": name,
            "scores": scores.detach().float().cpu(),
            "patch_scores": patch_scores.detach().float().cpu(),
            "topk_idx": topk_idx.detach().cpu(),
            "topk_scores": topk_scores.detach().float().cpu(),
            "num_special": num_special,
            "N": num_tokens,
            "topk_ratio": self.topk_ratio,
        }

        return torch.cat([special_tokens, selected_patch_tokens], dim=1)

    bridge._select_tokens = types.MethodType(_select_tokens_logged, bridge)


def split_files_for_protocol(data_root, domain, split):
    split_dir = Path(data_root) / "splits"
    if split == "protocol_test":
        # Match GDRBench(mode="test"), which evaluates target domains using
        # the union of train and crossval split files rather than *_test.txt.
        return [
            split_dir / f"{domain}_train.txt",
            split_dir / f"{domain}_crossval.txt",
        ]

    return [split_dir / f"{domain}_{split}.txt"]


def read_split(
    data_root,
    domain,
    split,
    max_samples=None,
    grade_min=None,
    image_rel_path=None,
):
    split_paths = split_files_for_protocol(data_root, domain, split)
    image_root = Path(data_root) / "images"

    missing_paths = [path for path in split_paths if not path.exists()]
    if missing_paths:
        missing_str = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Cannot find split file(s): {missing_str}")

    samples = []
    for split_path in split_paths:
        with open(split_path) as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                rel_path = parts[0]
                label = int(parts[1])

                if image_rel_path is not None and rel_path != image_rel_path:
                    continue

                if grade_min is not None and label < grade_min:
                    continue

                img_path = image_root / rel_path
                if img_path.exists():
                    samples.append((img_path, rel_path, label))

                if max_samples is not None and len(samples) >= max_samples:
                    return samples

    return samples


def preprocess_image(img_path, input_size):
    raw = Image.open(img_path).convert("RGB")

    display_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(input_size),
        ]
    )
    display_pil = display_transform(raw)

    model_transform = transforms.Compose([transforms.ToTensor(), get_normalize()])
    tensor = model_transform(display_pil).unsqueeze(0)
    display_np = np.array(display_pil).astype(np.float32) / 255.0

    return display_np, tensor


def load_single_mask(mask_path, input_size):
    mask = Image.open(mask_path).convert("L")
    mask_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.CenterCrop(input_size),
        ]
    )
    mask = mask_transform(mask)
    mask_np = np.array(mask)
    return (mask_np > 0).astype(np.float32)


def maybe_load_mask(mask_root, rel_path, input_size):
    if mask_root is None:
        return None

    mask_root = Path(mask_root)
    rel_path = Path(rel_path)
    candidates = [
        mask_root / rel_path,
        mask_root / rel_path.name,
        mask_root / rel_path.with_suffix(".png").name,
        mask_root / rel_path.with_suffix(".tif").name,
        mask_root / rel_path.with_suffix(".jpg").name,
    ]

    mask_path = next((path for path in candidates if path.exists()), None)
    if mask_path is None:
        return None

    return load_single_mask(mask_path, input_size)


def extract_idrid_case_id(rel_path):
    """Extract an IDRID case ID such as ``IDRiD_01`` from an image path."""
    stem = Path(rel_path).stem
    match = re.search(r"(IDRiD_\d+)", stem, flags=re.IGNORECASE)
    if match is None:
        return None

    case_id = match.group(1)
    _, number = case_id.split("_")
    return f"IDRiD_{int(number):02d}"


def load_idrid_merged_lesion_mask(idrid_lesion_root, rel_path, input_size):
    """Load and merge IDRID MA/HE/EX/SE masks, explicitly excluding optic disc."""
    if idrid_lesion_root is None:
        return None, {}

    case_id = extract_idrid_case_id(rel_path)
    if case_id is None:
        return None, {}

    idrid_lesion_root = Path(idrid_lesion_root)
    merged = None
    found = {}

    for suffix, folder in IDRID_LESION_FOLDERS.items():
        mask_path = idrid_lesion_root / folder / f"{case_id}_{suffix}.tif"
        if not mask_path.exists():
            found[suffix] = None
            continue

        mask_np = load_single_mask(mask_path, input_size)
        merged = mask_np.copy() if merged is None else np.maximum(merged, mask_np)
        found[suffix] = str(mask_path)

    if merged is None:
        return None, found

    return (merged > 0).astype(np.float32), found


def make_maps_from_cache(cache):
    """Convert cached router outputs into grid score and selection maps.

    The routed bridge selects visual tokens from ``feat_vit[:, num_special:, :]``.
    Therefore the visualization grid must use the same special-token prefix
    instead of assuming the sequence is always ``CLS + visual patches``. This is
    required for DINO variants that expose native register/special tokens.
    """
    scores = cache["scores"]
    topk_idx = cache["topk_idx"]
    num_special = int(cache["num_special"])
    num_tokens = int(cache["N"])

    if scores is None or topk_idx is None:
        raise RuntimeError("No router scores were cached.")

    visual_start = num_special
    num_visual_tokens = num_tokens - visual_start
    grid = int(round(math.sqrt(num_visual_tokens)))

    if grid * grid != num_visual_tokens:
        raise RuntimeError(
            f"Cannot reshape {num_visual_tokens} routed visual tokens into a square grid. "
            f"N={num_tokens}, num_special={num_special}, "
            f"visual_start={visual_start}, grid={grid}. "
            "This usually means the non-visual token prefix was parsed incorrectly."
        )

    scores_1d = scores[0].numpy()

    # Scores corresponding exactly to the visual tokens used by the router.
    visual_scores = scores_1d[visual_start : visual_start + num_visual_tokens]
    score_map = visual_scores.reshape(grid, grid)

    # topk_idx is already relative to patch_tokens = feat_vit[:, num_special:, :].
    selected_visual_idx = topk_idx[0].numpy()

    selected_map = np.zeros(num_visual_tokens, dtype=np.float32)
    selected_map[selected_visual_idx] = 1.0
    selected_map = selected_map.reshape(grid, grid)

    return score_map, selected_map, selected_visual_idx, grid


def normalize_map(values):
    values = values.astype(np.float32)
    values = values - np.nanmin(values)
    denom = np.nanmax(values) + 1e-8
    return values / denom


def make_green_overlay(image, mask_np, alpha=0.45):
    """Return an RGB image with the binary lesion mask overlaid in green."""
    if mask_np is None:
        return image

    out = image.copy()
    green = np.zeros_like(out)
    green[..., 1] = 1.0

    mask = mask_np.astype(bool)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * green[mask]
    return out


def draw_lesion_overlay(ax, image, mask_np):
    overlay = make_green_overlay(image, mask_np, alpha=0.45)
    ax.imshow(overlay)
    if mask_np is not None:
        ax.contour(mask_np, levels=[0.5], linewidths=1.0, colors="lime")
    ax.axis("off")


def lesion_token_map_from_mask(mask_np, grid, threshold=0.0):
    """Convert a pixel-level lesion mask into lesion-overlapping DINO cells."""
    if mask_np is None:
        return None

    height, width = mask_np.shape[:2]
    cell_h = height / grid
    cell_w = width / grid
    token_map = np.zeros((grid, grid), dtype=np.float32)

    for y_pos in range(grid):
        y0 = int(round(y_pos * cell_h))
        y1 = int(round((y_pos + 1) * cell_h))
        for x_pos in range(grid):
            x0 = int(round(x_pos * cell_w))
            x1 = int(round((x_pos + 1) * cell_w))
            cell = mask_np[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            ratio = float((cell > 0).sum()) / float(cell.size)
            if threshold <= 0.0:
                token_map[y_pos, x_pos] = 1.0 if ratio > 0.0 else 0.0
            else:
                token_map[y_pos, x_pos] = 1.0 if ratio >= threshold else 0.0

    return token_map


def draw_green_token_boxes(ax, token_map, image_shape, patches, linewidth=1.0):
    """Draw green token-cell boxes for lesion-overlapping DINO tokens."""
    if token_map is None:
        return

    grid = token_map.shape[0]
    height, width = image_shape[:2]
    cell_h = height / grid
    cell_w = width / grid
    ys, xs = np.where(token_map > 0)

    for y_pos, x_pos in zip(ys, xs):
        rect = patches.Rectangle(
            (x_pos * cell_w, y_pos * cell_h),
            cell_w,
            cell_h,
            linewidth=linewidth,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)


def draw_token_grid(ax, image, grid, lesion_token_map=None, patches=None):
    ax.imshow(image)
    height, width = image.shape[:2]
    for idx in range(1, grid):
        ax.axhline(idx * height / grid, linewidth=0.25, alpha=0.35)
        ax.axvline(idx * width / grid, linewidth=0.25, alpha=0.35)

    if lesion_token_map is not None and patches is not None:
        draw_green_token_boxes(ax, lesion_token_map, image.shape, patches, linewidth=1.0)

    ax.axis("off")


def draw_selected_boxes(
    ax,
    image,
    selected_map,
    patches,
    lesion_mask=None,
    lesion_token_map=None,
):
    ax.imshow(image)
    grid = selected_map.shape[0]
    height, width = image.shape[:2]
    cell_h = height / grid
    cell_w = width / grid

    if lesion_token_map is not None:
        draw_green_token_boxes(ax, lesion_token_map, image.shape, patches, linewidth=1.0)

    if lesion_mask is not None:
        ax.contour(lesion_mask, levels=[0.5], linewidths=0.8, colors="lime")

    ys, xs = np.where(selected_map > 0)
    for y_pos, x_pos in zip(ys, xs):
        rect = patches.Rectangle(
            (x_pos * cell_w, y_pos * cell_h),
            cell_w,
            cell_h,
            linewidth=0.9,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.axis("off")


def draw_score_overlay(
    ax,
    image,
    score_map,
    lesion_mask=None,
    lesion_token_map=None,
    patches=None,
):
    ax.imshow(image)
    score_norm = normalize_map(score_map)
    ax.imshow(
        score_norm,
        cmap="jet",
        alpha=0.45,
        extent=(0, image.shape[1], image.shape[0], 0),
    )

    if lesion_token_map is not None and patches is not None:
        draw_green_token_boxes(ax, lesion_token_map, image.shape, patches, linewidth=1.0)

    if lesion_mask is not None:
        ax.contour(lesion_mask, levels=[0.5], linewidths=0.8, colors="lime")

    ax.axis("off")


def run_visualization(args):
    patches, plt = import_matplotlib()

    set_seed(args.seed)

    requested_device = torch.device(args.device)
    device = requested_device if requested_device.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    cfg = build_cfg(args)
    algorithm = load_algorithm(cfg, args.checkpoint, device)

    network = algorithm.network
    network.eval()

    attach_router_cache(network.bridge2, "bridge2")
    attach_router_cache(network.bridge3, "bridge3")

    samples = read_split(
        data_root=args.data_root,
        domain=args.vis_domain,
        split=args.split,
        max_samples=args.num_samples,
        grade_min=args.grade_min,
        image_rel_path=args.image_rel_path,
    )

    if len(samples) == 0:
        raise RuntimeError("No samples found. Check domain/split/grade_min.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_lesion_col = args.lesion_mask_root is not None or args.idrid_lesion_root is not None
    n_cols = 5 if use_lesion_col else 4
    n_rows = len(samples)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    for row_idx, (img_path, rel_path, label) in enumerate(samples):
        display_np, inputs = preprocess_image(img_path, args.input_size)
        inputs = inputs.to(device)

        with torch.no_grad():
            if args.amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = network(x_cnn=inputs, x_vit=inputs)
            else:
                _ = network(x_cnn=inputs, x_vit=inputs)

        bridge = network.bridge3 if args.bridge == "bridge3" else network.bridge2
        cache = bridge.vis_cache
        score_map, selected_map, selected_idx, grid = make_maps_from_cache(cache)

        lesion_mask = None
        lesion_sources = {}

        if args.idrid_lesion_root is not None and args.vis_domain == "IDRID":
            lesion_mask, lesion_sources = load_idrid_merged_lesion_mask(
                args.idrid_lesion_root,
                rel_path,
                args.input_size,
            )

        if lesion_mask is None and args.lesion_mask_root is not None:
            lesion_mask = maybe_load_mask(args.lesion_mask_root, rel_path, args.input_size)

        lesion_token_map = lesion_token_map_from_mask(
            lesion_mask,
            grid,
            threshold=args.lesion_token_threshold,
        )

        axes[row_idx, 0].imshow(display_np)
        axes[row_idx, 0].set_title(f"Original\nlabel={label}")
        axes[row_idx, 0].axis("off")

        if use_lesion_col:
            draw_lesion_overlay(axes[row_idx, 1], display_np, lesion_mask)
            axes[row_idx, 1].set_title("Lesion label overlay\nMA+HE+EX+SE")
            grid_col = 2
            selected_col = 3
            score_col = 4
        else:
            grid_col = 1
            selected_col = 2
            score_col = 3

        draw_token_grid(
            axes[row_idx, grid_col],
            display_np,
            grid,
            lesion_token_map=lesion_token_map,
            patches=patches,
        )
        axes[row_idx, grid_col].set_title(f"DINO token grid\n{grid}×{grid}")

        draw_selected_boxes(
            axes[row_idx, selected_col],
            display_np,
            selected_map,
            patches,
            lesion_mask=lesion_mask,
            lesion_token_map=lesion_token_map,
        )
        axes[row_idx, selected_col].set_title(
            f"CNN-guided Top-K\n{int(selected_map.sum())} tokens"
        )

        draw_score_overlay(
            axes[row_idx, score_col],
            display_np,
            score_map,
            lesion_mask=lesion_mask,
            lesion_token_map=lesion_token_map,
            patches=patches,
        )
        axes[row_idx, score_col].set_title("Router score map")

        sample_name = Path(rel_path).stem
        if lesion_mask is not None:
            lesion_overlay = make_green_overlay(display_np, lesion_mask, alpha=0.45)
            overlay_uint8 = (np.clip(lesion_overlay, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(overlay_uint8).save(
                out_dir / f"{sample_name}_lesion_overlay_green.png"
            )

            lesion_uint8 = (lesion_mask * 255).astype(np.uint8)
            Image.fromarray(lesion_uint8).save(
                out_dir / f"{sample_name}_merged_lesion_mask.png"
            )

        np.savez(
            out_dir / f"{sample_name}_{args.bridge}_routing.npz",
            score_map=score_map,
            selected_map=selected_map,
            selected_idx=selected_idx,
            grid=grid,
            label=label,
            rel_path=str(rel_path),
            bridge=args.bridge,
            lesion_mask=lesion_mask if lesion_mask is not None else np.array([]),
            lesion_token_map=(
                lesion_token_map if lesion_token_map is not None else np.array([])
            ),
            lesion_sources=np.array([str(lesion_sources)], dtype=object),
        )

        if args.save_single_sample_panels:
            sample_fig, sample_axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 3.8))
            sample_axes = np.atleast_1d(sample_axes)

            sample_axes[0].imshow(display_np)
            sample_axes[0].set_title(f"Original\nlabel={label}")
            sample_axes[0].axis("off")

            if use_lesion_col:
                draw_lesion_overlay(sample_axes[1], display_np, lesion_mask)
                sample_axes[1].set_title("Lesion label overlay\nMA+HE+EX+SE")
                sample_grid_col = 2
                sample_selected_col = 3
                sample_score_col = 4
            else:
                sample_grid_col = 1
                sample_selected_col = 2
                sample_score_col = 3

            draw_token_grid(
                sample_axes[sample_grid_col],
                display_np,
                grid,
                lesion_token_map=lesion_token_map,
                patches=patches,
            )
            sample_axes[sample_grid_col].set_title(f"DINO token grid\n{grid}×{grid}")

            draw_selected_boxes(
                sample_axes[sample_selected_col],
                display_np,
                selected_map,
                patches,
                lesion_mask=lesion_mask,
                lesion_token_map=lesion_token_map,
            )
            sample_axes[sample_selected_col].set_title(
                f"CNN-guided Top-K\n{int(selected_map.sum())} tokens"
            )

            draw_score_overlay(
                sample_axes[sample_score_col],
                display_np,
                score_map,
                lesion_mask=lesion_mask,
                lesion_token_map=lesion_token_map,
                patches=patches,
            )
            sample_axes[sample_score_col].set_title("Router score map")

            sample_fig.tight_layout()
            sample_fig.savefig(
                out_dir / f"{sample_name}_{args.bridge}_single_sample_panel.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(sample_fig)

    plt.tight_layout()
    save_path = out_dir / f"token_selection_{args.source_domain}_{args.vis_domain}_{args.split}_{args.bridge}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize CNN-guided Top-K token routing in CASS_GDRNet."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--data-root",
        type=str,
        default="/datasets/work/hb-nhmrc-dhcp/work/liu275/DGDR/GDR_Formatted_Data",
    )

    parser.add_argument("--source-domain", type=str, default="IDRID", choices=ALL_DOMAINS)
    parser.add_argument("--vis-domain", type=str, default="IDRID", choices=ALL_DOMAINS)
    parser.add_argument(
        "--split",
        type=str,
        default="crossval",
        choices=["train", "crossval", "test", "protocol_test"],
        help=(
            "Split to read. Use protocol_test to match GDRBench(mode=\"test\"), "
            "which reads train+crossval for target domains."
        ),
    )

    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--bridge", type=str, default="bridge3", choices=["bridge2", "bridge3"])

    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--grade-min", type=int, default=None)
    parser.add_argument(
        "--image-rel-path",
        type=str,
        default=None,
        help=(
            "Relative image path exactly as written in the split file. "
            "If provided, visualize this single image."
        ),
    )

    parser.add_argument("--lesion-mask-root", type=str, default=None)
    parser.add_argument(
        "--idrid-lesion-root",
        type=str,
        default=None,
        help=(
            "Root directory of IDRID pixel-level lesion masks. Expected subfolders: "
            "1. Microaneurysms, 2. Haemorrhages, 3. Hard Exudates, "
            "4. Soft Exudates. Optic Disc is excluded."
        ),
    )
    parser.add_argument(
        "--lesion-token-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum lesion-pixel ratio for marking a DINO token cell as "
            "lesion-overlapping. Use 0.0 to mark any cell with lesion pixels."
        ),
    )
    parser.add_argument(
        "--save-single-sample-panels",
        action="store_true",
        help="Save per-sample visualization panels in addition to the combined figure.",
    )
    parser.add_argument("--out-dir", type=str, default="./visualizations/token_selection")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run_visualization(parse_args())
