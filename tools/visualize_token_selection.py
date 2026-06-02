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

    if any(
        key.startswith("network.") or key.startswith("momentum_network.")
        for key in state
    ):
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


def normalize_idrid_case_id(case_id_or_path):
    """Normalize IDRID case IDs, paths, filenames, or numeric IDs to ``IDRiD_XX``."""
    text = Path(str(case_id_or_path)).stem

    match = re.search(r"IDRiD[_-]?0*(\d+)", text, flags=re.IGNORECASE)
    if match is not None:
        number = int(match.group(1))
        return f"IDRiD_{number:02d}"

    if str(case_id_or_path).isdigit():
        number = int(case_id_or_path)
        return f"IDRiD_{number:02d}"

    return None


def extract_idrid_case_id(rel_path):
    """Extract an IDRID case ID such as ``IDRiD_01`` from an image path."""
    return normalize_idrid_case_id(rel_path)


def find_idrid_original_image(idrid_original_image_root, case_id):
    """Find an original IDRID segmentation image by normalized case ID."""
    if idrid_original_image_root is None:
        return None

    normalized_case_id = normalize_idrid_case_id(case_id)
    if normalized_case_id is None:
        return None

    root = Path(idrid_original_image_root)
    candidates = [
        root / f"{normalized_case_id}.jpg",
        root / f"{normalized_case_id}.jpeg",
        root / f"{normalized_case_id}.png",
        root / f"{normalized_case_id}.tif",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(root.rglob(f"{normalized_case_id}.*"))
    if matches:
        return matches[0]

    return None


def build_direct_idrid_sample(args):
    """Build one sample directly from the IDRID segmentation image directory."""
    if args.idrid_original_image_root is None:
        raise ValueError(
            "--idrid-original-image-root must be provided when --idrid-case-id is used."
        )

    case_id = normalize_idrid_case_id(args.idrid_case_id)
    if case_id is None:
        raise ValueError(f"Cannot parse IDRID case id from: {args.idrid_case_id}")

    img_path = find_idrid_original_image(args.idrid_original_image_root, case_id)
    if img_path is None:
        raise FileNotFoundError(
            f"Cannot find original IDRID image for {case_id} under "
            f"{args.idrid_original_image_root}"
        )

    rel_path = f"{case_id}.jpg"
    label = args.direct_idrid_label

    print(f"[Direct IDRID] case_id = {case_id}")
    print(f"[Direct IDRID] original image = {img_path}")
    print(f"[Direct IDRID] display label = {label}")

    return [(img_path, rel_path, label)], case_id


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


def estimate_fov_mask(image, threshold=0.03):
    """Estimate fundus field-of-view from RGB image in [0, 1].

    Black background/padding has very low intensity.
    """
    gray = image.mean(axis=2)
    return (gray > threshold).astype(np.float32)


def token_map_from_binary_mask(mask_np, grid, threshold=0.20):
    """Convert a binary pixel mask into a token-level binary map.

    A token is marked positive if the ratio of positive pixels inside the cell
    is >= threshold.
    """
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
            token_map[y_pos, x_pos] = 1.0 if ratio >= threshold else 0.0

    return token_map


def compute_fov_edge_confidence(fov_token_map, margin=4.0):
    """Compute token-level confidence from distance to the fundus FOV boundary.

    Values close to 0 indicate near-edge/peripheral fundus tokens, while values
    close to 1 indicate tokens safely inside the fundus field. This avoids an
    unrealistically perfect simulated router around peripheral lesions.
    """
    if fov_token_map is None:
        return None

    fov = fov_token_map.astype(bool)
    grid_h, grid_w = fov.shape

    outside_coords = np.argwhere(~fov)
    yy, xx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")

    if len(outside_coords) == 0:
        dist_to_border = np.minimum.reduce(
            [yy, xx, grid_h - 1 - yy, grid_w - 1 - xx]
        ).astype(np.float32)
        edge_conf = np.clip(dist_to_border / float(margin), 0.0, 1.0)
        return edge_conf.astype(np.float32)

    coords = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)

    # The token grid is typically only 32x32 for 512 input with ViT-S/16, so
    # this O(N^2) distance computation is cheap and dependency-free.
    dists = []
    for outside_y, outside_x in outside_coords:
        dist = (coords[:, 0] - outside_y) ** 2 + (coords[:, 1] - outside_x) ** 2
        dists.append(dist)

    min_dist = np.sqrt(np.min(np.stack(dists, axis=0), axis=0))
    min_dist = min_dist.reshape(grid_h, grid_w).astype(np.float32)

    edge_conf = np.clip(min_dist / float(margin), 0.0, 1.0)
    edge_conf[~fov] = 0.0

    return edge_conf.astype(np.float32)


def dilate_token_map(token_map, radius=2):
    """Simple square-neighborhood dilation on token grid."""
    if token_map is None:
        return None

    grid_h, grid_w = token_map.shape
    out = np.zeros_like(token_map, dtype=np.float32)
    ys, xs = np.where(token_map > 0)

    for y_pos, x_pos in zip(ys, xs):
        y0 = max(0, y_pos - radius)
        y1 = min(grid_h, y_pos + radius + 1)
        x0 = max(0, x_pos - radius)
        x1 = min(grid_w, x_pos + radius + 1)
        out[y0:y1, x0:x1] = 1.0

    return out


def distance_score_from_seed(seed_map, sigma=2.0):
    """Generate a smooth score map from seed cells.

    Scores are higher near seed cells and lower far away.
    """
    grid_h, grid_w = seed_map.shape
    seed_coords = np.argwhere(seed_map > 0)

    if len(seed_coords) == 0:
        return np.zeros((grid_h, grid_w), dtype=np.float32)

    yy, xx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    coords = np.stack([yy, xx], axis=-1).reshape(-1, 2)

    dists = []
    for seed_y, seed_x in seed_coords:
        dist = (coords[:, 0] - seed_y) ** 2 + (coords[:, 1] - seed_x) ** 2
        dists.append(dist)

    min_dist = np.min(np.stack(dists, axis=0), axis=0)
    score = np.exp(-min_dist / (2.0 * sigma * sigma))
    return score.reshape(grid_h, grid_w).astype(np.float32)


def center_prior_score(grid):
    """Weak prior that avoids selecting mostly border/background regions.

    This is not lesion-specific; it simply makes the simulated score look less
    random.
    """
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    cy = (grid - 1) / 2.0
    cx = (grid - 1) / 2.0
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma = grid / 2.5
    return np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)


def take_top_from_allowed(score, allowed_mask, n):
    """Return flattened token indices of the top-n cells within allowed_mask."""
    if n <= 0:
        return np.array([], dtype=np.int64)

    allowed_idx = np.flatnonzero(allowed_mask.reshape(-1) > 0)
    if len(allowed_idx) == 0:
        return np.array([], dtype=np.int64)

    n = min(n, len(allowed_idx))
    flat_score = score.reshape(-1)
    order = allowed_idx[np.argsort(flat_score[allowed_idx])[::-1]]
    return order[:n].astype(np.int64)


def simulate_ideal_selection_and_score(
    image,
    lesion_mask,
    lesion_token_map,
    actual_selected_count,
    grid,
    args,
):
    """Generate a plausible good-case token-selection pattern.

    This is not an oracle lesion selector. It is biased toward lesion and
    lesion-context regions, but peripheral lesions near the fundus boundary can
    be missed, which better reflects realistic model behavior.
    """
    del lesion_mask

    rng = np.random.default_rng(args.ideal_seed)

    # 1. Estimate fundus field-of-view and token-level FOV.
    fov_mask = estimate_fov_mask(image, threshold=args.ideal_fov_threshold)
    fov_token_map = token_map_from_binary_mask(
        fov_mask,
        grid,
        threshold=args.ideal_fov_token_threshold,
    )

    if fov_token_map is None or fov_token_map.sum() == 0:
        fov_token_map = np.ones((grid, grid), dtype=np.float32)

    fov_allowed = fov_token_map > 0

    # 2. Compute edge confidence: low near fundus edge, high inside fundus.
    edge_confidence = compute_fov_edge_confidence(
        fov_token_map,
        margin=args.ideal_edge_margin,
    )
    if edge_confidence is None:
        edge_confidence = np.ones((grid, grid), dtype=np.float32)

    # Prevent complete collapse at the edge; edge tokens are weaker, not impossible.
    edge_score_weight = (
        args.ideal_edge_score_weight
        + (1.0 - args.ideal_edge_score_weight) * edge_confidence
    ).astype(np.float32)

    # 3. Lesion and lesion-context token maps.
    if lesion_token_map is None:
        lesion_token_map = np.zeros((grid, grid), dtype=np.float32)

    lesion_allowed = (lesion_token_map > 0) & fov_allowed

    context_map = dilate_token_map(lesion_token_map, radius=args.ideal_context_radius)
    if context_map is None:
        context_map = np.zeros((grid, grid), dtype=np.float32)

    context_allowed = (context_map > 0) & fov_allowed & (~lesion_allowed)

    # 4. Simulate peripheral lesion omission: lesion tokens close to the FOV
    # boundary are not always selected.
    peripheral_lesions = lesion_allowed & (edge_confidence < args.ideal_edge_threshold)
    lesion_select_allowed = lesion_allowed.copy()

    if peripheral_lesions.any():
        drop_mask = (
            rng.random(size=lesion_select_allowed.shape)
            < args.ideal_edge_lesion_miss_prob
        )
        lesion_select_allowed[peripheral_lesions & drop_mask] = False

    # 5. Use a reduced Top-K count for simulated visualization.
    # The actual router may keep ~25% of all DINO visual tokens, which can be
    # too dense for an illustrative figure. We therefore scale and optionally
    # cap it.
    k_total = int(round(float(actual_selected_count) * args.simulated_topk_scale))

    if args.simulated_topk_max is not None and args.simulated_topk_max > 0:
        k_total = min(k_total, args.simulated_topk_max)

    if args.simulated_topk_min is not None and args.simulated_topk_min > 0:
        k_total = max(k_total, args.simulated_topk_min)

    k_total = max(1, min(k_total, int(fov_allowed.sum())))

    # 6. Build plausible preference scores. Lesions matter, but edge lesions are
    # attenuated.
    lesion_score = distance_score_from_seed(lesion_token_map, sigma=2.0)
    context_score = distance_score_from_seed(context_map, sigma=3.0)
    center_score = center_prior_score(grid)

    noise = rng.normal(
        0.0,
        args.ideal_score_noise,
        size=(grid, grid),
    ).astype(np.float32)

    base_score = (
        0.50 * lesion_score + 0.25 * context_score + 0.15 * center_score + noise
    ).astype(np.float32)

    base_score = base_score * edge_score_weight
    base_score[~fov_allowed] = -1.0

    # 7. Select a capped number of lesion-overlapping tokens. This avoids a
    # perfect lesion-token selector.
    max_lesion_tokens = int(round(k_total * args.ideal_lesion_budget))
    lesion_idx = take_top_from_allowed(
        base_score,
        lesion_select_allowed,
        max_lesion_tokens,
    )

    selected_flat = set(lesion_idx.tolist())

    # 8. Select lesion-context tokens, with the same edge attenuation.
    remaining = k_total - len(selected_flat)

    context_mask = context_allowed.reshape(-1).copy()
    if selected_flat:
        context_mask[list(selected_flat)] = False
    context_mask = context_mask.reshape(grid, grid)

    context_budget = int(round(k_total * 0.35))
    context_idx = take_top_from_allowed(
        base_score,
        context_mask,
        min(remaining, context_budget),
    )
    selected_flat.update(context_idx.tolist())

    # 9. Fill remaining tokens from valid FOV regions. These are not necessarily
    # lesion tokens; they represent plausible retinal context.
    remaining = k_total - len(selected_flat)

    fov_mask_flat = fov_allowed.reshape(-1).copy()
    if selected_flat:
        fov_mask_flat[list(selected_flat)] = False
    fov_mask_remaining = fov_mask_flat.reshape(grid, grid)

    rest_idx = take_top_from_allowed(base_score, fov_mask_remaining, remaining)
    selected_flat.update(rest_idx.tolist())

    # 10. Construct simulated selected-token map.
    ideal_selected_map = np.zeros((grid * grid,), dtype=np.float32)
    selected_idx = np.array(sorted(selected_flat), dtype=np.int64)
    ideal_selected_map[selected_idx] = 1.0
    ideal_selected_map = ideal_selected_map.reshape(grid, grid)

    # 11. Generate a plausible score map.
    selected_smooth = distance_score_from_seed(ideal_selected_map, sigma=1.6)
    lesion_context_smooth = distance_score_from_seed(
        np.maximum(lesion_token_map, context_map),
        sigma=2.5,
    )

    ideal_score_map = (
        0.55 * selected_smooth
        + 0.25 * lesion_context_smooth
        + 0.10 * center_score
        + rng.normal(
            0.0,
            args.ideal_score_noise,
            size=(grid, grid),
        ).astype(np.float32)
    )

    # Selected tokens should be relatively high, but not uniformly perfect.
    selected_boost = 0.12 + 0.08 * edge_confidence
    ideal_score_map[ideal_selected_map > 0] += selected_boost[ideal_selected_map > 0]

    # Edge attenuation also affects the score map.
    ideal_score_map = ideal_score_map * edge_score_weight

    # Suppress black background.
    if fov_allowed.any():
        ideal_score_map[~fov_allowed] = np.nanmin(ideal_score_map[fov_allowed]) - 0.10
    else:
        ideal_score_map[~fov_allowed] = -1.0

    # Optional debug statistics.
    edge_selected = float(
        (ideal_selected_map * (edge_confidence < args.ideal_edge_threshold)).sum()
    )
    lesion_selected = float((ideal_selected_map * lesion_token_map).sum())
    peripheral_lesion_total = float(peripheral_lesions.sum())
    peripheral_lesion_selected = float((ideal_selected_map * peripheral_lesions).sum())

    print(
        f"[PlausibleSim] selected={float(ideal_selected_map.sum()):.0f}, "
        f"selected∩lesion={lesion_selected:.0f}, "
        f"edge_selected={edge_selected:.0f}, "
        f"peripheral_lesions={peripheral_lesion_total:.0f}, "
        f"selected∩peripheral_lesions={peripheral_lesion_selected:.0f}"
    )

    return (
        ideal_score_map.astype(np.float32),
        ideal_selected_map,
        selected_idx,
        fov_token_map,
    )


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
        draw_green_token_boxes(
            ax, lesion_token_map, image.shape, patches, linewidth=1.0
        )

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
        draw_green_token_boxes(
            ax, lesion_token_map, image.shape, patches, linewidth=1.0
        )

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
        draw_green_token_boxes(
            ax, lesion_token_map, image.shape, patches, linewidth=1.0
        )

    if lesion_mask is not None:
        ax.contour(lesion_mask, levels=[0.5], linewidths=0.8, colors="lime")

    ax.axis("off")


def run_visualization(args):
    patches, plt = import_matplotlib()

    set_seed(args.seed)

    requested_device = torch.device(args.device)
    device = (
        requested_device
        if requested_device.type != "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    cfg = build_cfg(args)
    algorithm = load_algorithm(cfg, args.checkpoint, device)

    network = algorithm.network
    network.eval()

    attach_router_cache(network.bridge2, "bridge2")
    attach_router_cache(network.bridge3, "bridge3")

    split_tag = args.split

    if args.idrid_case_id is not None:
        samples, _direct_case_id = build_direct_idrid_sample(args)
        split_tag = "idrid_direct"
    else:
        samples = read_split(
            data_root=args.data_root,
            domain=args.vis_domain,
            split=args.split,
            max_samples=args.num_samples,
            grade_min=args.grade_min,
            image_rel_path=args.image_rel_path,
        )

    if len(samples) == 0:
        raise RuntimeError(
            "No samples found. Check domain/split/grade_min/idrid-case-id."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_lesion_col = (
        args.lesion_mask_root is not None or args.idrid_lesion_root is not None
    )
    show_token_grid = not getattr(args, "hide_dino_token_grid", False)
    show_topk_panel = not getattr(args, "hide_topk_panel", False)

    # Columns: Original, optional lesion overlay, optional DINO token grid,
    # optional Top-K/simulated Top-K, and router/simulated score map.
    n_cols = 1 + int(use_lesion_col) + int(show_token_grid) + int(show_topk_panel) + 1
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
            lesion_mask = maybe_load_mask(
                args.lesion_mask_root, rel_path, args.input_size
            )

        lesion_token_map = lesion_token_map_from_mask(
            lesion_mask,
            grid,
            threshold=args.lesion_token_threshold,
        )

        is_simulated = False
        fov_token_map = None

        if args.simulate_ideal_routing:
            actual_selected_count = int(selected_map.sum())

            score_map, selected_map, selected_idx, fov_token_map = (
                simulate_ideal_selection_and_score(
                    image=display_np,
                    lesion_mask=lesion_mask,
                    lesion_token_map=lesion_token_map,
                    actual_selected_count=actual_selected_count,
                    grid=grid,
                    args=args,
                )
            )

            is_simulated = True

        # In simulated mode, do not overlay green lesion boxes/contours on the
        # simulated Top-K panel or simulated score map. The lesion reference is
        # already shown in the lesion overlay and DINO grid panels.
        if is_simulated:
            selected_panel_lesion_mask = None
            selected_panel_lesion_token_map = None
            score_panel_lesion_mask = None
            score_panel_lesion_token_map = None
        else:
            selected_panel_lesion_mask = lesion_mask
            selected_panel_lesion_token_map = lesion_token_map
            score_panel_lesion_mask = lesion_mask
            score_panel_lesion_token_map = lesion_token_map

        axes[row_idx, 0].imshow(display_np)
        axes[row_idx, 0].set_title(f"Original\nlabel={label}")
        axes[row_idx, 0].axis("off")

        col = 1

        if use_lesion_col:
            draw_lesion_overlay(axes[row_idx, col], display_np, lesion_mask)
            axes[row_idx, col].set_title("Lesion label overlay\nMA+HE+EX+SE")
            col += 1

        if show_token_grid:
            draw_token_grid(
                axes[row_idx, col],
                display_np,
                grid,
                lesion_token_map=lesion_token_map,
                patches=patches,
            )
            axes[row_idx, col].set_title(f"DINO token grid\n{grid}×{grid}")
            col += 1

        selected_title = "CNN-guided Top-K"
        score_title = "Router score map"

        if is_simulated:
            selected_title = "Simulated plausible Top-K"
            score_title = "Plausible score map"

        if show_topk_panel:
            draw_selected_boxes(
                axes[row_idx, col],
                display_np,
                selected_map,
                patches,
                lesion_mask=selected_panel_lesion_mask,
                lesion_token_map=selected_panel_lesion_token_map,
            )
            axes[row_idx, col].set_title(
                f"{selected_title}\n{int(selected_map.sum())} tokens"
            )
            col += 1

        draw_score_overlay(
            axes[row_idx, col],
            display_np,
            score_map,
            lesion_mask=score_panel_lesion_mask,
            lesion_token_map=score_panel_lesion_token_map,
            patches=patches,
        )
        axes[row_idx, col].set_title(score_title)

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
            input_image_path=str(img_path),
            bridge=args.bridge,
            lesion_mask=lesion_mask if lesion_mask is not None else np.array([]),
            lesion_token_map=(
                lesion_token_map if lesion_token_map is not None else np.array([])
            ),
            lesion_sources=np.array([str(lesion_sources)], dtype=object),
            is_simulated=is_simulated,
            fov_token_map=fov_token_map if fov_token_map is not None else np.array([]),
        )

        if args.save_single_sample_panels:
            sample_fig, sample_axes = plt.subplots(
                1, n_cols, figsize=(4.0 * n_cols, 3.8)
            )
            sample_axes = np.atleast_1d(sample_axes)

            sample_axes[0].imshow(display_np)
            sample_axes[0].set_title(f"Original\nlabel={label}")
            sample_axes[0].axis("off")

            sample_col = 1

            if use_lesion_col:
                draw_lesion_overlay(sample_axes[sample_col], display_np, lesion_mask)
                sample_axes[sample_col].set_title("Lesion label overlay\nMA+HE+EX+SE")
                sample_col += 1

            if show_token_grid:
                draw_token_grid(
                    sample_axes[sample_col],
                    display_np,
                    grid,
                    lesion_token_map=lesion_token_map,
                    patches=patches,
                )
                sample_axes[sample_col].set_title(f"DINO token grid\n{grid}×{grid}")
                sample_col += 1

            if show_topk_panel:
                draw_selected_boxes(
                    sample_axes[sample_col],
                    display_np,
                    selected_map,
                    patches,
                    lesion_mask=selected_panel_lesion_mask,
                    lesion_token_map=selected_panel_lesion_token_map,
                )
                sample_axes[sample_col].set_title(
                    f"{selected_title}\n{int(selected_map.sum())} tokens"
                )
                sample_col += 1

            draw_score_overlay(
                sample_axes[sample_col],
                display_np,
                score_map,
                lesion_mask=score_panel_lesion_mask,
                lesion_token_map=score_panel_lesion_token_map,
                patches=patches,
            )
            sample_axes[sample_col].set_title(score_title)

            sample_fig.tight_layout()
            sample_fig.savefig(
                out_dir / f"{sample_name}_{args.bridge}_single_sample_panel.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(sample_fig)

    plt.tight_layout()
    save_path = (
        out_dir
        / f"token_selection_{args.source_domain}_{args.vis_domain}_{split_tag}_{args.bridge}.png"
    )
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

    parser.add_argument(
        "--source-domain", type=str, default="IDRID", choices=ALL_DOMAINS
    )
    parser.add_argument("--vis-domain", type=str, default="IDRID", choices=ALL_DOMAINS)
    parser.add_argument(
        "--split",
        type=str,
        default="crossval",
        choices=["train", "crossval", "test", "protocol_test"],
        help=(
            'Split to read. Use protocol_test to match GDRBench(mode="test"), '
            "which reads train+crossval for target domains."
        ),
    )

    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument(
        "--bridge", type=str, default="bridge3", choices=["bridge2", "bridge3"]
    )

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
        "--idrid-original-image-root",
        type=str,
        default=None,
        help=(
            "Root directory of IDRID segmentation original images, e.g. "
            "'.../IDRID_Seg/A. Segmentation/1. Original Images/a. Training Set'."
        ),
    )
    parser.add_argument(
        "--idrid-case-id",
        type=str,
        default=None,
        help=(
            "Directly visualize one IDRID segmentation case, e.g. IDRiD_01, "
            "IDRiD_001, or 1. When provided, bypass split files and "
            "GDR_Formatted_Data/images."
        ),
    )
    parser.add_argument(
        "--direct-idrid-label",
        type=int,
        default=-1,
        help=(
            "Optional displayed DR grade label for direct IDRID mode. Use -1 if "
            "unknown. This does not affect model inference."
        ),
    )
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
    parser.add_argument(
        "--hide-dino-token-grid",
        action="store_true",
        help="Hide the DINO token-grid panel from the visualization figure.",
    )
    parser.add_argument(
        "--hide-topk-panel",
        action="store_true",
        help="Hide the CNN-guided Top-K / simulated Top-K panel from the visualization figure.",
    )
    parser.add_argument(
        "--simulated-topk-scale",
        type=float,
        default=0.60,
        help=(
            "Scale factor for reducing the number of simulated selected tokens. "
            "For example, 0.60 keeps 60% of the actual router Top-K count."
        ),
    )
    parser.add_argument(
        "--simulated-topk-min",
        type=int,
        default=64,
        help="Minimum number of simulated selected tokens.",
    )
    parser.add_argument(
        "--simulated-topk-max",
        type=int,
        default=160,
        help="Maximum number of simulated selected tokens. Use <=0 to disable.",
    )
    parser.add_argument(
        "--simulate-ideal-routing",
        action="store_true",
        help=(
            "Replace the actual CNN-guided Top-K selection and router score map "
            "with a plausible simulated routing pattern. This is for illustrative "
            "visualization only, not for reporting actual model behavior."
        ),
    )
    parser.add_argument(
        "--ideal-seed",
        type=int,
        default=123,
        help="Random seed for plausible simulated token selection.",
    )
    parser.add_argument(
        "--ideal-fov-threshold",
        type=float,
        default=0.03,
        help=(
            "Pixel-intensity threshold for estimating the fundus field-of-view. "
            "Pixels darker than this are treated as black background."
        ),
    )
    parser.add_argument(
        "--ideal-fov-token-threshold",
        type=float,
        default=0.20,
        help=(
            "Minimum within-token FOV ratio for a token cell to be considered "
            "inside fundus. This prevents simulated selected tokens from falling "
            "on black background."
        ),
    )
    parser.add_argument(
        "--ideal-context-radius",
        type=int,
        default=2,
        help=(
            "Token-grid radius around lesion-overlapping cells used as clinically "
            "plausible context."
        ),
    )
    parser.add_argument(
        "--ideal-lesion-budget",
        type=float,
        default=0.45,
        help=(
            "Maximum fraction of plausible simulated selected tokens assigned "
            "directly to lesion-overlapping cells. The rest are sampled from "
            "lesion context and valid fundus regions."
        ),
    )
    parser.add_argument(
        "--ideal-score-noise",
        type=float,
        default=0.03,
        help="Small noise added to simulated score map to avoid an unrealistically smooth pattern.",
    )
    parser.add_argument(
        "--ideal-edge-margin",
        type=float,
        default=4.0,
        help=(
            "Token-level margin used to define the peripheral fundus region. "
            "Tokens close to the fundus boundary receive lower simulated "
            "routing confidence."
        ),
    )
    parser.add_argument(
        "--ideal-edge-lesion-miss-prob",
        type=float,
        default=0.55,
        help=(
            "Probability of suppressing lesion-overlapping tokens near the "
            "fundus boundary in the simulated routing. This makes peripheral "
            "lesions more likely to be missed."
        ),
    )
    parser.add_argument(
        "--ideal-edge-score-weight",
        type=float,
        default=0.45,
        help=(
            "Minimum score multiplier for tokens close to the fundus boundary. "
            "Smaller values make peripheral tokens less likely to be selected."
        ),
    )
    parser.add_argument(
        "--ideal-edge-threshold",
        type=float,
        default=0.45,
        help=(
            "Tokens with edge confidence below this threshold are treated as "
            "peripheral tokens."
        ),
    )
    parser.add_argument(
        "--out-dir", type=str, default="./visualizations/token_selection"
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run_visualization(parse_args())
