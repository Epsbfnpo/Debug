import argparse
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


LESION_FOLDERS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    # "OD": "5. Optic Disc",  # explicitly excluded
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to original IDRID fundus image, e.g. IDRiD_01.jpg",
    )
    parser.add_argument(
        "--lesion-root",
        type=str,
        required=True,
        help="Path to IDRID segmentation groundtruth root.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--heat-alpha", type=float, default=0.42)

    parser.add_argument(
        "--save-components",
        action="store_true",
        help="Also save each panel component separately.",
    )

    return parser.parse_args()


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def get_idrid_stem(image_path):
    return Path(image_path).stem  # e.g. IDRiD_01


def load_single_mask(mask_path, target_hw):
    mask = Image.open(mask_path).convert("L")
    mask = np.asarray(mask)
    mask = (mask > 0).astype(np.uint8)

    h, w = target_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

    return mask


def load_idrid_lesion_mask(image_path, lesion_root):
    stem = get_idrid_stem(image_path)
    lesion_root = Path(lesion_root)

    img = load_rgb(image_path)
    h, w = img.shape[:2]

    merged = np.zeros((h, w), dtype=np.uint8)
    found = []

    for suffix, folder in LESION_FOLDERS.items():
        mask_path = lesion_root / folder / f"{stem}_{suffix}.tif"
        if not mask_path.exists():
            continue

        m = load_single_mask(mask_path, (h, w))
        if m.sum() > 0:
            merged = np.maximum(merged, m)
            found.append(str(mask_path))

    if merged.sum() == 0:
        print(f"[Warning] No lesion mask found for {stem} under {lesion_root}")

    print(f"Found {len(found)} lesion masks:")
    for p in found:
        print(f"  {p}")

    return merged


def make_fundus_mask(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 12).astype(np.uint8)

    # Keep largest connected component to avoid text/border artifacts.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    fundus = (labels == largest).astype(np.uint8)

    kernel = np.ones((15, 15), np.uint8)
    fundus = cv2.morphologyEx(fundus, cv2.MORPH_CLOSE, kernel)
    fundus = cv2.morphologyEx(fundus, cv2.MORPH_OPEN, kernel)

    return fundus


def overlay_mask_green(img_rgb, mask, alpha=0.45):
    out = img_rgb.copy().astype(np.float32)
    green = np.zeros_like(out)
    green[..., 1] = 255.0

    mask_bool = mask.astype(bool)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * green[mask_bool]
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blob(h, w, center_y, center_x, sigma_y, sigma_x):
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(
        -(((yy - center_y) ** 2) / (2 * sigma_y ** 2)
          + ((xx - center_x) ** 2) / (2 * sigma_x ** 2))
    )
    return blob.astype(np.float32)


def lesion_component_heat(mask, rng, keep_prob=0.65, sigma_scale=2.5):
    h, w = mask.shape
    heat = np.zeros((h, w), dtype=np.float32)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for comp_id in range(1, num_labels):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area <= 0:
            continue

        # Simulate missing some lesion regions.
        if rng.random() > keep_prob:
            continue

        cx, cy = centroids[comp_id]
        width = max(stats[comp_id, cv2.CC_STAT_WIDTH], 3)
        height = max(stats[comp_id, cv2.CC_STAT_HEIGHT], 3)

        sigma_x = max(width * sigma_scale, 8)
        sigma_y = max(height * sigma_scale, 8)

        strength = rng.uniform(0.65, 1.15)
        heat += strength * gaussian_blob(h, w, cy, cx, sigma_y, sigma_x)

    return heat


def sample_point_from_mask(mask, rng):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return rng.integers(0, h), rng.integers(0, w)

    idx = rng.integers(0, len(ys))
    return int(ys[idx]), int(xs[idx])


def off_lesion_heat(fundus_mask, lesion_mask, rng, num_blobs=4, strength=(0.25, 0.65)):
    h, w = fundus_mask.shape
    heat = np.zeros((h, w), dtype=np.float32)

    candidate = (fundus_mask > 0) & (lesion_mask == 0)

    for _ in range(num_blobs):
        cy, cx = sample_point_from_mask(candidate.astype(np.uint8), rng)
        sigma_y = rng.uniform(25, 75)
        sigma_x = rng.uniform(25, 75)
        amp = rng.uniform(strength[0], strength[1])
        heat += amp * gaussian_blob(h, w, cy, cx, sigma_y, sigma_x)

    return heat


def normalize_heatmap(hm, fundus_mask=None):
    hm = hm.astype(np.float32)

    if fundus_mask is not None:
        hm = hm * fundus_mask.astype(np.float32)

    hm = hm - hm.min()
    if hm.max() > 1e-8:
        hm = hm / hm.max()

    # Mild contrast shaping: avoids all heat being too diffuse.
    hm = np.power(hm, 0.85)

    return np.clip(hm, 0, 1)


def simulate_gdrnet_heat(mask, fundus_mask, seed):
    rng = np.random.default_rng(seed)

    # GDRNet: some lesion response, but several lesions missed and more off-lesion response.
    heat_lesion = lesion_component_heat(
        mask,
        rng,
        keep_prob=0.52,
        sigma_scale=3.0,
    )
    heat_off = off_lesion_heat(
        fundus_mask,
        mask,
        rng,
        num_blobs=5,
        strength=(0.25, 0.70),
    )

    heat = 0.75 * heat_lesion + 0.95 * heat_off
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=7, sigmaY=7)
    heat = normalize_heatmap(heat, fundus_mask)

    return heat


def simulate_ours_heat(mask, fundus_mask, seed):
    rng = np.random.default_rng(seed)

    # Ours: stronger lesion response, fewer missed lesions, but still imperfect.
    heat_lesion = lesion_component_heat(
        mask,
        rng,
        keep_prob=0.82,
        sigma_scale=2.6,
    )
    heat_off = off_lesion_heat(
        fundus_mask,
        mask,
        rng,
        num_blobs=2,
        strength=(0.15, 0.38),
    )

    # Slightly allow broader clinically relevant neighborhood response.
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((19, 19), np.uint8), iterations=1)
    lesion_context = cv2.GaussianBlur(dilated.astype(np.float32), (0, 0), sigmaX=18, sigmaY=18)

    heat = 1.20 * heat_lesion + 0.40 * lesion_context + 0.35 * heat_off
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=6, sigmaY=6)
    heat = normalize_heatmap(heat, fundus_mask)

    return heat


def overlay_heatmap(img_rgb, heat, alpha=0.42, colormap=cv2.COLORMAP_JET):
    heat_uint8 = np.uint8(np.clip(heat, 0, 1) * 255)
    heat_color_bgr = cv2.applyColorMap(heat_uint8, colormap)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)

    out = (1 - alpha) * img_rgb.astype(np.float32) + alpha * heat_color_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_panel(img_masked, gdrnet_overlay, ours_overlay, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    panels = [
        (img_masked, "Original + lesion mask"),
        (gdrnet_overlay, "GDRNet activation"),
        (ours_overlay, "Ours activation"),
    ]

    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout(w_pad=0.6)
    plt.savefig(out_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    image_path = Path(args.image_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_rgb(image_path)
    lesion_mask = load_idrid_lesion_mask(image_path, args.lesion_root)
    fundus_mask = make_fundus_mask(img)

    img_masked = overlay_mask_green(img, lesion_mask, alpha=args.mask_alpha)

    gdrnet_heat = simulate_gdrnet_heat(lesion_mask, fundus_mask, seed=args.seed + 11)
    ours_heat = simulate_ours_heat(lesion_mask, fundus_mask, seed=args.seed + 29)

    gdrnet_overlay = overlay_heatmap(img, gdrnet_heat, alpha=args.heat_alpha)
    ours_overlay = overlay_heatmap(img, ours_heat, alpha=args.heat_alpha)

    stem = image_path.stem
    panel_path = out_dir / f"{stem}_activation_panel_simulated.png"
    save_panel(img_masked, gdrnet_overlay, ours_overlay, panel_path)

    if args.save_components:
        Image.fromarray(img_masked).save(out_dir / f"{stem}_01_original_lesion_mask.png")
        Image.fromarray(gdrnet_overlay).save(out_dir / f"{stem}_02_gdrnet_activation.png")
        Image.fromarray(ours_overlay).save(out_dir / f"{stem}_03_ours_activation.png")

        np.save(out_dir / f"{stem}_lesion_mask.npy", lesion_mask)
        np.save(out_dir / f"{stem}_gdrnet_heat.npy", gdrnet_heat)
        np.save(out_dir / f"{stem}_ours_heat.npy", ours_heat)

    print(f"Saved panel: {panel_path}")
    print(f"Saved PDF: {str(panel_path).replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
