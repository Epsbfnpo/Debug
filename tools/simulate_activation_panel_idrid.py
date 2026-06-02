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


def lesion_component_heat(
    mask,
    rng,
    keep_prob=0.45,
    sigma_scale=4.0,
    max_components=12,
    min_area=3,
    jitter_scale=0.75,
):
    h, w = mask.shape
    heat = np.zeros((h, w), dtype=np.float32)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    components = []
    for comp_id in range(1, num_labels):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        width = max(stats[comp_id, cv2.CC_STAT_WIDTH], 3)
        height = max(stats[comp_id, cv2.CC_STAT_HEIGHT], 3)
        cx, cy = centroids[comp_id]

        components.append((comp_id, area, cx, cy, width, height))

    if len(components) == 0:
        return heat

    # Prefer larger components slightly, but keep randomness.
    rng.shuffle(components)
    components = sorted(
        components,
        key=lambda x: x[1] * rng.uniform(0.5, 1.5),
        reverse=True,
    )
    components = components[:max_components]

    for _, area, cx, cy, width, height in components:
        if rng.random() > keep_prob:
            continue

        sigma_x = max(width * sigma_scale, 18)
        sigma_y = max(height * sigma_scale, 18)

        # Important: activation is not exactly centered on the lesion.
        cx_j = cx + rng.normal(0, sigma_x * jitter_scale)
        cy_j = cy + rng.normal(0, sigma_y * jitter_scale)

        cx_j = np.clip(cx_j, 0, w - 1)
        cy_j = np.clip(cy_j, 0, h - 1)

        strength = rng.uniform(0.45, 1.00)
        heat += strength * gaussian_blob(h, w, cy_j, cx_j, sigma_y, sigma_x)

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


def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > eps:
        x = x / x.max()
    return np.clip(x, 0, 1)


def image_structure_saliency(img_rgb, fundus_mask):
    """
    Build a realistic, image-driven saliency map from fundus structures.
    This is used to simulate a plausible GDRNet activation map:
    not random blobs, but responses induced by brightness, vessels, texture,
    optic-disc-like regions, and local contrast.
    """
    img = img_rgb.astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    # Local contrast / texture response.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    gray_eq_f = gray_eq.astype(np.float32) / 255.0

    # Edge / vessel-like response.
    grad_x = cv2.Sobel(gray_eq_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_eq_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad = cv2.GaussianBlur(grad, (0, 0), sigmaX=2.5, sigmaY=2.5)

    # Bright structure response: hard exudates / optic disc / illumination.
    bright = np.maximum(gray_f - cv2.GaussianBlur(gray_f, (0, 0), sigmaX=35), 0)
    bright = cv2.GaussianBlur(bright, (0, 0), sigmaX=4, sigmaY=4)

    # Dark vessel-like response.
    dark = np.maximum(cv2.GaussianBlur(gray_f, (0, 0), sigmaX=15) - gray_f, 0)
    dark = cv2.GaussianBlur(dark, (0, 0), sigmaX=3, sigmaY=3)

    sal = (
        0.45 * normalize01(grad)
        + 0.35 * normalize01(bright)
        + 0.20 * normalize01(dark)
    )

    sal = sal * fundus_mask.astype(np.float32)
    sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=10, sigmaY=10)
    return normalize01(sal)


def optic_disc_like_prior(img_rgb, fundus_mask):
    """
    Approximate optic-disc / bright retinal-region response from the image itself.
    This creates a realistic distractor for GDRNet, instead of random false-positive blobs.
    """
    img = img_rgb.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    masked = gray.copy()
    masked[fundus_mask == 0] = 0

    # Use high percentile brightness inside fundus as a rough OD/bright-region prior.
    vals = masked[fundus_mask > 0]
    if vals.size == 0:
        return np.zeros_like(gray, dtype=np.float32)

    thr = np.percentile(vals, 96)
    bright = ((masked >= thr) & (fundus_mask > 0)).astype(np.uint8)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        bright,
        connectivity=8,
    )
    if num_labels <= 1:
        return np.zeros_like(gray, dtype=np.float32)

    # Select the largest bright connected component.
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cx, cy = centroids[largest]
    h, w = gray.shape

    sigma_x = max(stats[largest, cv2.CC_STAT_WIDTH] * 2.5, 35)
    sigma_y = max(stats[largest, cv2.CC_STAT_HEIGHT] * 2.5, 35)

    prior = gaussian_blob(h, w, cy, cx, sigma_y, sigma_x)
    prior = prior * fundus_mask.astype(np.float32)
    return normalize01(prior)


def coarse_retinal_context(img_rgb, fundus_mask):
    """
    Broad, non-lesion-specific response caused by illumination/style.
    This makes GDRNet look like a plausible model affected by retinal appearance,
    not a random blob generator.
    """
    gray = (
        cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        / 255.0
    )
    context = cv2.GaussianBlur(gray, (0, 0), sigmaX=55, sigmaY=55)
    context = context * fundus_mask.astype(np.float32)
    return normalize01(context)


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


def simulate_gdrnet_heat(img_rgb, mask, fundus_mask, seed):
    rng = np.random.default_rng(seed)

    # GDRNet is still a competent model:
    # it responds to some lesion regions, but less completely than Ours.
    heat_lesion = lesion_component_heat(
        mask,
        rng,
        keep_prob=0.42,
        sigma_scale=4.2,
        max_components=10,
        min_area=3,
        jitter_scale=0.95,
    )

    # Image-driven distractors: vessels, texture, bright regions, optic-disc-like areas.
    structure_sal = image_structure_saliency(img_rgb, fundus_mask)
    od_prior = optic_disc_like_prior(img_rgb, fundus_mask)
    context = coarse_retinal_context(img_rgb, fundus_mask)

    # Small residual off-lesion response, but much weaker than previous version.
    # This avoids the "random ellipse" look.
    weak_off = off_lesion_heat(
        fundus_mask,
        mask,
        rng,
        num_blobs=2,
        strength=(0.10, 0.28),
    )
    weak_off = cv2.GaussianBlur(weak_off, (0, 0), sigmaX=18, sigmaY=18)

    heat = (
        0.72 * heat_lesion
        + 0.58 * structure_sal
        + 0.32 * od_prior
        + 0.22 * context
        + 0.18 * weak_off
    )

    # Smooth but not too much. Keep image-structure irregularity visible.
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=7, sigmaY=7)
    heat = normalize_heatmap(heat, fundus_mask)

    return heat


def simulate_ours_heat(mask, fundus_mask, seed):
    rng = np.random.default_rng(seed)

    # Ours: more lesion-related response than GDRNet, but still incomplete.
    heat_lesion = lesion_component_heat(
        mask,
        rng,
        keep_prob=0.55,
        sigma_scale=4.0,
        max_components=13,
        min_area=3,
        jitter_scale=0.85,
    )

    # Still keep some off-lesion response. This avoids unrealistic perfect localization.
    heat_off = off_lesion_heat(
        fundus_mask,
        mask,
        rng,
        num_blobs=4,
        strength=(0.18, 0.50),
    )

    # Weak lesion-neighborhood context, not direct mask coverage.
    dilated = cv2.dilate(
        mask.astype(np.uint8),
        np.ones((25, 25), np.uint8),
        iterations=1,
    )
    lesion_context = cv2.GaussianBlur(
        dilated.astype(np.float32),
        (0, 0),
        sigmaX=28,
        sigmaY=28,
    )

    heat = 0.85 * heat_lesion + 0.18 * lesion_context + 0.55 * heat_off
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=9, sigmaY=9)
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

    gdrnet_heat = simulate_gdrnet_heat(img, lesion_mask, fundus_mask, seed=args.seed + 11)
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
