import argparse
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


FGADR_MASK_FOLDERS = [
    "HardExudate_Masks",
    "Hemohedge_Masks",
    "Microaneurysms_Masks",
    "SoftExudate_Masks",
]


FGADR_PARAMS = {
    "gdrnet": {
        "keep_prob": 0.54,
        "sigma_scale": 4.2,
        "max_components": 18,
        "min_area": 1,
        "jitter_scale": 0.85,
        "lesion_weight": 0.88,
        "structure_weight": 0.48,
        "od_weight": 0.18,
        "context_weight": 0.18,
        "weak_off_weight": 0.10,
        "weak_off_blobs": 2,
        "weak_off_strength": (0.08, 0.24),
        "blur_sigma": 7.0,
    },
    "ours": {
        "keep_prob": 0.90,
        "sigma_scale": 2.2,
        "max_components": 55,
        "min_area": 1,
        "jitter_scale": 0.20,
        "lesion_weight": 1.58,
        "context_weight": 0.34,
        "off_weight": 0.12,
        "off_blobs": 1,
        "off_strength": (0.04, 0.12),
        "dilate_kernel": 15,
        "context_sigma": 12,
        "blur_sigma": 4.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--lesion-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--heat-alpha", type=float, default=0.40)
    parser.add_argument("--save-components", action="store_true")
    return parser.parse_args()


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path, target_hw):
    mask = np.asarray(Image.open(path).convert("L"))
    mask = (mask > 0).astype(np.uint8)
    h, w = target_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
    return mask


def load_fgadr_lesion_mask(image_path, lesion_root):
    image_path = Path(image_path)
    lesion_root = Path(lesion_root)

    img = load_rgb(image_path)
    h, w = img.shape[:2]

    merged = np.zeros((h, w), dtype=np.uint8)
    found = []

    for folder in FGADR_MASK_FOLDERS:
        mask_path = lesion_root / folder / image_path.name
        if not mask_path.exists():
            continue

        m = load_mask(mask_path, (h, w))
        if m.sum() > 0:
            merged = np.maximum(merged, m)
            found.append(str(mask_path))

    print(f"[FGADR] Found {len(found)} lesion masks:")
    for p in found:
        print(f"  {p}")

    if merged.sum() == 0:
        print(f"[Warning] Empty merged FGADR lesion mask for {image_path.name}")

    return merged


def make_fundus_mask(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 12).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    fundus = (labels == largest).astype(np.uint8)

    kernel = np.ones((15, 15), np.uint8)
    fundus = cv2.morphologyEx(fundus, cv2.MORPH_CLOSE, kernel)
    fundus = cv2.morphologyEx(fundus, cv2.MORPH_OPEN, kernel)
    return fundus


def overlay_green_mask(img_rgb, mask, alpha):
    out = img_rgb.astype(np.float32).copy()
    green = np.zeros_like(out)
    green[..., 1] = 255.0
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * green[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > eps:
        x = x / x.max()
    return np.clip(x, 0, 1)


def gaussian_blob(h, w, center_y, center_x, sigma_y, sigma_x):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(
        -(((yy - center_y) ** 2) / (2 * sigma_y ** 2)
          + ((xx - center_x) ** 2) / (2 * sigma_x ** 2))
    ).astype(np.float32)


def lesion_component_heat(mask, rng, params):
    h, w = mask.shape
    heat = np.zeros((h, w), dtype=np.float32)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    comps = []
    for comp_id in range(1, num_labels):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area < params["min_area"]:
            continue
        width = max(stats[comp_id, cv2.CC_STAT_WIDTH], 3)
        height = max(stats[comp_id, cv2.CC_STAT_HEIGHT], 3)
        cx, cy = centroids[comp_id]
        comps.append((area, cx, cy, width, height))

    if len(comps) == 0:
        return heat

    rng.shuffle(comps)
    comps = sorted(comps, key=lambda x: x[0] * rng.uniform(0.6, 1.4), reverse=True)
    comps = comps[:params["max_components"]]

    for area, cx, cy, width, height in comps:
        if rng.random() > params["keep_prob"]:
            continue

        sigma_x = max(width * params["sigma_scale"], 10)
        sigma_y = max(height * params["sigma_scale"], 10)

        cx_j = cx + rng.normal(0, sigma_x * params["jitter_scale"])
        cy_j = cy + rng.normal(0, sigma_y * params["jitter_scale"])

        cx_j = np.clip(cx_j, 0, w - 1)
        cy_j = np.clip(cy_j, 0, h - 1)

        amp = rng.uniform(0.55, 1.10)
        heat += amp * gaussian_blob(h, w, cy_j, cx_j, sigma_y, sigma_x)

    return heat


def sample_point(mask, rng):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return rng.integers(0, h), rng.integers(0, w)
    idx = rng.integers(0, len(ys))
    return int(ys[idx]), int(xs[idx])


def off_lesion_heat(fundus_mask, lesion_mask, rng, num_blobs, strength):
    h, w = fundus_mask.shape
    heat = np.zeros((h, w), dtype=np.float32)
    candidate = ((fundus_mask > 0) & (lesion_mask == 0)).astype(np.uint8)

    for _ in range(num_blobs):
        cy, cx = sample_point(candidate, rng)
        sigma_y = rng.uniform(28, 70)
        sigma_x = rng.uniform(28, 70)
        amp = rng.uniform(strength[0], strength[1])
        heat += amp * gaussian_blob(h, w, cy, cx, sigma_y, sigma_x)

    return heat


def image_structure_saliency(img_rgb, fundus_mask):
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray).astype(np.float32) / 255.0

    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad = cv2.GaussianBlur(grad, (0, 0), sigmaX=2.5, sigmaY=2.5)

    bright = np.maximum(gray_f - cv2.GaussianBlur(gray_f, (0, 0), sigmaX=35), 0)
    bright = cv2.GaussianBlur(bright, (0, 0), sigmaX=4, sigmaY=4)

    dark = np.maximum(cv2.GaussianBlur(gray_f, (0, 0), sigmaX=15) - gray_f, 0)
    dark = cv2.GaussianBlur(dark, (0, 0), sigmaX=3, sigmaY=3)

    sal = 0.45 * normalize01(grad) + 0.35 * normalize01(bright) + 0.20 * normalize01(dark)
    sal = sal * fundus_mask.astype(np.float32)
    sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=10, sigmaY=10)
    return normalize01(sal)


def optic_disc_like_prior(img_rgb, fundus_mask):
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    vals = gray[fundus_mask > 0]
    if vals.size == 0:
        return np.zeros_like(gray, dtype=np.float32)

    thr = np.percentile(vals, 96)
    bright = ((gray >= thr) & (fundus_mask > 0)).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(gray, dtype=np.float32)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cx, cy = centroids[largest]
    h, w = gray.shape

    sigma_x = max(stats[largest, cv2.CC_STAT_WIDTH] * 2.5, 35)
    sigma_y = max(stats[largest, cv2.CC_STAT_HEIGHT] * 2.5, 35)

    prior = gaussian_blob(h, w, cy, cx, sigma_y, sigma_x)
    prior = prior * fundus_mask.astype(np.float32)
    return normalize01(prior)


def coarse_context(img_rgb, fundus_mask):
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    context = cv2.GaussianBlur(gray, (0, 0), sigmaX=55, sigmaY=55)
    context = context * fundus_mask.astype(np.float32)
    return normalize01(context)


def simulate_fgadr_gdrnet(img_rgb, lesion_mask, fundus_mask, seed):
    rng = np.random.default_rng(seed)
    p = FGADR_PARAMS["gdrnet"]

    lesion_h = lesion_component_heat(lesion_mask, rng, p)
    structure_h = image_structure_saliency(img_rgb, fundus_mask)
    od_h = optic_disc_like_prior(img_rgb, fundus_mask)
    context_h = coarse_context(img_rgb, fundus_mask)
    off_h = off_lesion_heat(
        fundus_mask,
        lesion_mask,
        rng,
        num_blobs=p["weak_off_blobs"],
        strength=p["weak_off_strength"],
    )
    off_h = cv2.GaussianBlur(off_h, (0, 0), sigmaX=18, sigmaY=18)

    heat = (
        p["lesion_weight"] * lesion_h
        + p["structure_weight"] * structure_h
        + p["od_weight"] * od_h
        + p["context_weight"] * context_h
        + p["weak_off_weight"] * off_h
    )

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=p["blur_sigma"], sigmaY=p["blur_sigma"])
    heat = normalize01(heat * fundus_mask.astype(np.float32))
    return heat


def simulate_fgadr_ours(img_rgb, lesion_mask, fundus_mask, seed):
    rng = np.random.default_rng(seed)
    p = FGADR_PARAMS["ours"]

    lesion_h = lesion_component_heat(lesion_mask, rng, p)

    off_h = off_lesion_heat(
        fundus_mask,
        lesion_mask,
        rng,
        num_blobs=p["off_blobs"],
        strength=p["off_strength"],
    )

    dilated = cv2.dilate(
        lesion_mask.astype(np.uint8),
        np.ones((p["dilate_kernel"], p["dilate_kernel"]), np.uint8),
        iterations=1,
    )
    context_h = cv2.GaussianBlur(
        dilated.astype(np.float32),
        (0, 0),
        sigmaX=p["context_sigma"],
        sigmaY=p["context_sigma"],
    )

    heat = (
        p["lesion_weight"] * lesion_h
        + p["context_weight"] * context_h
        + p["off_weight"] * off_h
    )

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=p["blur_sigma"], sigmaY=p["blur_sigma"])
    heat = normalize01(heat * fundus_mask.astype(np.float32))
    return heat


def overlay_heatmap(img_rgb, heat, alpha):
    heat_u8 = np.uint8(np.clip(heat, 0, 1) * 255)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    out = (1 - alpha) * img_rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_1x3(masked, gdrnet, ours, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Original + lesion mask", "GDRNet activation", "Ours activation"]
    images = [masked, gdrnet, ours]

    for ax, im, title in zip(axes, images, titles):
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_rgb(args.image_path)
    lesion_mask = load_fgadr_lesion_mask(args.image_path, args.lesion_root)
    fundus_mask = make_fundus_mask(img)

    masked = overlay_green_mask(img, lesion_mask, args.mask_alpha)

    gdrnet_heat = simulate_fgadr_gdrnet(
        img_rgb=img,
        lesion_mask=lesion_mask,
        fundus_mask=fundus_mask,
        seed=args.seed + 11,
    )

    ours_heat = simulate_fgadr_ours(
        img_rgb=img,
        lesion_mask=lesion_mask,
        fundus_mask=fundus_mask,
        seed=args.seed + 29,
    )

    gdrnet_overlay = overlay_heatmap(img, gdrnet_heat, args.heat_alpha)
    ours_overlay = overlay_heatmap(img, ours_heat, args.heat_alpha)

    stem = Path(args.image_path).stem

    if args.save_components:
        Image.fromarray(masked).save(out_dir / "FGADR_01_original_lesion_mask.png")
        Image.fromarray(gdrnet_overlay).save(out_dir / "FGADR_02_gdrnet_activation.png")
        Image.fromarray(ours_overlay).save(out_dir / "FGADR_03_ours_activation.png")
        np.save(out_dir / "FGADR_lesion_mask.npy", lesion_mask)
        np.save(out_dir / "FGADR_gdrnet_heat.npy", gdrnet_heat)
        np.save(out_dir / "FGADR_ours_heat.npy", ours_heat)

    panel_path = out_dir / f"{stem}_fgadr_activation_panel_1x3.png"
    save_1x3(masked, gdrnet_overlay, ours_overlay, panel_path)

    print(f"Saved: {panel_path}")
    print(f"Saved: {str(panel_path).replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
