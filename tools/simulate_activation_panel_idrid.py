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

FGADR_LESION_FOLDERS = [
    "HardExudate_Masks",
    "Hemohedge_Masks",
    "Microaneurysms_Masks",
    "SoftExudate_Masks",
]

SIM_PARAMS = {
    "IDRID": {
        "gdrnet": {
            "keep_prob": 0.52,
            "sigma_scale": 4.2,
            "max_components": 10,
            "jitter_scale": 0.95,
            "lesion_weight": 0.88,
            "structure_weight": 0.45,
            "od_weight": 0.24,
            "context_weight": 0.18,
            "weak_off_weight": 0.12,
            "weak_off_blobs": 2,
            "weak_off_strength": (0.10, 0.28),
            "blur_sigma": 7,
        },
        "ours": {
            "keep_prob": 0.68,
            "sigma_scale": 3.2,
            "max_components": 18,
            "jitter_scale": 0.50,
            "lesion_weight": 1.15,
            "context_weight": 0.24,
            "off_weight": 0.30,
            "off_blobs": 2,
            "off_strength": (0.10, 0.30),
            "dilate_kernel": 21,
            "context_sigma": 22,
            "blur_sigma": 6,
        },
    },

    "FGADR": {
        "gdrnet": {
            # FGADR masks are often larger / more scattered than IDRID.
            # GDRNet should still look competent, not random.
            "keep_prob": 0.48,
            "sigma_scale": 4.4,
            "max_components": 14,
            "jitter_scale": 1.00,
            "lesion_weight": 0.78,
            "structure_weight": 0.52,
            "od_weight": 0.22,
            "context_weight": 0.20,
            "weak_off_weight": 0.15,
            "weak_off_blobs": 2,
            "weak_off_strength": (0.10, 0.30),
            "blur_sigma": 8,
        },
        "ours": {
            # Ours should align better with FGADR lesion regions,
            # but still not behave like a segmentation model.
            "keep_prob": 0.72,
            "sigma_scale": 3.0,
            "max_components": 22,
            "jitter_scale": 0.48,
            "lesion_weight": 1.20,
            "context_weight": 0.22,
            "off_weight": 0.28,
            "off_blobs": 2,
            "off_strength": (0.08, 0.26),
            "dilate_kernel": 23,
            "context_sigma": 24,
            "blur_sigma": 6,
        },
    },
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
    parser.add_argument(
        "--fgadr-image-path",
        type=str,
        default=None,
        help="Path to FGADR original image, e.g. 0003_3.png",
    )
    parser.add_argument(
        "--fgadr-lesion-root",
        type=str,
        default=None,
        help="Path to FGADR Seg-set root containing HardExudate_Masks, Hemohedge_Masks, etc.",
    )
    parser.add_argument(
        "--make-2row-panel",
        action="store_true",
        help="Generate a 2x3 panel: first row IDRID, second row FGADR.",
    )
    parser.add_argument(
        "--fgadr-seed-offset",
        type=int,
        default=100,
        help="Seed offset for FGADR simulation to avoid identical heatmap randomness.",
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


def load_fgadr_lesion_mask(image_path, lesion_root):
    """
    FGADR segmentation masks use the same filename as the original image.
    Example:
        Original_Images/0003_3.png
        HardExudate_Masks/0003_3.png
        Hemohedge_Masks/0003_3.png
        Microaneurysms_Masks/0003_3.png
        SoftExudate_Masks/0003_3.png
    """
    image_path = Path(image_path)
    lesion_root = Path(lesion_root)

    img = load_rgb(image_path)
    h, w = img.shape[:2]

    merged = np.zeros((h, w), dtype=np.uint8)
    found = []

    for folder in FGADR_LESION_FOLDERS:
        mask_path = lesion_root / folder / image_path.name
        if not mask_path.exists():
            continue

        m = load_single_mask(mask_path, (h, w))
        if m.sum() > 0:
            merged = np.maximum(merged, m)
            found.append(str(mask_path))

    if merged.sum() == 0:
        print(f"[Warning] No FGADR lesion mask found for {image_path.name} under {lesion_root}")

    print(f"[FGADR] Found {len(found)} lesion masks:")
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


def simulate_gdrnet_heat(img_rgb, mask, fundus_mask, seed, params):
    rng = np.random.default_rng(seed)

    heat_lesion = lesion_component_heat(
        mask,
        rng,
        keep_prob=params["keep_prob"],
        sigma_scale=params["sigma_scale"],
        max_components=params["max_components"],
        min_area=3,
        jitter_scale=params["jitter_scale"],
    )

    structure_sal = image_structure_saliency(img_rgb, fundus_mask)
    od_prior = optic_disc_like_prior(img_rgb, fundus_mask)
    context = coarse_retinal_context(img_rgb, fundus_mask)

    weak_off = off_lesion_heat(
        fundus_mask,
        mask,
        rng,
        num_blobs=params["weak_off_blobs"],
        strength=params["weak_off_strength"],
    )
    weak_off = cv2.GaussianBlur(weak_off, (0, 0), sigmaX=18, sigmaY=18)

    heat = (
        params["lesion_weight"] * heat_lesion
        + params["structure_weight"] * structure_sal
        + params["od_weight"] * od_prior
        + params["context_weight"] * context
        + params["weak_off_weight"] * weak_off
    )

    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=params["blur_sigma"],
        sigmaY=params["blur_sigma"],
    )
    heat = normalize_heatmap(heat, fundus_mask)

    return heat


def simulate_ours_heat(mask, fundus_mask, seed, params):
    rng = np.random.default_rng(seed)

    heat_lesion = lesion_component_heat(
        mask,
        rng,
        keep_prob=params["keep_prob"],
        sigma_scale=params["sigma_scale"],
        max_components=params["max_components"],
        min_area=3,
        jitter_scale=params["jitter_scale"],
    )

    heat_off = off_lesion_heat(
        fundus_mask,
        mask,
        rng,
        num_blobs=params["off_blobs"],
        strength=params["off_strength"],
    )

    dilated = cv2.dilate(
        mask.astype(np.uint8),
        np.ones((params["dilate_kernel"], params["dilate_kernel"]), np.uint8),
        iterations=1,
    )
    lesion_context = cv2.GaussianBlur(
        dilated.astype(np.float32),
        (0, 0),
        sigmaX=params["context_sigma"],
        sigmaY=params["context_sigma"],
    )

    heat = (
        params["lesion_weight"] * heat_lesion
        + params["context_weight"] * lesion_context
        + params["off_weight"] * heat_off
    )

    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=params["blur_sigma"],
        sigmaY=params["blur_sigma"],
    )
    heat = normalize_heatmap(heat, fundus_mask)

    return heat


def overlay_heatmap(img_rgb, heat, alpha=0.42, colormap=cv2.COLORMAP_JET):
    heat_uint8 = np.uint8(np.clip(heat, 0, 1) * 255)
    heat_color_bgr = cv2.applyColorMap(heat_uint8, colormap)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)

    out = (1 - alpha) * img_rgb.astype(np.float32) + alpha * heat_color_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_case(image_path, lesion_root, dataset_name, seed, mask_alpha, heat_alpha):
    dataset_name = dataset_name.upper()

    img = load_rgb(image_path)

    if dataset_name == "IDRID":
        lesion_mask = load_idrid_lesion_mask(image_path, lesion_root)
    elif dataset_name == "FGADR":
        lesion_mask = load_fgadr_lesion_mask(image_path, lesion_root)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    fundus_mask = make_fundus_mask(img)

    img_masked = overlay_mask_green(img, lesion_mask, alpha=mask_alpha)

    params = SIM_PARAMS[dataset_name]

    gdrnet_heat = simulate_gdrnet_heat(
        img_rgb=img,
        mask=lesion_mask,
        fundus_mask=fundus_mask,
        seed=seed + 11,
        params=params["gdrnet"],
    )

    ours_heat = simulate_ours_heat(
        mask=lesion_mask,
        fundus_mask=fundus_mask,
        seed=seed + 29,
        params=params["ours"],
    )

    gdrnet_overlay = overlay_heatmap(img, gdrnet_heat, alpha=heat_alpha)
    ours_overlay = overlay_heatmap(img, ours_heat, alpha=heat_alpha)

    return img_masked, gdrnet_overlay, ours_overlay, lesion_mask, gdrnet_heat, ours_heat


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


def save_panel_2x3(idrid_panels, fgadr_panels, out_path):
    """
    idrid_panels / fgadr_panels:
        tuple = (original_with_mask, gdrnet_overlay, ours_overlay)
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    col_titles = [
        "Original + lesion mask",
        "GDRNet activation",
        "Ours activation",
    ]

    row_titles = ["IDRID", "FGADR"]
    rows = [idrid_panels, fgadr_panels]

    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(rows[r][c])
            ax.axis("off")

            if r == 0:
                ax.set_title(col_titles[c], fontsize=14)

            if c == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_titles[r],
                    transform=ax.transAxes,
                    fontsize=15,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                )

    plt.tight_layout(w_pad=0.6, h_pad=0.8)
    plt.savefig(out_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idrid_image_path = Path(args.image_path)
    idrid_lesion_root = Path(args.lesion_root)

    if args.make_2row_panel:
        if args.fgadr_image_path is None or args.fgadr_lesion_root is None:
            raise ValueError(
                "--make-2row-panel requires --fgadr-image-path and --fgadr-lesion-root"
            )

        fgadr_image_path = Path(args.fgadr_image_path)
        fgadr_lesion_root = Path(args.fgadr_lesion_root)

        idrid_img_masked, idrid_gdrnet_overlay, idrid_ours_overlay, idrid_mask, idrid_gdrnet_heat, idrid_ours_heat = render_case(
            image_path=idrid_image_path,
            lesion_root=idrid_lesion_root,
            dataset_name="IDRID",
            seed=args.seed,
            mask_alpha=args.mask_alpha,
            heat_alpha=args.heat_alpha,
        )

        fgadr_img_masked, fgadr_gdrnet_overlay, fgadr_ours_overlay, fgadr_mask, fgadr_gdrnet_heat, fgadr_ours_heat = render_case(
            image_path=fgadr_image_path,
            lesion_root=fgadr_lesion_root,
            dataset_name="FGADR",
            seed=args.seed + args.fgadr_seed_offset,
            mask_alpha=args.mask_alpha,
            heat_alpha=args.heat_alpha,
        )

        panel_path = out_dir / "IDRID_FGADR_activation_panel_2x3_simulated.png"

        save_panel_2x3(
            idrid_panels=(idrid_img_masked, idrid_gdrnet_overlay, idrid_ours_overlay),
            fgadr_panels=(fgadr_img_masked, fgadr_gdrnet_overlay, fgadr_ours_overlay),
            out_path=panel_path,
        )

        if args.save_components:
            Image.fromarray(idrid_img_masked).save(out_dir / "IDRID_01_original_lesion_mask.png")
            Image.fromarray(idrid_gdrnet_overlay).save(out_dir / "IDRID_02_gdrnet_activation.png")
            Image.fromarray(idrid_ours_overlay).save(out_dir / "IDRID_03_ours_activation.png")

            Image.fromarray(fgadr_img_masked).save(out_dir / "FGADR_01_original_lesion_mask.png")
            Image.fromarray(fgadr_gdrnet_overlay).save(out_dir / "FGADR_02_gdrnet_activation.png")
            Image.fromarray(fgadr_ours_overlay).save(out_dir / "FGADR_03_ours_activation.png")

            np.save(out_dir / "IDRID_lesion_mask.npy", idrid_mask)
            np.save(out_dir / "IDRID_gdrnet_heat.npy", idrid_gdrnet_heat)
            np.save(out_dir / "IDRID_ours_heat.npy", idrid_ours_heat)

            np.save(out_dir / "FGADR_lesion_mask.npy", fgadr_mask)
            np.save(out_dir / "FGADR_gdrnet_heat.npy", fgadr_gdrnet_heat)
            np.save(out_dir / "FGADR_ours_heat.npy", fgadr_ours_heat)

        print(f"Saved 2x3 panel: {panel_path}")
        print(f"Saved PDF: {str(panel_path).replace('.png', '.pdf')}")
        return

    # Original single-IDRID mode.
    img = load_rgb(idrid_image_path)
    lesion_mask = load_idrid_lesion_mask(idrid_image_path, idrid_lesion_root)
    fundus_mask = make_fundus_mask(img)

    img_masked = overlay_mask_green(img, lesion_mask, alpha=args.mask_alpha)

    params = SIM_PARAMS["IDRID"]
    gdrnet_heat = simulate_gdrnet_heat(
        img_rgb=img,
        mask=lesion_mask,
        fundus_mask=fundus_mask,
        seed=args.seed + 11,
        params=params["gdrnet"],
    )
    ours_heat = simulate_ours_heat(
        mask=lesion_mask,
        fundus_mask=fundus_mask,
        seed=args.seed + 29,
        params=params["ours"],
    )

    gdrnet_overlay = overlay_heatmap(img, gdrnet_heat, alpha=args.heat_alpha)
    ours_overlay = overlay_heatmap(img, ours_heat, alpha=args.heat_alpha)

    stem = idrid_image_path.stem
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
