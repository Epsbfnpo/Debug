import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid-original", type=str, required=True)
    parser.add_argument("--idrid-gdrnet", type=str, required=True)
    parser.add_argument("--idrid-ours", type=str, required=True)
    parser.add_argument("--fgadr-original", type=str, required=True)
    parser.add_argument("--fgadr-gdrnet", type=str, required=True)
    parser.add_argument("--fgadr-ours", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--cell-width", type=int, default=520)
    parser.add_argument("--cell-height", type=int, default=390)
    parser.add_argument("--title-height", type=int, default=55)
    parser.add_argument("--row-label-width", type=int, default=85)
    parser.add_argument("--gap", type=int, default=10)
    return parser.parse_args()


def load_and_fit(path, width, height):
    img = Image.open(path).convert("RGB")
    img.thumbnail((width, height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (width, height), "white")
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_centered_text(draw, box, text, font, fill="black"):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - tw) // 2
    y = y0 + (y1 - y0 - th) // 2
    draw.text((x, y), text, font=font, fill=fill)


def main():
    args = parse_args()

    col_titles = ["Original + lesion mask", "GDRNet activation", "Ours activation"]
    row_titles = ["IDRID", "FGADR"]

    paths = [
        [args.idrid_original, args.idrid_gdrnet, args.idrid_ours],
        [args.fgadr_original, args.fgadr_gdrnet, args.fgadr_ours],
    ]

    width = args.row_label_width + 3 * args.cell_width + 2 * args.gap
    height = args.title_height + 2 * args.cell_height + args.gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        row_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except Exception:
        title_font = ImageFont.load_default()
        row_font = ImageFont.load_default()

    # Column titles.
    for c, title in enumerate(col_titles):
        x0 = args.row_label_width + c * (args.cell_width + args.gap)
        x1 = x0 + args.cell_width
        draw_centered_text(draw, (x0, 0, x1, args.title_height), title, title_font)

    # Images and row labels.
    for r in range(2):
        y0 = args.title_height + r * (args.cell_height + args.gap)

        label_img = Image.new("RGB", (args.row_label_width, args.cell_height), "white")
        label_draw = ImageDraw.Draw(label_img)
        draw_centered_text(
            label_draw,
            (0, 0, args.row_label_width, args.cell_height),
            row_titles[r],
            row_font,
        )
        label_img = label_img.rotate(90, expand=True)
        label_img = label_img.resize(
            (args.row_label_width, args.cell_height),
            Image.Resampling.BICUBIC,
        )
        canvas.paste(label_img, (0, y0))

        for c in range(3):
            x0 = args.row_label_width + c * (args.cell_width + args.gap)
            cell = load_and_fit(paths[r][c], args.cell_width, args.cell_height)
            canvas.paste(cell, (x0, y0))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    canvas.save(out_path.with_suffix(".pdf"))

    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
