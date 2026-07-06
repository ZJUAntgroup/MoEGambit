#!/usr/bin/env python3
"""Create ICSE/IEEE two-column sized outputs for the MoEGambit figure."""

import os
import re
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation


OUT_DIR = Path("/Users/zds/bsr/log_analysis/figures")
BASE = "moegambit_runtime_architecture"
IEEE_TWO_COLUMN_WIDTH_IN = 7.16
POINTS_PER_INCH = 72

# draw.io exports the configured page, not just the visible figure.
# These bounds crop out the now-removed slide-style title and page whitespace.
CROP_X = 0.0
CROP_Y = 0.0
CROP_W = 1466.0
CROP_H = 560.0


def read_svg_canvas_size():
    src = OUT_DIR / f"{BASE}.svg"
    data = src.read_text(encoding="utf-8")
    viewbox_match = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', data)
    if not viewbox_match:
        raise RuntimeError(f"Could not find SVG viewBox in {src}")
    return data, float(viewbox_match.group(1)), float(viewbox_match.group(2))


def make_sized_svg():
    src = OUT_DIR / f"{BASE}.svg"
    dst = OUT_DIR / f"{BASE}_icse.svg"
    data, _, _ = read_svg_canvas_size()
    height_in = IEEE_TWO_COLUMN_WIDTH_IN * CROP_H / CROP_W

    data = re.sub(
        r'viewBox="0 0 [0-9.]+ [0-9.]+"',
        f'viewBox="{CROP_X:g} {CROP_Y:g} {CROP_W:g} {CROP_H:g}"',
        data,
        count=1,
    )

    data = re.sub(
        r'width="[^"]+"\s+height="[^"]+"',
        f'width="{IEEE_TWO_COLUMN_WIDTH_IN:.2f}in" height="{height_in:.2f}in"',
        data,
        count=1,
    )
    dst.write_text(data, encoding="utf-8")
    print(f"Wrote {dst}")


def make_sized_pdf():
    src = OUT_DIR / f"{BASE}.pdf"
    dst = OUT_DIR / f"{BASE}_icse.pdf"
    reader = PdfReader(str(src))
    page = reader.pages[0]

    source_page_width_pt = float(page.mediabox.width)
    source_page_height_pt = float(page.mediabox.height)
    _, svg_width_px, svg_height_px = read_svg_canvas_size()
    sx = source_page_width_pt / svg_width_px
    sy = source_page_height_pt / svg_height_px

    crop_x0 = CROP_X * sx
    crop_x1 = (CROP_X + CROP_W) * sx
    crop_y0 = (svg_height_px - (CROP_Y + CROP_H)) * sy
    crop_y1 = (svg_height_px - CROP_Y) * sy
    crop_width_pt = crop_x1 - crop_x0
    crop_height_pt = crop_y1 - crop_y0

    page.add_transformation(Transformation().translate(-crop_x0, -crop_y0))
    page.mediabox.lower_left = (0, 0)
    page.mediabox.upper_right = (crop_width_pt, crop_height_pt)
    page.cropbox.lower_left = (0, 0)
    page.cropbox.upper_right = (crop_width_pt, crop_height_pt)

    target_width_pt = IEEE_TWO_COLUMN_WIDTH_IN * POINTS_PER_INCH
    scale = target_width_pt / crop_width_pt
    target_height_pt = crop_height_pt * scale

    page.add_transformation(Transformation().scale(scale, scale))
    page.mediabox.lower_left = (0, 0)
    page.mediabox.upper_right = (target_width_pt, target_height_pt)
    page.cropbox.lower_left = (0, 0)
    page.cropbox.upper_right = (target_width_pt, target_height_pt)

    writer = PdfWriter()
    writer.add_page(page)
    with dst.open("wb") as f:
        writer.write(f)
    print(f"Wrote {dst}")
    print(f"PDF size: {IEEE_TWO_COLUMN_WIDTH_IN:.2f}in x {target_height_pt / POINTS_PER_INCH:.2f}in")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_sized_svg()
    make_sized_pdf()
