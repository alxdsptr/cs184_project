#!/usr/bin/env python3
"""Quick side-by-side compositor: 2x2 grid of mid-motion frames."""
import sys
from PIL import Image, ImageDraw, ImageFont

frames = [
    ("native",     "screenshots/native-shimmer/native-motion_000015.png"),
    ("dlss-SR",    "screenshots/dlss-shimmer-before/dlss-motion-before_000015.png"),
    ("rr-before",  "screenshots/rr-shimmer-before/rr-motion-before_000015.png"),
    ("rr-after",   "screenshots/rr-shimmer-fix1/rr-motion-fix1_000015.png"),
]
imgs = [(label, Image.open(path).convert("RGB")) for label, path in frames]
w, h = imgs[0][1].size
canvas = Image.new("RGB", (w * 2, h * 2), (0, 0, 0))
positions = [(0, 0), (w, 0), (0, h), (w, h)]
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 32)
except Exception:
    font = ImageFont.load_default()
for (label, im), (x, y) in zip(imgs, positions):
    canvas.paste(im, (x, y))
    # White stroke + black fill text
    draw.text((x + 12, y + 8), label, fill=(255, 255, 255),
              stroke_width=3, stroke_fill=(0, 0, 0), font=font)
out = "screenshots/comparison_side_by_side.png"
canvas.save(out)
print(f"wrote {out}")
