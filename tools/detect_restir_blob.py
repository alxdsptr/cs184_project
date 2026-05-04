#!/usr/bin/env python3
"""detect_restir_blob.py — detect ReSTIR temporal "flash-and-decay" blobs.

Given a directory of consecutive captured PNGs (one per frame, static camera
preferred), this script

  1. computes per-frame mean luminance,
  2. computes a per-pixel temporal MEDIAN luminance over the whole sequence
     (median, not mean — mean would be biased by the spike itself),
  3. flags pixels whose per-frame luminance exceeds k * median for at least
     `min_consecutive` consecutive frames (the canonical ReSTIR-blob signature),
  4. writes:
       <out>/<tag>_metrics.csv      — per-frame {mean_lum, blob_pixel_count}
       <out>/<tag>_summary.txt      — total blob pixels, peak frame, etc.
       <out>/<tag>_heatmap.png      — log-scale heatmap of the temporal
                                      max(luminance / median) per pixel.
       <out>/<tag>_blob_mask.png    — binary mask of pixels that ever blobbed.

Usage:
    python tools/detect_restir_blob.py <dir> [--tag PREFIX] [--ratio 5.0]
                                              [--min-consecutive 2] [--out DIR]

The `--tag` filters the input PNGs by filename prefix (e.g. "restir-di"
captures from the capture script are named restir-di_NNNNNN.png). If
omitted, every *.png in the directory is treated as one sequence.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required (pip install pillow)", file=sys.stderr)
    sys.exit(1)

LUM_R, LUM_G, LUM_B = 0.2126, 0.7152, 0.0722


def load_png(p: Path) -> np.ndarray:
    """Return RGB float32 array, normalised to [0,1]. Strips alpha."""
    img = Image.open(p).convert("RGB")
    a = np.asarray(img, dtype=np.float32) / 255.0
    return a


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * LUM_R + rgb[..., 1] * LUM_G + rgb[..., 2] * LUM_B


def collect_frames(d: Path, tag: str | None):
    pat = re.compile(r"\d+")
    files = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        if tag is not None and not f.stem.startswith(tag):
            continue
        # Sort by trailing integer if present, else by name.
        m = pat.findall(f.stem)
        key = (int(m[-1]), f.name) if m else (-1, f.name)
        files.append((key, f))
    files.sort()
    return [f for _, f in files]


def detect(args):
    in_dir = Path(args.input).resolve()
    out_dir = Path(args.out).resolve() if args.out else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_frames(in_dir, args.tag)
    if len(files) < 5:
        print(f"WARNING: only {len(files)} frames found "
              f"(tag={args.tag}); need ≥5 for meaningful detection.")
    if not files:
        print("ERROR: no frames found.")
        return 2

    print(f"[detect_restir_blob] loaded {len(files)} frames "
          f"({files[0].name} … {files[-1].name})")

    # Stack: (T, H, W) float32 luminance. RGB stack would be too big to keep
    # for full HD; compute luminance per-frame.
    first = load_png(files[0])
    H, W, _ = first.shape
    T = len(files)
    stack = np.empty((T, H, W), dtype=np.float32)
    stack[0] = luminance(first)
    for i in range(1, T):
        stack[i] = luminance(load_png(files[i]))

    per_frame_mean = stack.mean(axis=(1, 2))

    # Per-pixel temporal MEDIAN — robust to flash spikes.
    # np.median over axis 0 returns shape (H, W).
    median = np.median(stack, axis=0)
    # Add a tiny floor so dark pixels (median≈0) don't trigger every spike.
    floor = max(0.005, np.median(median))  # 0.5% reflectance or scene-median
    denom = np.maximum(median, floor)

    ratio = stack / denom[None, :, :]            # (T, H, W)
    is_hot = ratio >= args.ratio                 # boolean

    # Per-pixel: any run of `min_consecutive` hot frames?
    if args.min_consecutive <= 1:
        ever_blob = is_hot.any(axis=0)
    else:
        # Temporal convolution-style: a run-length≥k means there is a window
        # of k consecutive True. Compute via cumulative sum trick.
        k = args.min_consecutive
        # Pad with zeros so the rolling window aligns.
        pad = np.zeros((1, H, W), dtype=np.int32)
        cum = np.cumsum(np.concatenate([pad, is_hot.astype(np.int32)],
                                        axis=0),
                         axis=0)
        # Window sum of length k: cum[t+k] - cum[t]; ever blob = max ≥ k.
        if T >= k:
            window = cum[k:] - cum[:-k]          # (T-k+1, H, W)
            ever_blob = (window >= k).any(axis=0)
        else:
            ever_blob = np.zeros((H, W), dtype=bool)

    # Per-frame "blob pixel count" = pixels that are hot in this frame AND
    # are flagged as ever_blob (so transient single-frame noise is excluded).
    blob_per_frame = (is_hot & ever_blob[None, :, :]).sum(axis=(1, 2))

    # Temporal max ratio per pixel — the heatmap.
    max_ratio = ratio.max(axis=0)
    max_ratio_clamped = np.minimum(max_ratio, 50.0)  # cap for vis

    # ── Outputs ─────────────────────────────────────────────────────────
    tag_str = args.tag or "all"

    csv_path = out_dir / f"{tag_str}_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("frame,filename,mean_lum,blob_pixel_count\n")
        for i, f in enumerate(files):
            fh.write(f"{i},{f.name},{per_frame_mean[i]:.6f},"
                     f"{int(blob_per_frame[i])}\n")
    print(f"[detect_restir_blob] wrote {csv_path}")

    # Summary
    total_blob_px = int(ever_blob.sum())
    peak_frame = int(np.argmax(blob_per_frame))
    peak_count = int(blob_per_frame.max())
    summary = (
        f"input            : {in_dir}\n"
        f"tag              : {tag_str}\n"
        f"frames           : {T}\n"
        f"resolution       : {W}x{H} ({W*H} pixels)\n"
        f"ratio threshold  : {args.ratio}× per-pixel temporal median\n"
        f"min consecutive  : {args.min_consecutive}\n"
        f"floor            : {floor:.4f}\n"
        f"---\n"
        f"total ever-blob pixels   : {total_blob_px} "
        f"({100.0 * total_blob_px / (W*H):.4f}%)\n"
        f"peak blob frame          : {peak_frame} ({files[peak_frame].name})\n"
        f"peak blob pixel count    : {peak_count} "
        f"({100.0 * peak_count / (W*H):.4f}%)\n"
        f"mean blob pixels / frame : {blob_per_frame.mean():.1f}\n"
        f"per-frame mean lum       : "
        f"min={per_frame_mean.min():.4f} "
        f"max={per_frame_mean.max():.4f} "
        f"std={per_frame_mean.std():.4f}\n"
    )
    sum_path = out_dir / f"{tag_str}_summary.txt"
    sum_path.write_text(summary, encoding="utf-8")
    print(f"[detect_restir_blob] wrote {sum_path}")
    print(summary)

    # Heatmap: log-scale of max_ratio_clamped, mapped to red→yellow.
    hm = np.log1p(max_ratio_clamped) / np.log1p(50.0)   # [0,1]
    hm = np.clip(hm, 0, 1)
    # Simple red-hot colormap: low=black, mid=red, high=yellow/white.
    r = np.clip(hm * 2, 0, 1)
    g = np.clip(hm * 2 - 0.5, 0, 1)
    b = np.clip(hm * 2 - 1.0, 0, 1)
    hm_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    Image.fromarray(hm_rgb).save(out_dir / f"{tag_str}_heatmap.png")
    print(f"[detect_restir_blob] wrote {out_dir / (tag_str + '_heatmap.png')}")

    # Blob mask: pure binary.
    mask = (ever_blob.astype(np.uint8) * 255)
    Image.fromarray(mask).save(out_dir / f"{tag_str}_blob_mask.png")
    print(f"[detect_restir_blob] wrote {out_dir / (tag_str + '_blob_mask.png')}")

    return 0 if total_blob_px == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="directory containing PNGs")
    p.add_argument("--tag", default=None,
                   help="filter PNGs by filename prefix (e.g. 'restir-di')")
    p.add_argument("--ratio", type=float, default=5.0,
                   help="ratio threshold over per-pixel temporal median (default 5.0)")
    p.add_argument("--min-consecutive", type=int, default=2,
                   help="min run length of hot frames to count as blob (default 2)")
    p.add_argument("--out", default=None,
                   help="output directory (default: same as input)")
    args = p.parse_args()
    return detect(args)


if __name__ == "__main__":
    sys.exit(main())
