#!/usr/bin/env python3
"""measure_shimmer.py

Quantify temporal stability ("shimmer") of a sequence of consecutive captured
frames. We compute frame-to-frame absolute-difference metrics: a stable
upscaler/denoiser should produce small frame-to-frame deltas under continuous
camera motion (the only valid signal is the camera motion itself, which
should be smooth). Shimmer manifests as pixels that flicker between frames
even when the camera moves smoothly.

Outputs (in --output-dir, default = input dir):

    shimmer.csv    Per-frame metrics:
                       frame_idx, mean_abs_diff, p50, p99, peak
                   followed by a SUMMARY row (mean of means, mean of p99,
                   peak of peaks, sequence length).

    shimmer_heatmap.png
                   Per-pixel max abs-diff across the whole sequence,
                   stretched to 0..255 with a viridis-style colormap.

Usage:
    python tools/measure_shimmer.py <dir> --pattern "dlss-motion_*.png"

Conventions:
    - "abs diff" is the per-pixel absolute difference between consecutive
      frames, averaged over RGB channels. Range [0, 255].
    - Higher value = more shimmer.
    - Compare numbers between modes captured with the same camera motion;
      the absolute number is only meaningful relative to a baseline.
"""

import argparse
import csv
import glob
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image


def load_frames(paths: List[str]) -> np.ndarray:
    """Load a list of PNGs into a uint8 (N,H,W,3) array. Drops alpha."""
    arrs = []
    ref_shape = None
    for p in paths:
        img = Image.open(p).convert("RGB")
        a = np.asarray(img, dtype=np.uint8)
        if ref_shape is None:
            ref_shape = a.shape
        elif a.shape != ref_shape:
            print(f"WARN: skipping {p}: shape {a.shape} != {ref_shape}",
                  file=sys.stderr)
            continue
        arrs.append(a)
    return np.stack(arrs, axis=0)


def per_frame_diff(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (per_pixel_diff, per_frame_stats) where:
       per_pixel_diff: (N-1,H,W) float32 mean-channel abs diff in [0,255].
       per_frame_stats: list of dicts with mean / p50 / p99 / peak.
    """
    if frames.shape[0] < 2:
        raise SystemExit("Need at least 2 frames to compute shimmer.")
    f = frames.astype(np.float32)
    # mean over RGB channels of |f[t] - f[t-1]|
    diff = np.abs(f[1:] - f[:-1]).mean(axis=-1)  # (N-1,H,W)
    stats = []
    for i, d in enumerate(diff):
        flat = d.reshape(-1)
        stats.append({
            "frame_idx": i + 1,            # diff is between frame i+1 and i
            "mean_abs_diff": float(flat.mean()),
            "p50": float(np.percentile(flat, 50.0)),
            "p99": float(np.percentile(flat, 99.0)),
            "peak": float(flat.max()),
        })
    return diff, stats


def make_heatmap(diff: np.ndarray, out_path: str, vmax_pct: float = 99.5):
    """Save a heatmap PNG showing per-pixel max abs-diff over the sequence.
    vmax is the vmax_pct percentile of the per-pixel max (so a few wild
    outliers don't blow out the contrast)."""
    per_pixel_max = diff.max(axis=0)  # (H,W) float32
    vmax = float(np.percentile(per_pixel_max, vmax_pct))
    vmax = max(vmax, 1.0)             # avoid div-by-zero
    norm = np.clip(per_pixel_max / vmax, 0.0, 1.0)
    # Simple "magma-ish" 3-stop ramp so we don't depend on matplotlib.
    # 0   -> (0,0,0) black
    # 0.5 -> (180,40,80) crimson
    # 1.0 -> (255,255,200) bright yellow
    n = norm
    r = np.where(n < 0.5, n * 2 * 180.0,                180 + (n - 0.5) * 2 * (255 - 180))
    g = np.where(n < 0.5, n * 2 *  40.0,                 40 + (n - 0.5) * 2 * (255 -  40))
    b = np.where(n < 0.5, n * 2 *  80.0,                 80 + (n - 0.5) * 2 * (200 -  80))
    rgb = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(out_path)
    return per_pixel_max, vmax


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="Directory containing the captured PNGs.")
    ap.add_argument("--pattern", default="*.png",
        help='Glob pattern within input_dir (default "*.png").')
    ap.add_argument("--output-dir", default=None,
        help="Where to write shimmer.csv / shimmer_heatmap.png "
             "(defaults to input_dir).")
    ap.add_argument("--csv-name", default="shimmer.csv")
    ap.add_argument("--heatmap-name", default="shimmer_heatmap.png")
    ap.add_argument("--label", default=None,
        help="Optional label printed in the summary line.")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir or args.input_dir)
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files match {args.pattern!r} in {in_dir}")
    print(f"[shimmer] {len(paths)} frames matching {args.pattern!r} in {in_dir}")

    frames = load_frames(paths)
    diff, stats = per_frame_diff(frames)

    csv_path = os.path.join(out_dir, args.csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "mean_abs_diff", "p50", "p99", "peak"])
        for s in stats:
            w.writerow([s["frame_idx"],
                        f"{s['mean_abs_diff']:.4f}",
                        f"{s['p50']:.4f}",
                        f"{s['p99']:.4f}",
                        f"{s['peak']:.4f}"])
        means = np.array([s["mean_abs_diff"] for s in stats])
        p99s  = np.array([s["p99"] for s in stats])
        peaks = np.array([s["peak"] for s in stats])
        w.writerow([])
        w.writerow(["SUMMARY",
                    f"mean_of_means={means.mean():.4f}",
                    f"mean_of_p99={p99s.mean():.4f}",
                    f"peak_of_peaks={peaks.max():.4f}",
                    f"n_pairs={len(stats)}"])

    heatmap_path = os.path.join(out_dir, args.heatmap_name)
    per_pixel_max, vmax = make_heatmap(diff, heatmap_path)

    label = f"[{args.label}] " if args.label else ""
    print(f"[shimmer] {label}n_pairs={len(stats)}  "
          f"mean_of_means={means.mean():.3f}  "
          f"mean_of_p99={p99s.mean():.3f}  "
          f"peak_of_peaks={peaks.max():.3f}")
    print(f"[shimmer] wrote {csv_path}")
    print(f"[shimmer] wrote {heatmap_path}  (vmax={vmax:.2f})")


if __name__ == "__main__":
    main()
