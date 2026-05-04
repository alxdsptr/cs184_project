"""
compare_quality.py — Quantitative quality comparison for path-tracer captures.

Two comparison modes:

  PER-FRAME-INDEX (recommended; --ref-tag)
    For each frame index N, compare <ref-tag>_NNNNNN.png against
    <test-tag>_NNNNNN.png. Both come from runs that share the same
    deterministic camera path (same warmup / frames / stride / motion), so
    frame N corresponds to the same camera pose across modes. The diff
    therefore reflects ALGORITHM noise, not geometric motion.

    Use this whenever your reference is itself a captured sequence (e.g. the
    "native" sweep produced by run_quality_sweep.sh). The convergence-baseline
    scenario — "is restir-pt closer to the long-spp reference than path-tracing
    is?" — is best done by running native at very high spp and using its
    sequence as the ref-tag.

  SINGLE-IMAGE (legacy; --ref)
    Compare every captured frame against ONE static reference PNG. Only makes
    sense when your reference is a converged still and the captured sequences
    are also static (single-frame captures). For a moving camera, this mode
    measures geometric drift, not algorithm quality.

Typical usage (per-frame-index, the common case)
------------------------------------------------

    python tools/compare_quality.py \
        --test-dir screenshots/sweep \
        --ref-tag native \
        --test-tag restir-di --test-tag restir-di-gi --test-tag restir-di-pt \
        --report screenshots/sweep/report.html

Legacy single-image use
------------------------

    python tools/compare_quality.py \
        --ref screenshots/converged_5000spp.png \
        --test-dir screenshots/sweep \
        --test-tag native --test-tag restir-di --test-tag restir-di-gi --test-tag restir-di-pt

Metrics
-------
MAPE — Mean Absolute Percentage Error per Lin et al. 2022 §9 fn. 13. L1, robust
       to fireflies. Lower is better.
PSNR — Peak Signal-to-Noise Ratio in dB. Higher is better.
SSIM — Structural Similarity (skimage). Higher is better. Optional (skipped if
       scikit-image not installed).

Dependencies: numpy, Pillow, optional scikit-image.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn  # type: ignore
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False


def load_image(path: Path) -> np.ndarray:
    """Load image as float32 in [0, 1], shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    # Rec. 709 luma — matches the engine's `restirLuminance`.
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def mape(test: np.ndarray, ref: np.ndarray) -> float:
    t = to_grayscale(test)
    g = to_grayscale(ref)
    denom = 0.01 * g.mean() + g
    return float(np.mean(np.abs(t - g) / np.maximum(denom, 1e-8)))


def psnr(test: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((test - ref) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def ssim(test: np.ndarray, ref: np.ndarray) -> Optional[float]:
    if not _HAVE_SKIMAGE:
        return None
    return float(ssim_fn(ref, test, channel_axis=-1, data_range=1.0))


def diff_heatmap(test: np.ndarray, ref: np.ndarray, scale: float = 5.0) -> np.ndarray:
    diff = np.abs(test - ref) * scale
    diff = np.clip(diff, 0.0, 1.0)
    return (diff * 255.0).astype(np.uint8)


@dataclass
class FrameResult:
    name: str                   # test-image basename
    ref_name: str               # reference-image basename (for per-frame mode)
    frame_index: int
    mape: float
    psnr: float
    ssim: Optional[float]
    diff_path: Optional[Path]


@dataclass
class TagResult:
    tag: str
    frames: List[FrameResult]
    meta: dict

    def mean_mape(self) -> float:
        return float(np.mean([f.mape for f in self.frames])) if self.frames else float("nan")

    def mean_psnr(self) -> float:
        vals = [f.psnr for f in self.frames if np.isfinite(f.psnr)]
        return float(np.mean(vals)) if vals else float("nan")

    def mean_ssim(self) -> Optional[float]:
        vals = [f.ssim for f in self.frames if f.ssim is not None]
        return float(np.mean(vals)) if vals else None


# Frame-index parser: <tag>_NNNNNN.png — return N, or None if it doesn't match.
_FRAME_RE = re.compile(r"^(?P<tag>.+)_(?P<idx>\d+)$")
def parse_frame_index(p: Path) -> Optional[int]:
    m = _FRAME_RE.match(p.stem)
    return int(m.group("idx")) if m else None


def collect_tag_images(test_dir: Path, tag: str) -> Dict[int, Path]:
    """Return frame_index -> path for every <tag>_NNNNNN.png in test_dir."""
    out: Dict[int, Path] = {}
    for p in sorted(test_dir.glob(f"{tag}_[0-9]*.png")):
        # Skip diff_ outputs from previous runs: they begin with `diff_<tag>_NNN`,
        # but glob `<tag>_*` wouldn't catch them anyway. Defensive parse.
        idx = parse_frame_index(p)
        if idx is not None:
            out[idx] = p
    return out


def load_meta(test_dir: Path, tag: str) -> dict:
    meta_path = test_dir / f"{tag}_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception as e:
        print(f"WARN: could not parse {meta_path}: {e}", file=sys.stderr)
        return {}


def compare_tag_per_frame(test_dir: Path,
                           ref_tag: str,
                           test_tag: str,
                           emit_diff: bool,
                           diff_scale: float) -> TagResult:
    """Pair <ref_tag>_N.png with <test_tag>_N.png for every common N."""
    refs  = collect_tag_images(test_dir, ref_tag)
    tests = collect_tag_images(test_dir, test_tag)
    common = sorted(set(refs.keys()) & set(tests.keys()))
    if not common:
        print(f"WARN: no overlapping frame indices for ref-tag='{ref_tag}', "
              f"test-tag='{test_tag}'", file=sys.stderr)

    frames: List[FrameResult] = []
    for idx in common:
        ref_arr  = load_image(refs[idx])
        test_arr = load_image(tests[idx])
        if ref_arr.shape != test_arr.shape:
            print(f"WARN: skipping frame {idx}: shape mismatch "
                  f"{test_arr.shape} vs {ref_arr.shape}", file=sys.stderr)
            continue
        m  = mape(test_arr, ref_arr)
        ps = psnr(test_arr, ref_arr)
        ss = ssim(test_arr, ref_arr)
        diff_path: Optional[Path] = None
        if emit_diff:
            heat = diff_heatmap(test_arr, ref_arr, scale=diff_scale)
            diff_path = test_dir / f"diff_{test_tag}_vs_{ref_tag}_{idx:06d}.png"
            Image.fromarray(heat).save(diff_path)
        frames.append(FrameResult(
            name=tests[idx].name,
            ref_name=refs[idx].name,
            frame_index=idx,
            mape=m, psnr=ps, ssim=ss,
            diff_path=diff_path))
    return TagResult(tag=test_tag, frames=frames, meta=load_meta(test_dir, test_tag))


def compare_tag_single_ref(ref: np.ndarray,
                            test_dir: Path,
                            tag: str,
                            emit_diff: bool,
                            diff_scale: float) -> TagResult:
    """Legacy: compare every frame of `tag` against one static `ref` image."""
    images = collect_tag_images(test_dir, tag)
    if not images:
        print(f"WARN: no images found for tag '{tag}' in {test_dir}", file=sys.stderr)

    frames: List[FrameResult] = []
    for idx in sorted(images.keys()):
        p = images[idx]
        test = load_image(p)
        if test.shape != ref.shape:
            print(f"WARN: skipping {p.name}: shape {test.shape} != ref shape {ref.shape}",
                  file=sys.stderr)
            continue
        m = mape(test, ref)
        ps = psnr(test, ref)
        ss = ssim(test, ref)
        diff_path: Optional[Path] = None
        if emit_diff:
            heat = diff_heatmap(test, ref, scale=diff_scale)
            diff_path = test_dir / f"diff_{p.stem}.png"
            Image.fromarray(heat).save(diff_path)
        frames.append(FrameResult(
            name=p.name, ref_name="<single>", frame_index=idx,
            mape=m, psnr=ps, ssim=ss, diff_path=diff_path))
    return TagResult(tag=tag, frames=frames, meta=load_meta(test_dir, tag))


def write_html_report(results: List[TagResult],
                       ref_label: str,
                       per_frame_mode: bool,
                       out: Path):
    rows = []
    rows.append("<!doctype html><meta charset='utf-8'><title>quality report</title>")
    rows.append("<style>")
    rows.append("body{font-family:system-ui,sans-serif;margin:24px;max-width:1200px}")
    rows.append("h2{margin-top:32px}")
    rows.append("table{border-collapse:collapse;margin:8px 0 24px}")
    rows.append("th,td{border:1px solid #bbb;padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums}")
    rows.append("th:first-child,td:first-child{text-align:left}")
    rows.append("th{background:#eee}")
    rows.append("img.thumb{width:240px;height:auto;border:1px solid #ccc;margin:2px;display:block}")
    rows.append(".tag{font-weight:bold;color:#06c}")
    rows.append(".pair{display:flex;gap:8px;align-items:flex-start;margin:8px 0;flex-wrap:wrap}")
    rows.append(".triplet{display:flex;flex-direction:column;align-items:center;font-size:11px}")
    rows.append(".triplet>div+div{margin-top:2px}")
    rows.append("</style>")
    mode_str = "per-frame-index" if per_frame_mode else "single-image"
    rows.append(f"<h1>Quality comparison ({mode_str}) — reference: <code>{ref_label}</code></h1>")

    # Summary table
    rows.append("<h2>Summary</h2>")
    rows.append("<table><tr><th>tag</th><th>n</th><th>MAPE&nbsp;(↓)</th><th>PSNR dB&nbsp;(↑)</th><th>SSIM&nbsp;(↑)</th><th>fps</th></tr>")
    for r in results:
        ssim_v = r.mean_ssim()
        ssim_s = f"{ssim_v:.4f}" if ssim_v is not None else "—"
        fps = r.meta.get("mean_fps", float("nan"))
        rows.append(
            f"<tr><td class='tag'>{r.tag}</td><td>{len(r.frames)}</td>"
            f"<td>{r.mean_mape():.4f}</td>"
            f"<td>{r.mean_psnr():.2f}</td>"
            f"<td>{ssim_s}</td>"
            f"<td>{fps:.1f}</td></tr>"
        )
    rows.append("</table>")

    # Per-frame breakdown.
    for r in results:
        rows.append(f"<h2>{r.tag} — per frame</h2>")
        rows.append("<table><tr><th>frame</th><th>test image</th><th>ref image</th>"
                    "<th>MAPE</th><th>PSNR dB</th><th>SSIM</th></tr>")
        for f in r.frames:
            ss = f"{f.ssim:.4f}" if f.ssim is not None else "—"
            rows.append(
                f"<tr><td>{f.frame_index:06d}</td>"
                f"<td>{f.name}</td><td>{f.ref_name}</td>"
                f"<td>{f.mape:.4f}</td>"
                f"<td>{f.psnr:.2f}</td>"
                f"<td>{ss}</td></tr>"
            )
        rows.append("</table>")

        # Thumbnail grid: ref | test | diff per frame.
        if r.frames and r.frames[0].diff_path is not None:
            rows.append("<div class='pair'>")
            base_dir = out.parent
            for f in r.frames:
                test_rel = os.path.relpath(out.parent / f.name, base_dir)
                diff_rel = os.path.relpath(f.diff_path, base_dir) if f.diff_path else None
                ref_rel = None
                if per_frame_mode:
                    ref_path = out.parent / f.ref_name
                    ref_rel = os.path.relpath(ref_path, base_dir)
                rows.append("<div class='triplet'>")
                rows.append(f"<div>frame {f.frame_index:06d}</div>")
                if ref_rel:
                    rows.append(f"<div>ref</div><img class='thumb' src='{ref_rel}'>")
                rows.append(f"<div>test</div><img class='thumb' src='{test_rel}'>")
                if diff_rel:
                    rows.append(f"<div>|test−ref|×{int(5)}</div><img class='thumb' src='{diff_rel}'>")
                rows.append("</div>")
            rows.append("</div>")

    out.write_text("\n".join(rows), encoding="utf-8")


def write_html_grid_report(results: List[TagResult],
                            ref_label: str,
                            ref_tag: Optional[str],
                            per_frame_mode: bool,
                            test_dir: Path,
                            out: Path):
    """Compact "one row per frame" layout.

    Columns (left → right):
      frame index | reference image | <each tag image with its MAPE/PSNR>

    Designed for visual side-by-side inspection — easier than the breakdown
    layout when you just want to scroll through and eyeball "is restir-pt
    visibly cleaner than native at this same camera pose?".
    """
    base_dir = out.parent

    # Collect all frame indices that appear across the results — union, then
    # sorted ascending. Some tags may have skipped frames (corrupt save, etc.)
    # so we render whatever we have per cell, with a placeholder when missing.
    frame_set: set = set()
    for r in results:
        for f in r.frames:
            frame_set.add(f.frame_index)
    frame_indices = sorted(frame_set)

    # Build a tag -> frame_idx -> FrameResult lookup for fast cell rendering.
    by_tag: Dict[str, Dict[int, FrameResult]] = {}
    for r in results:
        by_tag[r.tag] = {f.frame_index: f for f in r.frames}

    # For per-frame mode the reference image's path is recorded in each
    # FrameResult.ref_name; pick any non-empty one to derive the ref image
    # location, and use the FrameResult to locate it.
    def ref_image_path_for(idx: int) -> Optional[Path]:
        if not per_frame_mode or ref_tag is None:
            return None
        # Reference filename pattern matches the capture filenames.
        return test_dir / f"{ref_tag}_{idx:06d}.png"

    rows: List[str] = []
    rows.append("<!doctype html><meta charset='utf-8'><title>quality report (grid)</title>")
    rows.append("<style>")
    rows.append("body{font-family:system-ui,sans-serif;margin:24px}")
    rows.append("table.grid{border-collapse:collapse;margin:8px 0}")
    rows.append("table.grid th,table.grid td{border:1px solid #bbb;padding:4px;vertical-align:top;text-align:center}")
    rows.append("table.grid th{background:#eee;font-weight:600}")
    rows.append("table.grid td.frame{font-variant-numeric:tabular-nums;font-size:12px;color:#555}")
    rows.append("table.grid img{width:280px;height:auto;display:block;margin:0 auto}")
    rows.append("table.grid .metrics{font-size:11px;color:#333;font-variant-numeric:tabular-nums;margin-top:2px;line-height:1.3}")
    rows.append(".empty{color:#aaa;font-size:11px}")
    # Summary table reuses same styling as breakdown report.
    rows.append("table.sum{border-collapse:collapse;margin:8px 0 24px}")
    rows.append("table.sum th,table.sum td{border:1px solid #bbb;padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums}")
    rows.append("table.sum th:first-child,table.sum td:first-child{text-align:left}")
    rows.append("table.sum th{background:#eee}")
    rows.append(".tag{font-weight:bold;color:#06c}")
    rows.append("</style>")

    rows.append(f"<h1>Side-by-side quality grid — reference: <code>{ref_label}</code></h1>")

    # Top-of-page summary so you don't have to scroll down to see numbers.
    rows.append("<h2>Summary</h2>")
    rows.append("<table class='sum'><tr><th>tag</th><th>n</th>"
                "<th>MAPE&nbsp;(↓)</th><th>PSNR dB&nbsp;(↑)</th>"
                "<th>SSIM&nbsp;(↑)</th><th>fps</th></tr>")
    for r in results:
        ssim_v = r.mean_ssim()
        ssim_s = f"{ssim_v:.4f}" if ssim_v is not None else "—"
        fps = r.meta.get("mean_fps", float("nan"))
        rows.append(
            f"<tr><td class='tag'>{r.tag}</td><td>{len(r.frames)}</td>"
            f"<td>{r.mean_mape():.4f}</td>"
            f"<td>{r.mean_psnr():.2f}</td>"
            f"<td>{ssim_s}</td>"
            f"<td>{fps:.1f}</td></tr>"
        )
    rows.append("</table>")

    # ── The grid ─────────────────────────────────────────────────────────
    rows.append("<h2>Per-frame side-by-side</h2>")
    rows.append("<table class='grid'>")

    # Header row: frame | reference (if any) | tag1 | tag2 | ...
    rows.append("<tr><th>frame</th>")
    if per_frame_mode and ref_tag is not None:
        rows.append(f"<th>reference<br><span class='metrics'>{ref_tag}</span></th>")
    elif not per_frame_mode:
        rows.append("<th>reference</th>")
    for r in results:
        rows.append(f"<th>{r.tag}</th>")
    rows.append("</tr>")

    # Body: one row per frame index.
    for idx in frame_indices:
        rows.append("<tr>")
        rows.append(f"<td class='frame'>{idx:06d}</td>")

        # Reference cell.
        if per_frame_mode and ref_tag is not None:
            ref_p = ref_image_path_for(idx)
            if ref_p and ref_p.exists():
                ref_rel = os.path.relpath(ref_p, base_dir)
                rows.append(f"<td><img src='{ref_rel}'></td>")
            else:
                rows.append("<td class='empty'>—</td>")
        elif not per_frame_mode:
            # Single-ref mode: the same reference repeats in every row. To
            # keep the page light we still show it (matches breakdown report
            # behaviour where you can see what you're comparing against).
            rows.append(f"<td class='empty'>(see top — same ref for all rows)</td>")

        # Per-tag cells: image + per-frame metrics.
        for r in results:
            f = by_tag.get(r.tag, {}).get(idx)
            if f is None:
                rows.append("<td class='empty'>—</td>")
                continue
            test_p = test_dir / f.name
            test_rel = os.path.relpath(test_p, base_dir)
            ssim_s = f"  SSIM={f.ssim:.3f}" if f.ssim is not None else ""
            rows.append(
                f"<td><img src='{test_rel}'>"
                f"<div class='metrics'>"
                f"MAPE={f.mape:.3f}&nbsp;&nbsp;PSNR={f.psnr:.1f}dB{ssim_s}"
                f"</div></td>"
            )
        rows.append("</tr>")

    rows.append("</table>")

    out.write_text("\n".join(rows), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)

    # Mode 1: per-frame-index (PREFERRED for sequences captured along the same
    # camera path).
    ap.add_argument("--ref-tag", type=str, default=None,
                    help="Reference capture tag. For each frame index N, "
                         "<ref-tag>_NNNNNN.png is paired with <test-tag>_NNNNNN.png.")

    # Mode 2: single-image (legacy, for static reference comparisons).
    ap.add_argument("--ref", type=Path, default=None,
                    help="Reference image. Every captured frame is compared "
                         "against this single PNG. Use this only when both ref "
                         "and tests are static; for moving cameras prefer --ref-tag.")

    ap.add_argument("--test-dir", required=True, type=Path,
                    help="Directory containing the captured PNG sequences")
    ap.add_argument("--test-tag", required=True, action="append",
                    help="Capture tag to evaluate (may be specified multiple times)")
    ap.add_argument("--report", type=Path, default=None,
                    help="HTML report path (default: <test-dir>/report.html)")
    ap.add_argument("--report-style", choices=("grid", "breakdown", "both"),
                    default="grid",
                    help="HTML layout. 'grid' (default): one row per frame, "
                         "columns = reference + each tag, designed for visual "
                         "side-by-side. 'breakdown': per-tag sections with full "
                         "metric tables. 'both' writes <report>.html (grid) AND "
                         "<report>.breakdown.html.")
    ap.add_argument("--no-diff", action="store_true",
                    help="Skip writing per-frame diff heatmaps (only relevant "
                         "for breakdown report; grid layout doesn't show them)")
    ap.add_argument("--diff-scale", type=float, default=5.0,
                    help="Diff heatmap brightness multiplier (default 5)")
    args = ap.parse_args()

    if args.ref is None and args.ref_tag is None:
        ap.error("must pass either --ref-tag <tag> (preferred) or --ref <png>")
    if args.ref is not None and args.ref_tag is not None:
        ap.error("--ref and --ref-tag are mutually exclusive; pick one")

    if not _HAVE_SKIMAGE:
        print("(scikit-image not installed — SSIM column will be blank.\n"
              " `pip install scikit-image` to enable.)", file=sys.stderr)

    if not args.test_dir.is_dir():
        print(f"ERROR: test dir not found: {args.test_dir}", file=sys.stderr)
        sys.exit(1)

    results: List[TagResult] = []
    per_frame_mode = (args.ref_tag is not None)

    if per_frame_mode:
        # Use HTML-friendly text — angle brackets in the original would be
        # parsed as a tag by the browser and the label would vanish.
        ref_label = f"&lt;{args.ref_tag}&gt;_NNNNNN.png &nbsp;(per-frame-index pairing)"
        print(f"per-frame mode: ref-tag='{args.ref_tag}' in {args.test_dir}")
        for tag in args.test_tag:
            if tag == args.ref_tag:
                # Self-comparison would be all zeros — skip to avoid clutter
                # but tell the user why.
                print(f"\n--- tag: {tag} (skipped: same as --ref-tag) ---")
                continue
            print(f"\n--- tag: {tag} (vs {args.ref_tag}) ---")
            r = compare_tag_per_frame(
                args.test_dir, args.ref_tag, tag,
                emit_diff=(not args.no_diff),
                diff_scale=args.diff_scale)
            for f in r.frames:
                ss = f"{f.ssim:.4f}" if f.ssim is not None else "  —  "
                print(f"  frame {f.frame_index:06d}  {f.name:30s}  "
                      f"MAPE={f.mape:.4f}  PSNR={f.psnr:6.2f} dB  SSIM={ss}")
            ssim_v = r.mean_ssim()
            ssim_s = f"{ssim_v:.4f}" if ssim_v is not None else "  —  "
            print(f"  MEAN  ({len(r.frames)} pairs)         "
                  f"MAPE={r.mean_mape():.4f}  PSNR={r.mean_psnr():6.2f} dB  SSIM={ssim_s}")
            results.append(r)
    else:
        if not args.ref.exists():
            print(f"ERROR: reference image not found: {args.ref}", file=sys.stderr)
            sys.exit(1)
        ref = load_image(args.ref)
        ref_label = str(args.ref.name)
        print(f"single-ref mode: ref={args.ref}  shape={ref.shape}  "
              f"mean lum={to_grayscale(ref).mean():.4f}")
        for tag in args.test_tag:
            print(f"\n--- tag: {tag} ---")
            r = compare_tag_single_ref(ref, args.test_dir, tag,
                                        emit_diff=(not args.no_diff),
                                        diff_scale=args.diff_scale)
            for f in r.frames:
                ss = f"{f.ssim:.4f}" if f.ssim is not None else "  —  "
                print(f"  {f.name:30s}  MAPE={f.mape:.4f}  PSNR={f.psnr:6.2f} dB  SSIM={ss}")
            ssim_v = r.mean_ssim()
            ssim_s = f"{ssim_v:.4f}" if ssim_v is not None else "  —  "
            print(f"  MEAN  ({len(r.frames)} imgs)         "
                  f"MAPE={r.mean_mape():.4f}  PSNR={r.mean_psnr():6.2f} dB  SSIM={ssim_s}")
            results.append(r)

    # Default report path. For "both" we derive a sibling ".breakdown.html"
    # so the user only has to pass one --report.
    report = args.report or (args.test_dir / "report.html")

    # Pick which writers to invoke.
    do_grid = args.report_style in ("grid", "both")
    do_breakdown = args.report_style in ("breakdown", "both")

    if do_grid:
        grid_path = report
        if args.report_style == "both":
            # When generating both, the breakdown takes the user-supplied
            # name and the grid gets the canonical "report.html" — flipped
            # because grid is the new default and most useful.
            grid_path = report
        write_html_grid_report(
            results, ref_label,
            ref_tag=(args.ref_tag if per_frame_mode else None),
            per_frame_mode=per_frame_mode,
            test_dir=args.test_dir,
            out=grid_path)
        print(f"\nGrid report: {grid_path}")

    if do_breakdown:
        breakdown_path = report
        if args.report_style == "both":
            # Sibling file: report.html -> report.breakdown.html.
            breakdown_path = report.with_suffix(".breakdown.html")
        write_html_report(results, ref_label, per_frame_mode, breakdown_path)
        print(f"Breakdown report: {breakdown_path}")


if __name__ == "__main__":
    main()
