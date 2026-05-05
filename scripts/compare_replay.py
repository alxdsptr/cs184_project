#!/usr/bin/env python3
"""Render the same recorded camera path through several pipeline variants and
stitch the resulting frames into one side-by-side comparison video.

Variants (all share `--mode native`; the difference is which ReSTIR passes
are enabled):

  native       : --no-restir --no-restir-gi --no-restir-pt
  di           : ReSTIR DI only        (default; --no-restir-gi --no-restir-pt)
  di_gi        : ReSTIR DI + GI        (--restir-gi --no-restir-pt)
  di_gi_pt     : ReSTIR DI + GI + PT   (--restir-gi --restir-pt)

Each variant is rendered in its own subprocess (one scene load) into
`<run_dir>/<replay-out>/<variant>/frame_NNNNNN.png`. Then ffmpeg combines the
per-variant frame sequences into one video where each cell shows one variant
at the same camera pose.

Usage (from anywhere)::

    python scripts/compare_replay.py recordings/path_BistroInterior_shorter.json \\
        --scene ../../Bistro_v5_2/BistroInterior.fbx \\
        --out renders/replay_compare.mp4

The recording path and `--scene` are interpreted relative to the renderer's
working directory (`build-all/Release/` by default), matching how you'd type
them at the prompt there.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Variant table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    name: str          # short id, also used as subdir + label
    label: str         # human-readable label drawn on the video
    flags: tuple[str, ...]


VARIANTS: dict[str, Variant] = {
    "native":   Variant("native",   "native (no ReSTIR)",
                        ("--no-restir", "--no-restir-gi", "--no-restir-pt")),
    "di":       Variant("di",       "ReSTIR DI",
                        ("--no-restir-gi", "--no-restir-pt")),
    "di_gi":    Variant("di_gi",    "ReSTIR DI + GI",
                        ("--restir-gi", "--no-restir-pt")),
    "di_gi_pt": Variant("di_gi_pt", "ReSTIR DI + GI + PT",
                        ("--restir-gi", "--restir-pt")),
}

DEFAULT_ORDER = ["native", "di", "di_gi", "di_gi_pt"]


# ---------------------------------------------------------------------------
# Renderer driver
# ---------------------------------------------------------------------------

def run_replay(exe: Path, run_dir: Path, scene: str, recording: str,
               replay_out: Path, width: int, height: int, spp: int,
               stride: int, backend: str, mode: str,
               variant: Variant, extra: list[str], verbose: bool) -> None:
    """One subprocess = one scene load = N frames for one variant."""
    cmd: list[str] = [
        str(exe),
        scene,
        "-r", str(width), str(height),
        "--backend", backend,
        "--mode", mode,
        "--replay", recording,
        "--replay-out", str(replay_out),
        "--replay-spp", str(spp),
        "--replay-stride", str(stride),
        *variant.flags,
        *extra,
    ]
    print(f"  $ {' '.join(cmd)}")
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL
    proc = subprocess.run(cmd, cwd=str(run_dir), stdout=stdout, stderr=stderr)
    if proc.returncode != 0:
        # Re-run loud so failures are diagnosable.
        subprocess.run(cmd, cwd=str(run_dir))
        raise RuntimeError(f"pathtracer exited with code {proc.returncode} for variant {variant.name}")


# ---------------------------------------------------------------------------
# ffmpeg compose
# ---------------------------------------------------------------------------

def _ffmpeg_layout(n: int) -> tuple[int, int]:
    """Return (cols, rows) for n cells. 1->1x1, 2->2x1, 3->3x1, 4->2x2."""
    if n == 1:
        return 1, 1
    if n == 2:
        return 2, 1
    if n == 3:
        return 3, 1
    if n == 4:
        return 2, 2
    # Fallback: pack into roughly-square grid.
    cols = int(n ** 0.5 + 0.999)
    rows = (n + cols - 1) // cols
    return cols, rows


_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _autodetect_font() -> Path | None:
    for cand in _FONT_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return p
    return None


def _drawtext_path(p: Path) -> str:
    """Format a font path for ffmpeg's drawtext fontfile= option. ffmpeg
    treats ':' as an option separator inside the filtergraph, so on Windows
    the drive-letter colon needs to be escaped as 'C\\:'. Forward slashes
    are fine inside the value."""
    s = p.as_posix()
    return s.replace(":", r"\:")


def assemble_grid(variant_dirs: list[tuple[Variant, Path]], out_path: Path,
                  fps: float, width: int, height: int,
                  font_path: Path | None, no_labels: bool) -> None:
    """Build a grid video with one cell per variant. Each cell is labeled."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH")

    n = len(variant_dirs)
    cols, rows = _ffmpeg_layout(n)
    cell_w, cell_h = width, height

    args: list[str] = ["ffmpeg", "-y"]
    for _, frames_dir in variant_dirs:
        args += ["-framerate", str(fps),
                 "-i", str(frames_dir / "frame_%06d.png")]

    # Resolve a font: explicit > autodetect > none. Without a font, ffmpeg's
    # drawtext crashes on builds without fontconfig (typical on Windows), so
    # we drop labels entirely if we can't find one.
    resolved_font = font_path if (font_path and font_path.exists()) else _autodetect_font()
    do_labels = (not no_labels) and (resolved_font is not None)
    if not no_labels and resolved_font is None:
        print("[warn] no usable font found for labels; pass --font <ttf> to enable labels. "
              "Continuing without labels.", file=sys.stderr)

    chains: list[str] = []
    drawtext_font = f"fontfile='{_drawtext_path(resolved_font)}':" if do_labels else ""

    for i, (variant, _) in enumerate(variant_dirs):
        scale = f"[{i}:v]scale={cell_w}:{cell_h}:flags=lanczos"
        if do_labels:
            label = variant.label.replace("\\", r"\\").replace("'", r"\'")
            chains.append(
                f"{scale},"
                f"drawtext={drawtext_font}text='{label}':"
                f"x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=0x00000080:boxborderw=8"
                f"[v{i}]"
            )
        else:
            chains.append(f"{scale}[v{i}]")

    if n == 1:
        chains.append("[v0]copy[out]")
    else:
        # xstack layout cells: each cell's top-left is expressed as
        # "x_y" where x and y are sums of widths/heights of *prior* inputs in
        # the same row/column. e.g. for a 2x2 grid: 0_0 | w0_0 | 0_h0 | w0_h0.
        layout_parts: list[str] = []
        for i in range(n):
            cx = i % cols
            cy = i // cols
            x_expr = "0" if cx == 0 else "+".join(f"w{k}" for k in range(cx))
            y_expr = "0" if cy == 0 else "+".join(f"h{k * cols}" for k in range(cy))
            layout_parts.append(f"{x_expr}_{y_expr}")
        inputs = "".join(f"[v{i}]" for i in range(n))
        chains.append(f"{inputs}xstack=inputs={n}:layout={'|'.join(layout_parts)}[out]")

    filter_complex = ";".join(chains)
    args += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        str(out_path),
    ]
    print(f"\n[ffmpeg] composing {n} variants -> {out_path} ({cols}x{rows})")
    print(f"  $ {' '.join(args)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(args, check=True)


# ---------------------------------------------------------------------------
# Per-variant solo encode (handy for inspecting a single variant)
# ---------------------------------------------------------------------------

def assemble_solo(frames_dir: Path, out_path: Path, fps: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]
    subprocess.run(args, check=True)


# ---------------------------------------------------------------------------
# Recording duration -> playback fps
# ---------------------------------------------------------------------------

def recording_duration(path: Path) -> tuple[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = data.get("poses") or []
    if not poses:
        return 0, 0.0
    first_t = float(poses[0]["t"])
    last_t = float(poses[-1]["t"])
    return len(poses), max(0.0, last_t - first_t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_exe = project_root / "build-all" / "Release" / "pathtracer.exe"

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording",
                   help="Path to the recording JSON. Interpreted relative to --run-dir, "
                        "matching how you'd type it at the renderer prompt.")
    p.add_argument("--scene", default="../../Bistro_v5_2/BistroInterior.fbx",
                   help="Scene argument passed to pathtracer (relative to --run-dir). "
                        "Default: ../../Bistro_v5_2/BistroInterior.fbx")
    p.add_argument("--variants", nargs="+", default=DEFAULT_ORDER,
                   choices=list(VARIANTS.keys()),
                   help=f"Which variants to render and stack. Default: {' '.join(DEFAULT_ORDER)}")

    p.add_argument("-r", "--resolution", nargs=2, type=int, metavar=("W", "H"),
                   default=(1280, 720))
    p.add_argument("--spp", type=int, default=1, help="--replay-spp (default 1)")
    p.add_argument("--stride", type=int, default=1, help="--replay-stride (default 1)")
    p.add_argument("--backend", choices=["cuda", "optix"], default="optix")
    p.add_argument("--mode", default="native",
                   choices=["native", "nrd", "dlss", "dlssonly", "rr"],
                   help="Renderer pipeline mode passed via --mode (default native)")

    p.add_argument("--exe", type=Path, default=default_exe,
                   help=f"Path to pathtracer.exe (default: {default_exe})")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Working dir for pathtracer (default: exe's parent, i.e. build-all/Release)")

    p.add_argument("--out", type=Path, default=None,
                   help="Output comparison video path. Default: <run-dir>/replay_compare/<recording_stem>_compare.mp4")
    p.add_argument("--frames-root", type=Path, default=None,
                   help="Directory under which per-variant frames are written. "
                        "Default: <run-dir>/replay_compare/<recording_stem>/")
    p.add_argument("--fps", type=float, default=None,
                   help="Playback fps. Default: derived from recording duration.")
    p.add_argument("--font", type=Path, default=None,
                   help="Font file for cell labels. Auto-detects Arial / DejaVu Sans if omitted.")
    p.add_argument("--no-labels", action="store_true",
                   help="Skip the drawtext label overlay (useful if ffmpeg has no usable font)")
    p.add_argument("--keep-existing", action="store_true",
                   help="Skip rendering a variant if its output dir already has frames")
    p.add_argument("--solo", action="store_true",
                   help="Also write one mp4 per variant next to the comparison video")
    p.add_argument("--skip-render", action="store_true",
                   help="Skip rendering, only run the ffmpeg compose step (frames must already exist)")
    p.add_argument("--verbose", action="store_true",
                   help="Stream pathtracer stdout/stderr instead of swallowing it")

    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args appended to every pathtracer invocation (use after --extra)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    exe = args.exe.resolve()
    if not exe.exists():
        print(f"[error] pathtracer not found: {exe}", file=sys.stderr)
        return 2
    run_dir = (args.run_dir or exe.parent).resolve()
    if not run_dir.exists():
        print(f"[error] run dir not found: {run_dir}", file=sys.stderr)
        return 2

    # Resolve recording absolute path for duration probing, but pass the
    # user's string verbatim to pathtracer so cwd-relative paths work the same
    # as on the command line.
    rec_arg = args.recording
    rec_abs = (run_dir / rec_arg).resolve() if not Path(rec_arg).is_absolute() else Path(rec_arg).resolve()
    if not rec_abs.exists():
        print(f"[error] recording not found: {rec_abs}", file=sys.stderr)
        return 2

    n_poses, duration = recording_duration(rec_abs)
    if n_poses == 0:
        print(f"[error] recording has no poses: {rec_abs}", file=sys.stderr)
        return 2
    rendered_count = max(1, (n_poses + args.stride - 1) // args.stride)
    fps = args.fps if args.fps is not None else (rendered_count / duration if duration > 0 else 24.0)

    rec_stem = rec_abs.stem
    frames_root = (args.frames_root or (run_dir / "replay_compare" / rec_stem)).resolve()
    out_path = (args.out or (run_dir / "replay_compare" / f"{rec_stem}_compare.mp4")).resolve()

    selected = [VARIANTS[v] for v in args.variants]

    print(f"Renderer  : {exe}")
    print(f"Run dir   : {run_dir}")
    print(f"Scene     : {args.scene}")
    print(f"Recording : {rec_arg}  ({n_poses} poses, {duration:.2f} s)")
    print(f"Stride/SPP: {args.stride} / {args.spp}  -> ~{rendered_count} frames per variant")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}, backend={args.backend}, mode={args.mode}")
    print(f"Variants  : {', '.join(v.name for v in selected)}")
    print(f"Frames    : {frames_root}")
    print(f"Output    : {out_path}")
    print(f"Playback  : {fps:.2f} fps")

    variant_dirs: list[tuple[Variant, Path]] = []
    for v in selected:
        out_dir = frames_root / v.name
        variant_dirs.append((v, out_dir))

    if not args.skip_render:
        for v, out_dir in variant_dirs:
            existing = sorted(out_dir.glob("frame_*.png")) if out_dir.exists() else []
            if existing and args.keep_existing:
                print(f"\n[{v.name}] keeping existing {len(existing)} frame(s) in {out_dir}")
                continue
            print(f"\n[{v.name}] rendering -> {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)
            for stale in out_dir.glob("frame_*.png"):
                stale.unlink()
            # pathtracer's --replay-out is interpreted relative to its cwd.
            # Pass the absolute path so it lands exactly where we expect
            # regardless of run_dir layout.
            run_replay(
                exe=exe, run_dir=run_dir, scene=args.scene, recording=rec_arg,
                replay_out=out_dir, width=args.resolution[0], height=args.resolution[1],
                spp=args.spp, stride=args.stride, backend=args.backend, mode=args.mode,
                variant=v, extra=list(args.extra or []), verbose=args.verbose,
            )

    # Sanity-check every variant produced the same frame count before stacking.
    counts = []
    for v, out_dir in variant_dirs:
        c = len(sorted(out_dir.glob("frame_*.png")))
        counts.append((v.name, c))
        if c == 0:
            print(f"[error] variant '{v.name}' has no frames in {out_dir}", file=sys.stderr)
            return 1
    print("\nFrame counts:")
    for name, c in counts:
        print(f"  {name:>10}: {c}")
    if len({c for _, c in counts}) > 1:
        print("[warn] variants produced different frame counts; xstack will use the shortest stream",
              file=sys.stderr)

    assemble_grid(variant_dirs, out_path, fps,
                  args.resolution[0], args.resolution[1],
                  args.font, args.no_labels)
    print(f"\nWrote {out_path}")

    if args.solo:
        for v, out_dir in variant_dirs:
            solo_out = out_path.with_name(f"{out_path.stem}_{v.name}.mp4")
            assemble_solo(out_dir, solo_out, fps)
            print(f"Wrote {solo_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
