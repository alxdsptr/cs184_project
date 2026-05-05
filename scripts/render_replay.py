#!/usr/bin/env python3
"""Render a recorded camera path through one pipeline variant and produce a
single mp4 (no side-by-side, no labels, no compositing).

Same variant set as compare_replay.py:

  native    : --no-restir --no-restir-gi --no-restir-pt
  di        : ReSTIR DI only       (default; --no-restir-gi --no-restir-pt)
  di_gi     : ReSTIR DI + GI       (--restir-gi --no-restir-pt)
  di_gi_pt  : ReSTIR DI + GI + PT  (--restir-gi --restir-pt)
  custom    : pass exactly the renderer flags you want via --flags

Examples::

    # Defaults: --variant di, 1280x720, optix/native, 30 fps cap, crf 23
    python scripts/render_replay.py recordings/path_BistroInterior_shorter.json

    # ReSTIR DI+GI+PT at 1080p, slower preset
    python scripts/render_replay.py recordings/path_BistroInterior_shorter.json \\
        --variant di_gi_pt -r 1920 1080 --preset slower --out renders/dgp.mp4

    # Pure pass-through: tell the script exactly which flags to add
    python scripts/render_replay.py recordings/path_BistroInterior_shorter.json \\
        --variant custom --flags "--restir-gi --no-restir-pt --emissive-target 200"

The recording path and --scene are interpreted relative to the renderer's
working directory (build-all/Release/ by default), the same way you'd type
them at the renderer's prompt.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# Reuse the variant table and ffmpeg encode helper from the comparison script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_replay import VARIANTS, Variant, run_replay, assemble_solo  # noqa: E402


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
    p.add_argument("--variant", default="di",
                   choices=list(VARIANTS.keys()) + ["custom"],
                   help="Which preset to render. 'custom' means: do not add any ReSTIR flags, "
                        "use --flags verbatim instead. Default: di")
    p.add_argument("--flags", default="",
                   help="Extra flags appended to the pathtracer invocation. With --variant "
                        "custom this is the *only* source of mode flags (other than --mode). "
                        "Quote the whole string, e.g. --flags \"--restir-gi --no-restir-pt\".")

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
                   help="Working dir for pathtracer (default: exe's parent)")

    p.add_argument("--out", type=Path, default=None,
                   help="Output mp4 path. Default: <run-dir>/replay/<recording_stem>_<variant>.mp4")
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="Where to write per-frame PNGs. "
                        "Default: <run-dir>/replay/<recording_stem>_<variant>/")
    p.add_argument("--fps", type=float, default=None,
                   help="Playback fps. Default: derived from recording duration.")
    p.add_argument("--keep-existing", action="store_true",
                   help="Skip rendering if frames-dir already has frames (just re-encode)")
    p.add_argument("--skip-render", action="store_true",
                   help="Skip rendering, only run the ffmpeg encode (frames must already exist)")

    p.add_argument("--crf", type=int, default=23,
                   help="x264 CRF. Default 23. Higher = smaller, lower = sharper.")
    p.add_argument("--preset", default="slow",
                   choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                            "medium", "slow", "slower", "veryslow"],
                   help="x264 preset; slower = better compression at the same CRF. Default 'slow'.")
    p.add_argument("--max-fps", type=float, default=30.0,
                   help="Cap output framerate. Pass 0 to disable the cap. Default 30.")
    p.add_argument("--verbose", action="store_true",
                   help="Stream pathtracer stdout/stderr instead of swallowing it")
    return p.parse_args()


def recording_duration(path: Path) -> tuple[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = data.get("poses") or []
    if not poses:
        return 0, 0.0
    return len(poses), max(0.0, float(poses[-1]["t"]) - float(poses[0]["t"]))


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
    extra_flags = shlex.split(args.flags) if args.flags else []

    # Build the variant. For "custom" we synthesize one with no preset flags so
    # the only renderer flags come from --flags / --mode.
    if args.variant == "custom":
        variant = Variant("custom", "custom", tuple())
    else:
        variant = VARIANTS[args.variant]

    frames_dir = (args.frames_dir or (run_dir / "replay" / f"{rec_stem}_{variant.name}")).resolve()
    out_path = (args.out or (run_dir / "replay" / f"{rec_stem}_{variant.name}.mp4")).resolve()

    print(f"Renderer  : {exe}")
    print(f"Run dir   : {run_dir}")
    print(f"Scene     : {args.scene}")
    print(f"Recording : {rec_arg}  ({n_poses} poses, {duration:.2f} s)")
    print(f"Stride/SPP: {args.stride} / {args.spp}  -> ~{rendered_count} frames")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}, backend={args.backend}, mode={args.mode}")
    print(f"Variant   : {variant.name}  flags={list(variant.flags) + extra_flags}")
    print(f"Frames    : {frames_dir}")
    print(f"Output    : {out_path}")
    print(f"Playback  : {fps:.2f} fps  (cap={args.max_fps or 'off'})")

    if not args.skip_render:
        existing = sorted(frames_dir.glob("frame_*.png")) if frames_dir.exists() else []
        if existing and args.keep_existing:
            print(f"\n[render] keeping existing {len(existing)} frame(s) in {frames_dir}")
        else:
            print(f"\n[render] -> {frames_dir}")
            frames_dir.mkdir(parents=True, exist_ok=True)
            for stale in frames_dir.glob("frame_*.png"):
                stale.unlink()
            run_replay(
                exe=exe, run_dir=run_dir, scene=args.scene, recording=rec_arg,
                replay_out=frames_dir, width=args.resolution[0], height=args.resolution[1],
                spp=args.spp, stride=args.stride, backend=args.backend, mode=args.mode,
                variant=variant, extra=extra_flags, verbose=args.verbose,
            )

    n_frames = len(sorted(frames_dir.glob("frame_*.png")))
    if n_frames == 0:
        print(f"[error] no frames found in {frames_dir}", file=sys.stderr)
        return 1
    print(f"\nFrames    : {n_frames} png(s) in {frames_dir}")

    if shutil.which("ffmpeg") is None:
        print("[error] ffmpeg not found in PATH", file=sys.stderr)
        return 1

    max_fps = args.max_fps if args.max_fps and args.max_fps > 0 else None
    print(f"\n[ffmpeg] encoding -> {out_path} (crf={args.crf}, preset={args.preset})")
    assemble_solo(frames_dir, out_path, fps, args.crf, args.preset, max_fps)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out_path}  ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
