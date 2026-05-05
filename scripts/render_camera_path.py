#!/usr/bin/env python3
"""
Render an animated GIF along a camera trajectory using the pathtracer executable.

Trajectory is defined by a list of "keyframe" camera files (the same v1 format
the renderer reads/writes via --camera). The script interpolates intermediate
poses, renders one PNG per frame, and assembles them into a GIF.

----------------------------------------------------------------------------
Quick examples
----------------------------------------------------------------------------

1) Diamond scene, 60 frames between two keyframes you already saved:

   python scripts/render_camera_path.py \
       --scene assets/diamond/scene.gltf \
       --keyframes camera_diamond.txt cameras/diamond_side.txt \
       --frames 60 --fps 24 --spp 32 --resolution 960 540 \
       --out renders/diamond_orbit.gif

2) Use a JSON trajectory file (lets you set per-segment durations etc.):

   python scripts/render_camera_path.py \
       --scene assets/diamond/scene.gltf \
       --trajectory trajectories/diamond_loop.json \
       --out renders/diamond_loop.gif

   trajectory JSON schema:
   {
     "frames": 120,            // total frames (overridden by --frames)
     "fps": 24,                // playback FPS (overridden by --fps)
     "loop": true,             // wrap last keyframe back to first
     "interpolation": "catmull",  // "linear" | "catmull"
     "keyframes": [
        "camera_diamond.txt",
        "cameras/diamond_side.txt",
        {"position": [0,1,4.5], "yaw": 268.95, "pitch": -0.55,
         "fov_deg": 34.0, "aspect": 1.7778, "near": 0.05, "far": 100}
     ]
   }

3) Replay a recorded path. In the GUI, press F5 to start recording, fly around
   with WASD+mouse, press F5 again to stop. The app writes
   build-all/Release/recordings/path_<stamp>.json, which this script can replay:

   python scripts/render_camera_path.py \
       --scene Bistro_v5_2/BistroInterior.fbx \
       --recording build-all/Release/recordings/path_20260505_013422.json \
       --replay-stride 9 --resolution 1280 720 --spp 1 --mode native \
       --out renders/bistro_replay.gif

   In --recording mode the script drives pathtracer's in-process replay loop
   (one subprocess, one scene load, N frames). --replay-stride decides how
   many recorded poses to skip between rendered frames (1 = every pose).
   Playback duration defaults to the recording's wall-clock length; pass --fps
   only when you want slow-mo or speed-up.

   For stride=1 at 1280x720 the resulting GIF can be huge (hundreds of MB).
   Switch to mp4 with --out foo.mp4 (or --encoder ffmpeg) — typically 10-30x
   smaller for the same quality.

----------------------------------------------------------------------------
Notes
----------------------------------------------------------------------------
- The pathtracer is invoked headlessly via `-f out.png -s <spp>`. Each frame
  is an *independent* render, so high SPP per frame is the right knob for
  quality (denoisers/ReSTIR temporal accumulation across frames is not
  carried over).
- Pillow is required for GIF assembly. ffmpeg (if available) can also be
  used via --encoder ffmpeg to produce mp4 instead.
- yaw is interpolated along the shorter angular arc to avoid 359°->1° spins.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Camera file IO (matches Application::loadCameraFile / saveCameraFile format)
# ---------------------------------------------------------------------------

@dataclass
class CameraPose:
    position: tuple[float, float, float]
    yaw: float          # degrees
    pitch: float        # degrees
    fov_deg: float
    aspect: float
    near: float
    far: float

    @classmethod
    def default(cls) -> "CameraPose":
        return cls((0.0, 1.0, 4.5), 268.95, -0.55, 34.0736, 16.0 / 9.0, 0.05, 100.0)

    @classmethod
    def from_file(cls, path: Path) -> "CameraPose":
        text = path.read_text(encoding="utf-8")
        kv: dict[str, list[str]] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            kv[parts[0]] = parts[1:]
        base = cls.default()
        pos_tokens = kv.get("position", list(map(str, base.position)))
        return cls(
            position=(float(pos_tokens[0]), float(pos_tokens[1]), float(pos_tokens[2])),
            yaw=float(kv.get("yaw", [str(base.yaw)])[0]),
            pitch=float(kv.get("pitch", [str(base.pitch)])[0]),
            fov_deg=float(kv.get("fov_deg", [str(base.fov_deg)])[0]),
            aspect=float(kv.get("aspect", [str(base.aspect)])[0]),
            near=float(kv.get("near", [str(base.near)])[0]),
            far=float(kv.get("far", [str(base.far)])[0]),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "CameraPose":
        base = cls.default()
        pos = d.get("position", list(base.position))
        return cls(
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            yaw=float(d.get("yaw", base.yaw)),
            pitch=float(d.get("pitch", base.pitch)),
            fov_deg=float(d.get("fov_deg", base.fov_deg)),
            aspect=float(d.get("aspect", base.aspect)),
            near=float(d.get("near", base.near)),
            far=float(d.get("far", base.far)),
        )

    def to_file(self, path: Path) -> None:
        text = (
            "# path_tracer camera v1\n"
            f"position {self.position[0]} {self.position[1]} {self.position[2]}\n"
            f"yaw {self.yaw}\n"
            f"pitch {self.pitch}\n"
            f"fov_deg {self.fov_deg}\n"
            f"aspect {self.aspect}\n"
            f"near {self.near}\n"
            f"far {self.far}\n"
        )
        path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _wrap_angle_delta(a: float, b: float) -> float:
    """Shortest signed delta from a to b in degrees."""
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def lerp_angle(a: float, b: float, t: float) -> float:
    return a + _wrap_angle_delta(a, b) * t


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_pose(p0: CameraPose, p1: CameraPose, t: float) -> CameraPose:
    return CameraPose(
        position=(
            lerp(p0.position[0], p1.position[0], t),
            lerp(p0.position[1], p1.position[1], t),
            lerp(p0.position[2], p1.position[2], t),
        ),
        yaw=lerp_angle(p0.yaw, p1.yaw, t),
        pitch=lerp(p0.pitch, p1.pitch, t),
        fov_deg=lerp(p0.fov_deg, p1.fov_deg, t),
        aspect=lerp(p0.aspect, p1.aspect, t),
        near=lerp(p0.near, p1.near, t),
        far=lerp(p0.far, p1.far, t),
    )


def catmull_rom_3(p0: tuple[float, float, float],
                  p1: tuple[float, float, float],
                  p2: tuple[float, float, float],
                  p3: tuple[float, float, float],
                  t: float) -> tuple[float, float, float]:
    """Centripetal-style Catmull-Rom (uniform parameterisation; OK for
    gentle paths). t in [0,1] interpolates between p1 and p2."""
    t2 = t * t
    t3 = t2 * t
    out = []
    for i in range(3):
        a, b, c, d = p0[i], p1[i], p2[i], p3[i]
        v = 0.5 * ((2.0 * b)
                   + (-a + c) * t
                   + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                   + (-a + 3.0 * b - 3.0 * c + d) * t3)
        out.append(v)
    return tuple(out)  # type: ignore[return-value]


def sample_path(keyframes: list[CameraPose], n_frames: int,
                interpolation: str, loop: bool) -> list[CameraPose]:
    """Produce n_frames poses spread evenly along a piecewise path through the
    keyframes. With loop=True the last segment connects back to keyframes[0]."""
    if n_frames <= 0:
        return []
    if len(keyframes) == 1:
        return [keyframes[0]] * n_frames

    seq = list(keyframes) + ([keyframes[0]] if loop else [])
    n_segments = len(seq) - 1
    poses: list[CameraPose] = []
    for i in range(n_frames):
        # Map frame i -> global parameter u in [0, n_segments]. Both ends
        # inclusive when looping is off, else last frame stops just shy of
        # wrap-around so the GIF doesn't pause on the duplicated endpoint.
        denom = (n_frames - 1) if (n_frames > 1 and not loop) else max(n_frames, 1)
        u = (i / denom) * n_segments if n_segments > 0 else 0.0
        if u >= n_segments:
            u = n_segments - 1e-6
        seg = int(math.floor(u))
        t = u - seg
        a = seq[seg]
        b = seq[seg + 1]
        if interpolation == "catmull" and len(seq) >= 3:
            # Pick previous/next for tangents, clamping or wrapping at ends.
            if loop:
                prev_idx = (seg - 1) % len(keyframes)
                next_idx = (seg + 2) % len(keyframes)
                prev_p = keyframes[prev_idx]
                next_p = keyframes[next_idx]
            else:
                prev_p = seq[seg - 1] if seg - 1 >= 0 else seq[seg]
                next_p = seq[seg + 2] if seg + 2 < len(seq) else seq[seg + 1]
            pos = catmull_rom_3(prev_p.position, a.position, b.position, next_p.position, t)
            pose = CameraPose(
                position=pos,
                yaw=lerp_angle(a.yaw, b.yaw, t),
                pitch=lerp(a.pitch, b.pitch, t),
                fov_deg=lerp(a.fov_deg, b.fov_deg, t),
                aspect=lerp(a.aspect, b.aspect, t),
                near=lerp(a.near, b.near, t),
                far=lerp(a.far, b.far, t),
            )
        else:
            pose = lerp_pose(a, b, t)
        poses.append(pose)
    return poses


# ---------------------------------------------------------------------------
# Renderer driver
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    exe: Path
    run_dir: Path
    scene: Path
    width: int
    height: int
    spp: int
    backend: str          # "cuda" | "optix"
    mode: str             # "native" | "nrd" | "dlss" | "dlssonly" | "rr"
    sg: str | None        # passthrough for --sg
    extra_args: list[str]


def render_one(cfg: RenderConfig, camera_file: Path, out_png: Path,
               quiet: bool = True) -> None:
    """Invoke pathtracer.exe in headless mode for a single frame."""
    cmd: list[str] = [
        str(cfg.exe),
        str(cfg.scene.resolve()),
        "-r", str(cfg.width), str(cfg.height),
        "-s", str(cfg.spp),
        "-f", str(out_png.resolve()),
        "--backend", cfg.backend,
        "--mode", cfg.mode,
        "--camera", str(camera_file.resolve()),
    ]
    if cfg.sg:
        cmd += ["--sg", cfg.sg]
    cmd += cfg.extra_args

    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    proc = subprocess.run(cmd, cwd=str(cfg.run_dir), stdout=stdout, stderr=stderr)
    if proc.returncode != 0:
        # Re-run loud so the user sees what happened.
        subprocess.run(cmd, cwd=str(cfg.run_dir))
        raise RuntimeError(f"pathtracer exited with code {proc.returncode}")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def assemble_gif(frames: list[Path], out_gif: Path, fps: float, loop: int = 0) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for GIF assembly. Install with: pip install Pillow"
        ) from exc

    if not frames:
        raise SystemExit("No frames to assemble.")
    duration_ms = max(int(round(1000.0 / fps)), 20)
    images = [Image.open(p).convert("RGB") for p in frames]
    # Use adaptive palette for cleaner colors than the default web palette.
    palette_imgs = [im.convert("P", palette=Image.Palette.ADAPTIVE, dither=Image.Dither.FLOYDSTEINBERG)
                    for im in images]
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    palette_imgs[0].save(
        out_gif,
        save_all=True,
        append_images=palette_imgs[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
        disposal=2,
    )


def assemble_ffmpeg(frames_dir: Path, pattern: str, out_path: Path, fps: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Trajectory file loading
# ---------------------------------------------------------------------------

def load_keyframe(item, base_dir: Path) -> CameraPose:
    """An item is either a path string to a camera v1 file, or a dict literal."""
    if isinstance(item, str):
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        if not candidate.exists():
            raise SystemExit(f"Keyframe camera file not found: {candidate}")
        return CameraPose.from_file(candidate)
    if isinstance(item, dict):
        return CameraPose.from_dict(item)
    raise SystemExit(f"Unrecognised keyframe entry: {item!r}")


def load_trajectory(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "keyframes" not in data or not isinstance(data["keyframes"], list):
        raise SystemExit("Trajectory JSON must have a 'keyframes' list")
    return data


# ---------------------------------------------------------------------------
# Recording (F5 in-app capture) loading & resampling
# ---------------------------------------------------------------------------

def load_recording(path: Path) -> tuple[list[float], list[CameraPose]]:
    """Load a recording produced by Application::stopRecording. Returns
    (timestamps_seconds, poses)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("poses")
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"Recording {path} has no poses")
    times: list[float] = []
    poses: list[CameraPose] = []
    for entry in raw:
        times.append(float(entry["t"]))
        poses.append(CameraPose.from_dict(entry))
    # Defensive: timestamps must be monotone non-decreasing for the resampler.
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            raise SystemExit(f"Recording {path} has non-monotone timestamp at frame {i}")
    return times, poses


def resample_recording(times: list[float], poses: list[CameraPose],
                       target_fps: float, target_frames: int | None) -> list[CameraPose]:
    """Linearly interpolate poses at evenly spaced output times. If target_frames
    is None, derive frame count from duration * target_fps."""
    duration = times[-1] - times[0]
    if duration <= 0.0:
        # Degenerate single-instant recording: just repeat the pose.
        return [poses[0]] * (target_frames or 1)
    if target_frames is None:
        target_frames = max(1, int(round(duration * target_fps)))
    out: list[CameraPose] = []
    j = 0
    for i in range(target_frames):
        # Even spacing across [t0, tN] inclusive.
        t = times[0] + (duration * i / max(target_frames - 1, 1)) if target_frames > 1 else times[0]
        # Advance bracket so that times[j] <= t <= times[j+1].
        while j + 1 < len(times) - 1 and times[j + 1] < t:
            j += 1
        a, b = poses[j], poses[j + 1]
        ta, tb = times[j], times[j + 1]
        u = 0.0 if tb <= ta else (t - ta) / (tb - ta)
        u = max(0.0, min(1.0, u))
        out.append(lerp_pose(a, b, u))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, type=Path, help="Scene file (passed to pathtracer)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--keyframes", nargs="+",
                     help="One or more camera v1 files to interpolate between")
    src.add_argument("--trajectory", type=Path,
                     help="JSON trajectory file (see top-of-file docs)")
    src.add_argument("--recording", type=Path,
                     help="JSON recording produced by F5 in the GUI; replays the captured path")
    p.add_argument("--replay-stride", type=int, default=1,
                   help="With --recording: render every Nth recorded pose (default 1 = all). "
                        "Larger stride = fewer frames = faster render. Playback length is preserved.")

    p.add_argument("--frames", type=int, default=None,
                   help="Total frames to render. Ignored with --recording (use --replay-stride).")
    p.add_argument("--fps", type=float, default=None,
                   help="Playback FPS. With --recording, defaults to the recording's real-time pace; "
                        "set explicitly only for slow-mo / speed-up.")
    p.add_argument("--loop", action=argparse.BooleanOptionalAction, default=None,
                   help="Wrap last keyframe back to first (default off, or 'loop' field of trajectory)")
    p.add_argument("--interpolation", choices=["linear", "catmull"], default=None,
                   help="Position interpolation (default catmull when ≥3 keyframes, else linear)")

    p.add_argument("--resolution", "-r", nargs=2, type=int, metavar=("W", "H"),
                   default=(1280, 720))
    p.add_argument("--spp", type=int, default=32, help="Samples per frame (default 32)")
    p.add_argument("--backend", choices=["cuda", "optix"], default="optix")
    p.add_argument("--mode", default="native",
                   choices=["native", "nrd", "dlss", "dlssonly", "rr"],
                   help="Renderer pipeline (default native)")
    p.add_argument("--sg", default=None, help="--sg passthrough (off|heuristic|fbx-c4d|fbx-ue)")

    # Build/exe layout.
    project_root = Path(__file__).resolve().parents[1]
    p.add_argument("--exe", type=Path,
                   default=project_root / "build-all" / "Release" / "pathtracer.exe",
                   help="Path to pathtracer executable")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Working dir for pathtracer (defaults to exe's directory)")

    p.add_argument("--out", type=Path, required=True,
                   help="Output GIF (or mp4 with --encoder ffmpeg)")
    p.add_argument("--encoder", choices=["pillow", "ffmpeg"], default="pillow")
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="Where to write per-frame PNGs (default: temp dir, deleted at end)")
    p.add_argument("--keep-frames", action="store_true",
                   help="Keep the per-frame PNGs even when frames-dir is a temp dir")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Skip rendering frames < this index (useful for resume)")
    p.add_argument("--verbose", action="store_true",
                   help="Stream pathtracer stdout/stderr instead of swallowing it")

    # Anything after a literal `--` gets forwarded to the renderer untouched.
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args appended to every pathtracer invocation (use after --extra)")
    return p.parse_args()


def run_replay_subprocess(cfg: "RenderConfig", recording: Path, frames_dir: Path,
                          spp: int, stride: int, quiet: bool) -> int:
    """Drive pathtracer's in-process replay loop. Renders all selected poses in a
    single subprocess (one scene load) into frames_dir/frame_NNNNNN.png. Returns
    the number of PNGs actually written."""
    cmd: list[str] = [
        str(cfg.exe),
        str(cfg.scene.resolve()),
        "-r", str(cfg.width), str(cfg.height),
        "--backend", cfg.backend,
        "--mode", cfg.mode,
        "--replay", str(recording.resolve()),
        "--replay-out", str(frames_dir.resolve()),
        "--replay-spp", str(spp),
        "--replay-stride", str(stride),
    ]
    if cfg.sg:
        cmd += ["--sg", cfg.sg]
    cmd += cfg.extra_args

    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    proc = subprocess.run(cmd, cwd=str(cfg.run_dir), stdout=stdout, stderr=stderr)
    if proc.returncode != 0:
        # Re-run loud so the user sees what happened.
        subprocess.run(cmd, cwd=str(cfg.run_dir))
        raise RuntimeError(f"pathtracer exited with code {proc.returncode}")

    return len(sorted(frames_dir.glob("frame_*.png")))


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_dir = project_root  # for resolving relative keyframe paths

    keyframes: list[CameraPose] = []
    interp = "linear"
    loop = False

    # ── Recording mode: drive pathtracer's --replay loop in a single subprocess.
    # Skip the per-frame Python orchestration entirely.
    if args.recording:
        if args.frames is not None:
            print("[error] --frames is ignored with --recording. Use --replay-stride instead.",
                  file=sys.stderr)
            return 2
        times, recorded = load_recording(args.recording)
        recording_duration = times[-1] - times[0]
        if args.replay_stride < 1:
            print("[error] --replay-stride must be >= 1", file=sys.stderr)
            return 2

        # Frame storage
        if args.frames_dir:
            frames_dir = args.frames_dir.resolve()
            cleanup = False
        else:
            frames_dir = Path(tempfile.mkdtemp(prefix="path_render_")).resolve()
            cleanup = not args.keep_frames
        frames_dir.mkdir(parents=True, exist_ok=True)
        # Wipe any stale PNGs so the post-render glob counts only this run's output.
        for stale in frames_dir.glob("frame_*.png"):
            stale.unlink()

        exe = args.exe.resolve()
        if not exe.exists():
            print(f"[error] pathtracer not found: {exe}", file=sys.stderr)
            return 2
        run_dir = (args.run_dir or exe.parent).resolve()
        w, h = args.resolution
        cfg = RenderConfig(
            exe=exe, run_dir=run_dir, scene=args.scene,
            width=w, height=h, spp=args.spp,
            backend=args.backend, mode=args.mode, sg=args.sg,
            extra_args=list(args.extra or []),
        )

        print(f"Renderer : {exe}")
        print(f"Run dir  : {run_dir}")
        print(f"Scene    : {args.scene}")
        print(f"Source   : recording={args.recording.name} ({len(recorded)} poses, "
              f"{recording_duration:.2f} s)")
        print(f"Stride   : {args.replay_stride} (renders ~{(len(recorded) + args.replay_stride - 1) // args.replay_stride} frames)")
        print(f"Resolution: {w}x{h}, spp={args.spp}, backend={args.backend}, mode={args.mode}")
        print(f"Frames dir: {frames_dir} (cleanup={cleanup})")

        rendered_count = run_replay_subprocess(cfg, args.recording, frames_dir,
                                               args.spp, args.replay_stride,
                                               quiet=not args.verbose)
        if rendered_count == 0:
            print("[error] pathtracer wrote no frames", file=sys.stderr)
            return 1

        # Real-time playback: each output frame holds for (duration / N_frames).
        real_pace_fps = rendered_count / recording_duration if recording_duration > 0 else 24.0
        if args.fps is not None:
            playback_fps = args.fps
            if abs(playback_fps - real_pace_fps) > 0.05:
                print(f"[note] --fps {playback_fps} overrides real-time pace ({real_pace_fps:.2f} fps); "
                      f"playback will be {real_pace_fps / playback_fps:.2f}x of recorded speed.")
        else:
            playback_fps = real_pace_fps
            print(f"Playback : {playback_fps:.2f} fps (matches recording duration {recording_duration:.2f} s)")

        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        out_path = args.out.resolve()
        if args.encoder == "ffmpeg" or out_path.suffix.lower() in {".mp4", ".mov", ".webm"}:
            assemble_ffmpeg(frames_dir, "frame_%06d.png", out_path, playback_fps)
        else:
            assemble_gif(frame_paths, out_path, playback_fps)
        print(f"\nWrote {out_path}")

        if cleanup:
            shutil.rmtree(frames_dir, ignore_errors=True)
        else:
            print(f"(per-frame PNGs kept under {frames_dir})")
        return 0

    if args.trajectory:
        traj = load_trajectory(args.trajectory)
        base_dir = args.trajectory.parent
        keyframes = [load_keyframe(item, base_dir) for item in traj["keyframes"]]
        n_frames = args.frames if args.frames is not None else int(traj.get("frames", 60))
        fps = args.fps if args.fps is not None else float(traj.get("fps", 24.0))
        loop = args.loop if args.loop is not None else bool(traj.get("loop", False))
        interp = args.interpolation or str(traj.get("interpolation", "catmull"))
        poses = sample_path(keyframes, n_frames, interp, loop)
    else:
        keyframes = [CameraPose.from_file(Path(p)) for p in args.keyframes]
        n_frames = args.frames if args.frames is not None else 60
        fps = args.fps if args.fps is not None else 24.0
        loop = bool(args.loop) if args.loop is not None else False
        interp = args.interpolation or ("catmull" if len(keyframes) >= 3 else "linear")
        poses = sample_path(keyframes, n_frames, interp, loop)

    if len(keyframes) < 1:
        print("Need at least one keyframe / recorded frame.", file=sys.stderr)
        return 2
    if len(keyframes) < 2 and n_frames > 1 and not args.recording:
        print("[warn] Only one keyframe — every frame will be identical.", file=sys.stderr)

    # Force aspect from --resolution so it actually matches the rendered image
    # (otherwise a 16:9 keyframe rendered at 1:1 would look squashed).
    w, h = args.resolution
    target_aspect = w / h
    for pose in poses:
        pose.aspect = target_aspect

    exe = args.exe.resolve()
    if not exe.exists():
        print(f"[error] pathtracer not found: {exe}", file=sys.stderr)
        print("        Build with PATHTRACER_ENABLE_OPTIX=ON first, or pass --exe.", file=sys.stderr)
        return 2
    run_dir = (args.run_dir or exe.parent).resolve()

    # Frame storage
    if args.frames_dir:
        frames_dir = args.frames_dir.resolve()
        frames_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        frames_dir = Path(tempfile.mkdtemp(prefix="path_render_")).resolve()
        cleanup = not args.keep_frames

    # Recording mode returned earlier — only trajectory/keyframes reach here.
    if args.trajectory:
        src_desc = f"trajectory={args.trajectory.name} ({len(keyframes)} keyframes, {interp}{', loop' if loop else ''})"
    else:
        src_desc = f"keyframes={len(keyframes)} ({interp}{', loop' if loop else ''})"

    print(f"Renderer : {exe}")
    print(f"Run dir  : {run_dir}")
    print(f"Scene    : {args.scene}")
    print(f"Source   : {src_desc}")
    print(f"Frames   : {n_frames} @ {fps} fps  -> {n_frames / fps:.2f} s")
    print(f"Resolution: {w}x{h}, spp={args.spp}, backend={args.backend}, mode={args.mode}")
    print(f"Frames dir: {frames_dir} (cleanup={cleanup})")

    cfg = RenderConfig(
        exe=exe,
        run_dir=run_dir,
        scene=args.scene,
        width=w,
        height=h,
        spp=args.spp,
        backend=args.backend,
        mode=args.mode,
        sg=args.sg,
        extra_args=list(args.extra or []),
    )

    cam_dir = frames_dir / "cameras"
    cam_dir.mkdir(exist_ok=True)
    frame_paths: list[Path] = []
    for i, pose in enumerate(poses):
        cam_file = cam_dir / f"frame_{i:05d}.txt"
        png_file = frames_dir / f"frame_{i:05d}.png"
        pose.to_file(cam_file)
        frame_paths.append(png_file)
        if i < args.start_frame and png_file.exists():
            print(f"[{i+1:>4}/{n_frames}] skip (already exists)")
            continue
        print(f"[{i+1:>4}/{n_frames}] rendering -> {png_file.name}")
        render_one(cfg, cam_file, png_file, quiet=not args.verbose)

    # Assemble.
    out_path = args.out.resolve()
    if args.encoder == "ffmpeg" or out_path.suffix.lower() in {".mp4", ".mov", ".webm"}:
        assemble_ffmpeg(frames_dir, "frame_%05d.png", out_path, fps)
    else:
        assemble_gif(frame_paths, out_path, fps)
    print(f"\nWrote {out_path}")

    if cleanup:
        shutil.rmtree(frames_dir, ignore_errors=True)
    else:
        print(f"(per-frame PNGs kept under {frames_dir})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
