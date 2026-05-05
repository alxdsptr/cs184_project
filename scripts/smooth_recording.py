#!/usr/bin/env python3
"""
Smooth the per-frame jitter out of a recorded camera path (F5 in the GUI),
producing a new JSON in the same v1 format that --replay accepts.

Why this exists
---------------
The GUI's mouse/keyboard input quantises rotation to ~0.15 deg steps, and WASD
gives a slightly noisy translation. When that recording is replayed through
the path tracer, the staircase shows up directly in motion vectors, which
NRD/DLSS/ReSTIR use for history reprojection. A small temporal low-pass on
the camera path removes the stairstep without distorting the route.

Why low-pass instead of spline interpolation
--------------------------------------------
The recording is a *dense* sample (~28 fps, hundreds of poses), not sparse
keyframes. A Catmull-Rom spline would still pass through every wobble.
A Gaussian smoothing kernel on [x, y, z, yaw, pitch] removes the wobble
while leaving the path shape intact.

Static-block handling
---------------------
Application::runReplay() walks every pose and feeds (pose[i-1] -> pose[i])
into the denoiser as motion vectors. A run of byte-identical poses at the
start/end yields zero motion vectors, which lets NRD/DLSS/ReSTIR history
settle. Smearing motion into those holds defeats the purpose, so:
  - Leading/trailing identical-pose runs are detected and left untouched.
  - The Gaussian kernel still sees them (so the smoothed values at the
    boundary taper to match), but their values are never overwritten.

Yaw unwrap
----------
Yaw is unwrapped (consecutive deltas folded to (-180 deg, 180 deg]) before
smoothing, so the kernel never averages across a 359 deg->1 deg boundary.

Usage
-----
    python scripts/smooth_recording.py path_in.json
        # writes path_in_smoothed.json next to the input

    python scripts/smooth_recording.py path_in.json --sigma 8 --out smoothed.json
        # heavier smoothing, explicit output path

    python scripts/smooth_recording.py path_in.json --drop-static
        # also strip the leading/trailing held-still frames
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Smoothing primitives
# ---------------------------------------------------------------------------

def unwrap_angles_deg(angles: list[float]) -> list[float]:
    """Make a degree sequence continuous by folding each delta to (-180, 180]."""
    if not angles:
        return []
    out = [angles[0]]
    for a in angles[1:]:
        d = a - out[-1]
        # Fold into (-180, 180]
        d -= 360.0 * math.floor((d + 180.0) / 360.0)
        out.append(out[-1] + d)
    return out


def gaussian_smooth(values: list[float], sigma: float) -> list[float]:
    """1D Gaussian smoothing with edge-clamped padding. Sigma is in samples."""
    n = len(values)
    if n == 0 or sigma <= 0.0:
        return list(values)
    radius = max(1, int(math.ceil(sigma * 3.0)))
    weights = [math.exp(-0.5 * (k / sigma) ** 2) for k in range(-radius, radius + 1)]
    out: list[float] = []
    for i in range(n):
        num = 0.0
        den = 0.0
        for k in range(-radius, radius + 1):
            j = i + k
            if j < 0:
                j = 0
            elif j >= n:
                j = n - 1
            w = weights[k + radius]
            num += w * values[j]
            den += w
        out.append(num / den)
    return out


def boundary_blend(orig: list[float], smoothed: list[float],
                   head_end: int, tail_start: int, ramp: int) -> list[float]:
    """Smoothstep-blend smoothed -> orig within `ramp` frames of each
    static-block boundary. Without this, a Gaussian at the boundary frame
    averages over the moving section ahead and lands well off the frozen
    static value, causing a one-frame motion-vector spike right when motion
    starts/ends."""
    out = list(smoothed)
    if ramp <= 0:
        return out
    # Head boundary: at index head_end use orig (matches the static value's
    # natural first step); fade to fully smoothed by head_end + ramp.
    head_stop = min(head_end + ramp + 1, tail_start)
    for i in range(head_end, head_stop):
        t = (i - head_end) / float(ramp)
        if t > 1.0:
            t = 1.0
        w = t * t * (3.0 - 2.0 * t)  # smoothstep
        out[i] = orig[i] * (1.0 - w) + smoothed[i] * w
    # Tail boundary: mirror image.
    tail_start_idx = max(head_end, tail_start - 1 - ramp)
    for i in range(tail_start_idx, tail_start):
        t = (tail_start - 1 - i) / float(ramp)
        if t > 1.0:
            t = 1.0
        w = t * t * (3.0 - 2.0 * t)
        out[i] = orig[i] * (1.0 - w) + smoothed[i] * w
    return out


# ---------------------------------------------------------------------------
# Pose comparison (for static-block detection)
# ---------------------------------------------------------------------------

def pose_key(p: dict) -> tuple:
    """The fields that change with WASD / mouse. If two consecutive poses
    have identical keys, the camera was held still."""
    pos = p["position"]
    return (pos[0], pos[1], pos[2], p["yaw"], p["pitch"])


def find_static_blocks(poses: list[dict]) -> tuple[int, int]:
    """Return (head_end, tail_start) such that poses[0:head_end] all match
    poses[0] and poses[tail_start:] all match poses[-1]. The two ranges are
    not allowed to overlap — if the entire recording is static, head_end ==
    tail_start == len(poses)."""
    n = len(poses)
    if n == 0:
        return 0, 0
    first = pose_key(poses[0])
    head_end = 1
    while head_end < n and pose_key(poses[head_end]) == first:
        head_end += 1
    last = pose_key(poses[-1])
    tail_start = n - 1
    while tail_start > head_end and pose_key(poses[tail_start - 1]) == last:
        tail_start -= 1
    return head_end, tail_start


# ---------------------------------------------------------------------------
# Stats (so the user can see whether smoothing actually changed anything)
# ---------------------------------------------------------------------------

def max_consecutive_delta(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(abs(values[i] - values[i - 1]) for i in range(1, len(values)))


def report_stats(label: str, poses: list[dict]) -> None:
    if len(poses) < 2:
        print(f"  {label}: <2 poses, no deltas")
        return
    xs = [p["position"][0] for p in poses]
    ys = [p["position"][1] for p in poses]
    zs = [p["position"][2] for p in poses]
    yaws = unwrap_angles_deg([p["yaw"] for p in poses])
    pitches = [p["pitch"] for p in poses]
    pos_max = max(
        math.sqrt(
            (xs[i] - xs[i - 1]) ** 2
            + (ys[i] - ys[i - 1]) ** 2
            + (zs[i] - zs[i - 1]) ** 2
        )
        for i in range(1, len(poses))
    )
    print(
        f"  {label}: max |Dpos|={pos_max:.5f}  "
        f"max |Dyaw|={max_consecutive_delta(yaws):.3f} deg  "
        f"max |Dpitch|={max_consecutive_delta(pitches):.3f} deg"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def smooth_recording(input_path: Path, output_path: Path, sigma: float,
                     drop_static: bool) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    poses: list[dict] = data.get("poses", [])
    if not poses:
        raise SystemExit(f"No 'poses' array in {input_path}")

    n = len(poses)
    head_end, tail_start = find_static_blocks(poses)
    moving_count = max(0, tail_start - head_end)

    print(f"Input    : {input_path}  ({n} poses)")
    print(f"Static   : head=[0:{head_end}]  tail=[{tail_start}:{n}]  "
          f"moving={moving_count}")
    print(f"Sigma    : {sigma} frames  (kernel radius ~ {int(math.ceil(sigma * 3))})")
    print("Before:")
    report_stats("all   ", poses)
    if moving_count > 0:
        report_stats("moving", poses[head_end:tail_start])

    if moving_count == 0:
        print("[note] Entire recording is static — nothing to smooth.")
    else:
        # Smooth the entire sequence (so the kernel sees the static blocks
        # as natural anchors), then write back only into the moving frames.
        # Deep-copy values out of the dicts before we start mutating.
        xs = [p["position"][0] for p in poses]
        ys = [p["position"][1] for p in poses]
        zs = [p["position"][2] for p in poses]
        yaws_unwrapped = unwrap_angles_deg([p["yaw"] for p in poses])
        pitches = [p["pitch"] for p in poses]

        xs_s = gaussian_smooth(xs, sigma)
        ys_s = gaussian_smooth(ys, sigma)
        zs_s = gaussian_smooth(zs, sigma)
        yaws_s = gaussian_smooth(yaws_unwrapped, sigma)
        pitches_s = gaussian_smooth(pitches, sigma)

        ramp = max(1, int(math.ceil(sigma * 3.0)))
        xs_s = boundary_blend(xs, xs_s, head_end, tail_start, ramp)
        ys_s = boundary_blend(ys, ys_s, head_end, tail_start, ramp)
        zs_s = boundary_blend(zs, zs_s, head_end, tail_start, ramp)
        yaws_s = boundary_blend(yaws_unwrapped, yaws_s, head_end, tail_start, ramp)
        pitches_s = boundary_blend(pitches, pitches_s, head_end, tail_start, ramp)

        for i in range(head_end, tail_start):
            poses[i]["position"] = [xs_s[i], ys_s[i], zs_s[i]]
            poses[i]["yaw"] = yaws_s[i]
            poses[i]["pitch"] = pitches_s[i]

        print("After:")
        report_stats("all   ", poses)
        report_stats("moving", poses[head_end:tail_start])

    if drop_static:
        before = len(poses)
        poses = poses[head_end:tail_start]
        data["poses"] = poses
        print(f"Dropped static: {before} -> {len(poses)} poses")
        # Rebase timestamps so t starts near zero (purely cosmetic; --replay
        # doesn't read the field, but it keeps the metadata honest).
        if poses:
            t0 = poses[0].get("t", 0.0)
            for p in poses:
                if "t" in p:
                    p["t"] = p["t"] - t0
            data["duration_seconds"] = poses[-1].get("t", 0.0) if poses else 0.0
            data["recorded_frames"] = len(poses)

    # Write JSON: one pose per line so diffs against the original are readable.
    out_lines = ["{"]
    other_keys = [k for k in data.keys() if k != "poses"]
    for k in other_keys:
        out_lines.append(f"  {json.dumps(k)}: {json.dumps(data[k])},")
    out_lines.append('  "poses": [')
    for i, p in enumerate(poses):
        sep = "," if i + 1 < len(poses) else ""
        out_lines.append(f"    {json.dumps(p)}{sep}")
    out_lines.append("  ]")
    out_lines.append("}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote    : {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", type=Path, help="Recording JSON to smooth")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path (default: <input>_smoothed.json)")
    p.add_argument("--sigma", type=float, default=5.0,
                   help="Gaussian smoothing sigma in frames (default 5; try 2–10)")
    p.add_argument("--drop-static", action="store_true",
                   help="Strip leading/trailing held-still frames from the output. "
                        "By default they are preserved (zero motion vectors there "
                        "let denoiser/upscaler history settle before motion begins).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"[error] not found: {args.input}", file=sys.stderr)
        return 2
    if args.sigma < 0:
        print("[error] --sigma must be >= 0", file=sys.stderr)
        return 2
    out = args.out or args.input.with_name(args.input.stem + "_smoothed.json")
    smooth_recording(args.input, out, args.sigma, args.drop_static)
    return 0


if __name__ == "__main__":
    sys.exit(main())
