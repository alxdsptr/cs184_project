#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_reference_sweep.sh — render ONLY the reference sequence used by
# run_quality_sweep.sh as ground truth.
#
# Why this is a separate script:
#   The reference sweep dwells for many frames at each capture point (the
#   accumulator integrates a near-converged image per pose). That's
#   expensive, but it doesn't change between test runs as long as the
#   scene + camera path stay the same. Running this once and then
#   re-using its output across many test sweeps saves a lot of time.
#
# Output:
#   reference_NNNNNN.png in <out dir> (default: screenshots/sweep). The
#   filenames and frame indices line up frame-for-frame with whatever
#   run_quality_sweep.sh later renders for the test tags.
#
# Usage (run from anywhere — script resolves project root from $0):
#     tools/run_reference_sweep.sh <scene_file> [extra_pathtracer_flags ...]
#
# Examples:
#     tools/run_reference_sweep.sh assets/scene.fbx --camera mycam.txt
#     tools/run_reference_sweep.sh assets/scene.fbx --reference-dwell 4000
#     tools/run_reference_sweep.sh assets/scene.fbx --backend cuda
#
# Recognised script flags (consumed before forwarding):
#     --backend <cuda|optix>  Runtime backend. Default: optix.
#     --reference-dwell <N>   Dwell frames per capture point during the
#                             reference sweep (default 2000). Higher → more
#                             converged ref images, longer runtime.
#     --pathtracer <exe>      Override the pathtracer.exe path.
#                             Default: build-optix/Release/pathtracer.exe.
#     --out <dir>             Output directory.
#                             Defaults to <project_root>/screenshots/sweep.
#
# Anything not in that list is forwarded to the pathtracer (--camera,
# --capture-speed, --capture-orbit, --emissive-target, ...). The capture
# parameters (warmup/frames/stride/motion) MUST match those in
# run_quality_sweep.sh — they're kept identical here.
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
}

[[ $# -ge 1 ]] || usage

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENE_ARG="$1"
shift

BACKEND="optix"
REFERENCE_DWELL=2000
PATHTRACER_REL="build-optix/Release/pathtracer.exe"
OUTDIR_REL="screenshots/sweep"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)            BACKEND="$2"; shift 2 ;;
        --reference-dwell)    REFERENCE_DWELL="$2"; shift 2 ;;
        --pathtracer)         PATHTRACER_REL="$2"; shift 2 ;;
        --out)                OUTDIR_REL="$2"; shift 2 ;;
        --capture-tag)
            echo "warning: ignoring --capture-tag (this script always tags 'reference')" >&2
            shift 2 ;;
        --capture-out)
            echo "warning: ignoring --capture-out; use --out <dir> instead" >&2
            shift 2 ;;
        --capture-dwell)
            echo "warning: ignoring --capture-dwell (use --reference-dwell)" >&2
            shift 2 ;;
        --no-restir|--no-restir-gi|--no-restir-pt|--restir-gi|--restir-pt)
            echo "warning: ignoring '$1' (reference is always pure path-traced)" >&2
            shift ;;
        *) FORWARD_ARGS+=("$1"); shift ;;
    esac
done

case "${BACKEND}" in
    cuda|optix) ;;
    *) echo "ERROR: --backend must be 'cuda' or 'optix' (got '${BACKEND}')" >&2; exit 1 ;;
esac

if [[ -e "${SCENE_ARG}" ]]; then
    SCENE_ABS="$(cd -- "$(dirname -- "${SCENE_ARG}")" && pwd)/$(basename -- "${SCENE_ARG}")"
elif [[ -e "${PROJECT_ROOT}/${SCENE_ARG}" ]]; then
    SCENE_ABS="${PROJECT_ROOT}/${SCENE_ARG}"
else
    echo "ERROR: scene file not found: ${SCENE_ARG}" >&2
    exit 1
fi

PATHTRACER="${PROJECT_ROOT}/${PATHTRACER_REL}"
if [[ ! -x "${PATHTRACER}" ]]; then
    echo "ERROR: ${PATHTRACER} not found." >&2
    echo "       Build it first, or pass --pathtracer <path>." >&2
    exit 1
fi

OUTDIR="${PROJECT_ROOT}/${OUTDIR_REL}"
[[ "${OUTDIR_REL}" = /* || "${OUTDIR_REL}" =~ ^[A-Za-z]: ]] && OUTDIR="${OUTDIR_REL}"
mkdir -p "${OUTDIR}"

to_win() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        printf '%s' "$1"
    fi
}

maybe_winize() {
    local s="$1"
    if [[ -e "$s" ]]; then
        local abs
        abs="$(cd -- "$(dirname -- "$s")" && pwd)/$(basename -- "$s")"
        to_win "${abs}"
    else
        printf '%s' "$s"
    fi
}

FORWARD_W=()
for a in "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"; do
    FORWARD_W+=("$(maybe_winize "$a")")
done

PATHTRACER_W="$(to_win "${PATHTRACER}")"
SCENE_W="$(to_win "${SCENE_ABS}")"
OUTDIR_W="$(to_win "${OUTDIR}")"

# These MUST match run_quality_sweep.sh so frame indices align.
WARMUP=120
FRAMES=240
STRIDE=30
RES_ARGS=(-r 1280 720)
BACKEND_ARGS=(--backend "${BACKEND}")
MOTION_ARGS=(--capture-dolly --capture-speed 0.3)

echo
echo "=== Reference sweep: native, dwell=${REFERENCE_DWELL} per capture point ==="
"${PATHTRACER_W}" "${SCENE_W}" "${RES_ARGS[@]}" "${BACKEND_ARGS[@]}" \
    --no-restir --no-restir-gi --no-restir-pt \
    --capture-tag "reference" \
    --capture-out "${OUTDIR_W}" \
    --capture-warmup "${WARMUP}" \
    --capture-frames "${FRAMES}" \
    --capture-stride "${STRIDE}" \
    --capture-dwell "${REFERENCE_DWELL}" \
    "${MOTION_ARGS[@]}" \
    "${FORWARD_W[@]+"${FORWARD_W[@]}"}"

echo
echo "Reference sweep done. Frames in: ${OUTDIR}"
echo "Now run: tools/run_quality_sweep.sh <scene> [flags...]"
