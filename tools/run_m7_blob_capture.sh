#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_m7_blob_capture.sh — capture consecutive frames of MEASURE_SEVEN with
# a STATIC camera so the only thing varying frame-to-frame is reservoir state.
# Used to reproduce the ReSTIR temporal-reuse "flash-and-decay" artifact:
# a blob suddenly lights up then fades over ~mCap frames.
#
# Usage (run from anywhere):
#   tools/run_m7_blob_capture.sh [--pathtracer <exe>] [--out <dir>]
#                                [--frames N] [--warmup N]
#                                [--variant native|restir-di|restir-gi|restir-pt|all]
#
# Defaults:
#   pathtracer = build-all/Release/pathtracer.exe (the prompt's build target)
#                falls back to build-optix/Release/pathtracer.exe if missing
#   out        = screenshots/m7-blob
#   frames     = 90
#   warmup     = 30
#   variant    = all
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PATHTRACER_REL=""
OUTDIR_REL="screenshots/m7-blob"
FRAMES=90
WARMUP=30
VARIANT="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pathtracer) PATHTRACER_REL="$2"; shift 2 ;;
        --out)        OUTDIR_REL="$2"; shift 2 ;;
        --frames)     FRAMES="$2"; shift 2 ;;
        --warmup)     WARMUP="$2"; shift 2 ;;
        --variant)    VARIANT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' >&2
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Default pathtracer = build-all (prompt build), fall back to build-optix
# (existing build), then to anything called pathtracer.exe under build-*.
if [[ -z "${PATHTRACER_REL}" ]]; then
    for cand in build-all/Release/pathtracer.exe build-optix/Release/pathtracer.exe; do
        if [[ -x "${PROJECT_ROOT}/${cand}" ]]; then PATHTRACER_REL="${cand}"; break; fi
    done
fi
if [[ -z "${PATHTRACER_REL}" ]]; then
    echo "ERROR: no pathtracer.exe found in build-all/ or build-optix/." >&2
    exit 1
fi

PATHTRACER="${PROJECT_ROOT}/${PATHTRACER_REL}"
[[ -x "${PATHTRACER}" ]] || { echo "ERROR: ${PATHTRACER} not executable." >&2; exit 1; }

# The MEASURE_SEVEN scene + camera + emissive flags live in the parent
# project root (not inside the worktree). The worktree is on disk D: and
# shares assets with the parent, so we resolve absolutely.
SCENE="D:/examples/courses/cs184/final_project/assets/MEASURE_SEVEN/MEASURE_SEVEN_COLORED_LIGHTS.fbx"
CAMERA="D:/examples/courses/cs184/final_project/camera_000244.txt"

if [[ ! -f "${SCENE}" ]]; then
    echo "ERROR: scene not found: ${SCENE}" >&2
    exit 1
fi
if [[ ! -f "${CAMERA}" ]]; then
    echo "ERROR: camera file not found: ${CAMERA}" >&2
    exit 1
fi

OUTDIR="${PROJECT_ROOT}/${OUTDIR_REL}"
[[ "${OUTDIR_REL}" = /* || "${OUTDIR_REL}" =~ ^[A-Za-z]: ]] && OUTDIR="${OUTDIR_REL}"
mkdir -p "${OUTDIR}"

to_win() {
    if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

PATHTRACER_W="$(to_win "${PATHTRACER}")"
SCENE_W="$(to_win "${SCENE}")"
CAMERA_W="$(to_win "${CAMERA}")"
OUTDIR_W="$(to_win "${OUTDIR}")"

# Static camera: --capture-speed 0, dwell=1 so each capture point is one
# frame, stride=1 so we capture every frame after warmup.
COMMON_ARGS=(
    "${SCENE_W}"
    --camera "${CAMERA_W}"
    --sg fbx-c4d
    --emissive-target 5
    -r 1280 720
    --backend optix --mode native
    --capture-out "${OUTDIR_W}"
    --capture-warmup "${WARMUP}"
    --capture-frames "${FRAMES}"
    --capture-stride 1
    --capture-dwell 1
    --capture-speed 0
)

run_variant() {
    local tag="$1"; shift
    echo
    echo "=== Capture variant: ${tag} (frames=${FRAMES}, warmup=${WARMUP}) ==="
    "${PATHTRACER_W}" "${COMMON_ARGS[@]}" --capture-tag "${tag}" "$@"
}

case "${VARIANT}" in
    native)     run_variant "native"    --no-restir --no-restir-gi --no-restir-pt ;;
    restir-di)  run_variant "restir-di" --no-restir-gi --no-restir-pt ;;
    restir-gi)  run_variant "restir-gi" --no-restir --restir-gi --no-restir-pt ;;
    restir-pt)  run_variant "restir-pt" --no-restir --no-restir-gi --restir-pt ;;
    all)
        run_variant "native"    --no-restir --no-restir-gi --no-restir-pt
        run_variant "restir-di" --no-restir-gi --no-restir-pt
        run_variant "restir-gi" --no-restir --restir-gi --no-restir-pt
        run_variant "restir-pt" --no-restir --no-restir-gi --restir-pt
        ;;
    *) echo "ERROR: unknown variant '${VARIANT}'" >&2; exit 2 ;;
esac

echo
echo "Done. Output: ${OUTDIR}"
echo "Run: tools/detect_restir_blob.py ${OUTDIR}"
