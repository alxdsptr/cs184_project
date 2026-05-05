#!/usr/bin/env bash
# Render the seven ReSTIR/NRD variants of an animated FBX scene
# (default: MEASURE_SEVEN_COLORED_LIGHTS) into separate MP4s.
#
# Usage:
#   scripts/render_m7_animation_variants.sh [recording] [--out DIR] [--res WxH] [--scene PATH]
#
# Defaults:
#   recording = recordings/path_M7.json (relative to <run-dir>)
#   out       = renders/720p
#   res       = 1280x720
#   scene     = ../../assets/MEASURE_SEVEN/MEASURE_SEVEN_COLORED_LIGHTS.fbx
#
# Honors PATHTRACER_EXE / PATHTRACER_RUN_DIR for the binary location;
# defaults to build-all/Release/.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
EXE="${PATHTRACER_EXE:-$ROOT/build-all/Release/pathtracer.exe}"
RUN_DIR="${PATHTRACER_RUN_DIR:-$ROOT/build-all/Release}"

RECORDING="recordings/path_M7.json"
OUT="$ROOT/renders/720p"
RES_W=1280
RES_H=720
SCENE="$ROOT/assets/MEASURE_SEVEN/MEASURE_SEVEN_COLORED_LIGHTS.fbx"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)   OUT="$2"; shift 2;;
        --res)   IFS='x' read -r RES_W RES_H <<<"$2"; shift 2;;
        --scene) SCENE="$2"; shift 2;;
        --*)     echo "unknown flag: $1" >&2; exit 1;;
        *)       RECORDING="$1"; shift;;
    esac
done

mkdir -p "$OUT"

# Each row: <tag> <--variant> <--mode>
#
# Variants tested:
#   native       no-ReSTIR baseline (path-traced, no denoise)
#   di           ReSTIR DI only
#   di_gi        ReSTIR DI + GI
#   di_gi_pt     ReSTIR DI + GI + PT (longest path postfix)
#   nrd_only     NRD denoiser, no ReSTIR
#   nrd_di       NRD + ReSTIR DI
#   nrd_di_gi    NRD + ReSTIR DI + GI
#   nrd_pt       NRD + ReSTIR DI + GI + PT
VARIANTS=(
    "di         di         native"
    "di_gi      di_gi      native"
    "di_gi_pt   di_gi_pt   native"
    "nrd_only   native     nrd"
    "nrd_di     di         nrd"
    "nrd_di_gi  di_gi      nrd"
    "nrd_pt     di_gi_pt   nrd"
)

for row in "${VARIANTS[@]}"; do
    set -- $row
    TAG=$1
    VAR=$2
    MODE=$3
    echo "===== Rendering $TAG (variant=$VAR mode=$MODE) ====="
    python "$ROOT/scripts/render_replay.py" "$RECORDING" \
        --scene "$SCENE" \
        --variant "$VAR" \
        --mode "$MODE" \
        --backend optix \
        --flags "--sg fbx-c4d --emissive-target 5" \
        --play-anim --anim-fps 30 \
        -r "$RES_W" "$RES_H" \
        --spp 1 \
        --stride 1 \
        --exe "$EXE" \
        --run-dir "$RUN_DIR" \
        --out "$OUT/$TAG.mp4" \
        --frames-dir "$OUT/frames_$TAG" \
        --fps 30 \
        --max-fps 0 \
        --crf 18 \
        --preset medium 2>&1 | tail -8
    echo
done

echo "All renders done. Outputs in $OUT:"
ls -la "$OUT"/*.mp4
