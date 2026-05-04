#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# run_quality_sweep.sh — capture each ReSTIR mode along an identical
# camera path, then run tools/compare_quality.py for an HTML report.
#
# Workflow:
#   1. REFERENCE sweep — runs the exact same camera path as the tests but
#      tells the path tracer to dwell for REFERENCE_DWELL frames at every
#      capture point before saving (--capture-dwell). Motion is paused
#      during the dwell, so the engine's accumulator integrates many
#      samples into the same pose. The result is a per-pose near-converged
#      ground-truth sequence reference_NNNNNN.png that lines up frame-for-
#      frame with the test sweeps.
#   2. Native + ReSTIR DI / GI / PT sweeps along the same camera path,
#      with dwell=1 (one render per capture point).
#   3. Compare every captured frame against the reference frame at the
#      same index (per-frame-index mode in compare_quality.py).
#
# Usage (run from anywhere — script resolves project root from $0):
#     tools/run_quality_sweep.sh <scene_file> [extra_pathtracer_flags ...]
#
# Examples:
#     # default: optix backend, 2000-frame dwell reference, dolly motion
#     tools/run_quality_sweep.sh assets/scene.fbx --camera mycam.txt
#
#     # cuda backend
#     tools/run_quality_sweep.sh assets/scene.fbx --backend cuda
#
#     # crank up reference quality
#     tools/run_quality_sweep.sh assets/scene.fbx --reference-dwell 4000
#
#     # static camera (best for "is restir converging at this fixed pose?")
#     tools/run_quality_sweep.sh assets/scene.fbx --camera mycam.txt \
#         --capture-speed 0
#
#     # bring your own per-frame reference sequence and skip the ref sweep
#     tools/run_quality_sweep.sh assets/scene.fbx \
#         --skip-reference-sweep   # reuses prior reference_*.png in --out
#
# Recognised script flags (consumed before forwarding):
#     --backend <cuda|optix>  Which backend to drive at RUNTIME. Default:
#                             optix. Forwarded to the pathtracer as
#                             --backend <cuda|optix>; the build-optix exe
#                             supports both code paths so we always launch
#                             the same binary.
#     --reference-dwell <N>   Dwell frames per capture point during the
#                             reference sweep (default 2000). Higher → more
#                             converged ref images, longer runtime.
#     --skip-reference-sweep  Don't render a fresh reference (reuse a prior
#                             reference_*.png sequence in the out dir).
#     --pathtracer <exe>      Override the pathtracer.exe path.
#                             Default: build-optix/Release/pathtracer.exe.
#     --out <dir>             Output directory.
#                             Defaults to <project_root>/screenshots/sweep.
#     --skip-compare          Skip running compare_quality.py at the end.
#
# Anything not in that list is forwarded to EVERY sweep invocation
# (--camera, --capture-speed, --capture-orbit, --emissive-target, ...).
# Forwarded flags apply to BOTH the reference and the test sweeps so they
# share the same camera path. The script itself sets --capture-tag,
# --capture-out, --capture-warmup, --capture-frames, --capture-stride,
# --capture-dwell, the ReSTIR mode, and the dolly motion defaults — those
# cannot be overridden from the forwarded flags (and a warning is printed
# if you try).
# Designed for Git Bash / MSYS2 / WSL / Cygwin on Windows.
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

usage() {
    sed -n '2,66p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
}

[[ $# -ge 1 ]] || usage

# Resolve project root = parent of this script's directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# ── Parse args ───────────────────────────────────────────────────────────
SCENE_ARG="$1"
shift

BACKEND="optix"
REFERENCE_DWELL=2000
SKIP_REFERENCE_SWEEP=0
# The build-optix exe carries both backends; --backend selects which path
# is used at runtime. Override via --pathtracer if you have a different
# build layout.
PATHTRACER_REL="build-optix/Release/pathtracer.exe"
OUTDIR_REL="screenshots/sweep"
SKIP_COMPARE=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)                BACKEND="$2"; shift 2 ;;
        --reference-dwell)        REFERENCE_DWELL="$2"; shift 2 ;;
        --skip-reference-sweep)   SKIP_REFERENCE_SWEEP=1; shift ;;
        --pathtracer)             PATHTRACER_REL="$2"; shift 2 ;;
        --out)                    OUTDIR_REL="$2"; shift 2 ;;
        --skip-compare)           SKIP_COMPARE=1; shift ;;
        # The script itself owns these — silently override.
        --capture-tag)
            echo "warning: ignoring --capture-tag (script assigns per-mode tags)" >&2
            shift 2 ;;
        --capture-out)
            echo "warning: ignoring --capture-out; use --out <dir> instead" >&2
            shift 2 ;;
        --capture-dwell)
            echo "warning: ignoring --capture-dwell (use --reference-dwell for the ref sweep)" >&2
            shift 2 ;;
        --no-restir|--no-restir-gi|--no-restir-pt|--restir-gi|--restir-pt)
            echo "warning: ignoring '$1' (script assigns ReSTIR mode per sweep)" >&2
            shift ;;
        *) FORWARD_ARGS+=("$1"); shift ;;
    esac
done

case "${BACKEND}" in
    cuda|optix) ;;
    *) echo "ERROR: --backend must be 'cuda' or 'optix' (got '${BACKEND}')" >&2; exit 1 ;;
esac

# Resolve scene to absolute path so the exe finds it regardless of CWD.
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

# ── Path style for the .exe ──────────────────────────────────────────────
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

# ── Capture parameters (shared across all sweeps so frame indices line up) ──
WARMUP=120
FRAMES=240
STRIDE=30
RES_ARGS=(-r 1280 720)
BACKEND_ARGS=(--backend "${BACKEND}")

# Forward dolly: 0.3 units/sec (slow walk-through). Forwarded args may
# override --capture-speed / --capture-orbit downstream — that's fine, the
# pathtracer's last-arg-wins semantics handle it.
MOTION_ARGS=(--capture-dolly --capture-speed 0.3)

# ── Sweep launcher ───────────────────────────────────────────────────────
# Each invocation shares warmup/frames/stride/motion so frame index N maps
# to the same camera pose across all tags. Dwell varies: large for the
# reference (long-spp accumulation per pose), 1 for the tests.
run_sweep() {
    local tag="$1"; shift
    local dwell="$1"; shift
    echo
    echo "=== Sweep: ${tag} (dwell=${dwell}) ==="
    "${PATHTRACER_W}" "${SCENE_W}" "${RES_ARGS[@]}" "${BACKEND_ARGS[@]}" \
        "$@" \
        --capture-tag "${tag}" \
        --capture-out "${OUTDIR_W}" \
        --capture-warmup "${WARMUP}" \
        --capture-frames "${FRAMES}" \
        --capture-stride "${STRIDE}" \
        --capture-dwell "${dwell}" \
        "${MOTION_ARGS[@]}" \
        "${FORWARD_W[@]+"${FORWARD_W[@]}"}"
}

# ── Reference sweep ──────────────────────────────────────────────────────
# Same camera path as the tests, but each capture point dwells for
# REFERENCE_DWELL frames with motion paused, so the accumulator converges
# at every pose. Disabling all ReSTIR variants ensures the reference is
# pure path-traced ground truth.
if [[ "${SKIP_REFERENCE_SWEEP}" -eq 0 ]]; then
    echo
    echo "=== Reference sweep: native, dwell=${REFERENCE_DWELL} per capture point ==="
    run_sweep "reference" "${REFERENCE_DWELL}" \
        --no-restir --no-restir-gi --no-restir-pt
fi

# Quick sanity: at least one reference frame must exist for compare to work.
if [[ "${SKIP_COMPARE}" -eq 0 ]]; then
    if ! ls "${OUTDIR}"/reference_[0-9]*.png >/dev/null 2>&1; then
        echo "ERROR: no reference_*.png in ${OUTDIR}." >&2
        echo "       Drop --skip-reference-sweep so we can render one." >&2
        exit 1
    fi
fi

# ── Test sweeps (4 modes, identical camera path, dwell=1) ────────────────
run_sweep "native"        1 --no-restir --no-restir-gi --no-restir-pt
run_sweep "restir-di"     1 --no-restir-gi --no-restir-pt
run_sweep "restir-di-gi"  1 --restir-gi --no-restir-pt
run_sweep "restir-di-pt"  1 --no-restir-gi --restir-pt

# ── Comparison (per-frame-index against the reference sequence) ──────────
if [[ "${SKIP_COMPARE}" -eq 1 ]]; then
    echo
    echo "Captures done. Skipping comparison (--skip-compare). Output: ${OUTDIR}"
    exit 0
fi

PY=""
for cand in python python3 py; do
    if command -v "${cand}" >/dev/null 2>&1; then
        PY="${cand}"; break
    fi
done
if [[ -z "${PY}" ]]; then
    echo "ERROR: no python interpreter found in PATH (tried python, python3, py)." >&2
    exit 1
fi

echo
echo "=== Comparing every captured frame vs reference (per-frame-index) ==="
(
    cd "${PROJECT_ROOT}"
    "${PY}" tools/compare_quality.py \
        --test-dir "${OUTDIR}" \
        --ref-tag reference \
        --test-tag native \
        --test-tag restir-di \
        --test-tag restir-di-gi \
        --test-tag restir-di-pt \
        --report "${OUTDIR}/report.html"
)

echo
echo "Done. Open ${OUTDIR}/report.html"
