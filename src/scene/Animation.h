#pragma once
#include "core/Types.h"
#include <cstdint>
#include <string>
#include <vector>

// ── Scene-graph + animation data ──────────────────────────────
//
// Layout chosen for the MEASURE_SEVEN_COLORED_LIGHTS.fbx scene (10,740 rigid
// meshes, 619 animated nodes, no skinning, no morph targets — pure per-node
// rigid-body animation):
//
//   - Each TriangleMesh's geometry stays in *mesh-local* space (i.e. as the
//     authoring tool emitted it; SceneLoader no longer pre-transforms vertices).
//   - The hierarchy is captured in `SceneNode`s. Every leaf TriangleMesh is
//     attached to one `SceneNode` (the one Assimp originally placed it under).
//   - Animation = per-node TRS tracks. Sampling a clip at time t produces a
//     local-space transform per animated node; a single hierarchy walk turns
//     those into world-space transforms.
//   - Mesh `worldRest` is the world transform at t=0 (or the static rest-pose
//     for non-animated nodes). DeviceScene bakes vertices into world-space at
//     the rest pose so the existing single-buffer/single-GAS layout still
//     works for the static portion of the renderer.
//   - Each frame the animation evaluator produces `worldCurr` for every mesh
//     and a CUDA pose-update kernel re-poses vertices: world_pos =
//     worldCurr * inverse(worldRest) * worldRest_pos. We store the
//     pre-multiplied delta `restToCurr = worldCurr * inverse(worldRest)` to
//     keep the kernel cheap. `restToPrev` is kept around so the previous
//     frame's posed positions can be reconstructed for motion vectors without
//     a second buffer of world-space prev positions.
//
// Animated nodes generate one mesh transform per (mesh that descends from
// them). Nodes the FBX importer split apart with `$AssimpFbx$_*` pivot chains
// are collapsed back into a single composed transform per logical node, so we
// only sample one TRS track per logical node, not per intermediate.

struct AnimChannelTrack {
    // Each entry is (time-in-seconds, value). Times are pre-converted from
    // ticks during load. Sorted ascending; sampling does a linear scan within
    // the channel (channels are short — typically <50 keys for our scene).
    std::vector<float>  posTimes;
    std::vector<float3> posValues;

    std::vector<float>  rotTimes;
    // Quaternion (w, x, y, z). aiQuaternion is (w, x, y, z); we keep that
    // convention. Stored as float4 with .w = real part.
    std::vector<float4> rotValues;

    std::vector<float>  scaleTimes;
    std::vector<float3> scaleValues;
};

struct AnimationClip {
    std::string name;
    float       duration   = 0.0f;  // seconds
    float       ticksPerSecond = 30.0f;

    // Channel index N drives nodeIndices[N] in the SceneNode array. Multiple
    // tracks may target the same node (rare); we collapse them at load.
    std::vector<int>              nodeIndices;
    std::vector<AnimChannelTrack> channels;
};

struct SceneNode {
    std::string name;
    int         parent = -1;       // index into Scene::m_nodes; -1 for root

    // Local-space rest transform. For animated nodes this is the bind pose
    // (used as a fallback when the animation is paused at a time outside the
    // clip range, e.g. for non-animated children of an animated parent).
    float4x4    localRest = float4x4::identity();
    // World-space rest transform = parent.worldRest * localRest. Cached at
    // SceneLoader build time so DeviceScene can bake vertices to world space
    // without a second walk.
    float4x4    worldRest = float4x4::identity();

    // Animation channel feeding this node, or -1 if none. Only set for the
    // `logical` node after Assimp pivot-chain ($AssimpFbx$_*) collapsing.
    int         animChannel = -1;
    bool        animated    = false;  // set if this node OR any ancestor is animated

    // Number of TriangleMesh entries directly attached to this node. Used by
    // the loader and by motion-vector / instance bookkeeping.
    uint32_t    meshCount = 0;
};

// ── Per-mesh animation binding (consumed by DeviceScene) ──────
//
// Built alongside the meshes. nodeIndex is the SceneNode the mesh hangs off.
// vertexOffset is the starting vertex index inside the global flattened
// position/normal/uv buffer (matches the existing layout in DeviceScene).
struct MeshNodeBinding {
    int      nodeIndex     = -1;
    uint32_t vertexOffset  = 0;
    uint32_t vertexCount   = 0;
    bool     animated      = false;  // = SceneNode::animated for fast filtering
};
