#pragma once
#include "scene/Animation.h"
#include <vector>

class Scene;

// Sample an animation clip at time `t` (seconds, looped to clip.duration) and
// produce per-node local transforms, then propagate them into world transforms.
//
//   localOut[i] = animated nodes get TRS from the clip; non-animated nodes
//                 inherit `node.localRest`.
//   worldOut[i] = parent.worldOut * localOut[i] (root parent = identity).
//
// Both arrays must be pre-sized to scene.getNodes().size().
void evalAnimation(const Scene& scene,
                   const AnimationClip& clip,
                   float t,
                   std::vector<float4x4>& localOut,
                   std::vector<float4x4>& worldOut);

// Per-mesh transform delta = worldCurr[node] * inverse(worldRest[node]).
// Pre-multiplied at host so the GPU pose-update kernel needs only one mat-vec
// per vertex. Sized to scene.getMeshes().size().
void computeMeshDeltas(const Scene& scene,
                       const std::vector<float4x4>& worldCurr,
                       std::vector<float4x4>& restToCurrOut);

// Composed M*restToCurr for the upper 3x3, normalized for normal transform.
// We use the matrix's upper-3x3 inverse-transpose (== adjugate / det). Stored
// as float3x3 packed into 3 float4s for cheap GPU upload.
struct NormalMat34 {
    // Row-major 3 rows of (3 floats + 1 pad) = 12 floats. The pad is unused
    // by the kernel; alignment to float4 simplifies the device-side load.
    float4 row[3];
};
void computeNormalMats(const std::vector<float4x4>& restToCurr,
                       std::vector<NormalMat34>& out);
