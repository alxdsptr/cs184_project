#pragma once
#include "scene/Animation.h"
#include "scene/AnimationEval.h"
#include <cuda_runtime.h>
#include <vector>

// GPU-side animation buffers. Owned by DeviceScene; populated at upload time
// (rest-pose, mesh-index mapping, mesh-count) and refreshed each animation
// frame (mesh deltas, normal matrices).
struct PoseUpdateData {
    // Rest-pose world-space positions / normals. Built at upload by composing
    // each mesh's local-space vertices with its node's worldRest, so static
    // meshes are already in their final position and the pose-update kernel
    // only needs to touch animated vertices.
    float3* d_positionsRest = nullptr;   // worldRest * meshLocalPos
    float3* d_normalsRest   = nullptr;   // (normalMat of worldRest) * meshLocalNormal, normalized

    // Per-vertex mesh index. -1 (or specifically 0xFFFFFFFF) marks a static
    // vertex the kernel can skip. Otherwise indexes the per-mesh delta arrays.
    uint32_t* d_perVertexMeshIndex = nullptr;
    uint32_t  staticSentinel = 0xFFFFFFFFu;

    // Per-mesh transform delta = worldCurr * worldRest^-1 (4x3 row-major,
    // last row implicit (0,0,0,1)). Stored as 3 float4s per mesh = 12 floats.
    // d_meshDelta points to one entry per mesh in scene.getMeshes() order.
    float4*   d_meshDelta = nullptr;     // 3 rows per mesh, length = 3 * meshCount
    // 3x3 cofactor (= det * inverse-transpose) of the upper-3x3 of meshDelta,
    // used by the kernel to transform normals. 3 float4s per mesh.
    float4*   d_meshNormalMat = nullptr; // 3 rows per mesh, length = 3 * meshCount

    uint32_t  meshCount    = 0;
    uint32_t  vertexCount  = 0;          // total verts across all meshes
    uint32_t  animatedVertexCount = 0;   // verts whose meshIndex != staticSentinel

    // Output buffers consumed by the OptiX GAS and motion-vector code.
    // d_positionsCurr: world-space positions at the *current* animation time.
    // d_normalsCurr  : world-space normals at the current time.
    // d_positionsPrev: positions from the *previous* frame's pose, used by
    //                  closest-hit programs to reproject hit points for motion
    //                  vectors. Filled by the pose-update kernel (it reads
    //                  d_positionsCurr and writes it to d_positionsPrev before
    //                  computing the new positions).
    float3*   d_positionsCurr = nullptr;
    float3*   d_normalsCurr   = nullptr;
    float3*   d_positionsPrev = nullptr;

    // ── Light-BVH refit support (animated emitters) ─────────────────────
    // Aliased pointers into DeviceScene's m_data arrays so the per-frame
    // light-update + BVH-refit kernels can re-pose animated area lights and
    // rebuild the BVH bounds in place. The kernels write into the same
    // GPUAreaLight[] / LightBVHNode[] arrays the renderer already reads, so
    // there's nothing else to wire up downstream — NEE just sees fresh
    // values.
    void*       d_areaLights        = nullptr; // GPUAreaLight*  (forward decl)
    uint32_t    areaLightCount      = 0;
    void*       d_lightBVHNodes     = nullptr; // LightBVHNode*  (forward decl)
    uint32_t*   d_orderedLightIndices = nullptr; // owned by DeviceScene; aliased
    uint32_t    lightBVHRootIndex   = 0;
    uint32_t    lightBVHNodeCount   = 0;
    // For bottom-up refit of internal nodes: one entry per level (excluding
    // level 0 which is leaves, refreshed by the per-light kernel directly).
    // d_lightBVHLevel[i] holds the GPU array of node indices at internal
    // level (i+1); the refit driver launches each level in order with
    // `lightBVHLevelSize[i]` threads.
    std::vector<uint32_t*> d_lightBVHLevel;
    std::vector<uint32_t>  lightBVHLevelSize;
};

// Allocate the GPU buffers. Caller is responsible for filling
// d_positionsRest/d_normalsRest/d_perVertexMeshIndex from host data and for
// freeing via poseUpdateFree().
void poseUpdateAlloc(PoseUpdateData& d, uint32_t vertexCount, uint32_t meshCount);
void poseUpdateFree(PoseUpdateData& d);

// Upload current-frame mesh deltas + normal matrices. The host arrays are
// produced by AnimationEval::computeMeshDeltas / computeNormalMats.
void poseUpdateUploadDeltas(PoseUpdateData& d,
                            const std::vector<float4x4>& meshDelta,
                            const std::vector<NormalMat34>& meshNormalMat);

// Run the pose-update kernel: for each vertex, save its current world-space
// position into d_positionsPrev, then re-pose it from d_positionsRest using
// d_meshDelta[d_perVertexMeshIndex[v]]. Vertices flagged static keep their
// rest-pose values (which already equal worldRest * meshLocal), and their
// "prev" entry is just a copy of "curr" — no motion vector contribution.
//
// Pass `firstFrame=true` on the very first frame after upload so that
// d_positionsPrev is initialised to d_positionsCurr instead of stale data.
void poseUpdateLaunch(PoseUpdateData& d, bool firstFrame, cudaStream_t stream = 0);

// Re-pose every animated area light (those with meshIndex >= 0): apply the
// current frame's meshDelta to the rest-pose triangle, write back v0/e1/e2/
// normal, and atomically expand the corresponding BVH leaf's AABB. After
// this kernel returns the leaf nodes hold up-to-date bounds.
//
// Static lights (meshIndex == -1) are skipped — their leaf bounds were
// initialised at upload time and never go stale.
void lightUpdateLaunch(PoseUpdateData& d, cudaStream_t stream = 0);

// Bottom-up internal-node BVH refit. Each level reads the AABBs + weights
// of children written by the previous level and merges them into the parent.
// Internal-only (level 0 = leaves was already refreshed by lightUpdateLaunch).
// Levels are launched in order so each level's writes are visible to the next.
void lightBVHRefitLaunch(PoseUpdateData& d, cudaStream_t stream = 0);
