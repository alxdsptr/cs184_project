#include "gpu/PoseUpdate.h"
#include "gpu/AreaLightGPU.h"
#include "accel/LightBVHNode.h"
#include "util/CudaCheck.h"
#include "core/Math.h"

#include <cstring>

// ── Device kernel ────────────────────────────────────────────
// One thread per vertex. We read meshIndex; if it's the static sentinel we
// just copy current->prev (so motion vector from this vertex is zero, which
// is correct) and leave d_positionsCurr/d_normalsCurr unchanged. Otherwise we
// fetch the 3x4 mesh delta (3 float4 rows = 12 floats) plus the 3x3 normal
// matrix (3 float4 rows; .w padding ignored), apply them to the rest-pose
// position/normal, and write to the curr buffer.
//
// `firstFrame`: when true we skip writing d_positionsPrev = d_positionsCurr
// for animated verts (so the prev buffer's initial-zero contents don't show
// up as a giant velocity on the first rendered frame). The host code memcpys
// d_positionsCurr -> d_positionsPrev once before launching with firstFrame=true.
__global__ void poseUpdateKernel(
    const float3* __restrict__ d_positionsRest,
    const float3* __restrict__ d_normalsRest,
    const uint32_t* __restrict__ d_perVertexMeshIndex,
    uint32_t   staticSentinel,
    const float4* __restrict__ d_meshDelta,        // 3 rows per mesh
    const float4* __restrict__ d_meshNormalMat,    // 3 rows per mesh
    float3* __restrict__ d_positionsCurr,
    float3* __restrict__ d_normalsCurr,
    float3* __restrict__ d_positionsPrev,
    uint32_t vertexCount,
    int firstFrame)
{
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vertexCount) return;

    uint32_t mi = d_perVertexMeshIndex[v];

    // Save prev. For static verts the prev/curr are identical (and equal to
    // the rest pose since we don't touch them); a prev=curr copy is still
    // useful as a defensive default.
    if (firstFrame) {
        // First-frame: prev := curr (which currently holds the previous
        // launch's data — for the very first launch this is still the rest-
        // pose copy that DeviceScene seeded). The host did this with a
        // memcpy already, so we don't need to do it here. But to keep
        // motion-vector output near zero during animation init, also write
        // prev := the about-to-be-computed-curr below for animated verts.
    } else {
        d_positionsPrev[v] = d_positionsCurr[v];
    }

    if (mi == staticSentinel) {
        // Static vertex: position + normal stay at rest-pose values.
        d_positionsCurr[v] = d_positionsRest[v];
        d_normalsCurr[v]   = d_normalsRest[v];
        return;
    }

    // Fetch 3x4 row-major mesh delta.
    const float4 r0 = d_meshDelta[mi * 3 + 0];
    const float4 r1 = d_meshDelta[mi * 3 + 1];
    const float4 r2 = d_meshDelta[mi * 3 + 2];

    float3 P = d_positionsRest[v];
    float3 newP;
    newP.x = r0.x * P.x + r0.y * P.y + r0.z * P.z + r0.w;
    newP.y = r1.x * P.x + r1.y * P.y + r1.z * P.z + r1.w;
    newP.z = r2.x * P.x + r2.y * P.y + r2.z * P.z + r2.w;
    d_positionsCurr[v] = newP;

    // Normal: cofactor-3x3 (== det(M) * inverse-transpose). Renormalise.
    const float4 n0 = d_meshNormalMat[mi * 3 + 0];
    const float4 n1 = d_meshNormalMat[mi * 3 + 1];
    const float4 n2 = d_meshNormalMat[mi * 3 + 2];

    float3 N = d_normalsRest[v];
    float3 newN;
    newN.x = n0.x * N.x + n0.y * N.y + n0.z * N.z;
    newN.y = n1.x * N.x + n1.y * N.y + n1.z * N.z;
    newN.z = n2.x * N.x + n2.y * N.y + n2.z * N.z;
    float len = sqrtf(newN.x * newN.x + newN.y * newN.y + newN.z * newN.z);
    if (len > 1e-12f) {
        float inv = 1.0f / len;
        newN.x *= inv; newN.y *= inv; newN.z *= inv;
    } else {
        newN = N;
    }
    d_normalsCurr[v] = newN;

    if (firstFrame) {
        // Now that we've computed the new pose, seed prev with it so the
        // first frame's motion vectors are zero rather than a large jump.
        d_positionsPrev[v] = newP;
    }
}

void poseUpdateAlloc(PoseUpdateData& d, uint32_t vertexCount, uint32_t meshCount) {
    d.vertexCount = vertexCount;
    d.meshCount = meshCount;
    d.staticSentinel = 0xFFFFFFFFu;
    if (vertexCount == 0) return;

    CUDA_CHECK(cudaMalloc(&d.d_positionsRest, vertexCount * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d.d_normalsRest,   vertexCount * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d.d_perVertexMeshIndex, vertexCount * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&d.d_positionsCurr, vertexCount * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d.d_normalsCurr,   vertexCount * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d.d_positionsPrev, vertexCount * sizeof(float3)));

    if (meshCount > 0) {
        CUDA_CHECK(cudaMalloc(&d.d_meshDelta,     meshCount * 3 * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d.d_meshNormalMat, meshCount * 3 * sizeof(float4)));
    }
}

void poseUpdateFree(PoseUpdateData& d) {
    if (d.d_positionsRest)       { cudaFree(d.d_positionsRest);       d.d_positionsRest = nullptr; }
    if (d.d_normalsRest)         { cudaFree(d.d_normalsRest);         d.d_normalsRest   = nullptr; }
    if (d.d_perVertexMeshIndex)  { cudaFree(d.d_perVertexMeshIndex);  d.d_perVertexMeshIndex = nullptr; }
    if (d.d_meshDelta)           { cudaFree(d.d_meshDelta);           d.d_meshDelta = nullptr; }
    if (d.d_meshNormalMat)       { cudaFree(d.d_meshNormalMat);       d.d_meshNormalMat = nullptr; }
    if (d.d_positionsCurr)       { cudaFree(d.d_positionsCurr);       d.d_positionsCurr = nullptr; }
    if (d.d_normalsCurr)         { cudaFree(d.d_normalsCurr);         d.d_normalsCurr = nullptr; }
    if (d.d_positionsPrev)       { cudaFree(d.d_positionsPrev);       d.d_positionsPrev = nullptr; }

    // Light-BVH refit per-level index buffers. Owned by PoseUpdate.
    for (uint32_t* p : d.d_lightBVHLevel) {
        if (p) cudaFree(p);
    }
    d.d_lightBVHLevel.clear();
    d.lightBVHLevelSize.clear();

    // Aliased pointers — owned by DeviceScene, just clear references.
    d.d_areaLights        = nullptr;
    d.areaLightCount      = 0;
    d.d_lightBVHNodes     = nullptr;
    d.d_orderedLightIndices = nullptr;
    d.lightBVHRootIndex   = 0;
    d.lightBVHNodeCount   = 0;

    d.vertexCount = 0;
    d.meshCount = 0;
}

// Convert a host-side row-major 4x4 to the 3 float4 rows the kernel reads.
static void packDelta(const float4x4& M, float4* out3) {
    out3[0] = make_float4(M.m[0][0], M.m[0][1], M.m[0][2], M.m[0][3]);
    out3[1] = make_float4(M.m[1][0], M.m[1][1], M.m[1][2], M.m[1][3]);
    out3[2] = make_float4(M.m[2][0], M.m[2][1], M.m[2][2], M.m[2][3]);
}

void poseUpdateUploadDeltas(PoseUpdateData& d,
                            const std::vector<float4x4>& meshDelta,
                            const std::vector<NormalMat34>& meshNormalMat)
{
    if (d.meshCount == 0) return;
    if (meshDelta.size() != d.meshCount || meshNormalMat.size() != d.meshCount) return;

    // Pack into a host scratch buffer then upload in one shot.
    std::vector<float4> packed(d.meshCount * 3);
    for (uint32_t m = 0; m < d.meshCount; m++) {
        packDelta(meshDelta[m], &packed[m * 3]);
    }
    CUDA_CHECK(cudaMemcpy(d.d_meshDelta, packed.data(),
                          d.meshCount * 3 * sizeof(float4),
                          cudaMemcpyHostToDevice));

    for (uint32_t m = 0; m < d.meshCount; m++) {
        packed[m * 3 + 0] = meshNormalMat[m].row[0];
        packed[m * 3 + 1] = meshNormalMat[m].row[1];
        packed[m * 3 + 2] = meshNormalMat[m].row[2];
    }
    CUDA_CHECK(cudaMemcpy(d.d_meshNormalMat, packed.data(),
                          d.meshCount * 3 * sizeof(float4),
                          cudaMemcpyHostToDevice));
}

void poseUpdateLaunch(PoseUpdateData& d, bool firstFrame, cudaStream_t stream) {
    if (d.vertexCount == 0) return;
    int threads = 256;
    int blocks  = (int)((d.vertexCount + threads - 1) / threads);
    poseUpdateKernel<<<blocks, threads, 0, stream>>>(
        d.d_positionsRest, d.d_normalsRest,
        d.d_perVertexMeshIndex, d.staticSentinel,
        d.d_meshDelta, d.d_meshNormalMat,
        d.d_positionsCurr, d.d_normalsCurr,
        d.d_positionsPrev,
        d.vertexCount,
        firstFrame ? 1 : 0);
}

// ── Light update + Light-BVH refit ────────────────────────────
//
// Per-frame pipeline for animated emitters:
//
//   1. lightUpdateKernel   — one thread per area light. Skip static
//      (meshIndex == -1). For animated lights, apply the mesh's pose delta
//      to the rest triangle and write back v0/e1/e2/normal.
//
//   2. lightBVHLeafRefitKernel — one thread per leaf node. Walk the leaf's
//      `[primOffset, primOffset + primCount)` slice of orderedLightIndices,
//      look up each light's now-current world triangle, union their three
//      vertices into a fresh AABB, and write it back to node.bounds. (We
//      also re-sum `weight` from the same lights — for rigid animation the
//      sum doesn't change, but doing it keeps the path uniform with non-
//      rigid extensions.)
//
//   3. lightBVHInternalRefitKernel — one launch per internal level, bottom-
//      up. Each thread merges its node's leftChild + rightChild bounds /
//      weights into the parent. Levels are kicked off in `lightBVHRefit`
//      below in order, so each level's results are visible to the next.

__global__ void lightUpdateKernel(
    GPUAreaLight* __restrict__ d_areaLights,
    uint32_t lightCount,
    const float4* __restrict__ d_meshDelta,
    const float4* __restrict__ d_meshNormalMat)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lightCount) return;

    GPUAreaLight L = d_areaLights[i];   // local copy, write-back at end
    if (L.meshIndex < 0) return;        // static; skip

    int mi = L.meshIndex;
    const float4 r0 = d_meshDelta[mi * 3 + 0];
    const float4 r1 = d_meshDelta[mi * 3 + 1];
    const float4 r2 = d_meshDelta[mi * 3 + 2];

    // v0 is a point: full 4x3 transform.
    float3 v0 = make_float3(
        r0.x * L.v0_rest.x + r0.y * L.v0_rest.y + r0.z * L.v0_rest.z + r0.w,
        r1.x * L.v0_rest.x + r1.y * L.v0_rest.y + r1.z * L.v0_rest.z + r1.w,
        r2.x * L.v0_rest.x + r2.y * L.v0_rest.y + r2.z * L.v0_rest.z + r2.w);

    // e1, e2 are direction vectors (edges): rotate-only (no translation).
    float3 e1 = make_float3(
        r0.x * L.e1_rest.x + r0.y * L.e1_rest.y + r0.z * L.e1_rest.z,
        r1.x * L.e1_rest.x + r1.y * L.e1_rest.y + r1.z * L.e1_rest.z,
        r2.x * L.e1_rest.x + r2.y * L.e1_rest.y + r2.z * L.e1_rest.z);
    float3 e2 = make_float3(
        r0.x * L.e2_rest.x + r0.y * L.e2_rest.y + r0.z * L.e2_rest.z,
        r1.x * L.e2_rest.x + r1.y * L.e2_rest.y + r1.z * L.e2_rest.z,
        r2.x * L.e2_rest.x + r2.y * L.e2_rest.y + r2.z * L.e2_rest.z);

    // Normal: cofactor (= det * inverse-transpose). Renormalise.
    const float4 n0 = d_meshNormalMat[mi * 3 + 0];
    const float4 n1 = d_meshNormalMat[mi * 3 + 1];
    const float4 n2 = d_meshNormalMat[mi * 3 + 2];
    float3 N = make_float3(
        n0.x * L.normal_rest.x + n0.y * L.normal_rest.y + n0.z * L.normal_rest.z,
        n1.x * L.normal_rest.x + n1.y * L.normal_rest.y + n1.z * L.normal_rest.z,
        n2.x * L.normal_rest.x + n2.y * L.normal_rest.y + n2.z * L.normal_rest.z);
    float Nlen = sqrtf(N.x*N.x + N.y*N.y + N.z*N.z);
    if (Nlen > 1e-12f) {
        float inv = 1.0f / Nlen;
        N.x *= inv; N.y *= inv; N.z *= inv;
    } else {
        N = L.normal_rest;
    }

    // Write back. We deliberately don't update L.area / L.weight: rigid
    // transforms preserve them, and recomputing would force re-uploading the
    // CDF. If we ever support non-rigid scaling here, re-derive both.
    d_areaLights[i].v0     = v0;
    d_areaLights[i].e1     = e1;
    d_areaLights[i].e2     = e2;
    d_areaLights[i].normal = N;
}

__global__ void lightBVHLeafRefitKernel(
    LightBVHNode* __restrict__ nodes,
    uint32_t levelSize,
    const uint32_t* __restrict__ levelIndices,    // node index for thread `i`
    const GPUAreaLight* __restrict__ d_areaLights,
    const uint32_t* __restrict__ d_orderedLightIndices)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= levelSize) return;
    uint32_t nodeIdx = levelIndices[t];
    LightBVHNode n = nodes[nodeIdx];
    if (!n.isLeaf()) return;  // safety — level 0 should be leaves only

    float3 mn = make_float3( 1e30f,  1e30f,  1e30f);
    float3 mx = make_float3(-1e30f, -1e30f, -1e30f);
    float  wSum = 0.0f;

    for (uint32_t k = 0; k < n.primCount; k++) {
        uint32_t lightIdx = d_orderedLightIndices[n.primOffset + k];
        GPUAreaLight L = d_areaLights[lightIdx];
        float3 v0 = L.v0;
        float3 v1 = make_float3(L.v0.x + L.e1.x, L.v0.y + L.e1.y, L.v0.z + L.e1.z);
        float3 v2 = make_float3(L.v0.x + L.e2.x, L.v0.y + L.e2.y, L.v0.z + L.e2.z);
        mn.x = fminf(mn.x, fminf(v0.x, fminf(v1.x, v2.x)));
        mn.y = fminf(mn.y, fminf(v0.y, fminf(v1.y, v2.y)));
        mn.z = fminf(mn.z, fminf(v0.z, fminf(v1.z, v2.z)));
        mx.x = fmaxf(mx.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
        mx.y = fmaxf(mx.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
        mx.z = fmaxf(mx.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
        wSum += L.weight;
    }

    nodes[nodeIdx].bounds.bmin = mn;
    nodes[nodeIdx].bounds.bmax = mx;
    nodes[nodeIdx].weight      = wSum;
}

__global__ void lightBVHInternalRefitKernel(
    LightBVHNode* __restrict__ nodes,
    uint32_t levelSize,
    const uint32_t* __restrict__ levelIndices)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= levelSize) return;
    uint32_t nodeIdx = levelIndices[t];
    LightBVHNode n = nodes[nodeIdx];
    if (n.isLeaf()) return;  // safety — internal levels should not contain leaves

    LightBVHNode L = nodes[n.leftChild];
    LightBVHNode R = nodes[n.rightChild];

    float3 mn, mx;
    mn.x = fminf(L.bounds.bmin.x, R.bounds.bmin.x);
    mn.y = fminf(L.bounds.bmin.y, R.bounds.bmin.y);
    mn.z = fminf(L.bounds.bmin.z, R.bounds.bmin.z);
    mx.x = fmaxf(L.bounds.bmax.x, R.bounds.bmax.x);
    mx.y = fmaxf(L.bounds.bmax.y, R.bounds.bmax.y);
    mx.z = fmaxf(L.bounds.bmax.z, R.bounds.bmax.z);

    nodes[nodeIdx].bounds.bmin = mn;
    nodes[nodeIdx].bounds.bmax = mx;
    nodes[nodeIdx].weight      = L.weight + R.weight;
}

void lightUpdateLaunch(PoseUpdateData& d, cudaStream_t stream) {
    if (d.areaLightCount == 0 || !d.d_areaLights) return;
    int threads = 256;
    int blocks  = (int)((d.areaLightCount + threads - 1) / threads);
    lightUpdateKernel<<<blocks, threads, 0, stream>>>(
        (GPUAreaLight*)d.d_areaLights,
        d.areaLightCount,
        d.d_meshDelta,
        d.d_meshNormalMat);
}

void lightBVHRefitLaunch(PoseUpdateData& d, cudaStream_t stream) {
    if (d.lightBVHNodeCount == 0 || !d.d_lightBVHNodes) return;
    if (d.d_lightBVHLevel.empty()) return;

    // Level 0 is leaves — refresh from per-light world triangles.
    if (d.lightBVHLevelSize.size() > 0 && d.lightBVHLevelSize[0] > 0) {
        int threads = 128;
        int blocks  = (int)((d.lightBVHLevelSize[0] + threads - 1) / threads);
        lightBVHLeafRefitKernel<<<blocks, threads, 0, stream>>>(
            (LightBVHNode*)d.d_lightBVHNodes,
            d.lightBVHLevelSize[0],
            d.d_lightBVHLevel[0],
            (const GPUAreaLight*)d.d_areaLights,
            d.d_orderedLightIndices);
    }

    // Levels 1..N are internal — bottom-up wave-front merge of children.
    for (size_t lv = 1; lv < d.d_lightBVHLevel.size(); lv++) {
        if (d.lightBVHLevelSize[lv] == 0) continue;
        int threads = 128;
        int blocks  = (int)((d.lightBVHLevelSize[lv] + threads - 1) / threads);
        lightBVHInternalRefitKernel<<<blocks, threads, 0, stream>>>(
            (LightBVHNode*)d.d_lightBVHNodes,
            d.lightBVHLevelSize[lv],
            d.d_lightBVHLevel[lv]);
    }
}
