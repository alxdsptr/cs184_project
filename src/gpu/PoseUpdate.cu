#include "gpu/PoseUpdate.h"
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
