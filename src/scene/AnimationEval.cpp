#include "scene/AnimationEval.h"
#include "scene/Scene.h"
#include "core/Math.h"

#include <cmath>
#include <cstring>

// ── Track sampling ────────────────────────────────────────────
// Linear scan within a (short) sorted time array. If t is outside the range,
// clamp to the nearest end. Returns (lower-index, upper-index, alpha) for
// blending. For tracks with a single key it returns (0, 0, 0).
struct TrackBlend {
    int lo;
    int hi;
    float a;
};

static TrackBlend findKeys(const std::vector<float>& times, float t) {
    TrackBlend r{0, 0, 0.0f};
    if (times.empty()) return r;
    if (times.size() == 1) return r;
    if (t <= times.front()) { r.lo = 0; r.hi = 0; r.a = 0.0f; return r; }
    if (t >= times.back())  { int n = (int)times.size() - 1; r.lo = n; r.hi = n; r.a = 0.0f; return r; }
    // The channel is short (<50 keys typical). Linear scan is fine.
    for (int i = 0; i + 1 < (int)times.size(); i++) {
        if (t >= times[i] && t <= times[i + 1]) {
            r.lo = i;
            r.hi = i + 1;
            float dt = times[i + 1] - times[i];
            r.a = (dt > 1e-8f) ? (t - times[i]) / dt : 0.0f;
            return r;
        }
    }
    int n = (int)times.size() - 1;
    r.lo = n; r.hi = n; r.a = 0.0f; return r;
}

// Spherical linear interpolation. Quaternions are float4 with .w = real.
static float4 slerp(float4 a, float4 b, float t) {
    float cosTheta = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    if (cosTheta < 0.0f) {
        b.x = -b.x; b.y = -b.y; b.z = -b.z; b.w = -b.w;
        cosTheta = -cosTheta;
    }
    if (cosTheta > 0.9995f) {
        // Linear lerp + renormalise: cheap and accurate near identity.
        float4 r = make_float4(a.x + t*(b.x - a.x),
                               a.y + t*(b.y - a.y),
                               a.z + t*(b.z - a.z),
                               a.w + t*(b.w - a.w));
        float n = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z + r.w*r.w);
        if (n > 1e-12f) { r.x/=n; r.y/=n; r.z/=n; r.w/=n; }
        return r;
    }
    float theta = acosf(cosTheta);
    float sinTheta = sinf(theta);
    float wa = sinf((1.0f - t) * theta) / sinTheta;
    float wb = sinf(t * theta) / sinTheta;
    return make_float4(wa*a.x + wb*b.x,
                       wa*a.y + wb*b.y,
                       wa*a.z + wb*b.z,
                       wa*a.w + wb*b.w);
}

// Build a row-major 4x4 from translation, rotation (wxyz), scale.
static float4x4 composeTRS(float3 T, float4 Q, float3 S) {
    // Rotation matrix from quaternion (w,x,y,z).
    float xx = Q.x * Q.x;
    float yy = Q.y * Q.y;
    float zz = Q.z * Q.z;
    float xy = Q.x * Q.y;
    float xz = Q.x * Q.z;
    float yz = Q.y * Q.z;
    float wx = Q.w * Q.x;
    float wy = Q.w * Q.y;
    float wz = Q.w * Q.z;

    float r00 = 1.0f - 2.0f*(yy + zz);
    float r01 = 2.0f*(xy - wz);
    float r02 = 2.0f*(xz + wy);

    float r10 = 2.0f*(xy + wz);
    float r11 = 1.0f - 2.0f*(xx + zz);
    float r12 = 2.0f*(yz - wx);

    float r20 = 2.0f*(xz - wy);
    float r21 = 2.0f*(yz + wx);
    float r22 = 1.0f - 2.0f*(xx + yy);

    float4x4 M = float4x4::identity();
    M.m[0][0] = r00 * S.x; M.m[0][1] = r01 * S.y; M.m[0][2] = r02 * S.z; M.m[0][3] = T.x;
    M.m[1][0] = r10 * S.x; M.m[1][1] = r11 * S.y; M.m[1][2] = r12 * S.z; M.m[1][3] = T.y;
    M.m[2][0] = r20 * S.x; M.m[2][1] = r21 * S.y; M.m[2][2] = r22 * S.z; M.m[2][3] = T.z;
    M.m[3][0] = 0.0f; M.m[3][1] = 0.0f; M.m[3][2] = 0.0f; M.m[3][3] = 1.0f;
    return M;
}

void evalAnimation(const Scene& scene,
                   const AnimationClip& clip,
                   float t,
                   std::vector<float4x4>& localOut,
                   std::vector<float4x4>& worldOut)
{
    const auto& nodes = scene.getNodes();
    localOut.assign(nodes.size(), float4x4::identity());
    worldOut.assign(nodes.size(), float4x4::identity());

    // Loop time into [0, duration).
    float dur = (clip.duration > 1e-6f) ? clip.duration : 1.0f;
    float ts = t - dur * floorf(t / dur);

    // 1) Default every node to its rest local transform.
    for (size_t i = 0; i < nodes.size(); i++) {
        localOut[i] = nodes[i].localRest;
    }

    // 2) Override animated nodes with sampled TRS.
    for (size_t c = 0; c < clip.channels.size(); c++) {
        int ni = clip.nodeIndices[c];
        if (ni < 0 || (size_t)ni >= nodes.size()) continue;
        const AnimChannelTrack& tr = clip.channels[c];

        // Defaults: extract from the node's rest local matrix if a track is
        // missing one of T/R/S. For our scene every animated node has all
        // three, so this fallback is just defensive.
        float3 T = make_float3(nodes[ni].localRest.m[0][3],
                               nodes[ni].localRest.m[1][3],
                               nodes[ni].localRest.m[2][3]);
        float4 Q = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        float3 S = make_float3(1.0f, 1.0f, 1.0f);

        if (!tr.posTimes.empty()) {
            TrackBlend b = findKeys(tr.posTimes, ts);
            T = lerp(tr.posValues[b.lo], tr.posValues[b.hi], b.a);
        }
        if (!tr.rotTimes.empty()) {
            TrackBlend b = findKeys(tr.rotTimes, ts);
            Q = slerp(tr.rotValues[b.lo], tr.rotValues[b.hi], b.a);
        }
        if (!tr.scaleTimes.empty()) {
            TrackBlend b = findKeys(tr.scaleTimes, ts);
            S = lerp(tr.scaleValues[b.lo], tr.scaleValues[b.hi], b.a);
        }
        localOut[ni] = composeTRS(T, Q, S);
    }

    // 3) Propagate to world transforms. Nodes are stored in parent-before-
    //    child order by SceneLoader, so a single forward pass suffices.
    for (size_t i = 0; i < nodes.size(); i++) {
        int p = nodes[i].parent;
        if (p < 0) {
            worldOut[i] = localOut[i];
        } else {
            worldOut[i] = mat4_multiply(worldOut[p], localOut[i]);
        }
    }
}

void computeMeshDeltas(const Scene& scene,
                       const std::vector<float4x4>& worldCurr,
                       std::vector<float4x4>& restToCurrOut)
{
    const auto& bindings = scene.getMeshBindings();
    const auto& nodes    = scene.getNodes();
    restToCurrOut.assign(bindings.size(), float4x4::identity());

    // Cache invRest per node so we don't recompute for every mesh sharing a
    // node. Most nodes have only 1 mesh in this scene, but a handful host
    // dozens (the polySurface clusters).
    std::vector<float4x4> invRest(nodes.size(), float4x4::identity());
    std::vector<char>     invRestValid(nodes.size(), 0);

    for (size_t i = 0; i < bindings.size(); i++) {
        int ni = bindings[i].nodeIndex;
        if (ni < 0) {
            restToCurrOut[i] = float4x4::identity();
            continue;
        }
        if (!invRestValid[ni]) {
            invRest[ni] = mat4_inverse(nodes[ni].worldRest);
            invRestValid[ni] = 1;
        }
        restToCurrOut[i] = mat4_multiply(worldCurr[ni], invRest[ni]);
    }
}

void computeNormalMats(const std::vector<float4x4>& restToCurr,
                       std::vector<NormalMat34>& out)
{
    out.assign(restToCurr.size(), NormalMat34{});
    for (size_t i = 0; i < restToCurr.size(); i++) {
        const float4x4& M = restToCurr[i];
        // For rigid (rotation-only) transforms, the normal matrix equals the
        // upper-left 3x3. For uniform scale it's the same up to a positive
        // factor that re-normalisation eats. For non-uniform scale we'd need
        // inverse-transpose of the upper-3x3; FBX rigid-body keyframes here
        // include scale tracks but they're scalar 1.0 in practice. We compute
        // the full inverse-transpose to be safe.
        float a00 = M.m[0][0], a01 = M.m[0][1], a02 = M.m[0][2];
        float a10 = M.m[1][0], a11 = M.m[1][1], a12 = M.m[1][2];
        float a20 = M.m[2][0], a21 = M.m[2][1], a22 = M.m[2][2];
        // Cofactor matrix (= adjugate^T = det * inverse-transpose). For
        // normal transformation we want inverse-transpose, so we use the
        // cofactor matrix directly (the determinant scales away after
        // renormalising the result on the GPU).
        float c00 =  (a11 * a22 - a12 * a21);
        float c01 = -(a10 * a22 - a12 * a20);
        float c02 =  (a10 * a21 - a11 * a20);
        float c10 = -(a01 * a22 - a02 * a21);
        float c11 =  (a00 * a22 - a02 * a20);
        float c12 = -(a00 * a21 - a01 * a20);
        float c20 =  (a01 * a12 - a02 * a11);
        float c21 = -(a00 * a12 - a02 * a10);
        float c22 =  (a00 * a11 - a01 * a10);
        out[i].row[0] = make_float4(c00, c01, c02, 0.0f);
        out[i].row[1] = make_float4(c10, c11, c12, 0.0f);
        out[i].row[2] = make_float4(c20, c21, c22, 0.0f);
    }
}
