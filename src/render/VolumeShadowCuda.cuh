#pragma once
// CUDA-backend shadow-ray adapter for participating-medium NEE. Pairs with
// the backend-agnostic NEE template in render/VolumeNEE.cuh — that template
// receives this function as the `TraceShadowFn` callable so the medium
// integrator can stay backend-agnostic while we use the SAH-BVH on CUDA and
// the OptiX GAS on the OptiX backend.

#include "core/Math.h"
#include "gpu/MaterialGPU.h"
#include "gpu/RayTypes.h"
#include "gpu/DeviceScene.h"
#include "accel/BVH.h"

// Casts a shadow ray against the SAH-BVH and returns RGB attenuation for the
// segment [origin, origin + dir*dist]. Walks up to 8 transmissive surfaces
// (glass tints by albedo; near-white passes through). Returns (0,0,0) on
// opaque occlusion, (1,1,1) when the segment is unobstructed. Volumetric
// transmittance is NOT applied here — callers fold that in separately via
// volumeShadowTransmittance() in core/VolumeDevice.cuh.
//
// Signature matches the `TraceShadowFn` template parameter of
// volumeSingleScatterInScatter() in render/VolumeNEE.cuh.
__device__ inline float3 cudaTraceTransmissiveShadow(
    const DeviceSceneData& scene,
    float3 origin, float3 dir, float dist)
{
    if (!scene.d_bvhNodes || scene.totalTriangles == 0) {
        return make_float3(1.0f, 1.0f, 1.0f);
    }
    float3 atten = make_float3(1.0f, 1.0f, 1.0f);
    Ray sr;
    sr.origin = origin;
    sr.direction = dir;
    sr.tmin = 0.001f;
    sr.tmax = (dist >= 1e29f) ? 1e30f : fmaxf(dist - 0.002f, 0.001f);
    for (int step = 0; step < 8; step++) {
        HitRecord sh;
        sh.t = sr.tmax;
        if (!bvh_closestHit(sr, scene.d_bvhNodes, scene.bvhRootIndex,
                            scene.d_positions, scene.d_indices, scene.d_materialIndices,
                            sh)) {
            return atten;
        }
        if (sh.materialIndex < 0 || (uint32_t)sh.materialIndex >= scene.materialCount) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
        GPUMaterial sm = scene.d_materials[sh.materialIndex];
        if (sm.transmission <= 0.0f) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
        float albLum = 0.2126f * sm.albedo.x + 0.7152f * sm.albedo.y + 0.0722f * sm.albedo.z;
        if (albLum < 0.9f) atten = atten * sm.albedo;
        sr.origin = sh.position + dir * 0.002f;
        if (dist >= 1e29f) {
            sr.tmax = 1e30f;
        } else {
            sr.tmax = fmaxf(dist - length(sr.origin - origin) - 0.002f, 0.001f);
        }
    }
    return atten;
}
