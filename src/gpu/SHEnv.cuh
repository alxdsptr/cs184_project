#pragma once
// L2 (3rd-order, 9 coefficient) Spherical Harmonics helpers for diffuse
// environment irradiance. The basis ordering matches the one used in
// TextureManager::projectEnvToSH, so coefficient i here corresponds to
// coefficient i there.
//
// Diffuse irradiance convolution follows Ramamoorthi & Hanrahan 2001
// ("An Efficient Representation for Irradiance Environment Maps"): the
// cosine-lobe projection reduces to per-band scalars A_l applied to the
// radiance coefficients.

#include <cuda_runtime.h>
#include "core/Math.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// Ramamoorthi/Hanrahan cosine-lobe band factors. A0=π, A1=2π/3, A2=π/4.
__device__ __host__ inline float sh_cosineBandFactor(int l) {
    if (l == 0) return M_PI_F;
    if (l == 1) return 2.0f * M_PI_F / 3.0f;
    if (l == 2) return M_PI_F * 0.25f;
    return 0.0f;
}

// Real SH basis Y_l^m(n) for unit direction n. Nine bands in canonical order.
// Constants are the standard normalised real SH basis evaluated on x,y,z.
__device__ __host__ inline void sh_basis9(const float3& n, float out[9]) {
    float x = n.x, y = n.y, z = n.z;
    out[0] = 0.282094792f;                          // Y_0^0
    out[1] = 0.488602512f * y;                      // Y_1^-1
    out[2] = 0.488602512f * z;                      // Y_1^0
    out[3] = 0.488602512f * x;                      // Y_1^1
    out[4] = 1.092548431f * x * y;                  // Y_2^-2
    out[5] = 1.092548431f * y * z;                  // Y_2^-1
    out[6] = 0.315391565f * (3.0f * z * z - 1.0f);  // Y_2^0
    out[7] = 1.092548431f * x * z;                  // Y_2^1
    out[8] = 0.546274215f * (x * x - y * y);        // Y_2^2
}

// Evaluate diffuse irradiance E(n) from 9 radiance SH coefficients.
// Returns *irradiance* (∫ L(ω) max(n·ω, 0) dω); divide by π to turn it into
// the reflected radiance for a Lambertian surface with albedo 1.
__device__ inline float3 sh_evalIrradiance(const float3& n, const float3* coeffs) {
    if (!coeffs) return make_float3(0.0f, 0.0f, 0.0f);
    float basis[9];
    sh_basis9(n, basis);
    float3 E = make_float3(0.0f, 0.0f, 0.0f);
    // Band 0
    E = E + coeffs[0] * (basis[0] * sh_cosineBandFactor(0));
    // Band 1
    float a1 = sh_cosineBandFactor(1);
    E = E + coeffs[1] * (basis[1] * a1);
    E = E + coeffs[2] * (basis[2] * a1);
    E = E + coeffs[3] * (basis[3] * a1);
    // Band 2
    float a2 = sh_cosineBandFactor(2);
    E = E + coeffs[4] * (basis[4] * a2);
    E = E + coeffs[5] * (basis[5] * a2);
    E = E + coeffs[6] * (basis[6] * a2);
    E = E + coeffs[7] * (basis[7] * a2);
    E = E + coeffs[8] * (basis[8] * a2);
    if (E.x < 0.0f) E.x = 0.0f;
    if (E.y < 0.0f) E.y = 0.0f;
    if (E.z < 0.0f) E.z = 0.0f;
    return E;
}

// Reconstruct *radiance* from the SH coefficients along a direction (no
// cosine-lobe convolution). A low-order reconstruction: good enough for
// indirect diffuse rays that miss the scene, where a smooth approximation
// is preferable to the noisy HDR sample.
__device__ inline float3 sh_evalRadiance(const float3& dir, const float3* coeffs) {
    if (!coeffs) return make_float3(0.0f, 0.0f, 0.0f);
    float basis[9];
    sh_basis9(dir, basis);
    float3 L = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 9; i++) L = L + coeffs[i] * basis[i];
    if (L.x < 0.0f) L.x = 0.0f;
    if (L.y < 0.0f) L.y = 0.0f;
    if (L.z < 0.0f) L.z = 0.0f;
    return L;
}
