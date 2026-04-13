#pragma once
#include "core/Types.h"
#include "core/Math.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ── GGX/Trowbridge-Reitz Normal Distribution ────────────────
inline D float ggxD(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * denom * denom + 1e-7f);
}

// ── Schlick Fresnel ──────────────────────────────────────────
inline D float3 fresnelSchlick(float cosTheta, float3 F0) {
    float t = 1.0f - clampf(cosTheta, 0.0f, 1.0f);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return F0 + (make_float3(1,1,1) - F0) * t5;
}

// ── Smith-GGX Geometry ───────────────────────────────────────
inline D float smithG1(float NdotX, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotX / (NdotX * (1.0f - k) + k + 1e-7f);
}

inline D float smithG(float NdotL, float NdotV, float roughness) {
    return smithG1(NdotL, roughness) * smithG1(NdotV, roughness);
}

// ── Full Cook-Torrance evaluation ────────────────────────────
inline D float3 evaluateCookTorrance(
    const GPUMaterial& mat, float3 N, float3 V, float3 L, float3 albedo)
{
    float3 H = normalize(V + L);
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float LdotH = fmaxf(dot(L, H), 0.0f);

    if (NdotL < 1e-6f || NdotV < 1e-6f)
        return make_float3(0, 0, 0);

    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, mat.metallic);
    float3 F = fresnelSchlick(LdotH, F0);

    float D_val = ggxD(NdotH, mat.roughness);
    float G_val = smithG(NdotL, NdotV, mat.roughness);

    float3 specular = F * (D_val * G_val) * (1.0f / (4.0f * NdotL * NdotV + 1e-7f));

    float3 kd = (make_float3(1,1,1) - F) * (1.0f - mat.metallic);
    float3 diffuse = kd * albedo * (1.0f / M_PI_F);

    return (diffuse + specular) * NdotL;
}
