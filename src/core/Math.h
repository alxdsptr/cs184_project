#pragma once
#include "core/Types.h"

// ── float3 operations ────────────────────────────────────────
inline HD float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline HD float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline HD float3 operator*(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
inline HD float3 operator*(float3 a, float s)  { return make_float3(a.x*s, a.y*s, a.z*s); }
inline HD float3 operator*(float s, float3 a)  { return a * s; }
inline HD float3 operator/(float3 a, float s)  { float inv = 1.0f/s; return a * inv; }
inline HD float3 operator-(float3 a)           { return make_float3(-a.x, -a.y, -a.z); }
inline HD float3& operator+=(float3& a, float3 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
inline HD float3& operator*=(float3& a, float s)  { a.x*=s; a.y*=s; a.z*=s; return a; }

inline HD float  dot(float3 a, float3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline HD float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline HD float  length(float3 a)      { return sqrtf(dot(a, a)); }
inline HD float3 normalize(float3 a)   { return a / length(a); }
inline HD float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
inline HD float  clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

// ── float2 operations ────────────────────────────────────────
inline HD float2 operator+(float2 a, float2 b) { return make_float2(a.x+b.x, a.y+b.y); }
inline HD float2 operator-(float2 a, float2 b) { return make_float2(a.x-b.x, a.y-b.y); }
inline HD float2 operator*(float2 a, float s)  { return make_float2(a.x*s, a.y*s); }

// ── float4 operations ────────────────────────────────────────
inline HD float4 operator+(float4 a, float4 b) { return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
inline HD float4 operator*(float4 a, float s)  { return make_float4(a.x*s, a.y*s, a.z*s, a.w*s); }
inline HD float4 operator/(float4 a, float s)  { float inv = 1.0f/s; return a * inv; }
inline HD float4& operator+=(float4& a, float4 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; return a; }

// ── Matrix operations ────────────────────────────────────────
float4x4 mat4_multiply(const float4x4& a, const float4x4& b);
float4x4 mat4_inverse(const float4x4& m);
float4x4 mat4_lookAt(float3 eye, float3 center, float3 up);
float4x4 mat4_perspective(float fovY, float aspect, float nearP, float farP);

// Transform float3 by matrix (assumes w=1, perspective divide)
inline HD float3 mat4_transformPoint(const float4x4& m, float3 p) {
    float x = m.m[0][0]*p.x + m.m[0][1]*p.y + m.m[0][2]*p.z + m.m[0][3];
    float y = m.m[1][0]*p.x + m.m[1][1]*p.y + m.m[1][2]*p.z + m.m[1][3];
    float z = m.m[2][0]*p.x + m.m[2][1]*p.y + m.m[2][2]*p.z + m.m[2][3];
    float w = m.m[3][0]*p.x + m.m[3][1]*p.y + m.m[3][2]*p.z + m.m[3][3];
    if (fabsf(w) > 1e-7f) { x /= w; y /= w; z /= w; }
    return make_float3(x, y, z);
}

// Transform direction (no translation, no perspective divide)
inline HD float3 mat4_transformDir(const float4x4& m, float3 d) {
    return make_float3(
        m.m[0][0]*d.x + m.m[0][1]*d.y + m.m[0][2]*d.z,
        m.m[1][0]*d.x + m.m[1][1]*d.y + m.m[1][2]*d.z,
        m.m[2][0]*d.x + m.m[2][1]*d.y + m.m[2][2]*d.z
    );
}
