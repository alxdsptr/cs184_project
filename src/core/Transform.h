#pragma once
#include "core/Types.h"
#include "core/Math.h"

inline float4x4 mat4_translate(float3 t) {
    float4x4 m = float4x4::identity();
    m.m[0][3] = t.x;
    m.m[1][3] = t.y;
    m.m[2][3] = t.z;
    return m;
}

inline float4x4 mat4_scale(float3 s) {
    float4x4 m{};
    m.m[0][0] = s.x;
    m.m[1][1] = s.y;
    m.m[2][2] = s.z;
    m.m[3][3] = 1.0f;
    return m;
}
