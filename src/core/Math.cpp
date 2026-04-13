#include "core/Math.h"
#include <cmath>

float4x4 mat4_multiply(const float4x4& a, const float4x4& b) {
    float4x4 r{};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                r.m[i][j] += a.m[i][k] * b.m[k][j];
    return r;
}

float4x4 mat4_inverse(const float4x4& m) {
    float4x4 inv{};
    const float* s = &m.m[0][0];
    float* d = &inv.m[0][0];

    float s0 = s[0]*s[5]  - s[1]*s[4];
    float s1 = s[0]*s[6]  - s[2]*s[4];
    float s2 = s[0]*s[7]  - s[3]*s[4];
    float s3 = s[1]*s[6]  - s[2]*s[5];
    float s4 = s[1]*s[7]  - s[3]*s[5];
    float s5 = s[2]*s[7]  - s[3]*s[6];
    float c5 = s[10]*s[15] - s[11]*s[14];
    float c4 = s[9]*s[15]  - s[11]*s[13];
    float c3 = s[9]*s[14]  - s[10]*s[13];
    float c2 = s[8]*s[15]  - s[11]*s[12];
    float c1 = s[8]*s[14]  - s[10]*s[12];
    float c0 = s[8]*s[13]  - s[9]*s[12];

    float det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
    if (fabsf(det) < 1e-12f) return float4x4::identity();
    float invDet = 1.0f / det;

    d[0]  = ( s[5]*c5 - s[6]*c4 + s[7]*c3)  * invDet;
    d[1]  = (-s[1]*c5 + s[2]*c4 - s[3]*c3)  * invDet;
    d[2]  = ( s[13]*s5 - s[14]*s4 + s[15]*s3) * invDet;
    d[3]  = (-s[9]*s5  + s[10]*s4 - s[11]*s3) * invDet;
    d[4]  = (-s[4]*c5 + s[6]*c2 - s[7]*c1)  * invDet;
    d[5]  = ( s[0]*c5 - s[2]*c2 + s[3]*c1)  * invDet;
    d[6]  = (-s[12]*s5 + s[14]*s2 - s[15]*s1) * invDet;
    d[7]  = ( s[8]*s5  - s[10]*s2 + s[11]*s1) * invDet;
    d[8]  = ( s[4]*c4 - s[5]*c2 + s[7]*c0)  * invDet;
    d[9]  = (-s[0]*c4 + s[1]*c2 - s[3]*c0)  * invDet;
    d[10] = ( s[12]*s4 - s[13]*s2 + s[15]*s0) * invDet;
    d[11] = (-s[8]*s4  + s[9]*s2  - s[11]*s0) * invDet;
    d[12] = (-s[4]*c3 + s[5]*c1 - s[6]*c0)  * invDet;
    d[13] = ( s[0]*c3 - s[1]*c1 + s[2]*c0)  * invDet;
    d[14] = (-s[12]*s3 + s[13]*s1 - s[14]*s0) * invDet;
    d[15] = ( s[8]*s3  - s[9]*s1  + s[10]*s0) * invDet;

    return inv;
}

float4x4 mat4_lookAt(float3 eye, float3 center, float3 up) {
    float3 f = normalize(center - eye);
    float3 r = normalize(cross(f, up));
    float3 u = cross(r, f);

    float4x4 m = float4x4::identity();
    m.m[0][0] =  r.x; m.m[0][1] =  r.y; m.m[0][2] =  r.z; m.m[0][3] = -dot(r, eye);
    m.m[1][0] =  u.x; m.m[1][1] =  u.y; m.m[1][2] =  u.z; m.m[1][3] = -dot(u, eye);
    m.m[2][0] = -f.x; m.m[2][1] = -f.y; m.m[2][2] = -f.z; m.m[2][3] =  dot(f, eye);
    return m;
}

float4x4 mat4_perspective(float fovY, float aspect, float nearP, float farP) {
    float tanHalf = tanf(fovY * 0.5f);
    float4x4 m{};
    m.m[0][0] = 1.0f / (aspect * tanHalf);
    m.m[1][1] = 1.0f / tanHalf;
    m.m[2][2] = -(farP + nearP) / (farP - nearP);
    m.m[2][3] = -2.0f * farP * nearP / (farP - nearP);
    m.m[3][2] = -1.0f;
    return m;
}
