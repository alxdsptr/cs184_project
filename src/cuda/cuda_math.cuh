#ifndef CGL_CUDA_MATH_CUH
#define CGL_CUDA_MATH_CUH

#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================
// float3 arithmetic helpers
// ============================================================

__device__ __host__ inline float3 f3_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 f3_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 f3_mul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ inline float3 f3_mul_comp(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__ inline float3 f3_neg(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __host__ inline float f3_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float3 f3_cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __host__ inline float f3_length2(float3 a) {
    return f3_dot(a, a);
}

__device__ __host__ inline float f3_length(float3 a) {
    return sqrtf(f3_length2(a));
}

__device__ __host__ inline float3 f3_normalize(float3 a) {
    float len = f3_length(a);
    if (len > 0.0f) return f3_mul(a, 1.0f / len);
    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ __host__ inline float3 f3_zero() {
    return make_float3(0.0f, 0.0f, 0.0f);
}

// ============================================================
// 3x3 Matrix (stored as 3 column vectors)
// ============================================================

struct CUDAMatrix3x3 {
    float3 col0, col1, col2;
};

__device__ __host__ inline float3 mat3_mul_vec(const CUDAMatrix3x3& m, float3 v) {
    return make_float3(
        m.col0.x * v.x + m.col1.x * v.y + m.col2.x * v.z,
        m.col0.y * v.x + m.col1.y * v.y + m.col2.y * v.z,
        m.col0.z * v.x + m.col1.z * v.y + m.col2.z * v.z
    );
}

__device__ __host__ inline CUDAMatrix3x3 mat3_transpose(const CUDAMatrix3x3& m) {
    CUDAMatrix3x3 t;
    t.col0 = make_float3(m.col0.x, m.col1.x, m.col2.x);
    t.col1 = make_float3(m.col0.y, m.col1.y, m.col2.y);
    t.col2 = make_float3(m.col0.z, m.col1.z, m.col2.z);
    return t;
}

// Build a local coordinate system from a normal vector n.
// n is aligned with the Z axis of the returned basis.
// Mirrors make_coord_space() from bsdf.cpp.
__device__ inline void make_coord_space_device(CUDAMatrix3x3& o2w, float3 n) {
    float3 z = f3_normalize(n);
    float3 h = z;
    if (fabsf(h.x) <= fabsf(h.y) && fabsf(h.x) <= fabsf(h.z))
        h.x = 1.0f;
    else if (fabsf(h.y) <= fabsf(h.x) && fabsf(h.y) <= fabsf(h.z))
        h.y = 1.0f;
    else
        h.z = 1.0f;

    float3 y = f3_normalize(f3_cross(h, z));
    float3 x = f3_normalize(f3_cross(z, y));

    o2w.col0 = x;
    o2w.col1 = y;
    o2w.col2 = z;
}

// ============================================================
// Trigonometric helpers for local shading coordinates
// In local space, z = normal direction
// ============================================================

__device__ inline float cos_theta_d(float3 w) { return w.z; }
__device__ inline float abs_cos_theta_d(float3 w) { return fabsf(w.z); }
__device__ inline float sin_theta2_d(float3 w) { return fmaxf(0.0f, 1.0f - w.z * w.z); }
__device__ inline float sin_theta_d(float3 w) { return sqrtf(sin_theta2_d(w)); }

#endif // CGL_CUDA_MATH_CUH
