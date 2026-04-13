#pragma once

// On Windows, include windows.h early to avoid GL header conflicts
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

#ifdef __CUDACC__
#  define HD __host__ __device__
#  define D  __device__
#else
#  define HD
#  define D
#endif

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

struct float4x4 {
    float m[4][4];

    HD static float4x4 identity() {
        float4x4 r{};
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
        return r;
    }
};
