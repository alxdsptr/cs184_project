#pragma once
#include <cuda_runtime.h>
#include <cstdint>

void launchTonemapKernel(
    const float4* d_hdrInput,
    uchar4* d_ldrOutput,
    uint32_t width,
    uint32_t height,
    float exposure
);
