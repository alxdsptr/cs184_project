#include "render/Tonemapping.h"
#include "core/Math.h"
#include "util/CudaCheck.h"

// ACES filmic tonemapping
__device__ inline float3 acesTonemap(float3 x) {
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return make_float3(
        fminf(fmaxf((x.x*(a*x.x+b))/(x.x*(c*x.x+d)+e), 0.0f), 1.0f),
        fminf(fmaxf((x.y*(a*x.y+b))/(x.y*(c*x.y+d)+e), 0.0f), 1.0f),
        fminf(fmaxf((x.z*(a*x.z+b))/(x.z*(c*x.z+d)+e), 0.0f), 1.0f)
    );
}

__device__ inline float3 reinhardTonemap(float3 x) {
    return make_float3(
        x.x / (1.0f + x.x),
        x.y / (1.0f + x.y),
        x.z / (1.0f + x.z)
    );
}

// Linear to sRGB gamma
__device__ inline float linearToSRGB(float x) {
    return (x <= 0.0031308f) ? x * 12.92f : 1.055f * powf(x, 1.0f/2.4f) - 0.055f;
}

__global__ void tonemapKernel(
    const float4* d_hdrInput,
    uchar4* d_ldrOutput,
    uint32_t width,
    uint32_t height,
    float exposure,
    ToneMappingMode mode)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint32_t idx = y * width + x;
    float4 hdr = d_hdrInput[idx];

    // Exposure
    float3 color = make_float3(hdr.x, hdr.y, hdr.z) * exposure;

    if (mode == ToneMappingMode::Reinhard) {
        color = reinhardTonemap(color);
    } else if (mode == ToneMappingMode::ACES) {
        color = acesTonemap(color);
    }

    // sRGB gamma
    unsigned char r = (unsigned char)(linearToSRGB(color.x) * 255.0f + 0.5f);
    unsigned char g = (unsigned char)(linearToSRGB(color.y) * 255.0f + 0.5f);
    unsigned char b = (unsigned char)(linearToSRGB(color.z) * 255.0f + 0.5f);

    d_ldrOutput[idx] = make_uchar4(r, g, b, 255);
}

void launchTonemapKernel(
    const float4* d_hdrInput,
    uchar4* d_ldrOutput,
    uint32_t width,
    uint32_t height,
    float exposure,
    ToneMappingMode mode)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    tonemapKernel<<<grid, block>>>(d_hdrInput, d_ldrOutput, width, height, exposure, mode);
    CUDA_CHECK(cudaGetLastError());
}
