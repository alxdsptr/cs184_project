#include "render/DebugHeatmap.h"
#include "util/CudaCheck.h"

void DebugHeatmapBuffers::init(uint32_t width, uint32_t height) {
    m_width = width;
    m_height = height;
    size_t pixels = (size_t)width * height;
    CUDA_CHECK(cudaMalloc(&m_ptrs.d_pointLight,  pixels * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&m_ptrs.d_areaLight,   pixels * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&m_ptrs.d_environment, pixels * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&m_ptrs.d_indirect,    pixels * sizeof(float4)));
    reset();
}

void DebugHeatmapBuffers::resize(uint32_t width, uint32_t height) {
    if (width == m_width && height == m_height) return;
    free();
    init(width, height);
}

void DebugHeatmapBuffers::reset() {
    if (!m_ptrs.d_pointLight) return;
    size_t bytes = (size_t)m_width * m_height * sizeof(float4);
    CUDA_CHECK(cudaMemset(m_ptrs.d_pointLight,  0, bytes));
    CUDA_CHECK(cudaMemset(m_ptrs.d_areaLight,   0, bytes));
    CUDA_CHECK(cudaMemset(m_ptrs.d_environment, 0, bytes));
    CUDA_CHECK(cudaMemset(m_ptrs.d_indirect,    0, bytes));
}

void DebugHeatmapBuffers::free() {
    if (m_ptrs.d_pointLight)  { cudaFree(m_ptrs.d_pointLight);  m_ptrs.d_pointLight  = nullptr; }
    if (m_ptrs.d_areaLight)   { cudaFree(m_ptrs.d_areaLight);   m_ptrs.d_areaLight   = nullptr; }
    if (m_ptrs.d_environment) { cudaFree(m_ptrs.d_environment); m_ptrs.d_environment = nullptr; }
    if (m_ptrs.d_indirect)    { cudaFree(m_ptrs.d_indirect);    m_ptrs.d_indirect    = nullptr; }
    m_width = m_height = 0;
}

// ── Visualization kernel ──────────────────────────────────────────
__device__ inline uchar4 toLdr(float r, float g, float b) {
    auto sat = [] (float v) {
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return (unsigned char)(v * 255.0f + 0.5f);
    };
    return make_uchar4(sat(r), sat(g), sat(b), 255);
}

__device__ inline float lumOf(float3 c) {
    return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
}

__global__ void debugHeatmapKernel(
    const float4* pointLight,
    const float4* areaLight,
    const float4* environment,
    const float4* indirect,
    uchar4*       output,
    uint32_t width, uint32_t height,
    float invN,
    DebugHeatmapMode mode,
    float exposure)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t idx = y * width + x;

    float4 p = pointLight[idx]  * invN;
    float4 a = areaLight[idx]   * invN;
    float4 e = environment[idx] * invN;
    float4 i = indirect[idx]    * invN;

    float3 pc = make_float3(p.x, p.y, p.z) * exposure;
    float3 ac = make_float3(a.x, a.y, a.z) * exposure;
    float3 ec = make_float3(e.x, e.y, e.z) * exposure;
    float3 ic = make_float3(i.x, i.y, i.z) * exposure;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (mode == DebugHeatmapMode::Categorized) {
        // Map each bucket to a distinct hue, scaled by luminance. Reinhard-
        // tonemap the luminance so bright pixels don't all clip to 1.
        float pL = lumOf(pc), aL = lumOf(ac), eL = lumOf(ec), iL = lumOf(ic);
        auto rh = [] (float v) { return v / (1.0f + v); };
        // Palette: point=red, area/emissive=green, env=blue, indirect=gray.
        r = rh(pL) + 0.3f * rh(iL);
        g = rh(aL) + 0.3f * rh(iL);
        b = rh(eL) + 0.3f * rh(iL);
    } else {
        float3 src = make_float3(0,0,0);
        if      (mode == DebugHeatmapMode::PointLight)  src = pc;
        else if (mode == DebugHeatmapMode::AreaLight)   src = ac;
        else if (mode == DebugHeatmapMode::Environment) src = ec;
        else if (mode == DebugHeatmapMode::Indirect)    src = ic;
        // Reinhard-tonemap each channel so huge HDR values stay visible.
        r = src.x / (1.0f + src.x);
        g = src.y / (1.0f + src.y);
        b = src.z / (1.0f + src.z);
    }
    output[idx] = toLdr(r, g, b);
}

void launchDebugHeatmapKernel(
    const DebugHeatmapPtrs& buffers,
    uchar4*  d_ldrOutput,
    uint32_t width,
    uint32_t height,
    uint32_t sampleCount,
    DebugHeatmapMode mode,
    float    exposure)
{
    if (mode == DebugHeatmapMode::Off) return;
    if (!buffers.d_pointLight || !d_ldrOutput) return;
    uint32_t N = sampleCount < 1u ? 1u : sampleCount;
    float invN = 1.0f / (float)N;
    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    debugHeatmapKernel<<<grid, block>>>(
        buffers.d_pointLight, buffers.d_areaLight,
        buffers.d_environment, buffers.d_indirect,
        d_ldrOutput, width, height, invN, mode, exposure);
    CUDA_CHECK(cudaGetLastError());
}
