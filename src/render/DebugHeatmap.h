#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Per-pixel accumulation buffers that split the path-traced radiance by the
// kind of light source that contributed it. Used to produce a "where does the
// light in this pixel actually come from" visualization. Independent from the
// normal accumulation buffer -- these accumulate even while the main accum
// does, and are reset together with it via reset().
//
// All four buffers are float4 (rgb = accumulated radiance, a = sample count
// written by the kernel). Divided by the accumulator's sample count at
// visualization time.
struct DebugHeatmapPtrs {
    float4* d_pointLight = nullptr;   // point-light NEE contributions
    float4* d_areaLight  = nullptr;   // area-light NEE + emissive-hit contributions
    float4* d_environment = nullptr;  // env map / procedural sky
    float4* d_indirect   = nullptr;   // everything else (catch-all)

    // Per-pixel "which emitter is mainly lighting me" accumulator. rgb stores
    // SUM(color(emitterID) * luminance(contribution)); a stores
    // SUM(luminance(contribution)). Divide rgb by a at visualization time to
    // get the luminance-weighted average emitter color. Only populated for
    // area-light NEE and direct emissive hits (the ones with a meaningful
    // material ID); point lights are mapped through the same hash.
    float4* d_byEmitter  = nullptr;
};

enum class DebugHeatmapMode : uint32_t {
    Off          = 0,  // disable heatmap tracking entirely (no extra writes)
    Categorized  = 1,  // RGB = (point, area+emissive, env); indirect dimmed
    PointLight   = 2,  // show only the point-light bucket (grayscale)
    AreaLight    = 3,  // show only the area-light+emissive bucket
    Environment  = 4,  // show only the env/sky bucket
    Indirect     = 5,  // show only the "other" bucket
    ByEmitter    = 6,  // color pixels by the dominant area-light material ID
};

class DebugHeatmapBuffers {
public:
    void init(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void reset();
    void free();

    DebugHeatmapPtrs getPtrs() const { return m_ptrs; }
    bool valid() const { return m_ptrs.d_pointLight != nullptr; }

private:
    DebugHeatmapPtrs m_ptrs;
    uint32_t m_width = 0, m_height = 0;
};

// Visualize the heatmap buffers into the LDR swapchain output. `mode` must be
// non-Off. `sampleCount` is the current accumulation sample count (for the
// 1/N normalization).
void launchDebugHeatmapKernel(
    const DebugHeatmapPtrs& buffers,
    uchar4*  d_ldrOutput,
    uint32_t width,
    uint32_t height,
    uint32_t sampleCount,
    DebugHeatmapMode mode,
    float    exposure);
