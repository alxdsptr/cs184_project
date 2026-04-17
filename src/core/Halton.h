#pragma once
#include "core/Types.h"

// Low-discrepancy Halton sequence for sub-pixel jitter
inline HD float halton(int index, int base) {
    float f = 1.0f;
    float r = 0.0f;
    int i = index;
    while (i > 0) {
        f /= (float)base;
        r += f * (float)(i % base);
        i /= base;
    }
    return r;
}

// Returns jitter in [-0.5, 0.5] range for pixel-space offset.
// Default phases=16 matches the pre-feature pipeline for parity; DLSS
// recommends 32 but Renderer bumps `phases` explicitly in non-Native modes
// (see `Renderer::renderFrame`) so Native output stays bit-equivalent.
inline HD float2 haltonJitter(uint32_t frameIndex, uint32_t phases = 16) {
    uint32_t idx = (frameIndex % phases) + 1;
    return make_float2(halton(idx, 2) - 0.5f, halton(idx, 3) - 0.5f);
}
