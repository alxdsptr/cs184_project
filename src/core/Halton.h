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
//
// `phases` = number of unique sub-pixel samples before the sequence repeats.
// DLSS Programming Guide §3.7.1.1 sets the minimum phase count by quality:
//   Quality (1.5x) >= 18, Balanced (1.724x) >= 24, Performance (2.0x) >= 32,
//   Ultra Performance (3.0x) >= 72.  16 was below minimum for every DLSS
//   quality except DLAA — under camera motion the pattern repeated so fast
//   it caused a visible "screen door" shimmer. 32 satisfies up through
//   Performance; for Ultra Performance the caller should pass 72 explicitly.
inline HD float2 haltonJitter(uint32_t frameIndex, uint32_t phases = 32) {
    uint32_t idx = (frameIndex % phases) + 1;
    return make_float2(halton(idx, 2) - 0.5f, halton(idx, 3) - 0.5f);
}
