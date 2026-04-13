#pragma once
#include "core/Types.h"

// PCG32 minimal random number generator
inline D uint32_t pcg32(uint32_t& state) {
    uint32_t oldstate = state;
    state = oldstate * 747796405u + 2891336453u;
    uint32_t word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    return (word >> 22u) ^ word;
}

inline D float pcg32_float(uint32_t& state) {
    return (float)pcg32(state) / 4294967296.0f;
}

inline D uint32_t pcg32_seed(uint32_t pixelIndex, uint32_t frameIndex) {
    return pixelIndex * 1973u + frameIndex * 9277u + 6133u;
}
