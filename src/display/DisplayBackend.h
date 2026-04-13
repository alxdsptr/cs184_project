#pragma once
#include <cstdint>

// Abstract display interface.
// GLDisplay: current implementation (OpenGL PBO interop).
// VulkanDisplay: future stub for DLSS integration.
class DisplayBackend {
public:
    virtual ~DisplayBackend() = default;
    virtual void init(uint32_t width, uint32_t height) = 0;
    virtual void resize(uint32_t width, uint32_t height) = 0;

    // Map the display buffer for CUDA writing. Returns device pointer to uchar4 RGBA.
    virtual void* mapForCUDA() = 0;
    virtual void  unmapFromCUDA() = 0;

    // Render the mapped buffer to the screen.
    virtual void present() = 0;

    virtual void shutdown() = 0;
};
