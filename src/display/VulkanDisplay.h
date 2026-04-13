#pragma once
#include "display/DisplayBackend.h"

// Stub for future Vulkan display backend (required for DLSS integration).
// Not implemented -- placeholder to preserve the abstraction boundary.
class VulkanDisplay : public DisplayBackend {
public:
    void init(uint32_t, uint32_t) override {}
    void resize(uint32_t, uint32_t) override {}
    void* mapForCUDA() override { return nullptr; }
    void unmapFromCUDA() override {}
    void present() override {}
    void shutdown() override {}
};
