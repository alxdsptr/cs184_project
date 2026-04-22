#pragma once
#include <cstddef>
#include <cstdint>

struct GLFWwindow;
class VulkanDisplay;

class GUI {
public:
    // Hooks ImGui to the GLFW window for input and to the Vulkan display
    // backend for rendering. The display is used to borrow instance/device/
    // queue/renderpass and to register this GUI as its per-frame draw callback.
    void init(GLFWwindow* window, VulkanDisplay* display);

    void beginFrame();
    // Optional `renderMode` / `dlssQuality` appear as GUI controls when non-null.
    // `renderMode`: 0 = Native, 1 = NRD only, 2 = NRD + DLSS, 3 = DLSS only.
    // `dlssQuality`: 0 = Perf, 1 = Balanced, 2 = Quality, 3 = DLAA.
    // `renderResW/H` are informational readouts (the render resolution DLSS
    // picked). Callers may pass 0.
    bool render(float fps, uint32_t sampleCount, uint32_t width, uint32_t height,
                bool& enableEnvironment, bool& invertMouseY, uint32_t& maxBounces,
                float& exposure, int& toneMappingMode,
                float& moveSpeed,
                char* envMapPathBuf, size_t envMapPathBufSize, bool& loadEnvMapRequested,
                bool& debugShowPointLights,
                bool& debugShowEmissiveMeshes,
                bool& skipEmissiveInNEE,
                int& heatmapMode,
                int* renderMode = nullptr,
                int* dlssQuality = nullptr,
                uint32_t renderResW = 0,
                uint32_t renderResH = 0);
    void endFrame();
    void shutdown();

    bool wantCaptureMouse() const;
    bool wantCaptureKeyboard() const;

private:
    bool m_initialized = false;
    VulkanDisplay* m_display = nullptr;
};
