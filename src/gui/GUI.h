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
    bool render(float fps, uint32_t sampleCount, uint32_t width, uint32_t height,
                bool& enableEnvironment, bool& invertMouseY, uint32_t& maxBounces,
                float& exposure, int& toneMappingMode,
                float& moveSpeed,
                char* envMapPathBuf, size_t envMapPathBufSize, bool& loadEnvMapRequested);
    void endFrame();
    void shutdown();

    bool wantCaptureMouse() const;
    bool wantCaptureKeyboard() const;

private:
    bool m_initialized = false;
    VulkanDisplay* m_display = nullptr;
};
