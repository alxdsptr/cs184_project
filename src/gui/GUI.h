#pragma once
#include <cstdint>

struct GLFWwindow;

class GUI {
public:
    void init(GLFWwindow* window);
    void beginFrame();
    bool render(float fps, uint32_t sampleCount, uint32_t width, uint32_t height, bool& enableEnvironment, bool& invertMouseY);
    void endFrame();
    void shutdown();

    bool wantCaptureMouse() const;
    bool wantCaptureKeyboard() const;

private:
    bool m_initialized = false;
};
