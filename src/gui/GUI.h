#pragma once
#include <cstdint>

struct GLFWwindow;

class GUI {
public:
    void init(GLFWwindow* window);
    void beginFrame();
    void render(float fps, uint32_t sampleCount, uint32_t width, uint32_t height);
    void endFrame();
    void shutdown();

    bool wantCaptureMouse() const;
    bool wantCaptureKeyboard() const;

private:
    bool m_initialized = false;
};
