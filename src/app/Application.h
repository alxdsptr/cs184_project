#pragma once
#include "core/Camera.h"
#include "display/GLDisplay.h"
#include "gui/GUI.h"
#include "scene/Scene.h"
#include "scene/Texture.h"
#include "render/Renderer.h"
#include "backend/CUDABackend.h"
#include <cstdint>
#include <string>
#include <memory>

struct GLFWwindow;

class Application {
public:
    bool init(uint32_t width, uint32_t height, const std::string& title, bool enableGui = true);
    bool loadScene(const std::string& path);
    void run();
    void shutdown();
    void setMaxBounces(uint32_t maxBounces);
    void setHeadlessOutput(const std::string& outputPath, uint32_t sampleCount);

private:
    void processInput();
    void runGui();
    void runHeadless();
    void renderSceneSample(uchar4* d_pbo, bool timeHeadless);

    GLFWwindow* m_window = nullptr;
    uint32_t    m_width  = 1280;
    uint32_t    m_height = 720;

    Camera     m_camera;
    GLDisplay  m_display;
    GUI        m_gui;
    Scene      m_scene;
    TextureManager m_textures;
    Renderer   m_renderer;
    std::unique_ptr<CUDABackend> m_backend;

    bool       m_sceneLoaded = false;
    InputState m_input;
    float      m_lastFrameTime = 0.0f;
    uint32_t   m_frameIndex    = 0;

    // FPS tracking
    float    m_fps         = 0.0f;
    float    m_fpsTimer    = 0.0f;
    uint32_t m_fpsFrames   = 0;

    // Mouse state
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    bool   m_firstMouse = true;

    bool m_enableEnvironment = false;
    bool m_invertMouseY = false;
    bool m_prevF12Down = false;
    uint32_t m_maxBounces = 8;
    bool m_guiEnabled = true;
    std::string m_headlessOutputPath;
    uint32_t m_targetSamples = 1;
    double m_headlessRenderMs = 0.0;
    double m_headlessTotalMs = 0.0;
};
