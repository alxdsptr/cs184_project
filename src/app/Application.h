#pragma once
#include "core/Camera.h"
#include "display/VulkanDisplay.h"
#include "gui/GUI.h"
#include "scene/Scene.h"
#include "scene/SceneLoader.h"
#include "scene/Texture.h"
#include "render/Renderer.h"
#include "backend/RayTracingBackend.h"
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
    void setSamplesPerFrame(uint32_t spp);
    // 0 = Native, 1 = NRDOnly, 2 = NRDDLSS, 3 = DLSSOnly. Applied after init().
    void setInitialMode(int mode) { m_initialMode = mode; }
    // 0 = CUDA (default), 1 = OptiX. Applied during init().
    void setBackendKind(int kind) { m_backendKind = kind; }
    // Specular-Glossiness workflow mode for legacy FBX assets. Applied at
    // loadScene().
    void setSGWorkflowMode(SGWorkflowMode mode) { m_sgMode = mode; }
    // Target average linear luminance for textured emitters after adaptive
    // emissionStrength normalisation. Passed through to SceneLoader at
    // loadScene().
    void setEmissiveTargetLum(float v) { m_emissiveTargetLum = v; }
    void setHeadlessOutput(const std::string& outputPath, uint32_t sampleCount);
    void setEnvMap(const std::string& path);
    void loadCameraFile(const std::string& path) { m_cameraFilePath = path; }

private:
    static void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    void processInput();
    void runGui();
    void runHeadless();
    void renderSceneSample(uchar4* d_pbo, bool timeHeadless);
    void frameCameraToScene();

    GLFWwindow* m_window = nullptr;
    uint32_t    m_width  = 1280;
    uint32_t    m_height = 720;

    Camera        m_camera;
    VulkanDisplay m_display;
    GUI        m_gui;
    Scene      m_scene;
    TextureManager m_textures;
    Renderer   m_renderer;
    std::unique_ptr<RayTracingBackend> m_backend;
    int m_backendKind = 0;  // 0 = CUDA, 1 = OptiX
    SGWorkflowMode m_sgMode = SGWorkflowMode::Off;
    float m_emissiveTargetLum = 20.0f;

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

    // HDR environment map
    cudaTextureObject_t m_envMapTex = 0;
    char m_envMapPathBuf[512] = {};
    void loadEnvMap(const std::string& path);

    // Precomputed L2 SH radiance of the env map (9 RGB coeffs). Uploaded to
    // `m_d_shEnvCoeffs` on the device; null until an env map is loaded.
    // `m_useSHEnvIrradiance` toggles SH-shortcut sampling at runtime.
    void freeShEnvDevice();
    float3* m_d_shEnvCoeffs = nullptr;
    bool m_useSHEnvIrradiance = true;
    bool m_prevSHKeyDown = false;
    bool m_prevF12Down = false;
    bool m_prevF1Down = false;
    bool m_showGui = true;
    bool m_prevSpeedDownKey = false;
    bool m_prevSpeedUpKey = false;
    bool m_prevF2Down = false;
    bool m_prevF3Down = false;
    bool m_prevF4Down = false;
    uint32_t m_maxBounces = 8;
    uint32_t m_samplesPerFrame = 1;
    // Normal-map debug visualization. 0 = off; 1 = perturbed N; 2 = tangent
    // handedness; 3 = back-face-after-perturb flag. See DeviceSceneData.
    int m_debugNormalViz = 0;
    int m_initialMode = -1;  // -1 = leave as default (Native)
    bool m_guiEnabled = true;
    double m_pendingScrollY = 0.0;
    std::string m_headlessOutputPath;
    std::string m_cameraFilePath;
    uint32_t m_targetSamples = 1;
    double m_headlessRenderMs = 0.0;
    double m_headlessTotalMs = 0.0;
};
