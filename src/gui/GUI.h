#pragma once
#include <cstddef>
#include <cstdint>
#include <vector_types.h>  // for float4

struct GLFWwindow;
class VulkanDisplay;
struct CameraParams;
struct VolumeMedium;

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
                // Optional volumetric medium controls. Pass nullptr to hide
                // the section entirely.
                VolumeMedium* medium = nullptr,
                int* renderMode = nullptr,
                int* dlssQuality = nullptr,
                uint32_t renderResW = 0,
                uint32_t renderResH = 0,
                // Normal-map debug viz. 0 = off, 1 = perturbed N, 2 = tangent
                // handedness, 3 = back-face-after-perturbation flag.
                int* debugNormalViz = nullptr,
                // Master switch for normal maps (off = interpolated N only).
                bool* enableNormalMap = nullptr,
                // Normal-arrow overlay toggle + its parameters.
                bool* showNormalArrows = nullptr,
                int*  normalArrowStride = nullptr,
                float* normalArrowLength = nullptr,
                // ReSTIR toggles (DI = direct lighting, GI = indirect).
                // Pass nullptr to hide the section.
                bool* restirDIEnabled = nullptr,
                bool* restirGIEnabled = nullptr,
                // ReSTIR PT (Lin et al. 2022) toggle + path-length knob.
                // Pass nullptr to hide the controls.
                bool* restirPTEnabled = nullptr,
                int*  restirPTPathLength = nullptr,
                // FBX-clip animation playback. `playAnimation` toggles
                // play/pause; `animFps` is the rate at which the renderer
                // advances clip time per render frame. `animClipDurationSec`
                // is read-only context (length of the loaded clip). Pass
                // nullptr to any of them to hide the panel.
                bool*  playAnimation = nullptr,
                float* animFps = nullptr,
                float  animClipDurationSec = 0.0f,
                float  animCurrentTime = 0.0f);

    // Draw a sparse normal-arrow overlay on top of the path-traced image.
    // Call this once per frame between beginFrame() and endFrame(), only
    // when showNormalArrows is on. `arrows` is laid out as 2*N float4s:
    //   [2*i + 0].xyz = world pos,  .w = valid flag (1 = sample captured)
    //   [2*i + 1].xyz = world normal
    // `gridW/gridH` is the sample grid extent (width = ceil(screenW/stride)).
    // `camera` provides the world→clip matrix for projection.
    void drawNormalArrowsOverlay(
        const float4* arrows, int gridW, int gridH,
        const CameraParams& camera,
        uint32_t screenW, uint32_t screenH,
        float arrowLengthWorld);

    void endFrame();
    void shutdown();

    bool wantCaptureMouse() const;
    bool wantCaptureKeyboard() const;

private:
    bool m_initialized = false;
    VulkanDisplay* m_display = nullptr;
};
