#pragma once
#include "core/Camera.h"
#include "core/VolumeMedium.h"
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
#include <vector>

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
    // Volumetric-medium overrides applied at scene load time. Setting any of
    // these flips m_hasMediumOverride so the loader knows to use the
    // command-line-supplied parameters in place of whatever the scene file
    // declared (which is currently nothing — no loaders parse media).
    void setMediumEnabled(bool enabled) { m_medium.enabled = enabled; m_hasMediumOverride = true; }
    void setMediumSigmaA(const float3& sigmaA) { m_medium.sigmaA = sigmaA; m_hasMediumOverride = true; }
    void setMediumSigmaS(const float3& sigmaS) { m_medium.sigmaS = sigmaS; m_hasMediumOverride = true; }
    void setMediumDensity(float density) { m_medium.density = density; m_hasMediumOverride = true; }
    void setMediumAnisotropy(float g) { m_medium.anisotropy = g; m_hasMediumOverride = true; }
    void setMediumDensityKind(uint32_t k) { m_medium.densityKind = k; m_hasMediumOverride = true; }
    void setHeadlessOutput(const std::string& outputPath, uint32_t sampleCount);

    // Replay mode: load a recorded camera path (JSON written by F5 / stopRecording),
    // render one PNG per pose into outDir/frame_NNNNNN.png, then quit.
    // Scene loads exactly once → 100x faster than launching pathtracer per frame.
    // sppPerPose is the per-frame sample budget; for "1spp comparison" sweeps,
    // pass 1. The full ReSTIR/denoiser state from the CLI is honoured.
    struct ReplayOptions {
        std::string recordingPath;
        std::string outDir = "replay_frames";
        uint32_t    sppPerPose = 1;
        // If >0, replay only the first N poses of the recording (debug knob).
        uint32_t    maxPoses = 0;
        // If >1, render every Kth pose (drops the rest). Useful when the
        // recording is at GUI fps but the GIF target is lower.
        uint32_t    stride = 1;
    };
    void setReplayOptions(const ReplayOptions& o) { m_replayOpts = o; m_replayEnabled = true; }

    // ── Capture mode: deterministic camera path + periodic screenshot dump ─
    // Two motion modes, selected by CaptureOptions::motion:
    //
    //   Dolly  (default) — camera slides forward along its starting view
    //     direction at `dollySpeed` units/sec. Best for stress-testing ReSTIR
    //     temporal reprojection: linear translation along the view axis lets
    //     most pixels keep their reservoir history through the geometric
    //     reuse gates.
    //   Orbit            — camera circles a centre at `orbitPeriodSeconds`
    //     per revolution. Stresses the gates much harder; more dramatic
    //     visible difference between modes but harsher conditions for
    //     ReSTIR. Use this when you specifically want to evaluate behaviour
    //     under rotation.
    //
    // Other knobs:
    //   warmupFrames     — frames to render BEFORE motion starts, so ReSTIR
    //                       temporal can build up a healthy reservoir at the
    //                       start pose.
    //   captureFrames    — total frames in the motion phase.
    //   captureStride    — save one PNG every K frames during the motion.
    //
    // Output layout:  <outDir>/<tag>_NNNNNN.png  +  <outDir>/<tag>_meta.json
    //
    // Setting m_captureEnabled = true (via setCaptureOptions) activates the
    // pipeline; the app exits after the last frame is saved.
    enum class CaptureMotion { Dolly, Orbit };
    struct CaptureOptions {
        std::string tag = "capture";
        std::string outDir = "screenshots";
        uint32_t warmupFrames = 60;
        uint32_t captureFrames = 600;
        uint32_t captureStride = 30;

        // At every capture point, hold the camera still for this many frames
        // before saving. Motion is paused for the dwell (camera receives dt=0)
        // so the path-tracer accumulator keeps integrating into the same
        // pose's image. With dwell=1 (default) the loop behaves as before:
        // capture every Kth motion frame at 1 spp. With dwell=2000 the
        // reference sweep produces a near-converged image at each capture
        // point, sharing the exact same camera path as the test sweeps so
        // per-frame-index comparison stays meaningful.
        uint32_t dwellFrames = 1;

        // Fixed virtual frame rate driving the camera path during the motion
        // phase. We override the real `dt` inside the capture loop so that
        // frame N maps to a deterministic camera pose regardless of how fast
        // each ReSTIR mode actually renders. Without this, fps differences
        // (native ~100 fps vs restir-pt ~30 fps) cause the same frame index
        // across modes to land at very different camera positions, making
        // per-frame-index image comparison meaningless.
        float fixedStepFps = 60.0f;

        // Which motion to run. Default is Dolly — empirically friendlier to
        // ReSTIR temporal reuse and what most users want when they say
        // "render the scene with the camera moving slowly".
        CaptureMotion motion = CaptureMotion::Dolly;

        // Dolly knobs.
        float dollySpeed = 0.3f;    // world units / second along start fwd

        // Orbit knobs (only used when motion == Orbit).
        float    orbitPeriodSeconds = 12.0f;
        float    orbitRadius = 0.0f;     // 0 → derived from initial camera distance to scene centre
        float    orbitPitchDeg = 15.0f;
        float3   orbitCenter = make_float3(0, 0, 0);
        bool     orbitCenterFromScene = true; // if true, use scene AABB centre
    };
    void setCaptureOptions(const CaptureOptions& o) { m_captureOpts = o; m_captureEnabled = true; }
    // Toggle ReSTIR DI / GI passes. Pre-init these are stored and applied
    // after Renderer::init in Application::init().
    void setReSTIREnabled(bool on)    { m_pendingReSTIRDI = on; }
    void setReSTIRGIEnabled(bool on)  { m_pendingReSTIRGI = on; }
    void setReSTIRPTEnabled(bool on)  { m_pendingReSTIRPT = on; }
    void setEnvMap(const std::string& path);
    void loadCameraFile(const std::string& path) { m_cameraFilePath = path; }

private:
    static void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    void processInput();
    void runGui();
    void runHeadless();
    void runReplay();
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
    VolumeMedium m_medium;
    bool m_hasMediumOverride = false;

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
    bool m_prevF5Down = false;

    // Camera-path recording. Toggled with F5. While active, every GUI frame
    // appends (timestamp, pose) to m_recordedPath. Stopping flushes the
    // buffer to recordings/path_*.json — replayed by scripts/render_camera_path.py.
    struct RecordedPose {
        float    t;            // seconds since record-start
        float3   position;
        float    yaw;          // degrees
        float    pitch;        // degrees
        float    fovDeg;
        float    aspect;
        float    nearPlane;
        float    farPlane;
    };
    bool                       m_recording = false;
    double                     m_recordStartTime = 0.0;
    std::vector<RecordedPose>  m_recordedPath;
    void startRecording();
    void stopRecording();
    uint32_t m_maxBounces = 8;
    uint32_t m_samplesPerFrame = 1;
    // Pending ReSTIR toggles applied to m_renderer right after init().
    bool m_pendingReSTIRDI = true;
    bool m_pendingReSTIRGI = false;
    bool m_pendingReSTIRPT = false;
    // Normal-map debug visualization. 0 = off; 1 = perturbed N; 2 = tangent
    // handedness; 3 = back-face-after-perturb flag. See DeviceSceneData.
    int m_debugNormalViz = 0;
    // Master switch for tangent-space normal maps. Off = interpolated vertex
    // normals only. Mapped to DeviceSceneData::enableNormalMap each frame.
    bool m_enableNormalMap = true;
    // Normal-arrow debug overlay. When on, a sparse grid of world-space
    // (position, perturbed N) pairs is captured by the kernel, read back, and
    // drawn over the path-traced image by ImGui.
    bool    m_showNormalArrows = false;
    int     m_normalArrowStride = 24;   // one arrow every 24x24 pixels
    float   m_normalArrowLength = 0.25f; // world-space arrow length
    float4* m_d_debugArrows  = nullptr;
    size_t  m_debugArrowCapacityPairs = 0;  // #(pos,N) pairs currently allocated
    std::vector<float4> m_h_debugArrows;    // host-side copy, 2*N entries
    int     m_debugArrowGridW = 0;
    int     m_debugArrowGridH = 0;
    int m_initialMode = -1;  // -1 = leave as default (Native)
    bool m_guiEnabled = true;
    double m_pendingScrollY = 0.0;
    std::string m_headlessOutputPath;
    std::string m_cameraFilePath;
    uint32_t m_targetSamples = 1;
    double m_headlessRenderMs = 0.0;
    double m_headlessTotalMs = 0.0;

    // Capture-mode state.
    bool   m_captureEnabled = false;
    CaptureOptions m_captureOpts;
    uint32_t m_captureFramesElapsed = 0;
    uint32_t m_captureFramesSaved   = 0;
    // Per-saved-frame timing (ms). Dumped to <outDir>/<tag>_meta.json.
    std::vector<double> m_captureFrameMs;
    std::vector<uint32_t> m_captureSavedIndices;
    // Cumulative wall-clock at capture start so meta.json can report mean fps.
    double m_captureStartTime = 0.0;
    // Dwell/motion state machine (post-warmup). At each capture point we
    // dwell for dwellFrames frames with motion paused (so the path-tracer
    // accumulator integrates more samples at the same pose), then advance
    // the camera by `captureStride` motion frames at fixedStepFps before
    // the next dwell. Saved-file index uses m_captureMotionFrames so all
    // sweeps that share warmup/captureFrames/captureStride/fixedStepFps land
    // on the same per-pose camera path regardless of dwell.
    uint32_t m_captureMotionFrames    = 0;  // motion frames issued since warmup
    uint32_t m_captureDwellRemaining  = 0;  // >0 means we're in a dwell phase
    uint32_t m_captureMotionRemaining = 0;  // >0 means we're advancing motion

    bool          m_replayEnabled = false;
    ReplayOptions m_replayOpts;
};
