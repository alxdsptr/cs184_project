#include "app/Application.h"
#include "scene/SceneLoader.h"
#include "scene/AnimationEval.h"
#include "gpu/PoseUpdate.h"
#include "util/Log.h"
#include "util/CudaCheck.h"
#ifdef PATHTRACER_OPTIX_ENABLED
#include "backend/OptiXBackend.h"
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <chrono>
#include <ctime>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <map>
#include <unordered_map>

#ifdef _WIN32
// For GetModuleFileNameW — used in init() to resolve the .exe directory so
// runtime assets (optix_programs.optixir) are found regardless of CWD.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

static float3 normalizeOrFallback(float3 v, float3 fallback) {
    float len = length(v);
    if (len <= 1e-6f) {
        return fallback;
    }
    return v / len;
}

static float computeVerticalFovRadians(const SceneCamera& camera, float aspect) {
    float horizontal = camera.horizontalFovRadians > 1e-6f
        ? camera.horizontalFovRadians
        : 60.0f * 3.14159265358979323846f / 180.0f;
    float safeAspect = aspect > 1e-6f ? aspect : 1.0f;
    return 2.0f * atanf(tanf(horizontal * 0.5f) / safeAspect);
}

static float computeFitDistance(const AABB& bounds, float3 forward, float3 up, float aspect, float verticalFovRadians) {
    if (bounds.empty()) {
        return 3.0f;
    }

    forward = normalizeOrFallback(forward, make_float3(0.0f, 0.0f, -1.0f));
    up = normalizeOrFallback(up, make_float3(0.0f, 1.0f, 0.0f));
    float3 right = normalizeOrFallback(cross(forward, up), make_float3(1.0f, 0.0f, 0.0f));
    up = normalizeOrFallback(cross(right, forward), make_float3(0.0f, 1.0f, 0.0f));

    float tanHalfV = tanf(verticalFovRadians * 0.5f);
    float tanHalfH = tanHalfV * (aspect > 1e-6f ? aspect : 1.0f);
    if (tanHalfV <= 1e-6f) {
        tanHalfV = 1e-6f;
    }
    if (tanHalfH <= 1e-6f) {
        tanHalfH = 1e-6f;
    }

    const float3 center = bounds.center();
    const float3 corners[8] = {
        make_float3(bounds.bmin.x, bounds.bmin.y, bounds.bmin.z),
        make_float3(bounds.bmax.x, bounds.bmin.y, bounds.bmin.z),
        make_float3(bounds.bmin.x, bounds.bmax.y, bounds.bmin.z),
        make_float3(bounds.bmax.x, bounds.bmax.y, bounds.bmin.z),
        make_float3(bounds.bmin.x, bounds.bmin.y, bounds.bmax.z),
        make_float3(bounds.bmax.x, bounds.bmin.y, bounds.bmax.z),
        make_float3(bounds.bmin.x, bounds.bmax.y, bounds.bmax.z),
        make_float3(bounds.bmax.x, bounds.bmax.y, bounds.bmax.z),
    };

    float requiredDistance = 0.0f;
    for (const float3& corner : corners) {
        float3 rel = corner - center;
        float x = dot(rel, right);
        float y = dot(rel, up);
        float z = dot(rel, forward);
        requiredDistance = fmaxf(requiredDistance, fabsf(x) / tanHalfH - z);
        requiredDistance = fmaxf(requiredDistance, fabsf(y) / tanHalfV - z);
    }

    float extentLength = length(bounds.bmax - bounds.bmin);
    float paddedDistance = fmaxf(requiredDistance, extentLength * 0.5f);
    return fmaxf(paddedDistance * 1.15f, 0.25f);
}

static void glfwErrorCb(int code, const char* msg) {
    LOG_ERROR("GLFW [%d]: %s", code, msg);
}

static std::string lowerString(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return value;
}

static float horizontalToVerticalFovDeg(float horizontalFovRadians, float aspect) {
    float safeAspect = aspect > 1e-6f ? aspect : 1.0f;
    float verticalRadians = 2.0f * atanf(tanf(horizontalFovRadians * 0.5f) / safeAspect);
    return verticalRadians * 180.0f / 3.14159265358979323846f;
}

void Application::glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);

    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) {
        return;
    }

    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }

    app->m_pendingScrollY += yoffset;
}

void Application::setMaxBounces(uint32_t maxBounces) {
    m_maxBounces = maxBounces;
    if (m_maxBounces < 1) {
        m_maxBounces = 1;
    }
}

void Application::setSamplesPerFrame(uint32_t spp) {
    m_samplesPerFrame = spp < 1 ? 1 : spp;
}

void Application::setHeadlessOutput(const std::string& outputPath, uint32_t sampleCount) {
    m_headlessOutputPath = outputPath;
    m_targetSamples = sampleCount < 1 ? 1 : sampleCount;
    m_headlessTotalMs = 0.0;
}

void Application::setEnvMap(const std::string& path) {
    loadEnvMap(path);
    if (m_envMapTex != 0) {
        m_enableEnvironment = true;
        // Copy path to GUI buffer for display
        size_t len = path.size() < sizeof(m_envMapPathBuf) - 1 ? path.size() : sizeof(m_envMapPathBuf) - 1;
        memcpy(m_envMapPathBuf, path.c_str(), len);
        m_envMapPathBuf[len] = '\0';
    }
}

bool Application::init(uint32_t width, uint32_t height, const std::string& title, bool enableGui) {
    m_width  = width;
    m_height = height;
    m_guiEnabled = enableGui;

    // GLFW
    glfwSetErrorCallback(glfwErrorCb);
    if (!glfwInit()) { LOG_ERROR("glfwInit failed"); return false; }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, m_guiEnabled ? GLFW_TRUE : GLFW_FALSE);

    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!m_window) { LOG_ERROR("glfwCreateWindow failed"); return false; }
    glfwSetWindowUserPointer(m_window, this);

    // CUDA device
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { LOG_ERROR("No CUDA devices"); return false; }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    LOG_INFO("CUDA device: %s (SM %d.%d, %zu MB)",
             prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024*1024));
    CUDA_CHECK(cudaSetDevice(0));

    // Initialize subsystems — CUDA first so Vulkan can match its device UUID
    m_display.setWindow(m_window);
    m_display.init(width, height);
    if (m_guiEnabled) {
        m_gui.init(m_window, &m_display);
    }
    glfwSetScrollCallback(m_window, glfwScrollCallback);
    m_renderer.init(width, height);
    // Apply ReSTIR toggles set via main.cpp before init()
    m_renderer.setReSTIREnabled(m_pendingReSTIRDI);
    m_renderer.setReSTIRGIEnabled(m_pendingReSTIRGI);
    m_renderer.setReSTIRPTEnabled(m_pendingReSTIRPT);

    bool backendReady = false;
#ifdef PATHTRACER_OPTIX_ENABLED
    if (m_backendKind == 1) {
        auto optix = std::make_unique<OptiXBackend>();
        // Resolve the actual .exe directory (not the CWD). This used to use
        // current_path(), which broke when the exe was launched with a full
        // path from a different working directory (e.g. our capture-mode
        // batch script that runs `pathtracer.exe` while CWD is the project
        // root so scene paths resolve naturally). VulkanDisplay already does
        // the GetModuleFileNameW dance for shaders/; mirror it here.
        std::filesystem::path exeDir;
#ifdef _WIN32
        wchar_t buf[MAX_PATH];
        DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
        if (n > 0) exeDir = std::filesystem::path(std::wstring(buf, n)).parent_path();
#endif
        if (exeDir.empty()) exeDir = std::filesystem::current_path();
        std::filesystem::path irPath = exeDir / "optix_programs.optixir";
        if (!std::filesystem::exists(irPath)) {
            // Fall back to CWD-relative — preserved for backwards compat with
            // older invocations that copy the .optixir into the project root.
            irPath = std::filesystem::path("optix_programs.optixir");
        }
        if (optix->init(irPath.string())) {
            m_backend = std::move(optix);
            backendReady = true;
            LOG_INFO("Application: using OptiX backend");
        } else {
            LOG_ERROR("Application: OptiX init failed, falling back to CUDA");
        }
    }
#else
    if (m_backendKind == 1) {
        LOG_ERROR("Application: --backend optix requested but built without PATHTRACER_ENABLE_OPTIX; falling back to CUDA");
    }
#endif
    if (!backendReady) {
        m_backend = std::make_unique<CUDABackend>();
    }

    m_camera.init(
        make_float3(0, 1, 3),
        make_float3(0, 0, 0),
        60.0f,
        (float)width / height
    );

    m_enableEnvironment = false;

    m_lastFrameTime = (float)glfwGetTime();
    LOG_INFO("Application initialized (%ux%u)", width, height);

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    if (m_initialMode >= 0) {
        Renderer::Mode rm = Renderer::Mode::Native;
        if (m_initialMode == 1) rm = Renderer::Mode::NRDOnly;
        else if (m_initialMode == 2) rm = Renderer::Mode::NRDDLSS;
        else if (m_initialMode == 3) rm = Renderer::Mode::DLSSOnly;
        else if (m_initialMode == 4) rm = Renderer::Mode::DLSSRR;
        LOG_INFO("Applying initial renderer mode: %d", m_initialMode);
        m_renderer.setMode(rm, &m_display);
    }
#endif
    return true;
}

bool Application::loadScene(const std::string& path) {
    // Release previous scene textures before loading a new scene.
    m_textures.freeAll();

    m_scene = Scene{};
    if (!SceneLoader::load(path, m_scene, m_sgMode, m_emissiveTargetLum)) {
        LOG_ERROR("Failed to load scene: %s", path.c_str());
        return false;
    }

    // Volumetric-medium override: if the user set any --medium-* flag on the
    // command line, prefer those values over what the scene file declared
    // (no loader parses media today, so this is the only path to enable a
    // medium). Otherwise inherit the scene's medium (which is "off" until
    // set elsewhere) so subsequent toggles in the GUI start from defaults.
    if (m_hasMediumOverride) {
        m_scene.getMedium() = m_medium;
    } else {
        m_medium = m_scene.getMedium();
    }
    // Default the medium's bounding box to the scene AABB (slightly padded
    // so geometry on the boundary stays inside the medium). Skipped if the
    // user already configured bounds. Without this, a global "constant"
    // medium has no extent and delta tracking would loop on miss rays.
    if (!m_medium.bounded) {
        const AABB& sceneBounds = m_scene.getBounds();
        if (!sceneBounds.empty()) {
            float3 size = sceneBounds.bmax - sceneBounds.bmin;
            float3 pad = size * 0.05f;
            m_medium.bmin = sceneBounds.bmin - pad;
            m_medium.bmax = sceneBounds.bmax + pad;
            m_medium.bounded = true;
            // Sensible defaults for the height-falloff modes — yBase at the
            // scene floor, falloff height ~ 30% of scene height. These only
            // matter when the user picks a heterogeneous density kind.
            m_medium.yBase = sceneBounds.bmin.y;
            m_medium.falloffHeight = fmaxf(size.y * 0.3f, 1.0f);
            m_medium.fbmFrequency = 1.0f / fmaxf(fmaxf(size.x, size.z), 1.0f) * 4.0f;
        }
    }
    m_medium.recomputeMajorant();
    m_scene.getMedium() = m_medium;

    // Load textures and bind CUDA texture objects per material.
    // Cache key = (path, sRGB) since the same file may appear as both a
    // colour texture (needs sRGB decode) and a data texture (must stay
    // linear), and they need separate CUDA texture objects.
    std::map<std::pair<std::string, bool>, cudaTextureObject_t> textureCache;
    auto loadCachedTexture = [&](const std::string& texPath, bool sRGB) -> cudaTextureObject_t {
        if (texPath.empty()) {
            return 0;
        }
        auto key = std::make_pair(texPath, sRGB);
        auto it = textureCache.find(key);
        if (it != textureCache.end()) {
            return it->second;
        }
        cudaTextureObject_t obj = m_textures.loadTexture(texPath, sRGB);
        textureCache.emplace(key, obj);
        return obj;
    };

    auto& materials = m_scene.getMaterials();
    for (auto& mat : materials) {
        mat.albedoTexObj        = loadCachedTexture(mat.albedoTexPath,        /*sRGB=*/true);
        mat.normalTexObj        = loadCachedTexture(mat.normalTexPath,        /*sRGB=*/false);
        mat.metallicRoughTexObj = loadCachedTexture(mat.metallicRoughTexPath, /*sRGB=*/false);
        mat.emissiveTexObj      = loadCachedTexture(mat.emissiveTexPath,      /*sRGB=*/true);
        // Specular-gloss texture: legacy FBX (.dds) authors F0 in linear, not
        // sRGB. Loading as sRGB pushes slightly chromatic dielectric F0s into
        // saturated colours after Fresnel boost (the "purple pipes" symptom).
        // Alpha is glossiness, also linear — so a single linear binding is
        // correct for both channels.
        // Skip the upload entirely when the material didn't opt into the
        // Specular-Glossiness workflow (sgMode Off, or sgMode on but the
        // material failed the detection in SceneLoader). The kernel only
        // reads specularGlossTex when useSpecularGlossiness is true, so
        // loading it in any other case is dead weight — which matters on
        // asset-heavy scenes like Bistro that push the 8 GB envelope.
        if (mat.useSpecularGlossiness) {
            mat.specularGlossTexObj = loadCachedTexture(mat.specularGlossTexPath, /*sRGB=*/false);
        } else {
            mat.specularGlossTexObj = 0;
        }
    }

    // Back-fill emissive texture handles on area lights so NEE can fetch
    // per-texel emission for textured emitters.
    for (auto& areaLight : m_scene.getAreaLights()) {
        if (areaLight.materialIndex >= 0 &&
            (size_t)areaLight.materialIndex < materials.size()) {
            areaLight.emissiveTexObj = materials[areaLight.materialIndex].emissiveTexObj;
        }
    }

    // Build acceleration structure (uploads geometry + builds BVH)
    m_backend->buildAccelerationStructure(m_scene);
    m_sceneLoaded = true;
    m_renderer.resetAccumulation();
    // Scene swap genuinely invalidates ReSTIR reservoirs (the lights they
    // reference might no longer exist). Camera motion does NOT — that's
    // why resetAccumulation() no longer drops history on its own.
    m_renderer.invalidateReSTIRHistory();
    // A scene swap is also a true DLSS/NRD pipeline transition: prior
    // history references geometry that no longer exists.
    m_renderer.markPipelineNeedsReset();

    std::string ext = lowerString(std::filesystem::path(path).extension().string());
    const SceneCamera& sceneCamera = m_scene.getCamera();
    float aspect = m_height > 0 ? (float)m_width / (float)m_height : 1.0f;

    if (ext == ".dae") {
        frameCameraToScene();
    } else if (sceneCamera.valid) {
        float cameraAspect = sceneCamera.aspect > 1e-6f ? sceneCamera.aspect : aspect;
        float fovDeg = horizontalToVerticalFovDeg(sceneCamera.horizontalFovRadians, cameraAspect);
        m_camera.init(sceneCamera.position, sceneCamera.forward, sceneCamera.up, fovDeg, cameraAspect);
        m_camera.setClipPlanes(sceneCamera.nearPlane, sceneCamera.farPlane);
    }

    if (!m_cameraFilePath.empty()) {
        if (m_camera.loadFromFile(m_cameraFilePath)) {
            LOG_INFO("Loaded camera from: %s", m_cameraFilePath.c_str());
            // Camera teleport — DLSS/NRD reprojection has no valid history
            // for the new viewpoint. Tell them to RESTART on the next
            // pre-present.
            m_renderer.markPipelineNeedsReset();
        }
    }

    m_renderer.resetAccumulation();

    // ── Capture-mode: configure deterministic motion at scene load ────────
    // We DON'T enable auto-motion yet — leave the camera static for the
    // warmup phase so ReSTIR temporal can build a healthy reservoir at the
    // start pose. Motion kicks in after warmupFrames inside the run loop.
    if (m_captureEnabled) {
        const char* motionName =
            (m_captureOpts.motion == CaptureMotion::Dolly) ? "dolly" : "orbit";
        if (m_captureOpts.motion == CaptureMotion::Dolly) {
            LOG_INFO("Capture mode: tag='%s' motion=dolly speed=%.3f u/s "
                     "warmup=%u capture=%u stride=%u",
                     m_captureOpts.tag.c_str(),
                     m_captureOpts.dollySpeed,
                     m_captureOpts.warmupFrames,
                     m_captureOpts.captureFrames,
                     m_captureOpts.captureStride);
        } else {
            const AABB& bounds = m_scene.getBounds();
            float3 center = m_captureOpts.orbitCenterFromScene
                ? (bounds.empty() ? make_float3(0, 0, 0) : bounds.center())
                : m_captureOpts.orbitCenter;
            float radius = m_captureOpts.orbitRadius;
            if (radius <= 0.0f) {
                float3 d = m_camera.getPosition() - center;
                radius = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
                if (radius < 0.5f) radius = 0.5f;
            }
            LOG_INFO("Capture mode: tag='%s' motion=orbit center=(%.3f,%.3f,%.3f) "
                     "radius=%.3f period=%.2fs warmup=%u capture=%u stride=%u",
                     m_captureOpts.tag.c_str(),
                     center.x, center.y, center.z, radius,
                     m_captureOpts.orbitPeriodSeconds,
                     m_captureOpts.warmupFrames,
                     m_captureOpts.captureFrames,
                     m_captureOpts.captureStride);
        }
        (void)motionName;
    }

    return true;
}

void Application::frameCameraToScene() {
    const SceneCamera& sceneCamera = m_scene.getCamera();
    const AABB& bounds = m_scene.getBounds();
    float aspect = m_height > 0 ? (float)m_width / (float)m_height : 1.0f;

    float3 forward = sceneCamera.valid ? sceneCamera.forward : make_float3(0.0f, 0.0f, -1.0f);
    float3 up = sceneCamera.valid ? sceneCamera.up : make_float3(0.0f, 1.0f, 0.0f);
    float verticalFov = computeVerticalFovRadians(sceneCamera, aspect);

    float3 target = bounds.empty() ? make_float3(0.0f, 0.0f, 0.0f) : bounds.center();
    if (sceneCamera.valid) {
        float3 toTarget = normalizeOrFallback(target - sceneCamera.position, forward);
        if (dot(forward, toTarget) < 0.0f) {
            forward = -forward;
        }
    }
    float distance = computeFitDistance(bounds, forward, up, aspect, verticalFov);
    float3 position = target - normalizeOrFallback(forward, make_float3(0.0f, 0.0f, -1.0f)) * distance;

    m_camera.init(position, target, verticalFov * 180.0f / 3.14159265358979323846f, aspect);
}

void Application::processInput() {
    m_input = InputState{};

    if (!m_guiEnabled) {
        return;
    }

    bool speedDownKey = glfwGetKey(m_window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS;
    bool speedUpKey = glfwGetKey(m_window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS;

    if (!m_gui.wantCaptureKeyboard()) {
        m_input.forward  = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
        m_input.backward = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
        m_input.left     = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
        m_input.right    = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
        m_input.up       = glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS;
        m_input.down     = glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;

        if (speedDownKey && !m_prevSpeedDownKey) {
            m_camera.setMoveSpeed(m_camera.getMoveSpeed() * 0.9f);
        }
        if (speedUpKey && !m_prevSpeedUpKey) {
            m_camera.setMoveSpeed(m_camera.getMoveSpeed() * 1.1f);
        }
    }

    m_prevSpeedDownKey = speedDownKey;
    m_prevSpeedUpKey = speedUpKey;

    if (!m_gui.wantCaptureMouse()) {
        m_input.mouseHeld = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        double mx, my;
        glfwGetCursorPos(m_window, &mx, &my);
        if (m_firstMouse) {
            m_lastMouseX = mx;
            m_lastMouseY = my;
            m_firstMouse = false;
        }
        m_input.mouseDx = (float)(mx - m_lastMouseX);
        float rawDy = (float)(my - m_lastMouseY);
        m_input.mouseDy = m_invertMouseY ? -rawDy : rawDy;
        m_lastMouseX = mx;
        m_lastMouseY = my;
    }
}

void Application::freeShEnvDevice() {
    if (m_d_shEnvCoeffs) {
        cudaFree(m_d_shEnvCoeffs);
        m_d_shEnvCoeffs = nullptr;
    }
}

void Application::loadEnvMap(const std::string& path) {
    if (path.empty()) return;
    int w = 0, h = 0;
    cudaTextureObject_t tex = m_textures.loadHDRTexture(path, w, h);
    if (tex != 0) {
        m_envMapTex = tex;
        LOG_INFO("Environment map loaded: %s (%dx%d)", path.c_str(), w, h);

        // Precompute L2 SH diffuse irradiance coefficients for the new env map.
        // Cheap (one-time CPU pass); drops the noisy env-diffuse sampling path
        // on indirect bounces to a deterministic 9-coeff lookup.
        float sh[9][3];
        if (TextureManager::projectEnvToSH(path, sh)) {
            float3 host[9];
            for (int i = 0; i < 9; i++) {
                host[i] = make_float3(sh[i][0], sh[i][1], sh[i][2]);
            }
            freeShEnvDevice();
            CUDA_CHECK(cudaMalloc(&m_d_shEnvCoeffs, sizeof(float3) * 9));
            CUDA_CHECK(cudaMemcpy(m_d_shEnvCoeffs, host,
                                  sizeof(float3) * 9, cudaMemcpyHostToDevice));
        } else {
            freeShEnvDevice();
        }
    } else {
        LOG_ERROR("Failed to load environment map: %s", path.c_str());
    }
}

void Application::renderSceneSample(uchar4* d_pbo, bool timeHeadless) {
    if (m_sceneLoaded) {
        // Advance animation only on the first sample of each pose. The
        // accumulator's sample-count resets to 0 on camera move (replay loop,
        // capture motion-frame, GUI input). When it's >0 we're integrating
        // more samples at the same camera pose AND same animation time, so
        // touching the geometry would re-pose mid-accumulation and produce
        // ghosted frames. Also skip on GUI mode where the user is exploring;
        // they call --play-anim explicitly when they want playback.
        if (m_playAnimation && m_scene.hasAnimation() &&
            m_renderer.getSampleCount() == 0) {
            advanceAnimation(1.0f / std::max(1.0f, m_animFps));
        }

        CameraParams camParams = m_camera.getParams(m_frameIndex);
        DeviceSceneData sceneData = m_backend->getSceneData();
        sceneData.envMapTex = m_envMapTex;
        // Refresh medium each frame — the host m_medium can be tweaked at
        // runtime (CLI override at load, future GUI), and the backend's
        // cached copy is stale otherwise.
        sceneData.medium = m_medium;
        sceneData.d_shEnvCoeffs = m_d_shEnvCoeffs;
        sceneData.envUseSH = (m_useSHEnvIrradiance && m_d_shEnvCoeffs) ? 1 : 0;
        sceneData.debugNormalViz = m_debugNormalViz;
        sceneData.enableNormalMap = m_enableNormalMap ? 1 : 0;

        // Debug normal arrows: resize device buffer to match the current
        // window + stride, pre-clear validity flags, and hand the pointer to
        // the kernel. After the render we copy the samples back so the GUI
        // overlay can draw them.
        if (m_showNormalArrows && m_normalArrowStride > 0) {
            int stride = m_normalArrowStride;
            int gridW = (int)((m_width  + stride - 1) / stride);
            int gridH = (int)((m_height + stride - 1) / stride);
            size_t neededPairs = (size_t)gridW * (size_t)gridH;
            if (neededPairs > m_debugArrowCapacityPairs) {
                if (m_d_debugArrows) cudaFree(m_d_debugArrows);
                m_d_debugArrows = nullptr;
                CUDA_CHECK(cudaMalloc(&m_d_debugArrows,
                                      sizeof(float4) * 2 * neededPairs));
                m_debugArrowCapacityPairs = neededPairs;
            }
            // Zero-fill so stale "valid=1" from previous frames don't linger
            // in cells whose primary ray missed this frame.
            CUDA_CHECK(cudaMemset(m_d_debugArrows, 0,
                                  sizeof(float4) * 2 * neededPairs));
            m_debugArrowGridW = gridW;
            m_debugArrowGridH = gridH;
            sceneData.d_debugArrows    = m_d_debugArrows;
            sceneData.debugArrowStride = stride;
            sceneData.debugArrowWidth  = gridW;
            sceneData.debugArrowHeight = gridH;
        } else {
            sceneData.d_debugArrows    = nullptr;
            sceneData.debugArrowStride = 0;
            m_debugArrowGridW = 0;
            m_debugArrowGridH = 0;
        }

        m_renderer.renderFrame(camParams, sceneData, m_backend.get(), d_pbo,
                               m_enableEnvironment, m_maxBounces,
                               m_samplesPerFrame,
                               &m_display, m_frameIndex,
                               m_camera.hasMoved());

        // Read back the sparse arrow samples for the GUI overlay to draw.
        if (m_showNormalArrows && m_d_debugArrows && m_debugArrowGridW > 0) {
            size_t nPairs = (size_t)m_debugArrowGridW * (size_t)m_debugArrowGridH;
            m_h_debugArrows.resize(2 * nPairs);
            CUDA_CHECK(cudaMemcpy(m_h_debugArrows.data(), m_d_debugArrows,
                                  sizeof(float4) * 2 * nPairs,
                                  cudaMemcpyDeviceToHost));
        } else if (!m_h_debugArrows.empty()) {
            m_h_debugArrows.clear();
        }
    } else {
        CUDA_CHECK(cudaMemset(d_pbo, 40, m_width * m_height * sizeof(uchar4)));
    }
}

void Application::runGui() {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();

        float now = (float)glfwGetTime();
        float dt  = now - m_lastFrameTime;
        m_lastFrameTime = now;

        m_fpsFrames++;
        m_fpsTimer += dt;
        if (m_fpsTimer >= 0.5f) {
            m_fps = (float)m_fpsFrames / m_fpsTimer;
            m_fpsFrames = 0;
            m_fpsTimer  = 0.0f;
        }

        int fbW, fbH;
        glfwGetFramebufferSize(m_window, &fbW, &fbH);
        if (fbW > 0 && fbH > 0 && ((uint32_t)fbW != m_width || (uint32_t)fbH != m_height)) {
            m_width  = (uint32_t)fbW;
            m_height = (uint32_t)fbH;
            m_display.resize(m_width, m_height);
            m_renderer.resize(m_width, m_height);
            m_camera.setAspect((float)m_width / m_height);
            m_renderer.resetAccumulation();
        }

        processInput();
        // In capture mode, once we're in the motion phase, ignore the real
        // wall-clock dt and feed the camera a fixed virtual time step. This
        // keeps the camera path purely a function of frame index, so the
        // same frame N across different ReSTIR modes (each rendering at a
        // different real fps) lands at exactly the same camera pose.
        // Without this, frame N is at a different position in each mode and
        // per-frame-index image comparison is meaningless.
        //
        // While we're dwelling at a capture point (dwellRemaining > 0), feed
        // dt=0 so the auto-motion clock doesn't advance — the path-tracer
        // accumulator can keep integrating at the same pose. Motion only
        // advances during the explicit "advance to next capture point" phase.
        float updateDt = dt;
        if (m_captureEnabled && m_sceneLoaded && m_camera.isAutoMoving()) {
            float fps = m_captureOpts.fixedStepFps > 0.0f
                ? m_captureOpts.fixedStepFps : 60.0f;
            updateDt = (m_captureDwellRemaining > 0) ? 0.0f : (1.0f / fps);
        }
        m_camera.update(updateDt, m_input);

        if (fabs(m_pendingScrollY) > 1e-6) {
            float zoomFactor = powf(0.88f, (float)m_pendingScrollY);
            zoomFactor = clampf(zoomFactor, 0.25f, 4.0f);
            float newFov = m_camera.getFovDeg() * zoomFactor;
            m_camera.setFovDeg(newFov);
            m_pendingScrollY = 0.0;
            m_renderer.resetAccumulation();
        }

        if (m_camera.hasMoved()) {
            m_renderer.resetAccumulation();
        }

        if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(m_window, true);
        }

        bool f12Down = glfwGetKey(m_window, GLFW_KEY_F12) == GLFW_PRESS;
        bool saveScreenshot = f12Down && !m_prevF12Down;
        m_prevF12Down = f12Down;

        // F1 toggles the on-screen GUI overlay.
        bool f1Down = glfwGetKey(m_window, GLFW_KEY_F1) == GLFW_PRESS;
        if (f1Down && !m_prevF1Down) {
            m_showGui = !m_showGui;
        }
        m_prevF1Down = f1Down;

        // F2 freezes the current camera basis as the movement frame; WASD/
        // space/shift then translate along the saved axes regardless of where
        // the camera is looking. F3 releases it back to free-fly.
        bool f2Down = glfwGetKey(m_window, GLFW_KEY_F2) == GLFW_PRESS;
        if (f2Down && !m_prevF2Down) {
            m_camera.lockMovementFrame();
            LOG_INFO("Movement frame: LOCKED");
        }
        m_prevF2Down = f2Down;

        bool f3Down = glfwGetKey(m_window, GLFW_KEY_F3) == GLFW_PRESS;
        if (f3Down && !m_prevF3Down) {
            m_camera.unlockMovementFrame();
            LOG_INFO("Movement frame: FREE");
        }
        m_prevF3Down = f3Down;

        // F4 exports the current camera state to a file that can later be
        // passed back via --camera on the next run.
        bool f4Down = glfwGetKey(m_window, GLFW_KEY_F4) == GLFW_PRESS;
        if (f4Down && !m_prevF4Down) {
            std::filesystem::create_directories("cameras");
            std::ostringstream name;
            name << "cameras/camera_" << std::setw(6) << std::setfill('0') << m_frameIndex << ".txt";
            if (m_camera.saveToFile(name.str())) {
                LOG_INFO("Saved camera: %s", name.str().c_str());
            } else {
                LOG_ERROR("Failed to save camera: %s", name.str().c_str());
            }
        }
        m_prevF4Down = f4Down;

        // F5 toggles camera-path recording. When ON, the renderer appends
        // (timestamp, pose) every GUI frame to m_recordedPath. Stopping flushes
        // the buffer to recordings/path_*.json — replayed by
        // scripts/render_camera_path.py via --recording.
        bool f5Down = glfwGetKey(m_window, GLFW_KEY_F5) == GLFW_PRESS;
        if (f5Down && !m_prevF5Down) {
            if (m_recording) stopRecording();
            else             startRecording();
        }
        m_prevF5Down = f5Down;

        // 'H' toggles SH environment irradiance shortcut (only takes effect
        // when an env map is loaded and its SH has been precomputed).
        bool shKeyDown = glfwGetKey(m_window, GLFW_KEY_H) == GLFW_PRESS;
        if (shKeyDown && !m_prevSHKeyDown && !m_gui.wantCaptureKeyboard()) {
            m_useSHEnvIrradiance = !m_useSHEnvIrradiance;
            m_renderer.resetAccumulation();
            LOG_INFO("SH env irradiance shortcut: %s",
                     m_useSHEnvIrradiance ? "ON" : "OFF");
        }
        m_prevSHKeyDown = shKeyDown;

        uchar4* d_pbo = (uchar4*)m_display.mapForCUDA();
        renderSceneSample(d_pbo, false);
        m_display.unmapFromCUDA();

        if (saveScreenshot && m_sceneLoaded) {
            std::filesystem::create_directories("screenshots");
            std::ostringstream name;
            name << "screenshots/frame_" << std::setw(6) << std::setfill('0') << m_frameIndex << ".png";
            if (m_display.saveToPNG(name.str())) {
                LOG_INFO("Saved screenshot: %s", name.str().c_str());
            } else {
                LOG_ERROR("Failed to save screenshot: %s", name.str().c_str());
            }
        }

        // ── Capture-mode driver ──────────────────────────────────────────
        // Phase 1 (warmup): camera static, ReSTIR temporal hot-up.
        // Phase 2 (capture): alternate dwell (motion paused, accumulate) and
        //                    advance (motion runs forward by `stride` frames).
        //                    On the final frame of each dwell we save a PNG
        //                    indexed by motion-frame count, so all sweeps
        //                    share the same per-pose camera path.
        // Phase 3: write meta.json and quit.
        if (m_captureEnabled && m_sceneLoaded) {
            uint32_t capIdx = m_captureFramesElapsed;
            uint32_t dwellFrames = m_captureOpts.dwellFrames > 0
                ? m_captureOpts.dwellFrames : 1;
            if (capIdx == m_captureOpts.warmupFrames) {
                // Transition into capture phase: enable the configured motion
                // and queue up the first dwell at motion frame 0.
                if (m_captureOpts.motion == CaptureMotion::Dolly) {
                    m_camera.setAutoDolly(m_captureOpts.dollySpeed);
                    LOG_INFO("Capture: warmup done, dolly speed=%.3f for %u motion frames "
                             "(dwell=%u, stride=%u)",
                             m_captureOpts.dollySpeed,
                             m_captureOpts.captureFrames,
                             dwellFrames,
                             m_captureOpts.captureStride);
                } else {
                    const AABB& bounds = m_scene.getBounds();
                    float3 center = m_captureOpts.orbitCenterFromScene
                        ? (bounds.empty() ? make_float3(0, 0, 0) : bounds.center())
                        : m_captureOpts.orbitCenter;
                    float radius = m_captureOpts.orbitRadius;
                    if (radius <= 0.0f) {
                        float3 d = m_camera.getPosition() - center;
                        radius = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
                        if (radius < 0.5f) radius = 0.5f;
                    }
                    m_camera.setAutoOrbit(center, radius,
                                          m_captureOpts.orbitPeriodSeconds,
                                          m_captureOpts.orbitPitchDeg);
                    LOG_INFO("Capture: warmup done, orbit for %u motion frames "
                             "(dwell=%u, stride=%u)",
                             m_captureOpts.captureFrames,
                             dwellFrames,
                             m_captureOpts.captureStride);
                }
                m_captureStartTime = glfwGetTime();
                m_captureMotionFrames    = 0;
                m_captureDwellRemaining  = dwellFrames;
                m_captureMotionRemaining = 0;
            }
            if (capIdx >= m_captureOpts.warmupFrames) {
                m_captureFrameMs.push_back((double)dt * 1000.0);

                if (m_captureDwellRemaining > 0) {
                    // Final dwell frame at this capture point: save then queue
                    // up the next advance (or finish if we'd exceed the budget).
                    if (m_captureDwellRemaining == 1) {
                        std::filesystem::create_directories(m_captureOpts.outDir);
                        std::ostringstream name;
                        name << m_captureOpts.outDir << "/" << m_captureOpts.tag << "_"
                             << std::setw(6) << std::setfill('0')
                             << m_captureMotionFrames << ".png";
                        if (m_display.saveToPNG(name.str())) {
                            m_captureSavedIndices.push_back(m_captureMotionFrames);
                            m_captureFramesSaved++;
                        } else {
                            LOG_ERROR("Capture: failed to save %s", name.str().c_str());
                        }
                        // Decide whether another capture point fits.
                        uint32_t nextMotionIdx = m_captureMotionFrames
                                                + m_captureOpts.captureStride;
                        if (nextMotionIdx >= m_captureOpts.captureFrames) {
                            m_captureMotionRemaining = 0;
                            // Dump meta.json and quit.
                            double totalSec = glfwGetTime() - m_captureStartTime;
                            double meanFps = totalSec > 0.0
                                ? (double)m_captureFramesElapsed / totalSec
                                : 0.0;
                            std::filesystem::create_directories(m_captureOpts.outDir);
                            std::ostringstream meta;
                            meta << m_captureOpts.outDir << "/" << m_captureOpts.tag << "_meta.json";
                            std::ofstream mf(meta.str());
                            if (mf.is_open()) {
                                mf << "{\n";
                                mf << "  \"tag\": \"" << m_captureOpts.tag << "\",\n";
                                mf << "  \"warmup_frames\": " << m_captureOpts.warmupFrames << ",\n";
                                mf << "  \"capture_frames\": " << m_captureOpts.captureFrames << ",\n";
                                mf << "  \"capture_stride\": " << m_captureOpts.captureStride << ",\n";
                                mf << "  \"dwell_frames\": " << dwellFrames << ",\n";
                                mf << "  \"orbit_period_s\": " << m_captureOpts.orbitPeriodSeconds << ",\n";
                                mf << "  \"saved_count\": " << m_captureFramesSaved << ",\n";
                                mf << "  \"total_seconds\": " << totalSec << ",\n";
                                mf << "  \"mean_fps\": " << meanFps << ",\n";
                                mf << "  \"width\": " << m_width << ",\n";
                                mf << "  \"height\": " << m_height << ",\n";
                                mf << "  \"frame_ms\": [";
                                for (size_t i = 0; i < m_captureFrameMs.size(); i++) {
                                    if (i) mf << ", ";
                                    mf << m_captureFrameMs[i];
                                }
                                mf << "],\n";
                                mf << "  \"saved_indices\": [";
                                for (size_t i = 0; i < m_captureSavedIndices.size(); i++) {
                                    if (i) mf << ", ";
                                    mf << m_captureSavedIndices[i];
                                }
                                mf << "]\n";
                                mf << "}\n";
                                LOG_INFO("Capture: wrote %s (%u images, mean %.1f fps over %.2fs)",
                                         meta.str().c_str(), m_captureFramesSaved, meanFps, totalSec);
                            }
                            glfwSetWindowShouldClose(m_window, true);
                        } else {
                            m_captureMotionRemaining = m_captureOpts.captureStride;
                        }
                    }
                    m_captureDwellRemaining--;
                } else if (m_captureMotionRemaining > 0) {
                    // Motion advance frame: camera dt was 1/fps, so the camera
                    // pose moved one virtual step toward the next capture point.
                    m_captureMotionFrames++;
                    m_captureMotionRemaining--;
                    if (m_captureMotionRemaining == 0) {
                        m_captureDwellRemaining = dwellFrames;
                    }
                }
            }
            m_captureFramesElapsed++;
        }

        m_gui.beginFrame();
        bool envMapLoadRequested = false;
        float moveSpeed = m_camera.getMoveSpeed();
        float exposure = m_renderer.getExposure();
        int toneMappingMode = (int)m_renderer.getToneMappingMode();

#ifdef PATHTRACER_NRD_DLSS_ENABLED
        int guiMode = (int)m_renderer.getMode();
        int guiDlssQ = (int)m_renderer.getDLSSQuality();
        uint32_t rrW = m_renderer.getRenderWidth();
        uint32_t rrH = m_renderer.getRenderHeight();
        int* modePtr = &guiMode;
        int* qualityPtr = &guiDlssQ;
#else
        int* modePtr = nullptr;
        int* qualityPtr = nullptr;
        uint32_t rrW = 0, rrH = 0;
#endif

        bool envChanged = false;
        // Snapshot ReSTIR state into local bools so the GUI can mutate them;
        // we apply the result via the renderer's setters after render().
        bool guiReSTIRDI = m_renderer.isReSTIREnabled();
        bool guiReSTIRGI = m_renderer.isReSTIRGIEnabled();
        bool guiReSTIRPT = m_renderer.isReSTIRPTEnabled();
        int  guiReSTIRPTLen = (int)m_renderer.restirPT().pathLength();
        if (m_showGui) {
            envChanged = m_gui.render(
                m_fps,
                m_renderer.getSampleCount(),
                m_width,
                m_height,
                m_enableEnvironment,
                m_invertMouseY,
                m_maxBounces,
                exposure,
                toneMappingMode,
                moveSpeed,
                m_envMapPathBuf,
                sizeof(m_envMapPathBuf),
                envMapLoadRequested,
                modePtr, qualityPtr, rrW, rrH,
                &m_debugNormalViz,
                &m_enableNormalMap,
                &m_showNormalArrows,
                &m_normalArrowStride,
                &m_normalArrowLength,
                &guiReSTIRDI,
                &guiReSTIRGI,
                &guiReSTIRPT,
                &guiReSTIRPTLen);
            if (guiReSTIRDI != m_renderer.isReSTIREnabled()) {
                m_renderer.setReSTIREnabled(guiReSTIRDI);
            }
            if (guiReSTIRGI != m_renderer.isReSTIRGIEnabled()) {
                m_renderer.setReSTIRGIEnabled(guiReSTIRGI);
            }
            if (guiReSTIRPT != m_renderer.isReSTIRPTEnabled()) {
                m_renderer.setReSTIRPTEnabled(guiReSTIRPT);
            }
            if (guiReSTIRPTLen >= 0 &&
                (uint32_t)guiReSTIRPTLen != m_renderer.restirPT().pathLength()) {
                m_renderer.restirPT().setPathLength((uint32_t)guiReSTIRPTLen);
                m_renderer.restirPT().invalidateHistory();
            }

            // Arrow overlay renders under/over the same ImGui frame.
            if (m_showNormalArrows && !m_h_debugArrows.empty() &&
                m_debugArrowGridW > 0 && m_debugArrowGridH > 0)
            {
                CameraParams camForOverlay = m_camera.getParams(m_frameIndex);
                m_gui.drawNormalArrowsOverlay(
                    m_h_debugArrows.data(),
                    m_debugArrowGridW, m_debugArrowGridH,
                    camForOverlay, m_width, m_height,
                    m_normalArrowLength);
            }
        }

#ifdef PATHTRACER_NRD_DLSS_ENABLED
        if (modePtr && *modePtr != (int)m_renderer.getMode()) {
            m_renderer.setMode((Renderer::Mode)(*modePtr), &m_display);
        }
        if (qualityPtr && *qualityPtr != (int)m_renderer.getDLSSQuality()) {
            m_renderer.setDLSSQuality((Renderer::DLSSQuality)(*qualityPtr));
        }
#endif

        m_camera.setMoveSpeed(moveSpeed);
        m_renderer.setExposure(exposure);
        if (toneMappingMode < (int)ToneMappingMode::None) {
            toneMappingMode = (int)ToneMappingMode::None;
        }
        if (toneMappingMode > (int)ToneMappingMode::ACES) {
            toneMappingMode = (int)ToneMappingMode::ACES;
        }
        m_renderer.setToneMappingMode((ToneMappingMode)toneMappingMode);
        if (envMapLoadRequested) {
            loadEnvMap(std::string(m_envMapPathBuf));
            m_renderer.resetAccumulation();
            // Lighting changed → previous Lo/Le baked into reservoirs is
            // wrong. Drop history.
            m_renderer.invalidateReSTIRHistory();
        }
        if (envChanged) {
            m_renderer.resetAccumulation();
            m_renderer.invalidateReSTIRHistory();
        }
        m_gui.endFrame();

        m_display.present();
        if (m_recording) {
            RecordedPose rp;
            rp.t         = (float)(glfwGetTime() - m_recordStartTime);
            rp.position  = m_camera.getPosition();
            rp.yaw       = m_camera.getYawDeg();
            rp.pitch     = m_camera.getPitchDeg();
            rp.fovDeg    = m_camera.getFovDeg();
            rp.aspect    = m_camera.getAspect();
            rp.nearPlane = m_camera.getNearPlane();
            rp.farPlane  = m_camera.getFarPlane();
            m_recordedPath.push_back(rp);
        }
        // Snapshot current view/proj as "prev" for next frame's motion
        // vectors. Must come AFTER all getParams() calls of this frame
        // (renderer + any GUI overlay), otherwise the next frame's
        // reprojection would compare against this-frame matrices and
        // motion vectors would collapse to zero.
        m_camera.advanceFrame();
        m_frameIndex++;
    }
}

void Application::runHeadless() {
    const auto totalStart = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(m_window)) {
        uchar4* d_pbo = (uchar4*)m_display.mapForCUDA();
        renderSceneSample(d_pbo, true);
        m_display.unmapFromCUDA();
        // saveToPNG reads m_sampledImage, which is only populated by the
        // present-time composite path (NRD denoise + tonemap + blit). Without
        // present() the headless screenshot is uninitialised RGBA(0,0,0,0).
        m_display.present();

        if (m_sceneLoaded && m_renderer.getSampleCount() >= m_targetSamples) {
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto totalEnd = std::chrono::steady_clock::now();
            m_headlessTotalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
            LOG_INFO("Target samples reached: %u", m_renderer.getSampleCount());
            LOG_INFO("Headless total elapsed time: %.3f ms", m_headlessTotalMs);
            if (m_display.saveToPNG(m_headlessOutputPath)) {
                LOG_INFO("Saved image: %s", m_headlessOutputPath.c_str());
            } else {
                LOG_ERROR("Failed to save image: %s", m_headlessOutputPath.c_str());
            }
            glfwSetWindowShouldClose(m_window, true);
        }

        m_camera.advanceFrame();
        m_frameIndex++;
    }
}

// ─────────────────────────────────────────────────────────────
// Replay-mode JSON parsing.
//
// Recordings are produced by Application::stopRecording. Format is fixed and
// we control both writer and reader, so a tiny hand-rolled reader is enough —
// avoids pulling in a JSON library. We strip whitespace, then for every
// "{...}" object inside the "poses" array we extract the seven numeric fields
// by scanning for `"key":` and parsing the next number / number-list.
// ─────────────────────────────────────────────────────────────
namespace {
struct ReplayPose {
    float    t;
    float3   position;
    float    yawDeg;
    float    pitchDeg;
    float    fovDeg;
    float    aspect;
    float    nearPlane;
    float    farPlane;
};

bool readNumber(const std::string& s, size_t& i, float& out) {
    while (i < s.size() && (s[i] == ' ' || s[i] == ',' || s[i] == ':' || s[i] == '[' || s[i] == ']')) ++i;
    size_t start = i;
    while (i < s.size() && (isdigit((unsigned char)s[i]) || s[i] == '.' ||
                            s[i] == '-' || s[i] == '+' || s[i] == 'e' || s[i] == 'E')) ++i;
    if (i == start) return false;
    out = std::stof(s.substr(start, i - start));
    return true;
}

bool readKeyedFloat(const std::string& s, const char* key, float& out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t k = s.find(needle);
    if (k == std::string::npos) return false;
    k += needle.size();
    return readNumber(s, k, out);
}

bool readKeyedFloat3(const std::string& s, const char* key, float3& out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t k = s.find(needle);
    if (k == std::string::npos) return false;
    k += needle.size();
    return readNumber(s, k, out.x) && readNumber(s, k, out.y) && readNumber(s, k, out.z);
}

bool loadReplayPoses(const std::string& path, std::vector<ReplayPose>& outPoses) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        LOG_ERROR("Replay: failed to open %s", path.c_str());
        return false;
    }
    std::ostringstream ss; ss << in.rdbuf();
    std::string text = ss.str();

    // Slice out the poses array. The closing ']' must be matched against its
    // own '[' — naive find(']') would land on the inner "position": [x,y,z]
    // bracket and only the first pose's prefix would be parsed.
    size_t arr = text.find("\"poses\"");
    if (arr == std::string::npos) { LOG_ERROR("Replay: no 'poses' field in %s", path.c_str()); return false; }
    size_t lb = text.find('[', arr);
    if (lb == std::string::npos) { LOG_ERROR("Replay: no '[' after 'poses'"); return false; }
    int depth = 1;
    size_t rb = std::string::npos;
    for (size_t k = lb + 1; k < text.size(); ++k) {
        if (text[k] == '[') ++depth;
        else if (text[k] == ']') {
            if (--depth == 0) { rb = k; break; }
        }
    }
    if (rb == std::string::npos) {
        LOG_ERROR("Replay: unmatched '[' for poses array");
        return false;
    }
    std::string body = text.substr(lb + 1, rb - lb - 1);

    size_t cur = 0;
    while (true) {
        size_t obj_l = body.find('{', cur);
        if (obj_l == std::string::npos) break;
        size_t obj_r = body.find('}', obj_l);
        if (obj_r == std::string::npos) break;
        std::string obj = body.substr(obj_l, obj_r - obj_l + 1);

        ReplayPose p{};
        // Sensible defaults for any missing field.
        p.t = 0.0f; p.fovDeg = 60.0f; p.aspect = 16.0f / 9.0f;
        p.nearPlane = 0.001f; p.farPlane = 100.0f;
        readKeyedFloat (obj, "t",       p.t);
        readKeyedFloat3(obj, "position",p.position);
        readKeyedFloat (obj, "yaw",     p.yawDeg);
        readKeyedFloat (obj, "pitch",   p.pitchDeg);
        readKeyedFloat (obj, "fov_deg", p.fovDeg);
        readKeyedFloat (obj, "aspect",  p.aspect);
        readKeyedFloat (obj, "near",    p.nearPlane);
        readKeyedFloat (obj, "far",     p.farPlane);
        outPoses.push_back(p);
        cur = obj_r + 1;
    }
    return !outPoses.empty();
}
}  // namespace

void Application::runReplay() {
    std::vector<ReplayPose> poses;
    if (!loadReplayPoses(m_replayOpts.recordingPath, poses)) {
        LOG_ERROR("Replay: no usable poses in %s", m_replayOpts.recordingPath.c_str());
        return;
    }
    if (m_replayOpts.stride == 0) m_replayOpts.stride = 1;
    if (m_replayOpts.maxPoses > 0 && poses.size() > m_replayOpts.maxPoses) {
        poses.resize(m_replayOpts.maxPoses);
    }

    std::filesystem::create_directories(m_replayOpts.outDir);

    // Override aspect to match the actual render resolution. Otherwise a
    // recording made at one window size would render squashed at another.
    float renderAspect = m_height > 0 ? (float)m_width / (float)m_height : 1.0f;

    const uint32_t spp = m_replayOpts.sppPerPose > 0 ? m_replayOpts.sppPerPose : 1;
    LOG_INFO("Replay: %zu pose(s) (stride=%u → %zu frames), spp=%u, out=%s",
             poses.size(), m_replayOpts.stride,
             (poses.size() + m_replayOpts.stride - 1) / m_replayOpts.stride,
             spp, m_replayOpts.outDir.c_str());

    const auto totalStart = std::chrono::steady_clock::now();
    uint32_t saved = 0;

    for (size_t i = 0; i < poses.size(); i += m_replayOpts.stride) {
        if (glfwWindowShouldClose(m_window)) break;
        const ReplayPose& p = poses[i];

        // Inject pose. Aspect is overridden so headless render dimensions
        // dictate the projection regardless of what was recorded.
        m_camera.setPose(p.position, p.yawDeg, p.pitchDeg, p.fovDeg,
                         renderAspect, p.nearPlane, p.farPlane);

        // Each replay frame is independent — no temporal carry-over (matches
        // what the user expects when comparing 1spp renders across modes).
        m_renderer.resetAccumulation();
        m_renderer.invalidateReSTIRHistory();
        m_renderer.markPipelineNeedsReset();

        // Pump until we've rendered the requested spp at this pose.
        while (m_renderer.getSampleCount() < spp) {
            uchar4* d_pbo = (uchar4*)m_display.mapForCUDA();
            renderSceneSample(d_pbo, true);
            m_display.unmapFromCUDA();
            m_display.present();
            m_camera.advanceFrame();
            m_frameIndex++;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::ostringstream name;
        name << m_replayOpts.outDir << "/frame_" << std::setw(6)
             << std::setfill('0') << saved << ".png";
        if (!m_display.saveToPNG(name.str())) {
            LOG_ERROR("Replay: failed to save %s", name.str().c_str());
        }
        ++saved;

        if (saved % 10 == 0 || saved == 1) {
            LOG_INFO("Replay: %u frames written", saved);
        }
    }

    const auto totalEnd = std::chrono::steady_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    LOG_INFO("Replay: done, %u frame(s) in %.2f s (%.1f ms/frame)",
             saved, totalMs / 1000.0, saved > 0 ? totalMs / saved : 0.0);

    glfwSetWindowShouldClose(m_window, true);
}

void Application::run() {
    if (!m_guiEnabled && m_replayEnabled) {
        runReplay();
    } else if (!m_guiEnabled && !m_headlessOutputPath.empty()) {
        runHeadless();
    } else {
        runGui();
    }
}

void Application::shutdown() {
    if (m_recording) stopRecording();
    m_display.waitIdle();
    m_renderer.shutdown();
    m_textures.freeAll();
    freeShEnvDevice();
    if (m_d_debugArrows) {
        cudaFree(m_d_debugArrows);
        m_d_debugArrows = nullptr;
        m_debugArrowCapacityPairs = 0;
    }
    m_gui.shutdown();
    m_display.shutdown();
    glfwDestroyWindow(m_window);
    glfwTerminate();
    LOG_INFO("Application shutdown");
}

void Application::startRecording() {
    m_recordedPath.clear();
    m_recordStartTime = glfwGetTime();
    m_recording = true;
    LOG_INFO("Recording: STARTED (press F5 again to stop)");
}

void Application::stopRecording() {
    m_recording = false;
    if (m_recordedPath.empty()) {
        LOG_WARN("Recording: STOPPED with 0 frames captured (nothing written)");
        return;
    }
    std::filesystem::create_directories("recordings");

    // Build a filename containing the local date+time so multiple recordings
    // don't clobber each other across sessions.
    std::time_t now = std::time(nullptr);
    std::tm tm_local{};
#ifdef _WIN32
    localtime_s(&tm_local, &now);
#else
    localtime_r(&now, &tm_local);
#endif
    std::ostringstream stamp;
    stamp << std::put_time(&tm_local, "%Y%m%d_%H%M%S");

    std::ostringstream path;
    path << "recordings/path_" << stamp.str() << ".json";

    std::ofstream out(path.str(), std::ios::binary);
    if (!out) {
        LOG_ERROR("Recording: failed to open %s for write", path.str().c_str());
        return;
    }

    // Hand-rolled JSON — avoids pulling in a json library for one writer.
    // Format matches what scripts/render_camera_path.py expects via --recording.
    out << "{\n";
    out << "  \"version\": 1,\n";
    out << "  \"recorded_frames\": " << m_recordedPath.size() << ",\n";
    out << "  \"duration_seconds\": " << m_recordedPath.back().t << ",\n";
    out << "  \"poses\": [\n";
    for (size_t i = 0; i < m_recordedPath.size(); ++i) {
        const RecordedPose& p = m_recordedPath[i];
        out << "    {"
            << "\"t\": "          << p.t          << ", "
            << "\"position\": ["  << p.position.x << ", " << p.position.y << ", " << p.position.z << "], "
            << "\"yaw\": "        << p.yaw        << ", "
            << "\"pitch\": "      << p.pitch      << ", "
            << "\"fov_deg\": "    << p.fovDeg     << ", "
            << "\"aspect\": "     << p.aspect     << ", "
            << "\"near\": "       << p.nearPlane  << ", "
            << "\"far\": "        << p.farPlane
            << "}";
        if (i + 1 < m_recordedPath.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();

    LOG_INFO("Recording: STOPPED, saved %zu frames (%.2f s) to %s",
             m_recordedPath.size(), m_recordedPath.back().t, path.str().c_str());
    m_recordedPath.clear();
}

// ── Animation playback driver ────────────────────────────────
// Called once per render frame BEFORE the path-trace launch when animation
// is active. Evaluates the AnimationClip at the current `m_animTime`,
// computes per-mesh world deltas, uploads them, runs the GPU pose-update
// kernel, and refits the OptiX GAS so the launch sees the new geometry.
//
// No-ops cleanly when the scene has no clip or playback is off; the engine
// falls back to the static rest pose written into d_positions at upload.
void Application::advanceAnimation(float stepSeconds) {
    if (!m_playAnimation) return;
    if (!m_sceneLoaded) return;
    if (!m_scene.hasAnimation()) return;

#ifdef PATHTRACER_OPTIX_ENABLED
    auto* optix = dynamic_cast<OptiXBackend*>(m_backend.get());
    if (!optix) {
        // Animation only supported on the OptiX backend (CUDA backend uses a
        // CPU-built BVH that we'd have to rebuild every frame; not worth it
        // for this pass). Silently do nothing on the CUDA backend so the
        // user gets a static render rather than a hard error.
        return;
    }

    DeviceScene& dsc = optix->deviceScene();
    if (!dsc.hasAnimation()) return;  // no animated meshes in this scene

    // Sample time = m_animStartTime + m_animTime. The very first call (after
    // load) sees m_animTime == 0 → exactly the user-supplied start time.
    // After rendering, we advance by stepSeconds for the next call, so frame
    // N (0-indexed) is at time (animStartTime + N * stepSeconds).
    const float t = m_animStartTime + m_animTime;
    const AnimationClip& clip = m_scene.getAnimations().front();

    // 1) Evaluate per-node local + world transforms at this time.
    evalAnimation(m_scene, clip, t, m_animLocalScratch, m_animWorldScratch);

    // 2) Per-mesh delta = worldCurr * worldRest^-1.
    computeMeshDeltas(m_scene, m_animWorldScratch, m_animMeshDeltaScratch);

    // 3) Cofactor of the upper-3x3 of each delta, for normal transformation.
    computeNormalMats(m_animMeshDeltaScratch, m_animNormalMatScratch);

    // 4) Push to GPU + run the pose-update kernel.
    PoseUpdateData& pose = dsc.pose();
    poseUpdateUploadDeltas(pose, m_animMeshDeltaScratch, m_animNormalMatScratch);
    poseUpdateLaunch(pose, m_firstAnimFrame);
    m_firstAnimFrame = false;

    // 5) Refit the OptiX GAS so the next path-trace launch sees the new
    //    vertex positions.
    optix->refitGAS();

    // Advance for the next call. This ordering means: first invocation
    // renders at m_animStartTime, second at start + stepSeconds, etc.
    m_animTime += stepSeconds;

    // Note: caller already gated us on `sampleCount == 0` (see
    // renderSceneSample), so the accumulator is freshly reset from the
    // outer loop's camera-motion / replay-pose logic. We don't reset here;
    // doing so would also break GUI-driven playback where the user might
    // want to accumulate while paused.
#else
    (void)stepSeconds;
#endif
}

