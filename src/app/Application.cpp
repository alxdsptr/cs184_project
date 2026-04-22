#include "app/Application.h"
#include "scene/SceneLoader.h"
#include "util/Log.h"
#include "util/CudaCheck.h"
#ifdef PATHTRACER_OPTIX_ENABLED
#include "backend/OptiXBackend.h"
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <map>
#include <unordered_map>

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

    bool backendReady = false;
#ifdef PATHTRACER_OPTIX_ENABLED
    if (m_backendKind == 1) {
        auto optix = std::make_unique<OptiXBackend>();
        std::filesystem::path exeDir = std::filesystem::current_path();
        std::filesystem::path irPath = exeDir / "optix_programs.optixir";
        if (!std::filesystem::exists(irPath)) {
            // Fall back: some IDEs set CWD to project root, not exe dir.
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
        }
    }

    m_renderer.resetAccumulation();
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
        CameraParams camParams = m_camera.getParams(m_frameIndex);
        DeviceSceneData sceneData = m_backend->getSceneData();
        sceneData.envMapTex = m_envMapTex;
        sceneData.d_shEnvCoeffs = m_d_shEnvCoeffs;
        sceneData.envUseSH = (m_useSHEnvIrradiance && m_d_shEnvCoeffs) ? 1 : 0;
        m_renderer.renderFrame(camParams, sceneData, m_backend.get(), d_pbo,
                               m_enableEnvironment, m_maxBounces,
                               m_samplesPerFrame,
                               &m_display, m_frameIndex);
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
        m_camera.update(dt, m_input);

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
                modePtr, qualityPtr, rrW, rrH);
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
        }
        if (envChanged) {
            m_renderer.resetAccumulation();
        }
        m_gui.endFrame();

        m_display.present();
        m_frameIndex++;
    }
}

void Application::runHeadless() {
    const auto totalStart = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(m_window)) {
        uchar4* d_pbo = (uchar4*)m_display.mapForCUDA();
        renderSceneSample(d_pbo, true);
        m_display.unmapFromCUDA();

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

        m_frameIndex++;
    }
}

void Application::run() {
    if (!m_guiEnabled && !m_headlessOutputPath.empty()) {
        runHeadless();
    } else {
        runGui();
    }
}

void Application::shutdown() {
    m_display.waitIdle();
    m_renderer.shutdown();
    m_textures.freeAll();
    freeShEnvDevice();
    m_gui.shutdown();
    m_display.shutdown();
    glfwDestroyWindow(m_window);
    glfwTerminate();
    LOG_INFO("Application shutdown");
}
