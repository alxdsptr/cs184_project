#include "app/Application.h"
#include "scene/SceneLoader.h"
#include "util/Log.h"
#include "util/CudaCheck.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <filesystem>
#include <iomanip>
#include <sstream>

static void glfwErrorCb(int code, const char* msg) {
    LOG_ERROR("GLFW [%d]: %s", code, msg);
}

bool Application::init(uint32_t width, uint32_t height, const std::string& title) {
    m_width  = width;
    m_height = height;

    // GLFW
    glfwSetErrorCallback(glfwErrorCb);
    if (!glfwInit()) { LOG_ERROR("glfwInit failed"); return false; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!m_window) { LOG_ERROR("glfwCreateWindow failed"); return false; }
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(0);

    // GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { LOG_ERROR("glewInit failed"); return false; }

    // CUDA device
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { LOG_ERROR("No CUDA devices"); return false; }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    LOG_INFO("CUDA device: %s (SM %d.%d, %zu MB)",
             prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024*1024));
    CUDA_CHECK(cudaSetDevice(0));

    // Initialize subsystems
    m_display.init(width, height);
    m_gui.init(m_window);
    m_renderer.init(width, height);
    m_backend = std::make_unique<CUDABackend>();

    m_camera.init(
        make_float3(0, 1, 3),
        make_float3(0, 0, 0),
        60.0f,
        (float)width / height
    );

    m_enableEnvironment = false;

    m_lastFrameTime = (float)glfwGetTime();
    LOG_INFO("Application initialized (%ux%u)", width, height);
    return true;
}

bool Application::loadScene(const std::string& path) {
    m_scene = Scene{};
    if (!SceneLoader::load(path, m_scene)) {
        LOG_ERROR("Failed to load scene: %s", path.c_str());
        return false;
    }

    // Load textures
    auto& materials = m_scene.getMaterials();
    for (auto& mat : materials) {
        // Texture loading is handled but we store the objects per-material
        // The TextureManager loads and the DeviceScene upload will set them
        // For now, textures will be loaded when the scene is uploaded
    }

    // Build acceleration structure (uploads geometry + builds BVH)
    m_backend->buildAccelerationStructure(m_scene);
    m_sceneLoaded = true;
    m_renderer.resetAccumulation();

    // Auto-position camera based on scene bounds
    // For now, keep default camera position
    return true;
}

void Application::processInput() {
    m_input = InputState{};

    if (!m_gui.wantCaptureKeyboard()) {
        m_input.forward  = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
        m_input.backward = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
        m_input.left     = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
        m_input.right    = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
        m_input.up       = glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS;
        m_input.down     = glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
    }

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

void Application::run() {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();

        // Timing
        float now = (float)glfwGetTime();
        float dt  = now - m_lastFrameTime;
        m_lastFrameTime = now;

        // FPS counter
        m_fpsFrames++;
        m_fpsTimer += dt;
        if (m_fpsTimer >= 0.5f) {
            m_fps = (float)m_fpsFrames / m_fpsTimer;
            m_fpsFrames = 0;
            m_fpsTimer  = 0.0f;
        }

        // Handle resize
        int fbW, fbH;
        glfwGetFramebufferSize(m_window, &fbW, &fbH);
        if (fbW > 0 && fbH > 0 && ((uint32_t)fbW != m_width || (uint32_t)fbH != m_height)) {
            m_width  = (uint32_t)fbW;
            m_height = (uint32_t)fbH;
            m_display.resize(m_width, m_height);
            m_renderer.resize(m_width, m_height);
            m_camera.setAspect((float)m_width / m_height);
            m_renderer.resetAccumulation();
            glViewport(0, 0, fbW, fbH);
        }

        // Input
        processInput();
        m_camera.update(dt, m_input);
        if (m_camera.hasMoved()) {
            m_renderer.resetAccumulation();
        }

        // ESC to close
        if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(m_window, true);

        bool f12Down = glfwGetKey(m_window, GLFW_KEY_F12) == GLFW_PRESS;
        bool saveScreenshot = f12Down && !m_prevF12Down;
        m_prevF12Down = f12Down;

        // ── Render ─────────────────────────────────────────
        uchar4* d_pbo = (uchar4*)m_display.mapForCUDA();

        if (m_sceneLoaded) {
            CameraParams camParams = m_camera.getParams(m_frameIndex);
            DeviceSceneData sceneData = m_backend->getSceneData();
            m_renderer.renderFrame(camParams, sceneData, m_backend.get(), d_pbo, m_enableEnvironment);
        } else {
            // No scene: dark gray
            CUDA_CHECK(cudaMemset(d_pbo, 40, m_width * m_height * sizeof(uchar4)));
        }

        m_display.unmapFromCUDA();

        // Display
        glClear(GL_COLOR_BUFFER_BIT);
        m_display.present();

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

        // GUI overlay
        m_gui.beginFrame();
        bool envChanged = m_gui.render(
            m_fps,
            m_renderer.getSampleCount(),
            m_width,
            m_height,
            m_enableEnvironment,
            m_invertMouseY);
        if (envChanged) {
            m_renderer.resetAccumulation();
        }
        m_gui.endFrame();

        glfwSwapBuffers(m_window);

        m_frameIndex++;
    }
}

void Application::shutdown() {
    m_renderer.shutdown();
    m_textures.freeAll();
    m_gui.shutdown();
    m_display.shutdown();
    glfwDestroyWindow(m_window);
    glfwTerminate();
    LOG_INFO("Application shutdown");
}
