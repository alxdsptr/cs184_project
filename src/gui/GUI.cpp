#include "gui/GUI.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

void GUI::init(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    m_initialized = true;
}

void GUI::beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

bool GUI::render(float fps, uint32_t sampleCount, uint32_t width, uint32_t height,
                 bool& enableEnvironment, bool& invertMouseY, uint32_t& maxBounces,
                 float& moveSpeed,
                 char* envMapPathBuf, size_t envMapPathBufSize, bool& loadEnvMapRequested) {
    bool changed = false;
    loadEnvMapRequested = false;

    // Overlay window: top-left corner
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("Stats", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Samples: %u", sampleCount);
    ImGui::Text("Resolution: %ux%u", width, height);
    if (ImGui::Checkbox("Environment Light", &enableEnvironment)) {
        changed = true;
    }
    ImGui::Checkbox("Invert Mouse Y", &invertMouseY);

    ImGui::SliderFloat("Move Speed", &moveSpeed, 0.05f, 200.0f, "%.2f");
    ImGui::TextUnformatted("Adjust speed keys: [ / ]");

    int bounceValue = (int)maxBounces;
    if (ImGui::SliderInt("Max Bounces", &bounceValue, 1, 16)) {
        if (bounceValue < 1) {
            bounceValue = 1;
        }
        maxBounces = (uint32_t)bounceValue;
        changed = true;
    }

    // HDR environment map
    ImGui::Separator();
    ImGui::Text("HDR Environment Map");
    ImGui::PushItemWidth(200);
    ImGui::InputText("##hdr_path", envMapPathBuf, envMapPathBufSize);
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Load HDR")) {
        loadEnvMapRequested = true;
    }

    ImGui::End();
    return changed;
}

void GUI::endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool GUI::wantCaptureMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

bool GUI::wantCaptureKeyboard() const {
    return ImGui::GetIO().WantCaptureKeyboard;
}

void GUI::shutdown() {
    if (!m_initialized) return;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    m_initialized = false;
}
