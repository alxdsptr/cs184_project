#include "gui/GUI.h"
#include "display/VulkanDisplay.h"
#include "util/Log.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

static void imguiVulkanCheck(VkResult err) {
    if (err != VK_SUCCESS) {
        LOG_ERROR("ImGui Vulkan err: %d", (int)err);
    }
}

// Called by VulkanDisplay from inside the render pass each frame.
static void recordImGuiDraw(VkCommandBuffer cmd, void* /*user*/) {
    ImDrawData* dd = ImGui::GetDrawData();
    if (dd) {
        ImGui_ImplVulkan_RenderDrawData(dd, cmd);
    }
}

void GUI::init(GLFWwindow* window, VulkanDisplay* display) {
    m_display = display;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo init{};
    init.Instance       = display->instance();
    init.PhysicalDevice = display->physicalDevice();
    init.Device         = display->device();
    init.QueueFamily    = display->graphicsQueueFamily();
    init.Queue          = display->graphicsQueue();
    init.PipelineCache  = VK_NULL_HANDLE;
    init.DescriptorPool = display->descriptorPool();
    init.RenderPass     = display->renderPass();
    init.Subpass        = 0;
    init.MinImageCount  = display->minImageCount();
    init.ImageCount     = display->swapchainImageCount();
    init.MSAASamples    = VK_SAMPLE_COUNT_1_BIT;
    init.Allocator      = nullptr;
    init.CheckVkResultFn = imguiVulkanCheck;
    ImGui_ImplVulkan_Init(&init);

    // ImGui 1.91 uploads fonts lazily on first render; no manual upload needed.
    display->setImGuiRecorder(recordImGuiDraw, this);

    m_initialized = true;
}

void GUI::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

bool GUI::render(float fps, uint32_t sampleCount, uint32_t width, uint32_t height,
                 bool& enableEnvironment, bool& invertMouseY, uint32_t& maxBounces,
                 float& exposure, int& toneMappingMode,
                 float& moveSpeed,
                 char* envMapPathBuf, size_t envMapPathBufSize, bool& loadEnvMapRequested,
                 bool& debugShowPointLights,
                 bool& skipEmissiveInNEE,
                 int* renderMode,
                 int* dlssQuality,
                 uint32_t renderResW,
                 uint32_t renderResH) {
    bool changed = false;
    loadEnvMapRequested = false;

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
        if (bounceValue < 1) bounceValue = 1;
        maxBounces = (uint32_t)bounceValue;
        changed = true;
    }

    ImGui::Separator();
    ImGui::Text("Tone Mapping");
    ImGui::SliderFloat("Exposure", &exposure, 0.05f, 8.0f, "%.2f");
    const char* toneMappingItems[] = {"None", "Reinhard", "ACES"};
    ImGui::Combo("Mode", &toneMappingMode, toneMappingItems, IM_ARRAYSIZE(toneMappingItems));

    if (renderMode) {
        ImGui::Separator();
        ImGui::Text("Render Mode");
        // Order MUST match Renderer::Mode enum: Native, NRDOnly, NRDDLSS, DLSSOnly.
        const char* modes[] = {"Native", "NRD (denoise)", "NRD + DLSS", "DLSS only (no NRD)"};
        if (ImGui::Combo("Pipeline", renderMode, modes, IM_ARRAYSIZE(modes))) {
            changed = true;
        }
        // Both DLSS-using modes show the quality dropdown.
        if ((*renderMode == 2 || *renderMode == 3) && dlssQuality) {
            const char* q[] = {"Performance", "Balanced", "Quality", "DLAA"};
            if (ImGui::Combo("DLSS Quality", dlssQuality, q, IM_ARRAYSIZE(q))) {
                changed = true;
            }
        }
        if (*renderMode != 0 && renderResW && renderResH) {
            ImGui::Text("Render res: %ux%u", renderResW, renderResH);
        }
    }

    ImGui::Separator();
    ImGui::Text("Lighting Debug");
    ImGui::Checkbox("Show point lights (click box to toggle)", &debugShowPointLights);
    if (ImGui::Checkbox("Skip emissive in NEE", &skipEmissiveInNEE)) {
        changed = true;
    }

    ImGui::Separator();
    ImGui::Text("HDR Environment Map");
    ImGui::PushItemWidth(200);
    ImGui::InputText("##hdr_path", envMapPathBuf, envMapPathBufSize);
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Load HDR")) loadEnvMapRequested = true;

    ImGui::End();
    return changed;
}

void GUI::endFrame() {
    // Render data is recorded by VulkanDisplay's per-frame callback.
    ImGui::Render();
}

bool GUI::wantCaptureMouse() const    { return ImGui::GetIO().WantCaptureMouse; }
bool GUI::wantCaptureKeyboard() const { return ImGui::GetIO().WantCaptureKeyboard; }

void GUI::shutdown() {
    if (!m_initialized) return;
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    m_initialized = false;
}
