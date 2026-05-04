#include "gui/GUI.h"
#include "core/Camera.h"
#include "core/Math.h"
#include "display/VulkanDisplay.h"
#include "util/Log.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cmath>

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
                 int* renderMode,
                 int* dlssQuality,
                 uint32_t renderResW,
                 uint32_t renderResH,
                 int* debugNormalViz,
                 bool* enableNormalMap,
                 bool* showNormalArrows,
                 int*  normalArrowStride,
                 float* normalArrowLength,
                 bool* restirDIEnabled,
                 bool* restirGIEnabled,
                 bool* restirPTEnabled,
                 int*  restirPTPathLength) {
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
        // Order MUST match Renderer::Mode enum: Native, NRDOnly, NRDDLSS, DLSSOnly, DLSSRR.
        const char* modes[] = {
            "Native",
            "NRD (denoise)",
            "NRD + DLSS",
            "DLSS only (no NRD)",
            "DLSS-RR (Ray Reconstruction)"
        };
        if (ImGui::Combo("Pipeline", renderMode, modes, IM_ARRAYSIZE(modes))) {
            changed = true;
        }
        // All DLSS-using modes show the quality dropdown (DLSSRR uses the same
        // PerfQuality enum as DLSS-SR per RR §3.2).
        if ((*renderMode == 2 || *renderMode == 3 || *renderMode == 4) && dlssQuality) {
            const char* q[] = {"Performance", "Balanced", "Quality", "DLAA"};
            if (ImGui::Combo("DLSS Quality", dlssQuality, q, IM_ARRAYSIZE(q))) {
                changed = true;
            }
        }
        if (*renderMode != 0 && renderResW && renderResH) {
            ImGui::Text("Render res: %ux%u", renderResW, renderResH);
        }
    }

    if (debugNormalViz || enableNormalMap || showNormalArrows) {
        ImGui::Separator();
        ImGui::Text("Normal Map / Debug");
    }
    if (enableNormalMap) {
        if (ImGui::Checkbox("Enable normal maps", enableNormalMap)) {
            changed = true;
        }
    }
    if (debugNormalViz) {
        // Mutually-exclusive checkboxes: at most one debug mode active at a
        // time. Toggling any of them flags `changed=true` so the renderer
        // resets accumulation (debug output is deterministic but the accum
        // buffer is still holding the previous mode's values).
        bool vizN    = (*debugNormalViz == 1);
        bool vizHand = (*debugNormalViz == 2);
        bool vizBack = (*debugNormalViz == 3);
        if (ImGui::Checkbox("Viz: perturbed normal (RGB)", &vizN)) {
            *debugNormalViz = vizN ? 1 : 0;
            changed = true;
        }
        if (ImGui::Checkbox("Viz: tangent handedness", &vizHand)) {
            *debugNormalViz = vizHand ? 2 : 0;
            changed = true;
        }
        if (ImGui::Checkbox("Viz: back-face after perturb", &vizBack)) {
            *debugNormalViz = vizBack ? 3 : 0;
            changed = true;
        }
    }
    if (showNormalArrows) {
        // Overlay doesn't touch the path-traced accum buffer, so toggling it
        // doesn't need to reset accumulation — just enabling/disabling the
        // draw in this GUI call.
        ImGui::Checkbox("Overlay: normal arrows", showNormalArrows);
        if (*showNormalArrows) {
            if (normalArrowStride) {
                ImGui::SliderInt("Arrow stride (px)", normalArrowStride, 4, 128);
            }
            if (normalArrowLength) {
                ImGui::SliderFloat("Arrow length", normalArrowLength, 0.02f, 5.0f, "%.2f");
            }
        }
    }

    if (restirDIEnabled || restirGIEnabled || restirPTEnabled) {
        ImGui::Separator();
        ImGui::Text("ReSTIR");
        if (restirDIEnabled) {
            // Toggle the direct-lighting reservoir prepass. Disabling it
            // falls back to plain NEE at the primary hit.
            if (ImGui::Checkbox("ReSTIR DI", restirDIEnabled)) {
                changed = true;
            }
        }
        if (restirGIEnabled) {
            // Toggle the indirect-lighting reservoir prepass. Disabling it
            // restores the path tracer's own continuation bounces.
            if (ImGui::Checkbox("ReSTIR GI", restirGIEnabled)) {
                changed = true;
            }
        }
        if (restirPTEnabled) {
            // ReSTIR PT (Lin et al. 2022): multi-bounce path postfix per
            // reservoir. Subsumes GI when both are on (PT wins at consume
            // site to avoid double-counting). Path-length slider is the
            // number of bounces past the reconnection vertex.
            if (ImGui::Checkbox("ReSTIR PT", restirPTEnabled)) {
                changed = true;
            }
            if (*restirPTEnabled && restirPTPathLength) {
                if (ImGui::SliderInt("PT postfix bounces", restirPTPathLength, 0, 8)) {
                    if (*restirPTPathLength < 0) *restirPTPathLength = 0;
                    changed = true;
                }
            }
        }
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

// Project a world-space point through the viewProj matrix, returning whether
// the point is in front of the camera. `outScreen` is filled only when the
// point projects. `screenW/H` are the target pixel extents.
static bool projectToScreen(const float4x4& viewProj,
                            float3 worldPos,
                            uint32_t screenW, uint32_t screenH,
                            ImVec2& outScreen)
{
    float x = viewProj.m[0][0]*worldPos.x + viewProj.m[0][1]*worldPos.y + viewProj.m[0][2]*worldPos.z + viewProj.m[0][3];
    float y = viewProj.m[1][0]*worldPos.x + viewProj.m[1][1]*worldPos.y + viewProj.m[1][2]*worldPos.z + viewProj.m[1][3];
    float w = viewProj.m[3][0]*worldPos.x + viewProj.m[3][1]*worldPos.y + viewProj.m[3][2]*worldPos.z + viewProj.m[3][3];
    if (w < 1e-4f) return false;  // behind or at the camera
    float ndcX = x / w;
    float ndcY = y / w;
    outScreen = ImVec2(
        (ndcX * 0.5f + 0.5f) * (float)screenW,
        (1.0f - (ndcY * 0.5f + 0.5f)) * (float)screenH);
    return true;
}

void GUI::drawNormalArrowsOverlay(
    const float4* arrows, int gridW, int gridH,
    const CameraParams& camera,
    uint32_t screenW, uint32_t screenH,
    float arrowLengthWorld)
{
    if (!arrows || gridW <= 0 || gridH <= 0) return;

    // ImGui background draw list sits underneath any windows but on top of
    // the blitted path-traced image — perfect for our purpose. Window coords
    // match framebuffer coords here because we haven't applied DPI scaling.
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImU32 colTail = IM_COL32(255, 230, 60, 220);   // yellow stem
    const ImU32 colTip  = IM_COL32(255, 80,  80, 255);   // red arrowhead tick

    const int total = gridW * gridH;
    for (int i = 0; i < total; ++i) {
        float4 pos = arrows[2*i + 0];
        float4 nrm = arrows[2*i + 1];
        if (pos.w < 0.5f) continue;  // ray missed — no sample here

        float3 p0 = make_float3(pos.x, pos.y, pos.z);
        float3 p1 = make_float3(
            pos.x + nrm.x * arrowLengthWorld,
            pos.y + nrm.y * arrowLengthWorld,
            pos.z + nrm.z * arrowLengthWorld);

        ImVec2 s0, s1;
        if (!projectToScreen(camera.viewProjMatrix, p0, screenW, screenH, s0)) continue;
        if (!projectToScreen(camera.viewProjMatrix, p1, screenW, screenH, s1)) continue;

        // Stem
        dl->AddLine(s0, s1, colTail, 1.5f);
        // Small perpendicular tick at the tip to make orientation unambiguous
        // without computing a proper 3D arrowhead.
        float dx = s1.x - s0.x;
        float dy = s1.y - s0.y;
        float len = std::sqrt(dx*dx + dy*dy);
        if (len > 1e-3f) {
            float px = -dy / len * 4.0f;
            float py =  dx / len * 4.0f;
            dl->AddLine(ImVec2(s1.x - px, s1.y - py),
                        ImVec2(s1.x + px, s1.y + py),
                        colTip, 1.5f);
        }
    }
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
