#include "postfx/CompositePass.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct PushBlock {
    float exposure;
    int   tonemapMode;
};

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) return {};
    size_t size = (size_t)f.tellg();
    std::vector<char> bytes(size);
    f.seekg(0);
    f.read(bytes.data(), size);
    return bytes;
}

const char* modeFragShader(CompositePass::Mode m) {
    switch (m) {
        case CompositePass::COMPOSITE_TONEMAP:    return "composite_tonemap.frag.spv";
        case CompositePass::COMPOSITE_LINEAR_HDR: return "composite_linear.frag.spv";
        case CompositePass::TONEMAP_ONLY:         return "tonemap_only.frag.spv";
    }
    return nullptr;
}

} // namespace

CompositePass::~CompositePass() { shutdown(); }

VkShaderModule CompositePass::loadSpv(const char* path) {
    auto bytes = readFile(path);
    if (bytes.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(bytes.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    vkCreateShaderModule(m_device, &ci, nullptr, &mod);
    return mod;
}

void CompositePass::createSampler() {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    vkCreateSampler(m_device, &sci, nullptr, &m_sampler);
}

bool CompositePass::init(VkDevice device, VkRenderPass renderPass, Mode mode,
                         VkFormat outputFormat, const char* spirvDir)
{
    (void)outputFormat;
    m_device = device;
    m_renderPass = renderPass;
    m_mode   = mode;

    m_inputCount = (mode == TONEMAP_ONLY) ? 1u : 4u;
    m_usesPush   = (mode != COMPOSITE_LINEAR_HDR);  // linear HDR composite has no push constants

    createSampler();

    // ── Descriptor set layout: m_inputCount combined-image-sampler bindings.
    std::vector<VkDescriptorSetLayoutBinding> bindings(m_inputCount);
    for (uint32_t i = 0; i < m_inputCount; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = m_inputCount;
    dslci.pBindings    = bindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &dslci, nullptr, &m_setLayout) != VK_SUCCESS) {
        return false;
    }

    // ── Descriptor pool + set (size-1 pool, we write into it on every input change).
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_inputCount};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets       = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &poolSize;
    pci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    if (vkCreateDescriptorPool(m_device, &pci, nullptr, &m_pool) != VK_SUCCESS) return false;

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = m_pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &m_setLayout;
    if (vkAllocateDescriptorSets(m_device, &ai, &m_set) != VK_SUCCESS) return false;

    // ── Pipeline layout.
    VkPushConstantRange pr{};
    pr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pr.offset     = 0;
    pr.size       = sizeof(PushBlock);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &m_setLayout;
    if (m_usesPush) {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pr;
    }
    if (vkCreatePipelineLayout(m_device, &plci, nullptr, &m_pipelineLayout) != VK_SUCCESS) return false;

    // ── Shader modules (vertex reuses the shared fullscreen quad).
    std::string vertPath = std::string(spirvDir) + "/fullscreen_quad_vk.vert.spv";
    std::string fragPath = std::string(spirvDir) + "/" + modeFragShader(mode);
    VkShaderModule vert = loadSpv(vertPath.c_str());
    VkShaderModule frag = loadSpv(fragPath.c_str());
    if (!vert || !frag) {
        if (vert) vkDestroyShaderModule(m_device, vert, nullptr);
        if (frag) vkDestroyShaderModule(m_device, frag, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    // No vertex input — the vertex shader derives positions from gl_VertexIndex.
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments    = &cba;

    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.stageCount          = 2;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState   = &ms;
    gpci.pColorBlendState    = &cb;
    gpci.pDynamicState       = &dyn;
    gpci.layout              = m_pipelineLayout;
    gpci.renderPass          = renderPass;
    gpci.subpass             = 0;

    VkResult r = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &gpci, nullptr, &m_pipeline);

    vkDestroyShaderModule(m_device, vert, nullptr);
    vkDestroyShaderModule(m_device, frag, nullptr);
    return r == VK_SUCCESS;
}

void CompositePass::setInputs(VkImageView a, VkImageView b, VkImageView c, VkImageView d) {
    m_inputs[0] = a;
    m_inputs[1] = b;
    m_inputs[2] = c;
    m_inputs[3] = d;
    m_setDirty = true;
}

void CompositePass::writeDescriptorSet() {
    VkDescriptorImageInfo imgs[4]{};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < m_inputCount; ++i) {
        imgs[i].sampler     = m_sampler;
        imgs[i].imageView   = m_inputs[i];
        imgs[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        writes[i].sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet     = m_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &imgs[i];
    }
    vkUpdateDescriptorSets(m_device, m_inputCount, writes, 0, nullptr);
    m_setDirty = false;
}

void CompositePass::record(VkCommandBuffer cmd, VkExtent2D extent,
                           float exposure, int tonemapMode)
{
    if (m_setDirty) writeDescriptorSet();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    VkViewport vp{};
    vp.width = (float)extent.width;
    vp.height = (float)extent.height;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0, 0}, extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &m_set, 0, nullptr);
    if (m_usesPush) {
        PushBlock pb{ exposure, tonemapMode };
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pb), &pb);
    }
    // Fullscreen triangle-strip: 4 verts, 2 tris. (Matches fullscreen_quad_vk.vert.)
    vkCmdDraw(cmd, 4, 1, 0, 0);
}

void CompositePass::shutdown() {
    if (m_pipeline)       { vkDestroyPipeline(m_device, m_pipeline, nullptr); m_pipeline = VK_NULL_HANDLE; }
    if (m_pipelineLayout) { vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr); m_pipelineLayout = VK_NULL_HANDLE; }
    if (m_pool)           { vkDestroyDescriptorPool(m_device, m_pool, nullptr); m_pool = VK_NULL_HANDLE; }
    if (m_setLayout)      { vkDestroyDescriptorSetLayout(m_device, m_setLayout, nullptr); m_setLayout = VK_NULL_HANDLE; }
    if (m_sampler)        { vkDestroySampler(m_device, m_sampler, nullptr); m_sampler = VK_NULL_HANDLE; }
    m_set = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}
