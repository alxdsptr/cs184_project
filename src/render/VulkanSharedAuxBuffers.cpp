#include "render/VulkanSharedAuxBuffers.h"

// Channel format table matches the VkFormat / NRD ResourceType pairs for
// RELAX_DIFFUSE_SPECULAR. All images are 2D with 1 mip / 1 layer.
namespace {

cudaChannelFormatDesc chanRGBA16F() {
    return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
}
cudaChannelFormatDesc chanRGBA8Unorm() {
    // cudaReadModeNormalizedFloat doesn't matter for surface writes; only the
    // size fields do. 4×8 unsigned bytes.
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
}
cudaChannelFormatDesc chanR32F() {
    return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
}
cudaChannelFormatDesc chanRG16F() {
    return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindFloat);
}

// NRD reads these as STORAGE (compute R/W); composite samples them as SAMPLED.
// The SharedVulkanImage constructor ORs in both regardless, so usage passed
// here only needs to cover any extra bits (e.g. TRANSFER_* for debugging).
constexpr VkImageUsageFlags kDefaultUsage = 0;

} // namespace

bool VulkanSharedAuxBuffers::create(VkDevice device, VkPhysicalDevice phys,
                                    uint32_t width, uint32_t height)
{
    m_width  = width;
    m_height = height;

    const bool ok =
        m_diffRadHitDist.create(device, phys, width, height,
                                VK_FORMAT_R16G16B16A16_SFLOAT, kDefaultUsage, chanRGBA16F())
     && m_specRadHitDist.create(device, phys, width, height,
                                VK_FORMAT_R16G16B16A16_SFLOAT, kDefaultUsage, chanRGBA16F())
     && m_normalRoughness.create(device, phys, width, height,
                                 VK_FORMAT_R8G8B8A8_UNORM, kDefaultUsage, chanRGBA8Unorm())
     && m_viewZ.create(device, phys, width, height,
                       VK_FORMAT_R32_SFLOAT, kDefaultUsage, chanR32F())
     && m_motionVectors.create(device, phys, width, height,
                               VK_FORMAT_R16G16_SFLOAT, kDefaultUsage, chanRG16F())
     && m_albedo.create(device, phys, width, height,
                        VK_FORMAT_R8G8B8A8_UNORM, kDefaultUsage, chanRGBA8Unorm())
     && m_emissive.create(device, phys, width, height,
                          VK_FORMAT_R16G16B16A16_SFLOAT, kDefaultUsage, chanRGBA16F())
     && m_hdrColor.create(device, phys, width, height,
                          VK_FORMAT_R16G16B16A16_SFLOAT, kDefaultUsage, chanRGBA16F())
     && m_ndcDepth.create(device, phys, width, height,
                          VK_FORMAT_R32_SFLOAT, kDefaultUsage, chanR32F());

    if (!ok) {
        destroy();
        return false;
    }
    return true;
}

void VulkanSharedAuxBuffers::destroy() {
    m_diffRadHitDist.destroy();
    m_specRadHitDist.destroy();
    m_normalRoughness.destroy();
    m_viewZ.destroy();
    m_motionVectors.destroy();
    m_albedo.destroy();
    m_emissive.destroy();
    m_hdrColor.destroy();
    m_ndcDepth.destroy();
    m_width = m_height = 0;
}

bool VulkanSharedAuxBuffers::resize(VkDevice device, VkPhysicalDevice phys,
                                    uint32_t width, uint32_t height)
{
    if (width == m_width && height == m_height) return true;
    destroy();
    return create(device, phys, width, height);
}

SharedAuxSurfaces VulkanSharedAuxBuffers::surfaces() const {
    SharedAuxSurfaces s{};
    s.diffuseRadianceHitDist  = m_diffRadHitDist.surface();
    s.specularRadianceHitDist = m_specRadHitDist.surface();
    s.normalRoughness         = m_normalRoughness.surface();
    s.viewZ                   = m_viewZ.surface();
    s.motionVectors           = m_motionVectors.surface();
    s.albedo                  = m_albedo.surface();
    s.emissive                = m_emissive.surface();
    s.hdrColor                = m_hdrColor.surface();
    s.ndcDepth                = m_ndcDepth.surface();
    return s;
}
