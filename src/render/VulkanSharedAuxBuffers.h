#pragma once

// Vulkan-side aux buffers consumed by NRD (RELAX_DIFFUSE_SPECULAR) and later
// by the composite/DLSS passes. Each image is backed by exported memory so
// the path-trace kernel can write it directly via cudaSurfaceObject_t.
//
// Distinct from the CUDA-only `AuxBuffers` used by the Native fallback path.
// Only present when PATHTRACER_NRD_DLSS_ENABLED.

#include "interop/VulkanImageInterop.h"
#include <cstdint>

// Packed handles passed down to split path-trace kernel.
struct SharedAuxSurfaces {
    cudaSurfaceObject_t diffuseRadianceHitDist = 0; // RGBA16F, YCoCg+hitT packed (NRD frontend)
    cudaSurfaceObject_t specularRadianceHitDist = 0; // RGBA16F
    cudaSurfaceObject_t normalRoughness         = 0; // RGBA8_UNORM, NRD encoding 2 / roughness encoding 1
    cudaSurfaceObject_t viewZ                   = 0; // R32F (signed view-space Z, linear)
    cudaSurfaceObject_t motionVectors           = 0; // RG16F, screen-space pixel deltas
    cudaSurfaceObject_t albedo                  = 0; // RGBA8_UNORM, diffuse reflectance (demodulation factor)
    cudaSurfaceObject_t emissive                = 0; // RGBA16F, linear HDR
    // DLSSOnly: full HDR composite produced by the path-trace kernel directly,
    // bypassing NRD/composite. Render-resolution input to DLSS upscale.
    // Also reused as DLSS-RR's noisy color input (un-demodulated diff*albedo +
    // spec + emissive — what RR's DL prior denoises directly).
    cudaSurfaceObject_t hdrColor                = 0; // RGBA16F, linear HDR
    // DLSS requires NDC depth (clip.z / clip.w), not linear viewZ — see the
    // NRD-Sample's DlssBefore.cs.hlsl comment "SR doesn't support linear viewZ".
    // Separate from `viewZ` so NRD can keep consuming linear-meter viewZ.
    cudaSurfaceObject_t ndcDepth                = 0; // R32F, clip.z / clip.w in [0,1]
    // DLSS-RR specific guides. The DLSS-RR Integration Guide §3.4 disallows
    // RGBA8_UNORM for normals (RGB16F or 32F required) and requires per-pixel
    // specular albedo (EnvBRDFApprox2) and a per-pixel specular hit distance
    // — none of which the NRD path provides. Allocate these once, populate
    // them in Mode::DLSSRR, and leave them at zero in NRD modes.
    cudaSurfaceObject_t worldNormalRoughness    = 0; // RGBA16F: world normal.xyz + linear roughness in .w
    cudaSurfaceObject_t specAlbedo              = 0; // RGBA16F: EnvBRDFApprox2(F0, alpha, NoV)
    cudaSurfaceObject_t specHitT                = 0; // R32F:    world-space distance from primary hit to first secondary surface
};

class VulkanSharedAuxBuffers {
public:
    VulkanSharedAuxBuffers() = default;
    ~VulkanSharedAuxBuffers() = default;
    VulkanSharedAuxBuffers(const VulkanSharedAuxBuffers&) = delete;
    VulkanSharedAuxBuffers& operator=(const VulkanSharedAuxBuffers&) = delete;

    // Returns false on any allocation failure — caller may fall back to Native mode.
    bool create(VkDevice device, VkPhysicalDevice phys,
                uint32_t width, uint32_t height);

    // Safe to call multiple times / in any state.
    void destroy();

    bool resize(VkDevice device, VkPhysicalDevice phys,
                uint32_t width, uint32_t height);

    SharedAuxSurfaces surfaces() const;

    // Accessors for Vulkan compute (NRD) and composite pass.
    const SharedVulkanImage& diffuseRadianceHitDist() const { return m_diffRadHitDist; }
    const SharedVulkanImage& specularRadianceHitDist() const { return m_specRadHitDist; }
    const SharedVulkanImage& normalRoughness()         const { return m_normalRoughness; }
    const SharedVulkanImage& viewZ()                   const { return m_viewZ; }
    const SharedVulkanImage& motionVectors()           const { return m_motionVectors; }
    const SharedVulkanImage& albedo()                  const { return m_albedo; }
    const SharedVulkanImage& emissive()                const { return m_emissive; }
    const SharedVulkanImage& hdrColor()                const { return m_hdrColor; }
    const SharedVulkanImage& ndcDepth()                const { return m_ndcDepth; }
    const SharedVulkanImage& worldNormalRoughness()    const { return m_worldNormalRoughness; }
    const SharedVulkanImage& specAlbedo()              const { return m_specAlbedo; }
    const SharedVulkanImage& specHitT()                const { return m_specHitT; }

    uint32_t width()  const { return m_width; }
    uint32_t height() const { return m_height; }
    bool valid()      const { return m_width != 0; }

private:
    SharedVulkanImage m_diffRadHitDist;
    SharedVulkanImage m_specRadHitDist;
    SharedVulkanImage m_normalRoughness;
    SharedVulkanImage m_viewZ;
    SharedVulkanImage m_motionVectors;
    SharedVulkanImage m_albedo;
    SharedVulkanImage m_emissive;
    SharedVulkanImage m_hdrColor;
    SharedVulkanImage m_ndcDepth;
    SharedVulkanImage m_worldNormalRoughness;
    SharedVulkanImage m_specAlbedo;
    SharedVulkanImage m_specHitT;
    uint32_t m_width  = 0;
    uint32_t m_height = 0;
};
