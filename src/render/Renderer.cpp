#include "render/Renderer.h"
#include "render/Tonemapping.h"
#include "backend/RayTracingBackend.h"

void Renderer::init(uint32_t width, uint32_t height) {
    m_width  = width;
    m_height = height;
    m_accumBuffer.init(width, height);
    m_auxBuffers.init(width, height);
}

void Renderer::resize(uint32_t width, uint32_t height) {
    if (width == m_width && height == m_height) return;
    m_width  = width;
    m_height = height;
    m_accumBuffer.resize(width, height);
    m_auxBuffers.resize(width, height);
}

void Renderer::resetAccumulation() {
    m_accumBuffer.reset();
}

void Renderer::renderFrame(
    const CameraParams& camera,
    const DeviceSceneData& scene,
    RayTracingBackend* backend,
    uchar4* d_ldrOutput,
    bool enableEnvironment)
{
    uint32_t sampleIndex = m_accumBuffer.getSampleCount();

    // Path trace
    backend->launchPathTrace(
        scene, camera,
        m_accumBuffer.getAccumBuffer(),
        m_accumBuffer.getOutputBuffer(),
        m_auxBuffers.getPtrs(),
        m_width, m_height, sampleIndex,
        enableEnvironment
    );

    // Tonemap HDR -> LDR into the display PBO
    launchTonemapKernel(
        m_accumBuffer.getOutputBuffer(),
        d_ldrOutput,
        m_width, m_height,
        m_exposure
    );

    m_accumBuffer.incrementSamples();
}

void Renderer::shutdown() {
    m_accumBuffer.free();
    m_auxBuffers.free();
}
