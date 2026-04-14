#pragma once
#include "render/AccumulationBuffer.h"
#include "render/AuxBuffers.h"
#include "render/Tonemapping.h"
#include "core/Camera.h"
#include "gpu/DeviceScene.h"
#include <cstdint>

class RayTracingBackend;

class Renderer {
public:
    void init(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void resetAccumulation();

    // Render one progressive sample, tonemap into d_ldrOutput
    void renderFrame(
        const CameraParams& camera,
        const DeviceSceneData& scene,
        RayTracingBackend* backend,
        uchar4* d_ldrOutput,
        bool enableEnvironment,
        uint32_t maxBounces
    );

    uint32_t getSampleCount() const { return m_accumBuffer.getSampleCount(); }
    float getExposure() const { return m_exposure; }
    void setExposure(float exposure) { m_exposure = exposure; }
    ToneMappingMode getToneMappingMode() const { return m_toneMappingMode; }
    void setToneMappingMode(ToneMappingMode mode) { m_toneMappingMode = mode; }
    void shutdown();

private:
    AccumulationBuffer m_accumBuffer;
    AuxBuffers         m_auxBuffers;
    uint32_t m_width = 0, m_height = 0;
    float    m_exposure = 1.0f;
    ToneMappingMode m_toneMappingMode = ToneMappingMode::ACES;
};
