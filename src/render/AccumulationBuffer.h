#pragma once
#include <cuda_runtime.h>
#include <cstdint>

class AccumulationBuffer {
public:
    void init(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void reset();
    void free();

    float4* getAccumBuffer() const { return m_accumBuffer; }
    float4* getOutputBuffer() const { return m_outputBuffer; }
    uint32_t getSampleCount() const { return m_sampleCount; }
    void incrementSamples() { m_sampleCount++; }

private:
    float4* m_accumBuffer  = nullptr;
    float4* m_outputBuffer = nullptr;
    uint32_t m_width = 0, m_height = 0;
    uint32_t m_sampleCount = 0;
};
