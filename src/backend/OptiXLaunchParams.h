#pragma once
#include "gpu/DeviceScene.h"
#include "core/Camera.h"
#include "render/AuxBuffers.h"
#include <cuda_runtime.h>

#ifdef __CUDACC__
#include <optix.h>
#else
// Host side: we only need OptixTraversableHandle as an opaque 64-bit handle.
typedef unsigned long long OptixTraversableHandle;
#endif

struct LaunchParams {
    DeviceSceneData scene;
    CameraParams    camera;

    float4*         accum;
    float4*         output;
    AuxBufferPtrs   aux;

    unsigned int    width;
    unsigned int    height;
    unsigned int    sampleIndex;
    unsigned int    maxBounces;
    unsigned int    spp;
    unsigned int    enableEnvironment;

    OptixTraversableHandle handle;
};
