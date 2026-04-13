#ifndef CGL_CUDA_BUFFERS_H
#define CGL_CUDA_BUFFERS_H

#include "cuda_types.h"

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

// Forward declare curandState for non-CUDA compilation units
#ifndef __CUDACC__
struct curandState;
#endif

struct CUDASceneBuffers {
    // Scene data on device
    CUDABVHNode*    d_bvh_nodes;
    CUDAPrimitive*  d_primitives;
    CUDAMaterial*   d_materials;
    CUDALight*      d_lights;

    int num_bvh_nodes;
    int num_primitives;
    int num_materials;
    int num_lights;

    // Output buffers on device
    float*          d_hdr_buffer;    // 3 floats per pixel (RGB)
    int*            d_sample_count;  // per-pixel sample count
    curandState*    d_rng_states;    // per-pixel RNG state

    int output_w;
    int output_h;

    CUDASceneBuffers()
        : d_bvh_nodes(nullptr), d_primitives(nullptr)
        , d_materials(nullptr), d_lights(nullptr)
        , num_bvh_nodes(0), num_primitives(0)
        , num_materials(0), num_lights(0)
        , d_hdr_buffer(nullptr), d_sample_count(nullptr)
        , d_rng_states(nullptr)
        , output_w(0), output_h(0)
    {}
};

// Defined in cuda_pathtracer.cu
void cuda_upload_scene(CUDASceneBuffers& buf,
                       const CUDABVHNode* bvh_nodes, int num_nodes,
                       const CUDAPrimitive* primitives, int num_prims,
                       const CUDAMaterial* materials, int num_mats,
                       const CUDALight* lights, int num_lights);

void cuda_alloc_output(CUDASceneBuffers& buf, int w, int h);
void cuda_free_all(CUDASceneBuffers& buf);

void cuda_render(CUDASceneBuffers& buf,
                 const CUDACameraParams& cam,
                 const CUDARenderParams& params);

void cuda_download_results(const CUDASceneBuffers& buf,
                           float* host_hdr, int* host_sample_count,
                           int w, int h);

#endif // CGL_CUDA_BUFFERS_H
